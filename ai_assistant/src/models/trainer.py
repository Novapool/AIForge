import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd
from .model_manager import ProblemType

class ModelTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path("ai_assistant/models/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_epoch = 0
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

    def train_epoch(self, model, train_loader: DataLoader, criterion: nn.Module,
                   optimizer: optim.Optimizer, problem_type: ProblemType) -> Tuple[float, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, targets in train_loader:
            features, targets = features.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = model(features)
            
            loss, predictions = self._calculate_loss_and_predictions(
                outputs, targets, criterion, problem_type
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if problem_type != ProblemType.REGRESSION:
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if problem_type != ProblemType.REGRESSION else None
        
        return avg_loss, accuracy

    def validate(self, model, val_loader: DataLoader, criterion: nn.Module,
                problem_type: ProblemType) -> Tuple[float, float]:
        """Validate the model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = model(features)
                
                loss, predictions = self._calculate_loss_and_predictions(
                    outputs, targets, criterion, problem_type
                )
                
                total_loss += loss.item()
                if problem_type != ProblemType.REGRESSION:
                    correct += (predictions == targets).sum().item()
                    total += targets.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if problem_type != ProblemType.REGRESSION else None
        
        return avg_loss, accuracy

    def _calculate_loss_and_predictions(self, outputs, targets, criterion, problem_type):
        """Helper function to calculate loss and predictions based on problem type"""
        if problem_type == ProblemType.BINARY_CLASSIFICATION:
            outputs = outputs.squeeze()
            loss = criterion(outputs, targets)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
        elif problem_type == ProblemType.MULTICLASS_CLASSIFICATION:
            targets = targets.long()
            loss = criterion(outputs, targets)
            predictions = outputs.argmax(dim=1)
        else:  # REGRESSION
            outputs = outputs.squeeze()
            loss = criterion(outputs, targets)
            predictions = outputs
            
        return loss, predictions

    def train(self, model, dataloaders: Dict[str, DataLoader], problem_type: ProblemType,
             num_epochs: int = 100, learning_rate: float = 0.001,
             early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """Train the model"""
        checkpoint_path = self.checkpoint_dir / f"model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pth"
        
        # Initialize criterion and optimizer
        criterion = self._get_criterion(problem_type)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        patience_counter = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(
                model, dataloaders['train'], criterion, optimizer, problem_type
            )
            val_loss, val_acc = self.validate(
                model, dataloaders['val'], criterion, problem_type
            )
            
            # Update metrics history
            self._update_metrics(train_loss, val_loss, train_acc, val_acc)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            
            self.current_epoch = epoch
        
        # Load best model
        model.load_state_dict(torch.load(checkpoint_path))
        
        return self.metrics_history

    def _get_criterion(self, problem_type: ProblemType) -> nn.Module:
        """Get the appropriate criterion based on problem type"""
        if problem_type == ProblemType.REGRESSION:
            return nn.MSELoss()
        elif problem_type == ProblemType.BINARY_CLASSIFICATION:
            return nn.BCEWithLogitsLoss()
        else:  # MULTICLASS_CLASSIFICATION
            return nn.CrossEntropyLoss()

    def _update_metrics(self, train_loss, val_loss, train_acc, val_acc):
        """Update metrics history"""
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        if train_acc is not None:
            self.metrics_history['train_accuracy'].append(train_acc)
            self.metrics_history['val_accuracy'].append(val_acc)
