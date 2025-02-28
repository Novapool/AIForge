import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from .model_manager import ProblemType, ModelType
from .traditional_models import TraditionalModelWrapper

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
             early_stopping_patience: int = 10, **kwargs) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            model: Model to train
            dataloaders: Dictionary containing DataLoaders for train, val, and test
            problem_type: Type of problem (classification or regression)
            num_epochs: Number of epochs to train for
            learning_rate: Learning rate for optimizer
            early_stopping_patience: Number of epochs to wait before early stopping
            **kwargs: Additional parameters for training
            
        Returns:
            Dictionary containing training metrics
        """
        # Check if this is a traditional model
        if isinstance(model, TraditionalModelWrapper):
            return self.train_traditional_model(model, dataloaders, problem_type, **kwargs)
        
        # Neural network training
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
    
    def train_traditional_model(self, model_wrapper: TraditionalModelWrapper, 
                               dataloaders: Dict[str, DataLoader], 
                               problem_type: ProblemType,
                               **kwargs) -> Dict[str, List[float]]:
        """
        Train a traditional ML model
        
        Args:
            model_wrapper: TraditionalModelWrapper instance
            dataloaders: Dictionary containing DataLoaders for train, val, and test
            problem_type: Type of problem (classification or regression)
            **kwargs: Additional parameters for the model
            
        Returns:
            Dictionary containing training metrics
        """
        # Extract the raw model from the wrapper
        raw_model = model_wrapper.model
        
        # Convert PyTorch DataLoader to numpy arrays for training
        X_train, y_train = self._dataloader_to_numpy(dataloaders['train'])
        X_val, y_val = self._dataloader_to_numpy(dataloaders['val'])
        
        # Train the model
        raw_model.fit(X_train, y_train)
        
        # Evaluate on training and validation sets
        train_metrics = raw_model.evaluate(X_train, y_train)
        val_metrics = raw_model.evaluate(X_val, y_val)
        
        # Extract metrics based on problem type
        if problem_type == ProblemType.REGRESSION:
            train_loss = train_metrics['mse']
            val_loss = val_metrics['mse']
            train_acc = None
            val_acc = None
        else:  # Classification
            # For classification, use accuracy as the main metric
            train_loss = 1.0 - train_metrics['accuracy']  # Convert accuracy to loss
            val_loss = 1.0 - val_metrics['accuracy']
            train_acc = train_metrics['accuracy']
            val_acc = val_metrics['accuracy']
        
        # Update metrics history (just one "epoch" for traditional models)
        self._update_metrics(train_loss, val_loss, train_acc, val_acc)
        
        # Save the model
        checkpoint_path = self.checkpoint_dir / f"model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        raw_model.save(str(checkpoint_path))
        
        return self.metrics_history
    
    def _dataloader_to_numpy(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a PyTorch DataLoader to numpy arrays
        
        Args:
            dataloader: DataLoader to convert
            
        Returns:
            Tuple of (features, targets) as numpy arrays
        """
        features_list = []
        targets_list = []
        
        for features, targets in dataloader:
            features_list.append(features.numpy())
            targets_list.append(targets.numpy())
        
        return np.vstack(features_list), np.concatenate(targets_list)

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
