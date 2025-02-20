from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from enum import Enum
from .tabular_models import TabularModel

class ProblemType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"

class TabularDataset(Dataset):
    """Custom Dataset for tabular data"""
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class ModelManager:
    """Manages model creation, training, and evaluation"""
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.problem_type = None
        self.input_dim = None
        self.output_dim = None
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
    def prepare_data(self, 
                    df: pd.DataFrame,
                    target_column: str,
                    problem_type: ProblemType,
                    test_size: float = 0.2,
                    val_size: float = 0.2) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            problem_type: Type of problem (classification or regression)
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
        """
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Set input and output dimensions
        self.input_dim = X.shape[1]
        self.problem_type = problem_type
        
        if problem_type == ProblemType.REGRESSION:
            self.output_dim = 1
        elif problem_type == ProblemType.BINARY_CLASSIFICATION:
            self.output_dim = 1
            y = y.astype(int)
        else:  # MULTICLASS_CLASSIFICATION
            self.output_dim = len(y.unique())
            y = y.astype(int)
        
        # Split data into train, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
        
        # Convert to tensors
        train_data = {
            'features': torch.FloatTensor(X_train.values),
            'targets': torch.FloatTensor(y_train.values)
        }
        
        val_data = {
            'features': torch.FloatTensor(X_val.values),
            'targets': torch.FloatTensor(y_val.values)
        }
        
        test_data = {
            'features': torch.FloatTensor(X_test.values),
            'targets': torch.FloatTensor(y_test.values)
        }
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def create_model(self, 
                    hidden_dims: List[int] = [64, 32],
                    dropout_rate: float = 0.2) -> None:
        """Create and initialize the model"""
        if not self.input_dim or not self.output_dim:
            raise ValueError("Data must be prepared before creating model")
        
        self.model = TabularModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        ).to(self.device)
    
    def create_dataloaders(self,
                          data: Dict[str, Dict[str, torch.Tensor]],
                          batch_size: int = 32) -> Dict[str, DataLoader]:
        """Create DataLoaders for training, validation, and testing"""
        dataloaders = {}
        
        for split, split_data in data.items():
            dataset = TabularDataset(
                features=split_data['features'],
                targets=split_data['targets']
            )
            
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train')
            )
        
        return dataloaders
    
    def train_epoch(self,
                   train_loader: DataLoader,
                   criterion: nn.Module,
                   optimizer: optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, targets in train_loader:
            features, targets = features.to(self.device), targets.to(self.device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(features)
            
            # Reshape outputs and targets for loss calculation
            if self.problem_type == ProblemType.BINARY_CLASSIFICATION:
                outputs = outputs.squeeze()
                loss = criterion(outputs, targets)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
            elif self.problem_type == ProblemType.MULTICLASS_CLASSIFICATION:
                targets = targets.long()
                loss = criterion(outputs, targets)
                predictions = outputs.argmax(dim=1)
            else:  # REGRESSION
                outputs = outputs.squeeze()
                loss = criterion(outputs, targets)
                predictions = outputs
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item()
            if self.problem_type != ProblemType.REGRESSION:
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if self.problem_type != ProblemType.REGRESSION else None
        
        return avg_loss, accuracy
    
    def validate(self,
                val_loader: DataLoader,
                criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                
                outputs = self.model(features)
                
                # Calculate loss and metrics based on problem type
                if self.problem_type == ProblemType.BINARY_CLASSIFICATION:
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, targets)
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                elif self.problem_type == ProblemType.MULTICLASS_CLASSIFICATION:
                    targets = targets.long()
                    loss = criterion(outputs, targets)
                    predictions = outputs.argmax(dim=1)
                else:  # REGRESSION
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, targets)
                    predictions = outputs
                
                total_loss += loss.item()
                if self.problem_type != ProblemType.REGRESSION:
                    correct += (predictions == targets).sum().item()
                    total += targets.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if self.problem_type != ProblemType.REGRESSION else None
        
        return avg_loss, accuracy
    
    def train(self,
              data: Dict[str, Dict[str, torch.Tensor]],
              num_epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            data: Dictionary containing train, val, and test data
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            early_stopping_patience: Number of epochs to wait before early stopping
        """
        # Create dataloaders
        dataloaders = self.create_dataloaders(data, batch_size)
        
        # Initialize criterion and optimizer based on problem type
        if self.problem_type == ProblemType.REGRESSION:
            criterion = nn.MSELoss()
        elif self.problem_type == ProblemType.BINARY_CLASSIFICATION:
            criterion = nn.BCEWithLogitsLoss()
        else:  # MULTICLASS_CLASSIFICATION
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        patience_counter = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train and validate
            train_loss, train_acc = self.train_epoch(dataloaders['train'], criterion, optimizer)
            val_loss, val_acc = self.validate(dataloaders['val'], criterion)
            
            # Update metrics history
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_loss)
            if train_acc is not None:
                self.metrics_history['train_accuracy'].append(train_acc)
                self.metrics_history['val_accuracy'].append(val_acc)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            
            self.current_epoch = epoch
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return self.metrics_history
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Make predictions using the trained model"""
        self.model.eval()
        with torch.no_grad():
            features = features.to(self.device)
            outputs = self.model(features)
            
            if self.problem_type == ProblemType.BINARY_CLASSIFICATION:
                predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            elif self.problem_type == ProblemType.MULTICLASS_CLASSIFICATION:
                predictions = outputs.argmax(dim=1)
            else:  # REGRESSION
                predictions = outputs.squeeze()
            
            return predictions.cpu()
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data"""
        self.model.eval()
        criterion = nn.MSELoss() if self.problem_type == ProblemType.REGRESSION else nn.BCEWithLogitsLoss()
        
        test_loss, test_acc = self.validate(test_loader, criterion)
        
        metrics = {
            'test_loss': test_loss
        }
        
        if test_acc is not None:
            metrics['test_accuracy'] = test_acc
        
        return metrics