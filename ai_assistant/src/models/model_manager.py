from typing import Dict, List, Optional, Union, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from enum import Enum, auto
from pathlib import Path
from .tabular_models import TabularModel
from .traditional_models import (
    create_traditional_model, 
    TraditionalModelWrapper,
    BaseTraditionalModel
)

class DataPreprocessingError(Exception):
    """Custom exception for data preprocessing requirements"""
    def __init__(self, message: str, columns: List[str]):
        self.message = message
        self.columns = columns
        super().__init__(self.message)

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

class ModelType(Enum):
    NEURAL_NETWORK = "neural_network"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LINEAR = "linear"
    SVM = "svm"
    KNN = "knn"
    DECISION_TREE = "decision_tree"

class ModelManager:
    """Manages model creation and data preparation"""
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.problem_type = None
        self.input_dim = None
        self.output_dim = None
        self.model_type = None
        self.raw_model = None  # For traditional models, store the unwrapped model
    
    def prepare_data(self, 
                    df: pd.DataFrame,
                    target_column: str,
                    problem_type: str,
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
            
        Returns:
            Dictionary containing train, validation, and test data
        """
        # Convert string problem_type to enum
        problem_type = ProblemType(problem_type)
        
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)
        
        # Check for non-numeric data types in features
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            raise DataPreprocessingError(
                "Non-numeric data found in features. Please preprocess these columns using appropriate encoding methods.",
                non_numeric_cols
            )

        # Convert to tensors
        try:
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
        except (ValueError, TypeError) as e:
            raise DataPreprocessingError(
                "Failed to convert data to tensors. Please ensure all data is properly preprocessed and contains only numeric values.",
                non_numeric_cols if non_numeric_cols else []
            )
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def create_model(self, 
                    model_type: str = "neural_network",
                    hidden_dims: List[int] = [64, 32],
                    dropout_rate: float = 0.2,
                    **kwargs) -> None:
        """
        Create and initialize the model
        
        Args:
            model_type: Type of model to create (neural_network, random_forest, etc.)
            hidden_dims: List of hidden layer dimensions (for neural networks)
            dropout_rate: Dropout rate for regularization (for neural networks)
            **kwargs: Additional parameters for the model
        """
        if not self.input_dim or not self.output_dim:
            raise ValueError("Data must be prepared before creating model")
        
        # Convert string model_type to enum if needed
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        self.model_type = model_type
        
        if model_type == ModelType.NEURAL_NETWORK:
            # Create neural network model
            self.model = TabularModel(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate
            ).to(self.device)
        else:
            # Create traditional ML model
            self.create_traditional_model(model_type.value, **kwargs)
    
    def create_traditional_model(self, model_type: str, **kwargs) -> None:
        """
        Create and initialize a traditional ML model
        
        Args:
            model_type: Type of traditional model to create
            **kwargs: Additional parameters for the model
        """
        if not self.input_dim or not self.output_dim:
            raise ValueError("Data must be prepared before creating model")
        
        # Convert ProblemType enum to string for traditional models
        problem_type = "regression" if self.problem_type == ProblemType.REGRESSION else "classification"
        
        # Create the traditional model
        self.raw_model = create_traditional_model(model_type, problem_type, **kwargs)
        
        # Wrap the traditional model to make it compatible with PyTorch interface
        self.model = TraditionalModelWrapper(self.raw_model)
    
    def create_dataloaders(self,
                          data: Dict[str, Dict[str, torch.Tensor]],
                          batch_size: int = 32) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for training, validation, and testing
        
        Args:
            data: Dictionary containing data splits
            batch_size: Batch size for DataLoader
            
        Returns:
            Dictionary containing DataLoaders for each split
        """
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
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using the trained model
        
        Args:
            features: Input features as tensor
            
        Returns:
            Predictions as tensor
        """
        if self.model is None:
            raise ValueError("Model has not been created yet")
            
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
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current model configuration
        
        Returns:
            Dictionary containing model details
        """
        if self.model is None:
            raise ValueError("Model has not been created yet")
        
        summary = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'problem_type': self.problem_type.value if self.problem_type else None,
            'device': str(self.device),
            'model_type': self.model_type.value if self.model_type else None
        }
        
        # Add model-specific details
        if self.model_type == ModelType.NEURAL_NETWORK:
            # For neural networks, add architecture details
            if isinstance(self.model, TabularModel):
                summary['hidden_dims'] = [m.out_features for m in self.model.model if isinstance(m, nn.Linear)][:-1]
                summary['dropout_rate'] = next((m.p for m in self.model.model if isinstance(m, nn.Dropout)), None)
        elif self.raw_model is not None:
            # For traditional models, add model parameters
            summary['model_params'] = self.raw_model.get_params()
            
        return summary
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance for the model if available
        
        Returns:
            Dictionary mapping feature names to importance values, or None if not available
        """
        if self.model is None:
            raise ValueError("Model has not been created yet")
        
        if self.raw_model is not None and hasattr(self.raw_model, 'get_feature_importance'):
            return self.raw_model.get_feature_importance()
        
        return None
