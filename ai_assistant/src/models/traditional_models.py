import numpy as np
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.base import BaseEstimator
import joblib
import os
from pathlib import Path
import torch

# Import these conditionally to avoid errors if not installed
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class BaseTraditionalModel:
    """Base class for all traditional ML models"""
    
    def __init__(self, model_type: str, problem_type: str):
        self.model_type = model_type
        self.problem_type = problem_type
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> None:
        """
        Fit the model to the data
        
        Args:
            X: Features
            y: Target values
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        if isinstance(y, pd.Series):
            y = y.values
            
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """
        Get probability estimates for classification models
        
        Args:
            X: Features
            
        Returns:
            Probability estimates or None if not applicable
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
            
        if self.problem_type == "regression":
            return None
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Evaluate the model on test data
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if isinstance(y, pd.Series):
            y = y.values
            
        y_pred = self.predict(X)
        
        if self.problem_type == "regression":
            return {
                "r2": r2_score(y, y_pred),
                "mse": mean_squared_error(y, y_pred),
                "rmse": np.sqrt(mean_squared_error(y, y_pred))
            }
        else:
            return {
                "accuracy": accuracy_score(y, y_pred)
            }
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters
        
        Returns:
            Dictionary of model parameters
        """
        return self.model.get_params()
    
    def set_params(self, **params) -> None:
        """
        Set model parameters
        
        Args:
            **params: Parameters to set
        """
        self.model.set_params(**params)
    
    def save(self, path: str) -> None:
        """
        Save the model to disk
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and metadata
        joblib.dump({
            "model": self.model,
            "model_type": self.model_type,
            "problem_type": self.problem_type,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load the model from disk
        
        Args:
            path: Path to load the model from
        """
        data = joblib.load(path)
        self.model = data["model"]
        self.model_type = data["model_type"]
        self.problem_type = data["problem_type"]
        self.feature_names = data["feature_names"]
        self.is_fitted = data["is_fitted"]
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if available
        
        Returns:
            Dictionary mapping feature names to importance values, or None if not available
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
            
        if not hasattr(self.model, "feature_importances_"):
            return None
            
        importances = self.model.feature_importances_
        
        if self.feature_names is None:
            return {f"feature_{i}": importance for i, importance in enumerate(importances)}
        
        return {name: importance for name, importance in zip(self.feature_names, importances)}


class RandomForestModel(BaseTraditionalModel):
    """Random Forest model implementation"""
    
    def __init__(self, problem_type: str, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1, **kwargs):
        """
        Initialize Random Forest model
        
        Args:
            problem_type: Type of problem ("classification" or "regression")
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum number of samples required to split an internal node
            min_samples_leaf: Minimum number of samples required to be at a leaf node
            **kwargs: Additional parameters for the model
        """
        super().__init__("random_forest", problem_type)
        
        if problem_type == "regression":
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                **kwargs
            )
        else:  # classification
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                **kwargs
            )


class GradientBoostingModel(BaseTraditionalModel):
    """Gradient Boosting model implementation"""
    
    def __init__(self, problem_type: str, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2, min_samples_leaf: int = 1, **kwargs):
        """
        Initialize Gradient Boosting model
        
        Args:
            problem_type: Type of problem ("classification" or "regression")
            n_estimators: Number of boosting stages
            learning_rate: Learning rate shrinks the contribution of each tree
            max_depth: Maximum depth of the individual regression estimators
            min_samples_split: Minimum number of samples required to split an internal node
            min_samples_leaf: Minimum number of samples required to be at a leaf node
            **kwargs: Additional parameters for the model
        """
        super().__init__("gradient_boosting", problem_type)
        
        if problem_type == "regression":
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                **kwargs
            )
        else:  # classification
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                **kwargs
            )


class XGBoostModel(BaseTraditionalModel):
    """XGBoost model implementation"""
    
    def __init__(self, problem_type: str, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, **kwargs):
        """
        Initialize XGBoost model
        
        Args:
            problem_type: Type of problem ("classification" or "regression")
            n_estimators: Number of boosting rounds
            learning_rate: Step size shrinkage used to prevent overfitting
            max_depth: Maximum depth of a tree
            **kwargs: Additional parameters for the model
        """
        super().__init__("xgboost", problem_type)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install it with 'pip install xgboost'")
        
        if problem_type == "regression":
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                **kwargs
            )
        else:  # classification
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                **kwargs
            )


class LightGBMModel(BaseTraditionalModel):
    """LightGBM model implementation"""
    
    def __init__(self, problem_type: str, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = -1, num_leaves: int = 31, **kwargs):
        """
        Initialize LightGBM model
        
        Args:
            problem_type: Type of problem ("classification" or "regression")
            n_estimators: Number of boosting iterations
            learning_rate: Boosting learning rate
            max_depth: Maximum tree depth for base learners, -1 means no limit
            num_leaves: Maximum tree leaves for base learners
            **kwargs: Additional parameters for the model
        """
        super().__init__("lightgbm", problem_type)
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Please install it with 'pip install lightgbm'")
        
        if problem_type == "regression":
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_leaves=num_leaves,
                **kwargs
            )
        else:  # classification
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_leaves=num_leaves,
                **kwargs
            )


class LinearModel(BaseTraditionalModel):
    """Linear model implementation (Linear/Logistic Regression)"""
    
    def __init__(self, problem_type: str, model_subtype: str = "standard", 
                 C: float = 1.0, alpha: float = 1.0, **kwargs):
        """
        Initialize Linear model
        
        Args:
            problem_type: Type of problem ("classification" or "regression")
            model_subtype: Subtype of linear model ("standard", "ridge", "lasso")
            C: Inverse of regularization strength (for logistic regression)
            alpha: Regularization strength (for ridge and lasso regression)
            **kwargs: Additional parameters for the model
        """
        super().__init__("linear", problem_type)
        
        if problem_type == "regression":
            if model_subtype == "ridge":
                self.model = Ridge(alpha=alpha, **kwargs)
            elif model_subtype == "lasso":
                self.model = Lasso(alpha=alpha, **kwargs)
            else:  # standard
                self.model = LinearRegression(**kwargs)
        else:  # classification
            self.model = LogisticRegression(C=C, **kwargs)


class SVMModel(BaseTraditionalModel):
    """Support Vector Machine model implementation"""
    
    def __init__(self, problem_type: str, C: float = 1.0, kernel: str = "rbf", 
                 gamma: str = "scale", **kwargs):
        """
        Initialize SVM model
        
        Args:
            problem_type: Type of problem ("classification" or "regression")
            C: Regularization parameter
            kernel: Kernel type to be used
            gamma: Kernel coefficient
            **kwargs: Additional parameters for the model
        """
        super().__init__("svm", problem_type)
        
        if problem_type == "regression":
            self.model = SVR(C=C, kernel=kernel, gamma=gamma, **kwargs)
        else:  # classification
            self.model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, **kwargs)


class KNNModel(BaseTraditionalModel):
    """K-Nearest Neighbors model implementation"""
    
    def __init__(self, problem_type: str, n_neighbors: int = 5, weights: str = "uniform", 
                 algorithm: str = "auto", **kwargs):
        """
        Initialize KNN model
        
        Args:
            problem_type: Type of problem ("classification" or "regression")
            n_neighbors: Number of neighbors
            weights: Weight function used in prediction
            algorithm: Algorithm used to compute the nearest neighbors
            **kwargs: Additional parameters for the model
        """
        super().__init__("knn", problem_type)
        
        if problem_type == "regression":
            self.model = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                **kwargs
            )
        else:  # classification
            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                **kwargs
            )


class DecisionTreeModel(BaseTraditionalModel):
    """Decision Tree model implementation"""
    
    def __init__(self, problem_type: str, max_depth: Optional[int] = None, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1, **kwargs):
        """
        Initialize Decision Tree model
        
        Args:
            problem_type: Type of problem ("classification" or "regression")
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split an internal node
            min_samples_leaf: Minimum number of samples required to be at a leaf node
            **kwargs: Additional parameters for the model
        """
        super().__init__("decision_tree", problem_type)
        
        if problem_type == "regression":
            self.model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                **kwargs
            )
        else:  # classification
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                **kwargs
            )


def create_traditional_model(model_type: str, problem_type: str, **kwargs) -> BaseTraditionalModel:
    """
    Factory function to create a traditional ML model
    
    Args:
        model_type: Type of model to create
        problem_type: Type of problem ("classification" or "regression")
        **kwargs: Additional parameters for the model
        
    Returns:
        Initialized model instance
    """
    model_map = {
        "random_forest": RandomForestModel,
        "gradient_boosting": GradientBoostingModel,
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "linear": LinearModel,
        "svm": SVMModel,
        "knn": KNNModel,
        "decision_tree": DecisionTreeModel
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_map[model_type](problem_type=problem_type, **kwargs)


class TraditionalModelWrapper:
    """
    Wrapper class to make traditional ML models compatible with PyTorch interface
    This allows traditional models to be used with the existing training pipeline
    """
    
    def __init__(self, model: BaseTraditionalModel):
        """
        Initialize the wrapper
        
        Args:
            model: Traditional ML model to wrap
        """
        self.model = model
        self.device = torch.device("cpu")  # Traditional models always use CPU
    
    def to(self, device):
        """
        Dummy method to maintain compatibility with PyTorch models
        
        Args:
            device: Device to move the model to (ignored)
            
        Returns:
            Self for chaining
        """
        return self
    
    def train(self):
        """
        Set the model to training mode (no-op for traditional models)
        
        Returns:
            Self for chaining
        """
        return self
    
    def eval(self):
        """
        Set the model to evaluation mode (no-op for traditional models)
        
        Returns:
            Self for chaining
        """
        return self
    
    def __call__(self, x):
        """
        Forward pass (prediction)
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Convert PyTorch tensor to numpy array
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
        
        # Make prediction
        if self.model.problem_type == "regression":
            # For regression, return predictions directly
            y_pred = self.model.predict(x_np)
            return torch.tensor(y_pred, dtype=torch.float32).view(-1, 1)
        else:
            # For classification, return logits (log probabilities)
            y_proba = self.model.predict_proba(x_np)
            if y_proba is None:
                # If probabilities are not available, return dummy logits
                n_classes = 2  # Assume binary classification as fallback
                y_pred = self.model.predict(x_np)
                logits = torch.zeros((len(y_pred), n_classes), dtype=torch.float32)
                for i, pred in enumerate(y_pred):
                    logits[i, int(pred)] = 1.0
                return logits
            else:
                # Convert probabilities to logits
                return torch.tensor(np.log(y_proba + 1e-10), dtype=torch.float32)
    
    def parameters(self):
        """
        Dummy method to maintain compatibility with PyTorch models
        
        Returns:
            Empty list (traditional models don't have parameters in PyTorch sense)
        """
        return []
    
    def state_dict(self):
        """
        Dummy method to maintain compatibility with PyTorch models
        
        Returns:
            Empty dict (traditional models don't have state_dict)
        """
        return {}
    
    def load_state_dict(self, state_dict):
        """
        Dummy method to maintain compatibility with PyTorch models
        
        Args:
            state_dict: State dict to load (ignored)
        """
        pass
