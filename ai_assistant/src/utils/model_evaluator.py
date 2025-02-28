import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, mean_absolute_error, mean_squared_error, r2_score
)
from typing import Dict, List, Tuple, Union, Optional, Any
from models.model_manager import ProblemType, ModelType
from models.traditional_models import TraditionalModelWrapper

class ModelEvaluator:
    """Handles model evaluation metrics and analysis"""
    
    @staticmethod
    def get_feature_importance(model, feature_names: Optional[List[str]] = None) -> Optional[Dict[str, float]]:
        """
        Get feature importance from a model if available
        
        Args:
            model: Model to extract feature importance from
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary mapping feature names to importance values, or None if not available
        """
        # For traditional models wrapped in TraditionalModelWrapper
        if isinstance(model, TraditionalModelWrapper):
            if hasattr(model.model, 'get_feature_importance'):
                return model.model.get_feature_importance()
        
        # For sklearn-based models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            return {name: float(imp) for name, imp in zip(feature_names, importances)}
        
        # For linear models
        if hasattr(model, 'coef_'):
            coefs = model.coef_
            if len(coefs.shape) > 1:
                coefs = np.mean(np.abs(coefs), axis=0)
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(coefs))]
            return {name: float(abs(coef)) for name, coef in zip(feature_names, coefs)}
        
        return None
    
    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        is_binary: bool = False
    ) -> Dict:
        """
        Evaluate classification model performance
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for ROC/PR curves)
            is_binary: Whether this is a binary classification problem
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Basic metrics
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        results['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        # ROC and PR curves for binary classification
        if is_binary and y_prob is not None:
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            results['roc'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': auc(fpr, tpr)
            }
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            results['pr_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'avg_precision': np.mean(precision)
            }
        
        return results
    
    @staticmethod
    def evaluate_regression(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Evaluate regression model performance
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Calculate metrics
        results['mae'] = mean_absolute_error(y_true, y_pred)
        results['mse'] = mean_squared_error(y_true, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['r2'] = r2_score(y_true, y_pred)
        
        # Calculate residuals
        results['residuals'] = (y_true - y_pred).tolist()
        
        return results
    
    @staticmethod
    def evaluate_model(
        model,
        dataloader: torch.utils.data.DataLoader,
        problem_type: ProblemType,
        device: torch.device
    ) -> Dict:
        """
        Evaluate model on the provided dataloader
        
        Args:
            model: PyTorch model to evaluate
            dataloader: DataLoader containing evaluation data
            problem_type: Type of problem (classification or regression)
            device: Device to run evaluation on
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_targets = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for features, targets in dataloader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                
                # Process outputs based on problem type
                if problem_type == ProblemType.BINARY_CLASSIFICATION:
                    probs = torch.sigmoid(outputs.squeeze())
                    preds = (probs > 0.5).float()
                    all_probabilities.extend(probs.cpu().numpy())
                elif problem_type == ProblemType.MULTICLASS_CLASSIFICATION:
                    probs = torch.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1)
                    all_probabilities.extend(probs.cpu().numpy())
                else:  # REGRESSION
                    preds = outputs.squeeze()
                    all_probabilities = None
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities) if all_probabilities else None
        
        # Calculate metrics based on problem type
        if problem_type == ProblemType.REGRESSION:
            return ModelEvaluator.evaluate_regression(y_true, y_pred)
        else:
            is_binary = problem_type == ProblemType.BINARY_CLASSIFICATION
            return ModelEvaluator.evaluate_classification(y_true, y_pred, y_prob, is_binary)
