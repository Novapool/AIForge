from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os

class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self):
        self.scalers: Dict[str, Union[StandardScaler, MinMaxScaler, RobustScaler]] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.encoders: Dict[str, Union[LabelEncoder, OneHotEncoder]] = {}
        self.preprocessing_history: List[Dict] = []
        
    def _convert_to_native_types(self, obj: any) -> any:
        """Convert numpy types to native Python types"""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8, np.uint16,
            np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return [self._convert_to_native_types(x) for x in obj]
        elif isinstance(obj, dict):
            return {str(k): self._convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_native_types(x) for x in obj]
        return obj
        
    def handle_missing_values(self, 
                            df: pd.DataFrame,
                            strategy: str = 'mean',
                            columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: One of 'mean', 'median', 'most_frequent', or 'constant'
            columns: List of columns to process. If None, processes all numeric columns
        """
        transformation_details = {
            'operation': 'missing_values',
            'missing_value_rules': {}
        }

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col not in self.imputers:
                self.imputers[col] = SimpleImputer(strategy=strategy)
            
            # Transform the data
            col_data = df[[col]]
            df[col] = self.imputers[col].fit_transform(col_data)
            
            # Store only the rules with native Python types
            transformation_details['missing_value_rules'][str(col)] = {
                'strategy': strategy,
                'fill_value': float(self.imputers[col].statistics_[0])
            }
            
        return df, transformation_details
    
    def normalize_data(self,
                      df: pd.DataFrame,
                      method: str = 'standard',
                      columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize numerical data using specified method
        
        Args:
            df: Input DataFrame
            method: One of 'standard', 'minmax', or 'robust'
            columns: List of columns to normalize. If None, normalizes all numeric columns
        """
        transformation_details = {
            'operation': 'normalization',
            'normalizations': {}
        }

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            
        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        scaler = scaler_map.get(method)
        if not scaler:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Normalize selected columns
        df[columns] = scaler.fit_transform(df[columns])
        
        # Store scaling parameters with native Python types
        for i, col in enumerate(columns):
            params = {}
            if method == 'standard':
                params = {
                    'type': 'standard',
                    'mean': float(scaler.mean_[i]),
                    'std': float(scaler.scale_[i])
                }
            elif method == 'minmax':
                params = {
                    'type': 'minmax',
                    'min': float(scaler.min_[i]),
                    'scale': float(scaler.scale_[i])
                }
            elif method == 'robust':
                params = {
                    'type': 'robust',
                    'center': float(scaler.center_[i]),
                    'scale': float(scaler.scale_[i])
                }
            transformation_details['normalizations'][str(col)] = params
            
        return df, transformation_details
    
    def encode_categorical(self,
                         df: pd.DataFrame,
                         method: str = 'label',
                         columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            method: One of 'label' or 'onehot'
            columns: List of columns to encode. If None, encodes all object columns
        """
        transformation_details = {
            'operation': 'encoding',
            'encoding_maps': {}
        }

        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object', 'category']).columns
            
        if len(columns) == 0:
            print("No categorical columns found to encode!")
            return df_encoded, transformation_details
            
        if method == 'label':
            for col in columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                if col in df_encoded.columns and df_encoded[col].dtype in ['object', 'category']:
                    try:
                        # Get unique values and their encodings
                        unique_values = df_encoded[col].unique()
                        df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                        
                        # Create mapping with native Python types
                        mapping = {
                            str(key): int(value)
                            for key, value in zip(unique_values, 
                                               self.encoders[col].transform(unique_values.astype(str)))
                        }
                        
                        transformation_details['encoding_maps'][str(col)] = {
                            'type': 'label',
                            'values': mapping
                        }
                    except Exception as e:
                        print(f"Error encoding column {col}: {str(e)}")
                    
        elif method == 'onehot':
            for col in columns:
                if col not in self.encoders:
                    self.encoders[col] = OneHotEncoder(sparse_output=False)
                if col in df_encoded.columns and df_encoded[col].dtype in ['object', 'category']:
                    try:
                        # Get categories and transform
                        data_reshaped = df_encoded[[col]]
                        encoded_data = self.encoders[col].fit_transform(data_reshaped)
                        
                        # Get feature names and add columns
                        categories = [str(cat) for cat in self.encoders[col].categories_[0]]
                        encoded_cols = [f"{col}_{cat}" for cat in categories]
                        for i, new_col in enumerate(encoded_cols):
                            df_encoded[new_col] = encoded_data[:, i]
                        
                        # Store only the category information with native Python types
                        transformation_details['encoding_maps'][str(col)] = {
                            'type': 'onehot',
                            'categories': categories,
                            'encoded_columns': encoded_cols
                        }
                        
                        # Drop original column
                        df_encoded = df_encoded.drop(columns=[col])
                        
                    except Exception as e:
                        print(f"Error encoding column {col}: {str(e)}")
        
        return df_encoded, transformation_details
    
    def remove_outliers(self,
                       df: pd.DataFrame,
                       method: str = 'zscore',
                       threshold: float = 3.0,
                       columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Remove outliers from the dataset
        
        Args:
            df: Input DataFrame
            method: One of 'zscore' or 'iqr'
            threshold: Threshold for outlier detection
            columns: List of columns to process. If None, processes all numeric columns
        """
        transformation_details = {
            'operation': 'outliers',
            'outlier_rules': {}
        }

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            
        if method == 'zscore':
            for col in columns:
                mean_val = float(df[col].mean())
                std_val = float(df[col].std())
                z_scores = np.abs((df[col] - mean_val) / std_val)
                df = df[z_scores < threshold]
                
                transformation_details['outlier_rules'][str(col)] = {
                    'method': 'zscore',
                    'threshold': float(threshold),
                    'params': {
                        'mean': mean_val,
                        'std': std_val
                    }
                }
                
        elif method == 'iqr':
            for col in columns:
                Q1 = float(df[col].quantile(0.25))
                Q3 = float(df[col].quantile(0.75))
                IQR = float(Q3 - Q1)
                df = df[~((df[col] < (Q1 - 1.5 * IQR)) | 
                         (df[col] > (Q3 + 1.5 * IQR)))]
                
                transformation_details['outlier_rules'][str(col)] = {
                    'method': 'iqr',
                    'params': {
                        'Q1': Q1,
                        'Q3': Q3,
                        'IQR': IQR
                    }
                }
                
        return df, transformation_details
    
    def get_preprocessing_summary(self) -> Dict:
        """Return summary of all preprocessing operations performed"""
        return {
            'total_operations': len(self.preprocessing_history),
            'operations': self.preprocessing_history
        }
