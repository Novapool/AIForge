from typing import Dict, List, Optional, Union
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
        
    def handle_missing_values(self, 
                            df: pd.DataFrame,
                            strategy: str = 'mean',
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: One of 'mean', 'median', 'most_frequent', or 'constant'
            columns: List of columns to process. If None, processes all numeric columns
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col not in self.imputers:
                self.imputers[col] = SimpleImputer(strategy=strategy)
                
            col_data = df[[col]]
            df[col] = self.imputers[col].fit_transform(col_data)
            
        self.preprocessing_history.append({
            'operation': 'missing_values',
            'strategy': strategy,
            'columns': list(columns)
        })
        
        return df
    
    def normalize_data(self,
                      df: pd.DataFrame,
                      method: str = 'standard',
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize numerical data using specified method
        
        Args:
            df: Input DataFrame
            method: One of 'standard', 'minmax', or 'robust'
            columns: List of columns to normalize. If None, normalizes all numeric columns
        """
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
            
        # Normalize selected columns in place
        df[columns] = scaler.fit_transform(df[columns])
        
        self.preprocessing_history.append({
            'operation': 'normalization',
            'method': method,
            'columns': list(columns)
        })
        
        return df
    
    def encode_categorical(self,
                         df: pd.DataFrame,
                         method: str = 'label',
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            method: One of 'label' or 'onehot'
            columns: List of columns to encode. If None, encodes all object columns
        """
        # Create a copy to ensure we don't modify the original
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object', 'category']).columns
            
        if len(columns) == 0:
            print("No categorical columns found to encode!")
            return df_encoded
            
        if method == 'label':
            for col in columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                # Check if column exists and has categorical data
                if col in df_encoded.columns and df_encoded[col].dtype in ['object', 'category']:
                    try:
                        # Store original values for mapping
                        original_values = df_encoded[col].unique()
                        # Fit and transform the data
                        df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                        # Store the mapping for reference
                        self.encoders[f"{col}_mapping"] = dict(zip(original_values, 
                                                                 self.encoders[col].transform(original_values.astype(str))))
                    except Exception as e:
                        print(f"Error encoding column {col}: {str(e)}")
                    
        elif method == 'onehot':
            for col in columns:
                if col not in self.encoders:
                    self.encoders[col] = OneHotEncoder(sparse_output=False)
                if col in df_encoded.columns and df_encoded[col].dtype in ['object', 'category']:
                    try:
                        # Store original values
                        original_values = df_encoded[col].unique()
                        # Reshape and encode
                        data_reshaped = df_encoded[[col]]
                        encoded_data = self.encoders[col].fit_transform(data_reshaped)
                        
                        # Get feature names
                        encoded_cols = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0]]
                        
                        # Add encoded columns
                        for i, new_col in enumerate(encoded_cols):
                            df_encoded[new_col] = encoded_data[:, i]
                        
                        # Store mapping
                        categories = self.encoders[col].categories_[0]
                        encoded_matrix = np.eye(len(categories))
                        mapping = {cat: encoded_matrix[i].tolist() for i, cat in enumerate(categories)}
                        self.encoders[f"{col}_mapping"] = mapping
                        
                        # Drop original column
                        df_encoded = df_encoded.drop(columns=[col])
                        
                    except Exception as e:
                        print(f"Error encoding column {col}: {str(e)}")
        
        self.preprocessing_history.append({
            'operation': 'encoding',
            'method': method,
            'columns': list(columns)
        })
        
        return df
    
    def remove_outliers(self,
                       df: pd.DataFrame,
                       method: str = 'zscore',
                       threshold: float = 3.0,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove outliers from the dataset
        
        Args:
            df: Input DataFrame
            method: One of 'zscore' or 'iqr'
            threshold: Threshold for outlier detection
            columns: List of columns to process. If None, processes all numeric columns
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            
        if method == 'zscore':
            for col in columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
                
        elif method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[col] < (Q1 - 1.5 * IQR)) | 
                         (df[col] > (Q3 + 1.5 * IQR)))]
                
        self.preprocessing_history.append({
            'operation': 'outliers',
            'method': method,
            'threshold': threshold,
            'columns': list(columns)
        })
        
        return df
    
    def get_preprocessing_summary(self) -> Dict:
        """Return summary of all preprocessing operations performed"""
        return {
            'total_operations': len(self.preprocessing_history),
            'operations': self.preprocessing_history
        }
