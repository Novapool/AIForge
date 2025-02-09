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
            
        # Create a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Normalize selected columns
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
        
        self.preprocessing_history.append({
            'operation': 'normalization',
            'method': method,
            'columns': list(columns)
        })
        
        return df_copy
    
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
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=['object', 'category']).columns
            
        if len(columns) == 0:
            print("No categorical columns found to encode!")
            return df_copy
            
        if method == 'label':
            for col in columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                # Check if column exists and has categorical data
                if col in df_copy.columns and df_copy[col].dtype in ['object', 'category']:
                    try:
                        df_copy[col] = self.encoders[col].fit_transform(df_copy[col].astype(str))
                    except Exception as e:
                        print(f"Error encoding column {col}: {str(e)}")
                    
        elif method == 'onehot':
            for col in columns:
                if col not in self.encoders:
                    # Updated OneHotEncoder initialization
                    self.encoders[col] = OneHotEncoder(sparse_output=False)
                # Check if column exists and has categorical data
                if col in df_copy.columns and df_copy[col].dtype in ['object', 'category']:
                    try:
                        # Reshape data for OneHotEncoder
                        data_reshaped = df_copy[[col]]
                        encoded_data = self.encoders[col].fit_transform(data_reshaped)
                        
                        # Get feature names from encoder
                        encoded_cols = [f"{col}_{cat}" for cat in 
                                      self.encoders[col].categories_[0]]
                        
                        # Add encoded columns
                        for i, new_col in enumerate(encoded_cols):
                            df_copy[new_col] = encoded_data[:, i]
                        
                        # Drop original column
                        df_copy = df_copy.drop(columns=[col])
                        
                    except Exception as e:
                        print(f"Error encoding column {col}: {str(e)}")
        
        self.preprocessing_history.append({
            'operation': 'encoding',
            'method': method,
            'columns': list(columns)
        })
        
        return df_copy
    
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


def test_car_dataset_encoding():
    """Test categorical encoding specifically for the car price dataset"""
    
    # Read the CSV file
    try:
        df = pd.read_csv('car_price_dataset.csv')
        print("Original Dataset Info:")
        print("\nShape:", df.shape)
        print("\nData Types:")
        print(df.dtypes)
        
        # Show sample of original categorical columns
        categorical_cols = ['Brand', 'Model', 'Fuel_Type', 'Transmission']
        print("\nSample of original categorical columns:")
        print(df[categorical_cols].head())
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Apply label encoding
        df_encoded = preprocessor.encode_categorical(
            df, 
            method='label',
            columns=categorical_cols
        )
        
        print("\n" + "="*50)
        print("After Label Encoding:")
        print("\nShape:", df_encoded.shape)
        print("\nData Types:")
        print(df_encoded.dtypes)
        
        # Show sample of encoded columns
        print("\nSample of encoded categorical columns:")
        print(df_encoded[categorical_cols].head())
        
        # Show encoding mappings
        print("\nEncoding Mappings:")
        for col in categorical_cols:
            if col in preprocessor.encoders:
                print(f"\n{col} mapping:")
                for i, label in enumerate(preprocessor.encoders[col].classes_):
                    print(f"{label}: {i}")
        
        # Verify changes
        print("\nVerification:")
        for col in categorical_cols:
            original_unique = len(df[col].unique())
            encoded_unique = len(df_encoded[col].unique())
            print(f"{col}: {original_unique} unique values → {encoded_unique} encoded values")
            
        return df_encoded
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    encoded_df = test_car_dataset_encoding()
    
    
    if encoded_df is not None:
        # Save the encoded dataset
        try:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f'processed_car_price_dataset_{timestamp}.csv'
            encoded_df.to_csv(output_filename, index=False)
            print(f"\nSaved encoded dataset to: {output_filename}")
        except Exception as e:
            print(f"Error saving file: {str(e)}")