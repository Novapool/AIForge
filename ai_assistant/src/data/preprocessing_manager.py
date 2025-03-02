from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
from .data_preprocessor import DataPreprocessor

class PreprocessingManager:
    """Manages preprocessing state and operations through JSON files"""
    
    def __init__(self, base_path: str = "ai_assistant/preprocessing_states"):
        """Initialize the PreprocessingManager with a base path for state files"""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.preprocessor = DataPreprocessor()
    
    def _generate_file_id(self, dataset_name: str) -> str:
        """Generate a unique file ID using dataset name and timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{dataset_name.split('.')[0]}_{timestamp}"
    
    def _get_state_file_path(self, file_id: str) -> Path:
        """Get the full path for a state file"""
        return self.base_path / f"preprocessing_state_{file_id}.json"
    
    def _convert_to_serializable(self, obj: any) -> any:
        """Convert numpy and pandas types to JSON serializable types"""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8, np.uint16,
            np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return obj
    
    def save_state(self, dataset_name: str, operations: List[Dict]) -> str:
        """
        Save preprocessing operations to a JSON state file
        
        Args:
            dataset_name: Name of the dataset being processed
            operations: List of preprocessing operations to apply
            
        Returns:
            file_id: Unique identifier for the state file
        """
        file_id = self._generate_file_id(dataset_name)
        state_file = self._get_state_file_path(file_id)
        
        current_time = datetime.now().isoformat()
        state = {
            "dataset_name": dataset_name,
            "file_id": file_id,
            "created_at": current_time,
            "last_modified": current_time,
            "status": "pending",
            "operations": [
                {
                    **self._convert_to_serializable(operation),
                    "timestamp": current_time
                }
                for operation in operations
            ],
            "transformation_rules": {},
            "completion_info": {
                "timestamp": None,
                "output_file": None
            }
        }
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except TypeError as e:
            print(f"JSON serialization error: {str(e)}")
            raise
            
        return file_id
    
    def load_state(self, file_id: str) -> Optional[Dict]:
        """Load preprocessing state from JSON file"""
        state_file = self._get_state_file_path(file_id)
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error reading state file {state_file}: {str(e)}")
                return None
        return None
    
    def list_states(self) -> List[Dict]:
        """List all preprocessing state files with their basic information"""
        states = []
        for state_file in self.base_path.glob("preprocessing_state_*.json"):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    states.append({
                        "file_id": state["file_id"],
                        "dataset_name": state["dataset_name"],
                        "created_at": state["created_at"],
                        "last_modified": state["last_modified"],
                        "status": state["status"],
                        "operation_count": len(state["operations"])
                    })
            except Exception as e:
                print(f"Error reading state file {state_file}: {str(e)}")
        
        return sorted(states, key=lambda x: x["created_at"], reverse=True)
    
    def delete_state(self, file_id: str) -> bool:
        """Delete a preprocessing state file"""
        state_file = self._get_state_file_path(file_id)
        if state_file.exists():
            state_file.unlink()
            return True
        return False
    
    def view_state_history(self, file_id: str) -> Optional[Dict]:
        """View the complete history of operations for a state file"""
        return self.load_state(file_id)
    
    def apply_operations(self, df: pd.DataFrame, file_id: str) -> pd.DataFrame:
        """
        Apply preprocessing operations from state file to the DataFrame and save to CSV
        
        Args:
            df: Input DataFrame
            file_id: ID of the state file containing operations
            
        Returns:
            Processed DataFrame
        """
        state = self.load_state(file_id)
        if not state or state["status"] != "pending":
            raise ValueError(f"Invalid state or status for file_id: {file_id}")
        
        try:
            # Apply each operation in sequence and collect transformation rules
            transformation_rules = {
                'encoding_maps': {},
                'normalizations': {},
                'missing_value_rules': {},
                'outlier_rules': {}
            }
            
            for operation in state["operations"]:
                op_type = operation["operation"]
                
                if op_type == "encoding":
                    df, details = self.preprocessor.encode_categorical(
                        df,
                        method=operation["method"],
                        columns=operation["columns"]
                    )
                    if details.get('encoding_maps'):
                        transformation_rules['encoding_maps'].update(details['encoding_maps'])
                
                elif op_type == "normalization":
                    df, details = self.preprocessor.normalize_data(
                        df,
                        method=operation["method"],
                        columns=operation["columns"]
                    )
                    if details.get('normalizations'):
                        transformation_rules['normalizations'].update(details['normalizations'])
                
                elif op_type == "missing_values":
                    df, details = self.preprocessor.handle_missing_values(
                        df,
                        strategy=operation["strategy"],
                        columns=operation["columns"]
                    )
                    if details.get('missing_value_rules'):
                        transformation_rules['missing_value_rules'].update(details['missing_value_rules'])
                
                elif op_type == "outliers":
                    df, details = self.preprocessor.remove_outliers(
                        df,
                        method=operation["method"],
                        threshold=operation.get("threshold", 3.0),
                        columns=operation["columns"]
                    )
                    if details.get('outlier_rules'):
                        transformation_rules['outlier_rules'].update(details['outlier_rules'])
            
            # Save processed DataFrame to CSV
            processed_data_dir = Path("ai_assistant/processed_data")
            processed_data_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"processed_{state['dataset_name']}_{timestamp}.csv"
            output_path = processed_data_dir / output_filename
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            # Update state file with transformation rules and output file path
            state["status"] = "completed"
            state["transformation_rules"] = self._convert_to_serializable(transformation_rules)
            state["completion_info"]["timestamp"] = datetime.now().isoformat()
            state["completion_info"]["output_file"] = str(output_path)
            
            try:
                with open(self._get_state_file_path(file_id), 'w') as f:
                    json.dump(state, f, indent=2)
            except TypeError as e:
                print(f"JSON serialization error: {str(e)}")
                raise
            
            return df
            
        except Exception as e:
            # Update state file with error status
            state["status"] = "error"
            state["completion_info"]["error"] = str(e)
            
            try:
                with open(self._get_state_file_path(file_id), 'w') as f:
                    json.dump(state, f, indent=2)
            except TypeError as e:
                print(f"JSON serialization error: {str(e)}")
                raise
            
            raise e
    
    def apply_json_to_csv(self, input_csv: str, state_file_id: str, output_csv: Optional[str] = None) -> str:
        """
        Apply transformations from a JSON state file to a CSV file
        
        Args:
            input_csv: Path to input CSV file
            state_file_id: ID of the state file containing transformations
            output_csv: Optional path for output CSV. If None, will generate one
            
        Returns:
            Path to the transformed CSV file
        """
        # Load state file
        state = self.load_state(state_file_id)
        if not state or not state.get("transformation_rules"):
            raise ValueError(f"No transformation rules found in state file: {state_file_id}")
        
        # Read input CSV
        df = pd.read_csv(input_csv)
        
        # Generate output path if not provided
        if output_csv is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = f"{input_csv}_{timestamp}.csv"
        
        try:
            rules = state["transformation_rules"]
            
            # Apply encoding transformations
            for col, mapping in rules.get('encoding_maps', {}).items():
                if mapping['type'] == 'label':
                    df[col] = df[col].astype(str).map(mapping['values'])
                elif mapping['type'] == 'onehot':
                    # Create zero-filled columns for all categories
                    for cat in mapping['categories']:
                        df[f"{col}_{cat}"] = 0
                    # Set 1 for matching categories
                    for idx, val in df[col].items():
                        if str(val) in mapping['categories']:
                            df.at[idx, f"{col}_{val}"] = 1
                    df = df.drop(columns=[col])
            
            # Apply normalization transformations
            for col, params in rules.get('normalizations', {}).items():
                if params['type'] == 'standard':
                    df[col] = (df[col] - params['mean']) / params['std']
                elif params['type'] == 'minmax':
                    df[col] = (df[col] - params['min']) / params['scale']
                elif params['type'] == 'robust':
                    df[col] = (df[col] - params['center']) / params['scale']
            
            # Apply missing value rules
            for col, rule in rules.get('missing_value_rules', {}).items():
                df[col] = df[col].fillna(rule['fill_value'])
            
            # Apply outlier rules
            for col, rule in rules.get('outlier_rules', {}).items():
                if rule['method'] == 'zscore':
                    z_scores = np.abs((df[col] - rule['params']['mean']) / rule['params']['std'])
                    df = df[z_scores < rule['threshold']]
                elif rule['method'] == 'iqr':
                    Q1, Q3 = rule['params']['Q1'], rule['params']['Q3']
                    IQR = rule['params']['IQR']
                    df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
            
            # Save transformed DataFrame
            df.to_csv(output_csv, index=False)
            
            # Update state file with output file info
            state["completion_info"]["output_file"] = output_csv
            try:
                with open(self._get_state_file_path(state_file_id), 'w') as f:
                    json.dump(state, f, indent=2)
            except TypeError as e:
                print(f"JSON serialization error: {str(e)}")
                raise
            
            return output_csv
            
        except Exception as e:
            state["status"] = "error"
            state["completion_info"]["error"] = str(e)
            try:
                with open(self._get_state_file_path(state_file_id), 'w') as f:
                    json.dump(state, f, indent=2)
            except TypeError as e:
                print(f"JSON serialization error: {str(e)}")
                raise
            raise e
    
    def get_state_status(self, file_id: str) -> Optional[str]:
        """Get the current status of a state file"""
        state = self.load_state(file_id)
        return state["status"] if state else None
