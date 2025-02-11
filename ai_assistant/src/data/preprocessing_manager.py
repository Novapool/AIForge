from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Optional
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
                    **operation,
                    "timestamp": current_time
                }
                for operation in operations
            ],
            "completion_info": {
                "timestamp": None,
                "output_file": None
            }
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
            
        return file_id
    
    def load_state(self, file_id: str) -> Optional[Dict]:
        """Load preprocessing state from JSON file"""
        state_file = self._get_state_file_path(file_id)
        
        if state_file.exists():
            with open(state_file, 'r') as f:
                return json.load(f)
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
        Apply preprocessing operations from state file to the DataFrame
        
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
            # Apply each operation in sequence
            for operation in state["operations"]:
                op_type = operation["operation"]
                
                if op_type == "encoding":
                    df = self.preprocessor.encode_categorical(
                        df,
                        method=operation["method"],
                        columns=operation["columns"]
                    )
                elif op_type == "normalization":
                    df = self.preprocessor.normalize_data(
                        df,
                        method=operation["method"],
                        columns=operation["columns"]
                    )
                elif op_type == "missing_values":
                    df = self.preprocessor.handle_missing_values(
                        df,
                        strategy=operation["strategy"],
                        columns=operation["columns"]
                    )
                elif op_type == "outliers":
                    df = self.preprocessor.remove_outliers(
                        df,
                        method=operation["method"],
                        threshold=operation.get("threshold", 3.0),
                        columns=operation["columns"]
                    )
            
            # Update state file with completion info
            state["status"] = "completed"
            state["completion_info"]["timestamp"] = datetime.now().isoformat()
            
            with open(self._get_state_file_path(file_id), 'w') as f:
                json.dump(state, f, indent=2)
            
            return df
            
        except Exception as e:
            # Update state file with error status
            state["status"] = "error"
            state["completion_info"]["error"] = str(e)
            
            with open(self._get_state_file_path(file_id), 'w') as f:
                json.dump(state, f, indent=2)
            
            raise e
    
    def get_state_status(self, file_id: str) -> Optional[str]:
        """Get the current status of a state file"""
        state = self.load_state(file_id)
        return state["status"] if state else None
