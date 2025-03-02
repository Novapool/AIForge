from typing import Dict, List, Optional, Any, Union
import pandas as pd
import streamlit as st
from datetime import datetime

class SessionStateManager:
    """
    Centralized manager for Streamlit session state.
    Handles initialization, access, and updates to session state variables.
    """
    
    @staticmethod
    def initialize():
        """Initialize all session state variables with default values"""
        # File management
        if 'file_registry' not in st.session_state:
            st.session_state['file_registry'] = {}
        
        if 'active_file_id' not in st.session_state:
            st.session_state['active_file_id'] = None
            
        # Directory management
        if 'directory_path' not in st.session_state:
            st.session_state['directory_path'] = ''
            
        # Model management
        if 'model_config' not in st.session_state:
            st.session_state['model_config'] = {
                'initialized': False,
                'type': None,
                'params': {},
                'training_status': None
            }
            
        # Training management
        if 'training_history' not in st.session_state:
            st.session_state['training_history'] = []
            
        # Tab navigation
        if 'current_tab' not in st.session_state:
            st.session_state['current_tab'] = "Data Management"
            
        # Preprocessing states
        if 'preprocessing_registry' not in st.session_state:
            st.session_state['preprocessing_registry'] = {}
    
    @staticmethod
    def register_file(file_id: str, dataframe: pd.DataFrame, source: str, metadata: Optional[Dict] = None) -> str:
        """
        Register a new file in the file registry
        
        Args:
            file_id: Unique identifier for the file
            dataframe: The pandas DataFrame containing the file data
            source: Source of the file (e.g., 'upload', 'directory', 'preprocessing')
            metadata: Optional additional metadata about the file
            
        Returns:
            The file_id of the registered file (may be different if a duplicate was found)
        """
        if 'file_registry' not in st.session_state:
            SessionStateManager.initialize()
        
        # Check if a file with the same original filename already exists
        # Only do this check for non-preprocessing sources to avoid duplicating uploaded files
        if source != 'preprocessing' and metadata and 'original_filename' in metadata:
            original_filename = metadata.get('original_filename')
            for existing_id, existing_file in st.session_state['file_registry'].items():
                if (existing_file['metadata'].get('original_filename') == original_filename and 
                    existing_file['source'] == source):
                    # Update the existing file instead of creating a new one
                    st.session_state['file_registry'][existing_id]['dataframe'] = dataframe
                    st.session_state['file_registry'][existing_id]['last_accessed'] = datetime.now()
                    st.session_state['file_registry'][existing_id]['shape'] = dataframe.shape
                    st.session_state['file_registry'][existing_id]['columns'] = dataframe.columns.tolist()
                    st.session_state['file_registry'][existing_id]['dtypes'] = {
                        col: str(dtype) for col, dtype in dataframe.dtypes.items()
                    }
                    # Update metadata if needed
                    if metadata:
                        st.session_state['file_registry'][existing_id]['metadata'].update(metadata)
                    
                    SessionStateManager.set_active_file(existing_id)
                    return existing_id
            
        # Create file entry with metadata
        st.session_state['file_registry'][file_id] = {
            'dataframe': dataframe,
            'source': source,
            'upload_time': datetime.now(),
            'last_accessed': datetime.now(),
            'shape': dataframe.shape,
            'columns': dataframe.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in dataframe.dtypes.items()},
            'metadata': metadata or {},
            'preprocessing_history': [],
            'visualization_config': {},
            'model_association': None
        }
        
        # Set as active file if no active file exists
        if not st.session_state['active_file_id']:
            SessionStateManager.set_active_file(file_id)
            
        return file_id
    
    @staticmethod
    def set_active_file(file_id: str) -> None:
        """
        Set the active file
        
        Args:
            file_id: ID of the file to set as active
        """
        if file_id in st.session_state['file_registry']:
            st.session_state['active_file_id'] = file_id
            # Update last accessed time
            st.session_state['file_registry'][file_id]['last_accessed'] = datetime.now()
    
    @staticmethod
    def get_active_file() -> Optional[Dict]:
        """
        Get the currently active file data
        
        Returns:
            Dictionary containing file data and metadata, or None if no active file
        """
        if not st.session_state.get('active_file_id'):
            return None
            
        file_id = st.session_state['active_file_id']
        if file_id not in st.session_state['file_registry']:
            st.session_state['active_file_id'] = None
            return None
            
        return {
            'id': file_id,
            **st.session_state['file_registry'][file_id]
        }
    
    @staticmethod
    def get_dataframe(file_id: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get a dataframe by ID or the active dataframe if no ID is provided
        
        Args:
            file_id: Optional ID of the file to retrieve
            
        Returns:
            The requested DataFrame or None if not found
        """
        if file_id is None:
            active_file = SessionStateManager.get_active_file()
            return active_file['dataframe'] if active_file else None
            
        if file_id in st.session_state['file_registry']:
            return st.session_state['file_registry'][file_id]['dataframe']
            
        return None
    
    @staticmethod
    def update_dataframe(dataframe: pd.DataFrame, file_id: Optional[str] = None) -> None:
        """
        Update a dataframe in the registry
        
        Args:
            dataframe: The updated DataFrame
            file_id: Optional ID of the file to update (uses active file if None)
        """
        target_id = file_id or st.session_state.get('active_file_id')
        if not target_id or target_id not in st.session_state['file_registry']:
            return
            
        # Update the dataframe and related metadata
        st.session_state['file_registry'][target_id]['dataframe'] = dataframe
        st.session_state['file_registry'][target_id]['last_modified'] = datetime.now()
        st.session_state['file_registry'][target_id]['shape'] = dataframe.shape
        st.session_state['file_registry'][target_id]['columns'] = dataframe.columns.tolist()
        st.session_state['file_registry'][target_id]['dtypes'] = {
            col: str(dtype) for col, dtype in dataframe.dtypes.items()
        }
    
    @staticmethod
    def delete_file(file_id: str) -> bool:
        """
        Delete a file from the registry
        
        Args:
            file_id: ID of the file to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if file_id not in st.session_state['file_registry']:
            return False
            
        # Remove the file
        del st.session_state['file_registry'][file_id]
        
        # Update active file if needed
        if st.session_state['active_file_id'] == file_id:
            if st.session_state['file_registry']:
                # Set first available file as active
                st.session_state['active_file_id'] = next(iter(st.session_state['file_registry']))
            else:
                st.session_state['active_file_id'] = None
                
        return True
    
    @staticmethod
    def list_files() -> List[Dict]:
        """
        List all files in the registry with their metadata
        
        Returns:
            List of dictionaries containing file information
        """
        return [
            {
                'id': file_id,
                'upload_time': info['upload_time'],
                'last_accessed': info['last_accessed'],
                'source': info['source'],
                'shape': info['shape'],
                'is_active': file_id == st.session_state.get('active_file_id')
            }
            for file_id, info in st.session_state.get('file_registry', {}).items()
        ]
    
    @staticmethod
    def add_preprocessing_step(file_id: str, operation: Dict) -> None:
        """
        Add a preprocessing step to a file's history
        
        Args:
            file_id: ID of the file
            operation: Dictionary describing the preprocessing operation
        """
        if file_id not in st.session_state['file_registry']:
            return
            
        # Add operation with timestamp
        operation_with_time = {
            **operation,
            'timestamp': datetime.now()
        }
        
        st.session_state['file_registry'][file_id]['preprocessing_history'].append(operation_with_time)
    
    @staticmethod
    def set_model_config(config: Dict) -> None:
        """
        Set the model configuration
        
        Args:
            config: Dictionary containing model configuration
        """
        st.session_state['model_config'] = {
            **st.session_state.get('model_config', {}),
            **config,
            'initialized': True
        }
    
    @staticmethod
    def get_model_config() -> Dict:
        """
        Get the current model configuration
        
        Returns:
            Dictionary containing model configuration
        """
        return st.session_state.get('model_config', {'initialized': False})
    
    @staticmethod
    def associate_model_with_file(file_id: str) -> None:
        """
        Associate the current model with a specific file
        
        Args:
            file_id: ID of the file to associate with the model
        """
        if file_id not in st.session_state['file_registry']:
            return
            
        st.session_state['file_registry'][file_id]['model_association'] = {
            'timestamp': datetime.now(),
            'model_config': st.session_state.get('model_config', {})
        }
    
    @staticmethod
    def set_current_tab(tab_name: str) -> None:
        """
        Set the current tab
        
        Args:
            tab_name: Name of the current tab
        """
        st.session_state['current_tab'] = tab_name
    
    @staticmethod
    def get_current_tab() -> str:
        """
        Get the current tab name
        
        Returns:
            Name of the current tab
        """
        return st.session_state.get('current_tab', "Data Management")
        
    @staticmethod
    def migrate_legacy_data():
        """
        Migrate data from the old session state structure to the new one.
        This helps with backward compatibility during the transition.
        """
        # Migrate current_dfs to file_registry
        if 'current_dfs' in st.session_state and st.session_state['current_dfs']:
            for filename, df in st.session_state['current_dfs'].items():
                # Check if this file is already in the registry
                if not any(file_id.startswith(filename) for file_id in st.session_state.get('file_registry', {}).keys()):
                    # Generate a unique file ID
                    file_id = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Register the file
                    SessionStateManager.register_file(
                        file_id=file_id,
                        dataframe=df,
                        source='legacy_migration',
                        metadata={
                            'original_filename': filename,
                            'migrated_from': 'current_dfs'
                        }
                    )
            
            # If active_dataset is set, try to find the corresponding file in the registry
            if 'active_dataset' in st.session_state and st.session_state['active_dataset']:
                active_name = st.session_state['active_dataset']
                for file_id in st.session_state.get('file_registry', {}).keys():
                    if file_id.startswith(active_name):
                        SessionStateManager.set_active_file(file_id)
                        break
