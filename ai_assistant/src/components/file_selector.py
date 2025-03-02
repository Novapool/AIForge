import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Callable
import sys
import os

# Add the parent directory to the path so we can import the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.session_state_manager import SessionStateManager

def file_selector(key: str = "file_selector", on_change: Optional[Callable] = None) -> Optional[str]:
    """
    Reusable file selector component
    
    Args:
        key: Unique key for the selectbox
        on_change: Optional callback function when selection changes
        
    Returns:
        Selected file ID or None if no files available
    """
    # Get list of files
    files = SessionStateManager.list_files()
    
    if not files:
        st.warning("No files available. Please upload a file first.")
        return None
    
    # Create options for selectbox
    file_options = [f"{f['id']} ({f['shape'][0]} rows, {f['shape'][1]} cols)" for f in files]
    file_ids = [f['id'] for f in files]
    
    # Find index of active file
    active_file = SessionStateManager.get_active_file()
    default_index = 0
    if active_file:
        try:
            default_index = file_ids.index(active_file['id'])
        except ValueError:
            default_index = 0
    
    # Define callback function
    def update_active_file():
        selected_idx = file_options.index(st.session_state[key])
        selected_id = file_ids[selected_idx]
        SessionStateManager.set_active_file(selected_id)
        if on_change:
            on_change()
    
    # Create selectbox
    selected_option = st.selectbox(
        "Select File:",
        options=file_options,
        index=default_index,
        key=key,
        on_change=update_active_file
    )
    
    if selected_option:
        selected_idx = file_options.index(selected_option)
        return file_ids[selected_idx]
    
    return None

def file_info_display():
    """Display information about the currently active file"""
    active_file = SessionStateManager.get_active_file()
    
    if not active_file:
        st.info("No active file selected.")
        return
    
    # Display file information
    st.success(f"Active File: {active_file['id']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", active_file['shape'][0])
    with col2:
        st.metric("Columns", active_file['shape'][1])
    with col3:
        st.metric("Source", active_file['source'])
    
    # Display additional metadata if available
    if active_file.get('metadata'):
        with st.expander("File Metadata"):
            for key, value in active_file['metadata'].items():
                st.write(f"**{key}:** {value}")
    
    # Display preprocessing history if available
    if active_file.get('preprocessing_history'):
        with st.expander("Preprocessing History"):
            for i, step in enumerate(active_file['preprocessing_history']):
                st.write(f"**Step {i+1}:** {step['operation']} ({step['timestamp']})")
                st.write(f"Parameters: {', '.join([f'{k}={v}' for k, v in step.items() if k not in ['operation', 'timestamp']])}")
