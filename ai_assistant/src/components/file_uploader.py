from pathlib import Path
import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Tuple
import sys
import os
from datetime import datetime

# Add the parent directory to the path so we can import the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.session_state_manager import SessionStateManager
from utils.directory_handler import DirectoryHandler

def file_uploader_component():
    """Component for uploading files and adding them to the registry"""
    st.write("### Upload Data")
    
    upload_type = st.radio(
        "Choose upload type:",
        ["Single File", "Directory"],
        horizontal=True
    )
    
    if upload_type == "Single File":
        uploaded_file = st.file_uploader(
            "Upload your dataset", 
            type=['csv', 'xlsx', 'png', 'jpg', 'jpeg'],
            help="Supported formats: CSV, Excel, Images"
        )
        
        if uploaded_file:
            try:
                # Process the uploaded file
                if uploaded_file.name.endswith(('.csv', '.xlsx')):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    # Generate a unique file ID
                    file_id = f"{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Save a copy of the file to disk for preprocessing
                    processed_data_dir = Path("ai_assistant/processed_data")
                    processed_data_dir.mkdir(parents=True, exist_ok=True)
                    file_path = processed_data_dir / uploaded_file.name
                    df.to_csv(file_path, index=False)
                    
                    # Register the file
                    registered_id = SessionStateManager.register_file(
                        file_id=file_id,
                        dataframe=df,
                        source='upload',
                        metadata={
                            'original_filename': uploaded_file.name,
                            'file_size': uploaded_file.size,
                            'upload_method': 'single_file',
                            'file_path': str(file_path)
                        }
                    )
                    
                    st.success(f"Successfully loaded dataset: {uploaded_file.name}")
                    
                    # Display file preview
                    st.write("### Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    
                else:  # Image file
                    st.image(uploaded_file, caption=uploaded_file.name)
                    st.info("Image preview shown. Image processing functionality coming soon.")
                    
            except Exception as e:
                # Handle the case where the error is a dictionary
                if isinstance(e, dict) or (hasattr(e, 'args') and e.args and isinstance(e.args[0], dict)):
                    error_dict = e if isinstance(e, dict) else e.args[0]
                    st.error("Error loading file: Data format issue. Please check the file format.")
                    st.write(error_dict)  # Use st.write instead of st.json
                else:
                    st.error(f"Error loading file: {str(e)}")
                
    else:  # Directory upload
        col1, col2 = st.columns([3, 1])
        with col1:
            directory_path = st.text_input(
                "Selected Directory:", 
                value=st.session_state.get('directory_path', ''),
                disabled=True,
                key='directory_display'
            )
        with col2:
            if st.button("Browse Directory", key="browse_btn"):
                directory = DirectoryHandler.select_directory()
                if directory:
                    st.session_state['directory_path'] = directory
                    st.rerun()  # Updated from experimental_rerun()
        
        if st.session_state.get('directory_path', ''):
            st.write("### Dataset Structure")
            # Get and display directory structure
            structure = DirectoryHandler.get_directory_structure(st.session_state['directory_path'])
            if structure:
                DirectoryHandler.display_structure(structure)
                
                # Load and process data files
                data_files = DirectoryHandler.load_data_files(structure)
                
                if data_files:
                    st.write("### Data Files")
                    # Create tabs for each data file
                    file_tabs = st.tabs([f"File: {name}" for name in data_files.keys()])
                    
                    for name, df in data_files.items():
                        with file_tabs[list(data_files.keys()).index(name)]:
                            # Generate a unique file ID
                            file_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            
                            # Save a copy of the file to disk for preprocessing
                            processed_data_dir = Path("ai_assistant/processed_data")
                            processed_data_dir.mkdir(parents=True, exist_ok=True)
                            file_path = processed_data_dir / name
                            df.to_csv(file_path, index=False)
                            
                            # Register the file
                            registered_id = SessionStateManager.register_file(
                                file_id=file_id,
                                dataframe=df,
                                source='directory',
                                metadata={
                                    'original_filename': name,
                                    'directory_path': st.session_state['directory_path'],
                                    'upload_method': 'directory',
                                    'file_path': str(file_path)
                                }
                            )
                            
                            st.success(f"Successfully loaded dataset with shape: {df.shape}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Data Preview")
                                st.dataframe(df.head(), use_container_width=True)
                            
                            with col2:
                                st.subheader("Data Info")
                                from utils.visualization import DataVisualizer
                                summary_stats = DataVisualizer.generate_summary_stats(df)
                                st.write("Summary Statistics:")
                                for key, value in summary_stats.items():
                                    if key == 'memory_usage':
                                        st.metric(key.replace('_', ' ').title(), f"{value:.2f} MB")
                                    elif key not in ['numeric_stats', 'categorical_stats', 'dtypes', 'memory_usage_by_column']:
                                        st.metric(key.replace('_', ' ').title(), value)

def file_management_section():
    """Component for managing files in the registry"""
    st.write("### File Management")
    
    files = SessionStateManager.list_files()
    if not files:
        st.info("No files in registry. Upload files to get started.")
        return
    
    # Display files in a table
    file_data = []
    for file in files:
        file_data.append({
            "ID": file['id'],
            "Rows": file['shape'][0],
            "Columns": file['shape'][1],
            "Source": file['source'],
            "Upload Time": file['upload_time'].strftime("%Y-%m-%d %H:%M:%S"),
            "Active": "✓" if file['is_active'] else ""
        })
    
    st.dataframe(pd.DataFrame(file_data), use_container_width=True)
    
    # File operations
    st.write("### File Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select file to operate on
        from components.file_selector import file_selector
        selected_file_id = file_selector(key="file_ops_selector")
        
    with col2:
        if selected_file_id:
            if st.button("Delete Selected File"):
                if SessionStateManager.delete_file(selected_file_id):
                    st.success(f"File {selected_file_id} deleted")
                    st.rerun()
                else:
                    st.error("Failed to delete file")
