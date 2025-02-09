import streamlit as st
import torch
import pandas as pd
from pathlib import Path
import os
import numpy as np
import warnings

from utils.visualization import DataVisualizer
from utils.directory_handler import DirectoryHandler

warnings.filterwarnings('ignore', message='Examining the path of torch.classes.*')



# Page config should be the first streamlit command
st.set_page_config(
    page_title="AIForge",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Novapool/AIForge',
        'Report a bug': 'https://github.com/Novapool/AIForge/issues',
        'About': 'AIForge - Making AI Development Accessible'
    }
)

def main():
    # Sidebar
    with st.sidebar:
        st.title("AIForge")
        
        # Main navigation
        page = st.radio(
            "Navigation",
            ["Data Management", "Model Development", "Training", "Results"],
            index=0
        )
        
        # System info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### System Info")
        if torch.cuda.is_available():
            st.sidebar.success(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.sidebar.warning("GPU: Not Available")
        
    # Main content
    if page == "Data Management":
        render_data_management()
    elif page == "Model Development":
        render_model_development()
    elif page == "Training":
        render_training()
    else:
        render_results()

def render_data_management():
    st.header("Data Management")
    
    # Initialize session state
    if 'directory_path' not in st.session_state:
        st.session_state['directory_path'] = ''
    
    tabs = st.tabs(["Upload & Preview", "Analysis & Visualization", "Preprocessing", "TensorBoard"])
    
    with tabs[0]:  # Upload & Preview
        st.write("### Data Upload")
        upload_type = st.radio(
            "Choose upload type:",
            ["Single File", "Directory"],
            horizontal=True
        )
        
        if upload_type == "Directory":
            col1, col2 = st.columns([3, 1])
            with col1:
                # Display the path from session state
                st.text_input(
                    "Selected Directory:", 
                    value=st.session_state['directory_path'],
                    disabled=True,
                    key='directory_display'
                )
            with col2:
                if st.button("Browse Directory", key="browse_btn"):
                    directory = DirectoryHandler.select_directory()
                    if directory:
                        st.session_state['directory_path'] = directory
                        st.rerun()  # Updated from experimental_rerun()
            
            if st.session_state['directory_path']:
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
                        
                        processed_dfs = []
                        for name, df in data_files.items():
                            with file_tabs[list(data_files.keys()).index(name)]:
                                processed_dfs.append(df)
                                
                                st.success(f"Successfully loaded dataset with shape: {df.shape}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("Data Preview")
                                    st.dataframe(df.head(), use_container_width=True)
                                
                                with col2:
                                    st.subheader("Data Info")
                                    summary_stats = DataVisualizer.generate_summary_stats(df)
                                    st.write("Summary Statistics:")
                                    for key, value in summary_stats.items():
                                        if key == 'memory_usage':
                                            st.metric(key.replace('_', ' ').title(), f"{value:.2f} MB")
                                        else:
                                            st.metric(key.replace('_', ' ').title(), value)
                        
                        # Add batch summary if multiple files are loaded
                        if len(processed_dfs) > 1:
                            st.write("### Batch Summary")
                            batch_stats = DataVisualizer.process_batch_files(processed_dfs)
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Files", batch_stats["total_files"])
                            with col2:
                                st.metric("Total Rows", batch_stats["total_rows"])
                            with col3:
                                st.metric("Total Memory Usage", f"{batch_stats['memory_usage']:.2f} MB")
                        
                        # Store in session state
                        st.session_state['current_dfs'] = data_files
                
        else:  # Single File upload logic remains the same
            uploaded_file = st.file_uploader(
                "Upload your dataset", 
                type=['csv', 'xlsx', 'png', 'jpg', 'jpeg'],
                help="Supported formats: CSV, Excel, Images"
            )
            if uploaded_file:
                if uploaded_file.name.endswith(('.csv', '.xlsx')):
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        st.session_state['current_dfs'] = {uploaded_file.name: df}
                        
                        st.success(f"Successfully loaded dataset with shape: {df.shape}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Data Preview")
                            st.dataframe(df.head(), use_container_width=True)
                        
                        with col2:
                            st.subheader("Data Info")
                            summary_stats = DataVisualizer.generate_summary_stats(df)
                            st.write("Summary Statistics:")
                            for key, value in summary_stats.items():
                                if key == 'memory_usage':
                                    st.metric(key.replace('_', ' ').title(), f"{value:.2f} MB")
                                else:
                                    st.metric(key.replace('_', ' ').title(), value)
                    
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
                else:
                    st.image(uploaded_file, caption=uploaded_file.name)

def render_model_development():
    st.header("Model Development")
    st.info("Model development features coming soon!")
    
    # Placeholder for model configuration
    with st.expander("Model Configuration"):
        st.selectbox("Model Architecture", ["ResNet18", "ResNet50", "VGG16"])
        st.number_input("Batch Size", min_value=1, value=32)
        st.number_input("Learning Rate", min_value=0.0001, value=0.001, format="%f")

def render_training():
    st.header("Training")
    st.info("Training features coming soon!")
    
    # Placeholder for training progress
    with st.expander("Training Progress"):
        progress_bar = st.progress(0)
        st.metric("Training Loss", "0.325", "-0.015")
        st.metric("Validation Accuracy", "89.5%", "+1.2%")

def render_results():
    st.header("Results")
    st.info("Results visualization coming soon!")

if __name__ == "__main__":
    main()