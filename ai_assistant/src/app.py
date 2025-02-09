import streamlit as st
import torch
import pandas as pd
from pathlib import Path
import os
import numpy as np
import warnings
import json

from utils.visualization import DataVisualizer
from utils.directory_handler import DirectoryHandler
from data.data_preprocessor import DataPreprocessor


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
    with tabs[1]:  # Analysis & Visualization
        st.write("Analysis & Visualization features coming soon!")
        
    with tabs[2]:  # Preprocessing
        render_preprocessing_tab()
        
    with tabs[3]:  # TensorBoard
        st.write("TensorBoard integration coming soon!")

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

def render_preprocessing_tab():
    """Render the preprocessing tab in the Streamlit interface"""
    st.subheader("Data Preprocessing")
    
    # Check if data is loaded
    if 'current_dfs' not in st.session_state:
        st.warning("Please load data first in the Upload & Preview tab")
        return
        
    # Select dataset if multiple are loaded
    dataset_names = list(st.session_state['current_dfs'].keys())
    selected_dataset = st.selectbox("Select Dataset", dataset_names)
    df = st.session_state['current_dfs'][selected_dataset].copy()  # Important: create a copy
    
    # Initialize preprocessor if not exists
    if 'preprocessor' not in st.session_state:
        st.session_state['preprocessor'] = DataPreprocessor()
    
    # Create tabs for different preprocessing operations
    preprocess_tabs = st.tabs([
        "Categorical Encoding",
        "Normalization",
        "Missing Values",
        "Outlier Removal"
    ])
    
    # Track if any preprocessing was applied
    preprocessing_applied = False
    
    # Categorical Encoding Tab
    with preprocess_tabs[0]:
        st.write("### Encode Categorical Variables")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            col1, col2 = st.columns(2)
            with col1:
                selected_method = st.selectbox(
                    "Select Method",
                    ["label", "onehot"],
                    key="encode_method"
                )
            with col2:
                selected_cols = st.multiselect(
                    "Select Columns",
                    categorical_cols,
                    default=categorical_cols,
                    key="encode_cols"
                )
                
            if st.button("Apply Encoding"):
                try:
                    df = st.session_state['preprocessor'].encode_categorical(
                        df,
                        method=selected_method,
                        columns=selected_cols
                    )
                    # Update the DataFrame in session state
                    st.session_state['current_dfs'][selected_dataset] = df
                    preprocessing_applied = True
                    
                    # Show encoding mappings
                    st.write("### Encoding Mappings")
                    for col in selected_cols:
                        if col in st.session_state['preprocessor'].encoders:
                            encoder = st.session_state['preprocessor'].encoders[col]
                            if hasattr(encoder, 'classes_'):
                                st.write(f"\n{col} mapping:")
                                mapping_df = pd.DataFrame({
                                    'Original': encoder.classes_,
                                    'Encoded': range(len(encoder.classes_))
                                })
                                st.dataframe(mapping_df)
                    
                    st.success("Encoding applied successfully!")
                    
                except Exception as e:
                    st.error(f"Error applying encoding: {str(e)}")
        else:
            st.info("No categorical columns found in the dataset")
    
    # Normalization Tab
    with preprocess_tabs[1]:
        st.write("### Normalize Data")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            selected_method = st.selectbox(
                "Select Method",
                ["standard", "minmax", "robust"],
                key="norm_method"
            )
        with col2:
            selected_cols = st.multiselect(
                "Select Columns",
                numeric_cols,
                default=numeric_cols,
                key="norm_cols"
            )
            
        if st.button("Apply Normalization"):
            try:
                df = st.session_state['preprocessor'].normalize_data(
                    df,
                    method=selected_method,
                    columns=selected_cols
                )
                st.session_state['current_dfs'][selected_dataset] = df
                preprocessing_applied = True
                st.success("Normalization applied successfully!")
                
                # Show sample of normalized data
                st.write("### Sample of Normalized Data")
                st.dataframe(df[selected_cols].head())
            except Exception as e:
                st.error(f"Error applying normalization: {str(e)}")
    
    # Missing Values Tab
    with preprocess_tabs[2]:
        st.write("### Handle Missing Values")
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if missing_cols:
            col1, col2 = st.columns(2)
            with col1:
                selected_strategy = st.selectbox(
                    "Select Strategy",
                    ["mean", "median", "most_frequent", "constant"],
                    key="missing_strategy"
                )
            with col2:
                selected_cols = st.multiselect(
                    "Select Columns",
                    missing_cols,
                    default=missing_cols,
                    key="missing_cols"
                )
            
            if st.button("Apply Missing Value Treatment"):
                try:
                    df = st.session_state['preprocessor'].handle_missing_values(
                        df,
                        strategy=selected_strategy,
                        columns=selected_cols
                    )
                    st.session_state['current_dfs'][selected_dataset] = df
                    preprocessing_applied = True
                    st.success("Missing values handled successfully!")
                    
                    # Show impact of missing value treatment
                    st.write("### Impact of Missing Value Treatment")
                    for col in selected_cols:
                        missing_before = df[col].isnull().sum()
                        if missing_before == 0:
                            st.write(f"✅ {col}: All missing values handled")
                        else:
                            st.warning(f"⚠️ {col}: {missing_before} missing values remain")
                            
                except Exception as e:
                    st.error(f"Error handling missing values: {str(e)}")
        else:
            st.info("No missing values found in the dataset")
    
    # Outlier Removal Tab
    with preprocess_tabs[3]:
        st.write("### Remove Outliers")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_method = st.selectbox(
                "Select Method",
                ["zscore", "iqr"],
                key="outlier_method"
            )
        with col2:
            threshold = st.number_input(
                "Threshold",
                value=3.0,
                step=0.1,
                key="outlier_threshold"
            )
        with col3:
            selected_cols = st.multiselect(
                "Select Columns",
                numeric_cols,
                default=numeric_cols,
                key="outlier_cols"
            )
            
        if st.button("Remove Outliers"):
            try:
                original_shape = df.shape[0]
                df = st.session_state['preprocessor'].remove_outliers(
                    df,
                    method=selected_method,
                    threshold=threshold,
                    columns=selected_cols
                )
                st.session_state['current_dfs'][selected_dataset] = df
                preprocessing_applied = True
                st.success("Outliers removed successfully!")
                
                # Show impact of outlier removal
                st.write("### Impact of Outlier Removal")
                new_shape = df.shape[0]
                removed_rows = original_shape - new_shape
                st.write(f"Rows removed: {removed_rows} ({(removed_rows/original_shape)*100:.2f}% of data)")
                
                # Show statistics for each column
                st.write("### Column Statistics After Outlier Removal")
                stats_df = df[selected_cols].describe()
                st.dataframe(stats_df)
                
            except Exception as e:
                st.error(f"Error removing outliers: {str(e)}")
    
    # Display preprocessing summary
    if preprocessing_applied:
        st.write("---")
        st.write("### Preprocessing Summary")
        summary = st.session_state['preprocessor'].get_preprocessing_summary()
        st.write(f"Total operations performed: {summary['total_operations']}")
        
        if summary['operations']:
            for op in summary['operations']:
                st.write(f"- {op['operation'].title()}: {op['method']} "
                        f"on {len(op['columns'])} columns")
    
    # Save Changes Section
    st.write("---")
    st.write("### Save Processed Data")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        save_dir = st.text_input(
            "Save Directory",
            value="ai_assistant/processed_data",
            disabled=True
        )
    with col2:
        if st.button("Save Changes"):
            if selected_dataset and df is not None:
                # Pass the current DataFrame to save_processed_data
                save_processed_data(df, selected_dataset)
            else:
                st.warning("No data to save. Please load and process data first.")

def save_processed_data(df: pd.DataFrame, original_filename: str) -> None:
    """Save the processed DataFrame to a CSV file"""
    try:
        # Create processed_data directory if it doesn't exist
        save_dir = Path("ai_assistant/processed_data")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_{original_filename.split('.')[0]}_{timestamp}.csv"
        save_path = save_dir / filename
        
        # Important: Ensure all columns are properly formatted
        # Convert any categorical columns that were encoded to integers
        categorical_columns = [col for col in df.columns 
                             if df[col].dtype in ['int64', 'float64'] 
                             and col in st.session_state['preprocessor'].encoders]
        
        for col in categorical_columns:
            df[col] = df[col].astype('int64')
        
        # Save with index=False to avoid extra index column
        df.to_csv(save_path, index=False)
        
        # Save the encoding mappings if they exist
        if hasattr(st.session_state, 'preprocessor') and st.session_state['preprocessor'].encoders:
            mapping_file = save_dir / f"mappings_{timestamp}.json"
            mappings = {}
            for col, encoder in st.session_state['preprocessor'].encoders.items():
                if hasattr(encoder, 'classes_'):
                    mappings[col] = {
                        str(label): int(i) 
                        for i, label in enumerate(encoder.classes_)
                    }
            
            with open(mapping_file, 'w') as f:
                json.dump(mappings, f, indent=2)
            
            st.success(f"Successfully saved processed data to: {save_path}")
            st.info(f"Encoding mappings saved to: {mapping_file}")
            
            # Verify the save by reading back the file
            saved_df = pd.read_csv(save_path)
            if saved_df.equals(df):
                st.success("✅ Verification: Saved data matches processed data")
            else:
                st.warning("⚠️ Warning: Saved data might differ from processed data")
                
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")

def render_results():


    st.header("Results")
    st.info("Results visualization coming soon!")

if __name__ == "__main__":
    main()