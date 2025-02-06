import streamlit as st
import torch
import pandas as pd
from pathlib import Path
import os
import numpy as np
import warnings

from utils.visualization import DataVisualizer

warnings.filterwarnings('ignore', message='Examining the path of torch.classes.*')


# Page config should be the first streamlit command
st.set_page_config(
    page_title="AI Development Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Novapool/AIForge',
        'Report a bug': 'https://github.com/Novapool/AIForge/issues',
        'About': 'AI Development Assistant - Making AI Development Accessible'
    }
)

def main():
    # Sidebar
    with st.sidebar:
        st.title("🤖 AI Dev Assistant")
        
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
    
    tabs = st.tabs(["Upload & Preview", "Analysis & Visualization", "Preprocessing", "TensorBoard"])
    
    with tabs[0]:  # Upload & Preview
        uploaded_file = st.file_uploader(
            "Upload your dataset", 
            type=['csv', 'xlsx', 'png', 'jpg', 'jpeg'],
            help="Supported formats: CSV, Excel, Images"
        )
        
        if uploaded_file is not None:
            if uploaded_file.name.endswith(('.csv', '.xlsx')):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state['current_df'] = df  # Store in session state
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
        if 'current_df' in st.session_state:
            df = st.session_state['current_df']
            
            st.subheader("Data Analysis")
            
            # Correlation Matrix
            st.write("### Correlation Matrix")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_fig = DataVisualizer.generate_correlation_matrix(df, numeric_cols)
                st.plotly_chart(corr_fig, use_container_width=True)
            
            # Distribution Plots
            st.write("### Distribution Analysis")
            selected_column = st.selectbox("Select column for distribution analysis", numeric_cols)
            if selected_column:
                dist_fig = DataVisualizer.generate_distribution_plot(df, selected_column)
                st.plotly_chart(dist_fig, use_container_width=True)
            
            # Missing Values
            st.write("### Missing Values Analysis")
            missing_fig = DataVisualizer.generate_missing_values_chart(df)
            st.plotly_chart(missing_fig, use_container_width=True)
    
    with tabs[2]:  # Preprocessing
        if 'current_df' in st.session_state:
            st.subheader("Data Preprocessing")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Handle Missing Values")
                missing_strategy = st.selectbox(
                    "Strategy for missing values",
                    ["Drop rows", "Mean/Mode imputation", "Forward fill", "Backward fill"]
                )
                
                if st.button("Apply Missing Values Strategy"):
                    # TODO: Implement missing values handling
                    pass
            
            with col2:
                st.write("### Normalization")
                normalize_method = st.selectbox(
                    "Normalization method",
                    ["Min-Max Scaling", "Standard Scaling", "Robust Scaling"]
                )
                
                if st.button("Apply Normalization"):
                    # TODO: Implement normalization
                    pass
    
    with tabs[3]:  # TensorBoard
        st.subheader("TensorBoard Integration")
        if st.button("Launch TensorBoard"):
            try:
                import webbrowser
                from torch.utils.tensorboard import SummaryWriter
                
                # Create a logs directory if it doesn't exist
                log_dir = "logs"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                # Launch TensorBoard
                writer = SummaryWriter(log_dir)
                webbrowser.open("http://localhost:6006")
                st.success("TensorBoard launched! If it doesn't open automatically, visit http://localhost:6006")
            except Exception as e:
                st.error(f"Error launching TensorBoard: {str(e)}")

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