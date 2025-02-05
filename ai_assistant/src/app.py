import streamlit as st
import torch
import pandas as pd
from pathlib import Path

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
    
    # File upload section
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
                
                st.success(f"Successfully loaded dataset with shape: {df.shape}")
                
                # Basic data preview
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                
                with col2:
                    st.subheader("Data Info")
                    buffer = pd.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            # Image preview
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