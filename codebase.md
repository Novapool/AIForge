# .gitignore

```
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# UV
#   Similar to Pipfile.lock, it is generally recommended to include uv.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#uv.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#pdm.lock
#   pdm stores project-wide configurations in .pdm.toml, but it is recommended to not include it
#   in version control.
#   https://pdm.fming.dev/latest/usage/project/#working-with-version-control
.pdm.toml
.pdm-python
.pdm-build/

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#  and can be added to the global gitignore or merged into this file.  For a more nuclear
#  option (not recommended) you can uncomment the following to ignore the entire idea folder.
#.idea/

# PyPI configuration file
.pypirc

```

# ai_assistant\assets\AIForge_Logo.png

This is a binary file of the type: Image

# ai_assistant\configs\config.yaml

```yaml

```

# ai_assistant\notebooks\examples.ipynb

```ipynb

```

# ai_assistant\setup.py

```py

```

# ai_assistant\src\__init__.py

```py

```

# ai_assistant\src\app.py

```py
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
                    st.success("Missing values handled successfully!")
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
                df = st.session_state['preprocessor'].remove_outliers(
                    df,
                    method=selected_method,
                    threshold=threshold,
                    columns=selected_cols
                )
                st.session_state['current_dfs'][selected_dataset] = df
                st.success("Outliers removed successfully!")
                
                # Show impact of outlier removal
                st.write("### Impact of Outlier Removal")
                original_shape = st.session_state['current_dfs'][selected_dataset].shape[0]
                new_shape = df.shape[0]
                removed_rows = original_shape - new_shape
                st.write(f"Rows removed: {removed_rows} ({(removed_rows/original_shape)*100:.2f}% of data)")
            except Exception as e:
                st.error(f"Error removing outliers: {str(e)}")
    
    # Display preprocessing summary
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
        
        # Important: Convert any integer-encoded categorical columns to integers
        # before saving to ensure proper data type preservation
        categorical_columns = df.select_dtypes(include=['int64']).columns
        df[categorical_columns] = df[categorical_columns].astype('int64')
        
        # Save with explicit data types
        df.to_csv(save_path, index=False)
        
        # Verify the save
        saved_df = pd.read_csv(save_path)
        print("Verification of saved data types:")
        print(saved_df.dtypes)
        
        st.success(f"Successfully saved processed data to: {save_path}")
        
        # Save the encoding mappings if they exist
        if 'preprocessor' in st.session_state:
            mapping_file = save_dir / f"mappings_{timestamp}.json"
            mappings = {}
            for col, encoder in st.session_state['preprocessor'].encoders.items():
                if hasattr(encoder, 'classes_'):
                    mappings[col] = {label: int(i) for i, label in enumerate(encoder.classes_)}
            
            with open(mapping_file, 'w') as f:
                json.dump(mappings, f, indent=2)
                
            st.info(f"Encoding mappings saved to: {mapping_file}")
            
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")

def render_results():


    st.header("Results")
    st.info("Results visualization coming soon!")

if __name__ == "__main__":
    main()
```

# ai_assistant\src\data\__init__.py

```py

```

# ai_assistant\src\data\data_loader.py

```py

```

# ai_assistant\src\data\data_preprocessor.py

```py
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
```

# ai_assistant\src\data\data_utils.py

```py

```

# ai_assistant\src\models\__init__.py

```py

```

# ai_assistant\src\models\base_model.py

```py

```

# ai_assistant\src\models\trainer.py

```py

```

# ai_assistant\src\utils\__init__.py

```py

```

# ai_assistant\src\utils\directory_handler.py

```py
import streamlit as st
import os
import platform
from tkinter import Tk, filedialog
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List

class DirectoryHandler:
    """Handles directory selection, structure display, and file processing"""
    
    SUPPORTED_FORMATS = {
        'data': ('.csv', '.xlsx'),
        'image': ('.png', '.jpg', '.jpeg')
    }
    
    @staticmethod
    def select_directory() -> Optional[str]:
        """Opens native file explorer for directory selection based on OS"""
        if platform.system() == "Darwin":  # macOS
            try:
                # Use AppleScript to open native folder picker
                cmd = (
                    'osascript -e \'tell application "System Events"\' '
                    '-e \'activate\' '
                    '-e \'return POSIX path of (choose folder with prompt "Select a folder:")\' '
                    '-e \'end tell\''
                )
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
                return None
            except Exception:
                return DirectoryHandler._fallback_tkinter_picker()
        else:  # Windows and other platforms
            return DirectoryHandler._fallback_tkinter_picker()
    
    @staticmethod
    def _fallback_tkinter_picker() -> Optional[str]:
        """Fallback directory picker using tkinter"""
        root = Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        
        directory = filedialog.askdirectory()
        root.destroy()
        
        return directory if directory else None
    
    @staticmethod
    def get_directory_structure(path: str) -> Dict:
        """Returns a dictionary representing the directory structure"""
        structure = {}
        
        try:
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                # Skip hidden files and directories
                if item.startswith('.'):
                    continue
                    
                if os.path.isdir(full_path):
                    sub_structure = DirectoryHandler.get_directory_structure(full_path)
                    if sub_structure:  # Only add if not empty
                        structure[item] = {
                            'type': 'directory',
                            'contents': sub_structure
                        }
                else:
                    # Only add files with supported extensions
                    ext = os.path.splitext(item)[1].lower()
                    if any(ext in formats for formats in DirectoryHandler.SUPPORTED_FORMATS.values()):
                        structure[item] = {
                            'type': 'file',
                            'path': full_path,
                            'format': 'data' if ext in DirectoryHandler.SUPPORTED_FORMATS['data'] else 'image'
                        }
        except PermissionError:
            st.error(f"Permission denied for {path}")
        except Exception as e:
            st.error(f"Error accessing {path}: {str(e)}")
        
        return structure
    
    @staticmethod
    def display_structure(structure: Dict, indent: int = 0) -> None:
        """Displays the directory structure in Streamlit"""
        for name, info in sorted(structure.items()):
            if info['type'] == 'directory':
                st.markdown(
                    f"{'&nbsp;' * (indent * 4)}📁 **{name}**",
                    unsafe_allow_html=True
                )
                DirectoryHandler.display_structure(info['contents'], indent + 1)
            else:
                icon = "📊" if info['format'] == 'data' else "🖼️"
                st.markdown(
                    f"{'&nbsp;' * (indent * 4)}{icon} {name}",
                    unsafe_allow_html=True
                )
    
    @staticmethod
    def load_data_files(structure: Dict, base_path: str = "") -> Dict[str, pd.DataFrame]:
        """Recursively loads all data files from the directory structure"""
        data_files = {}
        
        for name, info in structure.items():
            if info['type'] == 'directory':
                # Recursively process subdirectories
                sub_path = os.path.join(base_path, name) if base_path else name
                sub_files = DirectoryHandler.load_data_files(info['contents'], sub_path)
                data_files.update(sub_files)
            elif info['format'] == 'data':
                try:
                    if name.endswith('.csv'):
                        df = pd.read_csv(info['path'])
                    else:  # .xlsx
                        df = pd.read_excel(info['path'])
                    
                    # Use relative path as key
                    rel_path = os.path.join(base_path, name) if base_path else name
                    data_files[rel_path] = df
                except Exception as e:
                    st.error(f"Error loading {name}: {str(e)}")
        
        return data_files
```

# ai_assistant\src\utils\visualization.py

```py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class DataVisualizer:
    """Handles all data visualization functionality"""
    
    @staticmethod
    def generate_distribution_plot(df: pd.DataFrame, column: str) -> go.Figure:
        """Generate distribution plot for numerical data"""
        fig = px.histogram(df, x=column, marginal="box")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    
    @staticmethod
    def generate_correlation_matrix(df: pd.DataFrame, numerical_columns: List[str]) -> go.Figure:
        """Generate correlation matrix heatmap"""
        corr_matrix = df[numerical_columns].corr()
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    
    @staticmethod
    def generate_missing_values_chart(df: pd.DataFrame) -> go.Figure:
        """Generate missing values visualization"""
        missing_values = df.isnull().sum()
        missing_percentages = (missing_values / len(df)) * 100
        
        fig = px.bar(
            x=missing_percentages.index,
            y=missing_percentages.values,
            labels={'x': 'Column', 'y': 'Missing Values (%)'},
            title="Missing Values Distribution"
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    
    @staticmethod
    def process_batch_files(files: List[pd.DataFrame]) -> Dict:
        """Process multiple files and generate combined statistics"""
        combined_stats = {
            "total_files": len(files),
            "total_rows": sum(len(df) for df in files),
            "file_sizes": [len(df) for df in files],
            "memory_usage": sum(df.memory_usage(deep=True).sum() / 1024**2 for df in files)
        }
        return combined_stats

    @staticmethod
    def generate_summary_stats(df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "missing_values": df.isnull().sum().sum(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2  # in MB
        }
        return summary
```

# ai_assistant\tests\__init__.py

```py

```

# ai_assistant\tests\test_data_loader.py

```py

```

# ai_assistant\tests\test_models.py

```py

```

# logs\events.out.tfevents.1738965040.Laiths-MacBook-Air-2.local.64172.0

This is a binary file of the type: Binary

# README.md

```md
# AIForge - AI Development Assistant

AIForge is a Streamlit-based web application designed to simplify the AI model development process for developers. It provides an intuitive interface for data management, model development, and training visualization.

## 🚀 Current Features

### Data Management
- **File Upload Support**
  - Single file upload (CSV, Excel, Images)
  - Directory-based batch upload
  - Automatic file type detection and handling

- **Data Preview & Analysis**
  - Interactive data preview with basic statistics
  - Memory usage tracking
  - Column type inference
  - Basic data profiling
    - Row and column counts
    - Data type distribution
    - Missing value detection
    - Memory usage metrics

- **Batch Processing**
  - Multi-file handling
  - Aggregate statistics for batch uploads
  - Directory structure visualization
  - Recursive file processing

### Visualization
- Basic data statistics display
- Dataset structure visualization
- Memory usage tracking
- Performance metrics display

## 🛠 Technical Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **ML Framework:** PyTorch
- **Development Tools:** 
  - Black (formatting)
  - Flake8 (linting)
  - Pytest (testing)

## 📦 Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/yourusername/AIForge.git
cd AIForge
\`\`\`

2. Create a virtual environment:
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## 🚀 Getting Started

1. Activate your virtual environment:
\`\`\`bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

2. Run the application:
\`\`\`bash
streamlit run ai_assistant/src/app.py
\`\`\`

3. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## 🗂 Project Structure

\`\`\`
ai_assistant/
├── assets/
│   └── AIForge_Logo.png
├── configs/
│   └── config.yaml
├── notebooks/
│   └── examples.ipynb
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── data/               # Data handling modules
│   ├── models/             # ML model implementations
│   └── utils/              # Utility functions
│       ├── directory_handler.py  # File/directory management
│       └── visualization.py      # Data visualization tools
└── tests/                  # Test suite
\`\`\`

## 🔜 Upcoming Features

- Data Preprocessing
  - Missing value handling
  - Feature scaling
  - Encoding categorical variables
  - Feature selection

- Model Development
  - Model architecture selection
  - Hyperparameter configuration
  - Training pipeline setup
  - Model validation

- Training Visualization
  - Real-time training metrics
  - Performance visualization
  - Model checkpointing
  - Early stopping

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📫 Contact

For questions and feedback, please open an issue on GitHub.

```

# requirements.txt

```txt
torch
torchvision
torchaudio
streamlit
pandas
numpy
plotly
scikit-learn
pillow
pytest  # for testing
black   # for code formatting
flake8  # for linting
python-dotenv  # for environment variables
tensorboard
openpyxl  # for Excel file support
matplotlib  # for additional plotting capabilities

```

