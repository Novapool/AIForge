import streamlit as st
import torch
import pandas as pd
from pathlib import Path
import os
import numpy as np
import warnings
import json
import plotly.express as px

# Import our new session state manager and components
from utils.session_state_manager import SessionStateManager
from components.file_selector import file_selector, file_info_display
from components.file_uploader import file_uploader_component, file_management_section

# Import existing modules
from utils.visualization import DataVisualizer
from utils.directory_handler import DirectoryHandler
from utils.model_evaluator import ModelEvaluator
from data.data_preprocessor import DataPreprocessor
from data.preprocessing_manager import PreprocessingManager
from models.model_manager import ModelManager, ProblemType, DataPreprocessingError
from models.tabular_models import TabularModel
from models.trainer import ModelTrainer

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
    # Initialize session state
    SessionStateManager.initialize()
    
    # Migrate legacy data if needed
    if 'current_dfs' in st.session_state and not st.session_state.get('file_registry'):
        SessionStateManager.migrate_legacy_data()
    
    # Sidebar
    with st.sidebar:
        st.title("AIForge")
        
        # Display active file if available
        active_file = SessionStateManager.get_active_file()
        if active_file:
            st.sidebar.success(f"Active Dataset: {active_file['id']}")
        
        # Main navigation
        page = st.radio(
            "Navigation",
            ["Data Management", "Model Development", "Training", "Results"],
            index=0,
            key="navigation",
            on_change=lambda: SessionStateManager.set_current_tab(st.session_state["navigation"])
        )
        
        # System info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### System Info")
        if torch.cuda.is_available():
            st.sidebar.success(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.sidebar.warning("GPU: Not Available")
    
    # Main content based on selected page
    current_tab = SessionStateManager.get_current_tab()
    
    if current_tab == "Data Management":
        render_data_management()
    elif current_tab == "Model Development":
        render_model_development()
    elif current_tab == "Training":
        render_training()
    else:
        render_results()

def render_data_management():
    st.header("Data Management")
    
    tabs = st.tabs(["Upload & Preview", "Analysis & Visualization", "Preprocessing"])
    
    with tabs[0]:  # Upload & Preview
        # Use our new file uploader component
        file_uploader_component()
        
        # Display file management section
        file_management_section()
    
    with tabs[1]:  # Analysis & Visualization
        st.write("### Data Analysis")
        
        # Check if we have any files
        if not SessionStateManager.list_files():
            st.warning("Please load data first in the Upload & Preview tab")
            return
        
        # Use file selector for analysis
        selected_file_id = file_selector(key="analysis_file_selector")
        
        if selected_file_id:
            # Get the dataframe
            df = SessionStateManager.get_dataframe(selected_file_id)
            
            if df is not None:
                # Analysis tabs remain similar but use our session state manager
                analysis_tabs = st.tabs([
                    "Data Overview",
                    "Distribution Analysis",
                    "Correlation Analysis",
                    "Missing Data Analysis",
                    "Feature Analysis"
                ])
                
                with analysis_tabs[0]:  # Data Overview
                    st.write("### Data Overview")
                    
                    # Generate and display summary statistics
                    summary_stats = DataVisualizer.generate_summary_stats(df)
                    
                    # Basic metrics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", summary_stats["total_rows"])
                        st.metric("Numeric Columns", summary_stats["numeric_columns"])
                    with col2:
                        st.metric("Total Columns", summary_stats["total_columns"])
                        st.metric("Categorical Columns", summary_stats["categorical_columns"])
                    with col3:
                        st.metric("Missing Values", summary_stats["missing_values"])
                        st.metric("Memory Usage (MB)", f"{summary_stats['memory_usage']:.2f}")
                    
                    # Detailed statistics
                    st.write("### Detailed Statistics")
                    
                    # Memory usage by column
                    memory_usage_df = pd.DataFrame.from_dict(
                        summary_stats["memory_usage_by_column"],
                        orient='index',
                        columns=['Memory (bytes)']
                    )
                    memory_usage_df['Memory (MB)'] = memory_usage_df['Memory (bytes)'] / (1024**2)
                    
                    fig_memory = px.bar(
                        memory_usage_df,
                        y='Memory (MB)',
                        title="Memory Usage by Column"
                    )
                    st.plotly_chart(fig_memory, use_container_width=True)
                    
                    # Data types distribution
                    dtypes_dist = pd.Series(summary_stats["dtypes"]).value_counts()
                    fig_dtypes = px.pie(
                        values=dtypes_dist.values,
                        names=dtypes_dist.index,
                        title="Data Types Distribution"
                    )
                    st.plotly_chart(fig_dtypes, use_container_width=True)
                
                with analysis_tabs[1]:  # Distribution Analysis
                    st.write("### Distribution Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Column selection
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                        
                        data_type = st.radio("Select Data Type", ["Numeric", "Categorical"])
                        if data_type == "Numeric":
                            selected_col = st.selectbox("Select Column", numeric_cols)
                            plot_type = st.selectbox(
                                "Select Plot Type",
                                ["histogram", "box", "violin"]
                            )
                            if plot_type == "histogram":
                                kde = st.checkbox("Show KDE", value=True)
                            else:
                                kde = False
                        else:
                            selected_col = st.selectbox("Select Column", categorical_cols)
                            plot_type = st.selectbox("Select Plot Type", ["bar", "pie"])
                    
                    with col2:
                        if data_type == "Numeric":
                            fig = DataVisualizer.generate_distribution_plot(
                                df, selected_col, plot_type, kde
                            )
                        else:
                            fig = DataVisualizer.generate_categorical_plot(
                                df, selected_col, plot_type
                            )
                        st.plotly_chart(fig, use_container_width=True)
                
                with analysis_tabs[2]:  # Correlation Analysis
                    st.write("### Correlation Analysis")
                    
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) < 2:
                        st.warning("Need at least 2 numeric columns for correlation analysis")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            corr_method = st.selectbox(
                                "Correlation Method",
                                ["pearson", "spearman", "kendall"]
                            )
                            selected_cols = st.multiselect(
                                "Select Columns",
                                numeric_cols,
                                default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
                            )
                        
                        if len(selected_cols) >= 2:
                            # Correlation matrix
                            fig_corr = DataVisualizer.generate_correlation_matrix(
                                df, selected_cols, corr_method
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                            # Scatter matrix for selected columns
                            if st.checkbox("Show Scatter Matrix"):
                                n_samples = st.number_input(
                                    "Number of samples to plot (0 for all)",
                                    min_value=0,
                                    max_value=len(df),
                                    value=min(1000, len(df))
                                )
                                fig_scatter = DataVisualizer.generate_scatter_matrix(
                                    df,
                                    selected_cols,
                                    n_samples if n_samples > 0 else None
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True)
                
                with analysis_tabs[3]:  # Missing Data Analysis
                    st.write("### Missing Data Analysis")
                    
                    # Missing values chart
                    fig_missing = DataVisualizer.generate_missing_values_chart(df)
                    st.plotly_chart(fig_missing, use_container_width=True)
                    
                    # Missing values patterns
                    missing_cols = df.columns[df.isnull().any()].tolist()
                    if missing_cols:
                        st.write("#### Missing Value Patterns")
                        missing_pattern = df[missing_cols].isnull().astype(int)
                        pattern_counts = missing_pattern.value_counts()
                        
                        st.write(f"Found {len(pattern_counts)} distinct missing value patterns:")
                        for pattern, count in pattern_counts.items():
                            pattern_str = ", ".join([col for col, is_missing in zip(missing_cols, pattern) if is_missing])
                            if not pattern_str:
                                pattern_str = "No missing values"
                            st.write(f"- Pattern (Count: {count}): {pattern_str}")
                
                with analysis_tabs[4]:  # Feature Analysis
                    st.write("### Feature Analysis")
                    
                    # Outlier detection
                    st.write("#### Outlier Detection")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        selected_col = st.selectbox("Select Column for Outlier Analysis", numeric_cols)
                        outlier_method = st.selectbox("Detection Method", ["zscore", "iqr"])
                        threshold = st.slider(
                            "Detection Threshold",
                            min_value=1.0,
                            max_value=5.0,
                            value=3.0,
                            step=0.1
                        )
                    
                    # Generate and display outlier plot
                    fig_outliers = DataVisualizer.generate_outlier_plot(
                        df, selected_col, outlier_method, threshold
                    )
                    st.plotly_chart(fig_outliers, use_container_width=True)
                    
                    # Basic statistics for the selected column
                    if selected_col in summary_stats.get("numeric_stats", {}):
                        st.write("#### Basic Statistics")
                        stats_df = pd.DataFrame(
                            summary_stats["numeric_stats"][selected_col],
                            index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                        ).T
                        st.dataframe(stats_df)
    
    with tabs[2]:  # Preprocessing
        render_preprocessing_tab()

def render_preprocessing_tab():
    """Render the preprocessing tab in the Streamlit interface"""
    st.subheader("Data Preprocessing")
    
    # Initialize preprocessing manager if not exists
    if 'preprocessing_manager' not in st.session_state:
        st.session_state['preprocessing_manager'] = PreprocessingManager()
    
    # Check if we have any files
    if not SessionStateManager.list_files():
        st.warning("Please load data first in the Upload & Preview tab")
        return
    
    # Use file selector for preprocessing
    selected_file_id = file_selector(key="preprocessing_file_selector")
    
    if selected_file_id:
        # Get the dataframe
        df = SessionStateManager.get_dataframe(selected_file_id)
        
        if df is not None:
            # Preprocessing tabs remain similar but use our session state manager
            preprocess_tabs = st.tabs([
                "Categorical Encoding",
                "Normalization",
                "Missing Values",
                "Outlier Removal",
                "Preprocessing States"
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
                        
                    if st.button("Add Encoding Operation"):
                        operation = {
                            'operation': 'encoding',
                            'method': selected_method,
                            'columns': selected_cols
                        }
                        
                        # Check if there's already a pending state for this file
                        existing_states = [
                            state for state in st.session_state['preprocessing_manager'].list_states()
                            if state['dataset_name'] == selected_file_id 
                            and state['status'] == 'pending'
                        ]
                        
                        if existing_states:
                            # Use the existing state file
                            existing_state = existing_states[-1]
                            # Load the state
                            state_data = st.session_state['preprocessing_manager'].load_state(existing_state['file_id'])
                            # Add the new operation
                            state_data['operations'].append({
                                **operation,
                                'timestamp': pd.Timestamp.now().isoformat()
                            })
                            # Save the updated state
                            with open(st.session_state['preprocessing_manager']._get_state_file_path(existing_state['file_id']), 'w') as f:
                                json.dump(state_data, f, indent=2)
                            state_file_id = existing_state['file_id']
                        else:
                            # Create a new state file
                            state_file_id = st.session_state['preprocessing_manager'].save_state(
                                selected_file_id,  # Use file_id instead of dataset name
                                [operation]
                            )
                        
                        # Add to file's preprocessing history
                        SessionStateManager.add_preprocessing_step(
                            selected_file_id,
                            operation
                        )
                        
                        st.success(f"Encoding operation added to preprocessing state")
            
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
                    
                if st.button("Add Normalization Operation"):
                    operation = {
                        'operation': 'normalization',
                        'method': selected_method,
                        'columns': selected_cols
                    }
                    
                    # Check if there's already a pending state for this file
                    existing_states = [
                        state for state in st.session_state['preprocessing_manager'].list_states()
                        if state['dataset_name'] == selected_file_id 
                        and state['status'] == 'pending'
                    ]
                    
                    if existing_states:
                        # Use the existing state file
                        existing_state = existing_states[-1]
                        # Load the state
                        state_data = st.session_state['preprocessing_manager'].load_state(existing_state['file_id'])
                        # Add the new operation
                        state_data['operations'].append({
                            **operation,
                            'timestamp': pd.Timestamp.now().isoformat()
                        })
                        # Save the updated state
                        with open(st.session_state['preprocessing_manager']._get_state_file_path(existing_state['file_id']), 'w') as f:
                            json.dump(state_data, f, indent=2)
                        state_file_id = existing_state['file_id']
                    else:
                        # Create a new state file
                        state_file_id = st.session_state['preprocessing_manager'].save_state(
                            selected_file_id,  # Use file_id instead of dataset name
                            [operation]
                        )
                    
                    # Add to file's preprocessing history
                    SessionStateManager.add_preprocessing_step(
                        selected_file_id,
                        operation
                    )
                    
                    st.success(f"Normalization operation added to preprocessing state")
            
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
                    
                    if st.button("Add Missing Value Operation"):
                        operation = {
                            'operation': 'missing_values',
                            'strategy': selected_strategy,
                            'columns': selected_cols
                        }
                        
                        # Check if there's already a pending state for this file
                        existing_states = [
                            state for state in st.session_state['preprocessing_manager'].list_states()
                            if state['dataset_name'] == selected_file_id 
                            and state['status'] == 'pending'
                        ]
                        
                        if existing_states:
                            # Use the existing state file
                            existing_state = existing_states[-1]
                            # Load the state
                            state_data = st.session_state['preprocessing_manager'].load_state(existing_state['file_id'])
                            # Add the new operation
                            state_data['operations'].append({
                                **operation,
                                'timestamp': pd.Timestamp.now().isoformat()
                            })
                            # Save the updated state
                            with open(st.session_state['preprocessing_manager']._get_state_file_path(existing_state['file_id']), 'w') as f:
                                json.dump(state_data, f, indent=2)
                            state_file_id = existing_state['file_id']
                        else:
                            # Create a new state file
                            state_file_id = st.session_state['preprocessing_manager'].save_state(
                                selected_file_id,  # Use file_id instead of dataset name
                                [operation]
                            )
                        
                        # Add to file's preprocessing history
                        SessionStateManager.add_preprocessing_step(
                            selected_file_id,
                            operation
                        )
                        
                        st.success(f"Missing value operation added to preprocessing state")
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
                    
                if st.button("Add Outlier Operation"):
                    operation = {
                        'operation': 'outliers',
                        'method': selected_method,
                        'threshold': threshold,
                        'columns': selected_cols
                    }
                    
                    # Check if there's already a pending state for this file
                    existing_states = [
                        state for state in st.session_state['preprocessing_manager'].list_states()
                        if state['dataset_name'] == selected_file_id 
                        and state['status'] == 'pending'
                    ]
                    
                    if existing_states:
                        # Use the existing state file
                        existing_state = existing_states[-1]
                        # Load the state
                        state_data = st.session_state['preprocessing_manager'].load_state(existing_state['file_id'])
                        # Add the new operation
                        state_data['operations'].append({
                            **operation,
                            'timestamp': pd.Timestamp.now().isoformat()
                        })
                        # Save the updated state
                        with open(st.session_state['preprocessing_manager']._get_state_file_path(existing_state['file_id']), 'w') as f:
                            json.dump(state_data, f, indent=2)
                        state_file_id = existing_state['file_id']
                    else:
                        # Create a new state file
                        state_file_id = st.session_state['preprocessing_manager'].save_state(
                            selected_file_id,  # Use file_id instead of dataset name
                            [operation]
                        )
                    
                    # Add to file's preprocessing history
                    SessionStateManager.add_preprocessing_step(
                        selected_file_id,
                        operation
                    )
                    
                    st.success(f"Outlier removal operation added to preprocessing state")
            
            # Preprocessing States Tab
            with preprocess_tabs[4]:
                st.write("### Manage Preprocessing States")
                
                # List all preprocessing states
                states = st.session_state['preprocessing_manager'].list_states()
                
                if states:
                    for state in states:
                        with st.expander(f"{state['dataset_name']} - {state['created_at']}"):
                            st.write(f"Status: {state['status']}")
                            st.write(f"Operations: {state['operation_count']}")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if st.button("View Details", key=f"view_{state['file_id']}"):
                                    details = st.session_state['preprocessing_manager'].view_state_history(
                                        state['file_id']
                                    )
                                    if details:
                                        st.json(details)
                            
                            with col2:
                                if st.button("Delete", key=f"delete_{state['file_id']}"):
                                    if st.session_state['preprocessing_manager'].delete_state(
                                        state['file_id']
                                    ):
                                        st.success("State file deleted")
                                        st.rerun()
                else:
                    st.info("No preprocessing states found")
            
            # Apply and Save Section
            st.write("---")
            st.write("### Apply and Save Preprocessing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Apply Preprocessing"):
                    try:
                        # Get all pending states for this dataset
                        states = [
                            state for state in st.session_state['preprocessing_manager'].list_states()
                            if state['dataset_name'] == selected_file_id 
                            and state['status'] == 'pending'
                        ]
                        
                        if not states:
                            st.warning("No pending preprocessing operations found")
                            return
                        
                        # Get the original file information
                        active_file = SessionStateManager.get_active_file()
                        if not active_file:
                            st.error("No active file found")
                            return
                        
                        # Get the original file path from metadata if available
                        original_file_path = None
                        if 'metadata' in active_file and 'output_csv_path' in active_file['metadata']:
                            # If this is already a processed file, use its CSV path
                            original_file_path = active_file['metadata']['output_csv_path']
                        elif 'metadata' in active_file and 'original_filename' in active_file['metadata']:
                            # Try to construct a path based on the original filename
                            original_filename = active_file['metadata']['original_filename']
                            # This is a simplification - in a real implementation, you'd need to track the actual file path
                            original_file_path = f"ai_assistant/processed_data/{original_filename}"
                        
                        # If we can't determine the original file path, create a temporary CSV
                        if not original_file_path:
                            # Create a temporary CSV file from the dataframe
                            temp_dir = Path("ai_assistant/processed_data")
                            temp_dir.mkdir(parents=True, exist_ok=True)
                            temp_csv = temp_dir / f"temp_{selected_file_id}.csv"
                            df = SessionStateManager.get_dataframe(selected_file_id)
                            df.to_csv(temp_csv, index=False)
                            original_file_path = str(temp_csv)
                        
                        # Apply operations using the JSON state file directly
                        last_state = states[-1]
                        
                        # Use apply_json_to_csv instead of apply_operations
                        output_file_path = st.session_state['preprocessing_manager'].apply_json_to_csv(
                            input_csv=original_file_path,
                            state_file_id=last_state['file_id']
                        )
                        
                        if output_file_path and os.path.exists(output_file_path):
                            # Read the processed CSV file
                            processed_df = pd.read_csv(output_file_path)
                            
                            # Generate a unique file ID for the processed data
                            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                            processed_file_id = f"processed_{selected_file_id}_{timestamp}"
                            
                            # Register the processed dataframe
                            SessionStateManager.register_file(
                                file_id=processed_file_id,
                                dataframe=processed_df,
                                source='preprocessing',
                                metadata={
                                    'original_file_id': selected_file_id,
                                    'preprocessing_timestamp': timestamp,
                                    'preprocessing_operations': len(states),
                                    'output_csv_path': output_file_path
                                }
                            )
                            
                            st.success(f"Preprocessing applied and new file created: {processed_file_id}")
                            st.success(f"CSV file saved to: {output_file_path}")
                            
                            # Button to switch to the processed data
                            if st.button("Use Processed Data"):
                                SessionStateManager.set_active_file(processed_file_id)
                                st.success(f"Now using processed dataset: {processed_file_id}")
                                st.rerun()
                        else:
                            st.warning("Preprocessing completed but no output file was generated or found.")
                        
                    except Exception as e:
                        st.error(f"Error applying preprocessing: {str(e)}")
                        st.error(f"Exception details: {type(e).__name__}: {str(e)}")
            
            with col2:
                if st.button("Reset All"):
                    # Get all states for this dataset
                    states = [
                        state for state in st.session_state['preprocessing_manager'].list_states()
                        if state['dataset_name'] == selected_file_id
                    ]
                    
                    # Delete each state
                    for state in states:
                        st.session_state['preprocessing_manager'].delete_state(state['file_id'])
                    
                    st.success("All preprocessing states cleared")
                    st.rerun()

def render_model_development():
    st.header("Model Development")
    
    # Check if we have any files
    if not SessionStateManager.list_files():
        st.warning("Please load data first in the Data Management tab")
        return
    
    # Initialize model manager if not exists
    if 'model_manager' not in st.session_state:
        st.session_state['model_manager'] = ModelManager()
        st.session_state['model_trainer'] = ModelTrainer()
    
    # Use file selector for model development
    selected_file_id = file_selector(key="model_development_file_selector")
    
    if selected_file_id:
        # Get the dataframe
        df = SessionStateManager.get_dataframe(selected_file_id)
        
        if df is not None:
            tabs = st.tabs(["Model Configuration", "Framework Selection", "Training Configuration"])
            
            with tabs[0]:
                st.subheader("Model Configuration")
                
                col1, col2 = st.columns(2)
                with col1:
                    target_column = st.selectbox(
                        "Select Target Column",
                        df.columns.tolist(),
                        help="Select the column you want to predict"
                    )
                    
                    problem_type = st.selectbox(
                        "Select Problem Type",
                        ["binary_classification", "multiclass_classification", "regression"],
                        help="Select the type of machine learning problem"
                    )
                    
                    model_type = st.selectbox(
                        "Select Model Type",
                        ["neural_network", "random_forest", "gradient_boosting", "xgboost", 
                         "lightgbm", "linear", "svm", "knn", "decision_tree"],
                        help="Select the type of model to use"
                    )
                
                with col2:
                    # Model-specific configuration
                    if model_type == "neural_network":
                        # Neural network configuration
                        st.write("#### Neural Network Configuration")
                        num_layers = st.number_input("Number of Hidden Layers", min_value=1, max_value=5, value=2)
                        hidden_dims = []
                        for i in range(num_layers):
                            dim = st.number_input(f"Hidden Layer {i+1} Dimension", 
                                                min_value=8, max_value=512, value=64 // (2**i))
                            hidden_dims.append(dim)
                        
                        dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.1)
                        model_params = {
                            "hidden_dims": hidden_dims,
                            "dropout_rate": dropout_rate
                        }
                    elif model_type in ["random_forest", "gradient_boosting", "decision_tree"]:
                        # Tree-based model configuration
                        st.write(f"#### {model_type.replace('_', ' ').title()} Configuration")
                        n_estimators = st.number_input("Number of Estimators", min_value=10, max_value=1000, value=100, step=10)
                        max_depth = st.number_input("Maximum Depth", min_value=1, max_value=100, value=10)
                        min_samples_split = st.number_input("Minimum Samples Split", min_value=2, max_value=20, value=2)
                        model_params = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split
                        }
                    elif model_type in ["xgboost", "lightgbm"]:
                        # Gradient boosting configuration
                        st.write(f"#### {model_type.replace('_', ' ').title()} Configuration")
                        n_estimators = st.number_input("Number of Estimators", min_value=10, max_value=1000, value=100, step=10)
                        learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.1, format="%.3f")
                        max_depth = st.number_input("Maximum Depth", min_value=1, max_value=100, value=6)
                        model_params = {
                            "n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "max_depth": max_depth
                        }
                    elif model_type == "linear":
                        # Linear model configuration
                        st.write("#### Linear Model Configuration")
                        linear_model_type = st.selectbox(
                            "Linear Model Type",
                            ["standard", "ridge", "lasso"],
                            help="Select the type of linear model"
                        )
                        if linear_model_type in ["ridge", "lasso"]:
                            alpha = st.number_input("Regularization Strength (Alpha)", min_value=0.001, max_value=10.0, value=1.0, format="%.3f")
                            model_params = {
                                "model_subtype": linear_model_type,
                                "alpha": alpha
                            }
                        else:
                            model_params = {
                                "model_subtype": linear_model_type
                            }
                    elif model_type == "svm":
                        # SVM configuration
                        st.write("#### SVM Configuration")
                        kernel = st.selectbox(
                            "Kernel",
                            ["rbf", "linear", "poly", "sigmoid"],
                            help="Select the kernel type"
                        )
                        C = st.number_input("Regularization Parameter (C)", min_value=0.1, max_value=100.0, value=1.0, format="%.2f")
                        model_params = {
                            "kernel": kernel,
                            "C": C
                        }
                    elif model_type == "knn":
                        # KNN configuration
                        st.write("#### K-Nearest Neighbors Configuration")
                        n_neighbors = st.number_input("Number of Neighbors", min_value=1, max_value=100, value=5)
                        weights = st.selectbox(
                            "Weight Function",
                            ["uniform", "distance"],
                            help="Select the weight function"
                        )
                        model_params = {
                            "n_neighbors": n_neighbors,
                            "weights": weights
                        }
                    else:
                        model_params = {}
                
                if st.button("Initialize Model"):
                    try:
                        with st.spinner("Preparing data and initializing model..."):
                            # Prepare data
                            data = st.session_state['model_manager'].prepare_data(
                                df, target_column, problem_type
                            )
                            
                            # Create model
                            st.session_state['model_manager'].create_model(
                                model_type=model_type,
                                **model_params
                            )
                            
                            # Store in session state
                            st.session_state['prepared_data'] = data
                            
                            # Store model config in session state manager
                            SessionStateManager.set_model_config({
                                'type': model_type,
                                'params': model_params,
                                'target_column': target_column,
                                'problem_type': problem_type
                            })
                            
                            # Associate model with file
                            SessionStateManager.associate_model_with_file(selected_file_id)
                            
                            st.success("Model initialized successfully!")
                            
                            # Display model summary
                            st.write("### Model Summary")
                            model_summary = st.session_state['model_manager'].get_model_summary()
                            st.write(f"Model Type: {model_type.replace('_', ' ').title()}")
                            st.write(f"Problem Type: {problem_type}")
                            st.write(f"Input Features: {model_summary['input_dim']}")
                            st.write(f"Output Dimension: {model_summary['output_dim']}")
                            
                            # Display model-specific details
                            if model_type == "neural_network":
                                st.write(f"Hidden Layers: {model_params['hidden_dims']}")
                                st.write(f"Dropout Rate: {model_params['dropout_rate']}")
                    
                    except DataPreprocessingError as e:
                        st.error("⚠️ Data Preprocessing Required")
                        st.error(f"{e.message}")
                        if e.columns:
                            st.write("Columns requiring preprocessing:")
                            for col in e.columns:
                                st.write(f"- {col}")
                            st.info("💡 Go to Data Management → Preprocessing → Categorical Encoding to process these columns.")
                    except Exception as e:
                        st.error(f"Error initializing model: {str(e)}")
    
            with tabs[1]:  # Framework Selection
                st.subheader("Framework Selection")
                
                col1, col2 = st.columns(2)
                with col1:
                    selected_framework = st.selectbox(
                        "Select Framework",
                        ["PyTorch (Current)", "TensorFlow", "Scikit-learn", "Custom Framework"],
                        help="Choose the deep learning framework for model implementation"
                    )
                    
                    framework_version = st.text_input(
                        "Framework Version",
                        value="2.0.1" if selected_framework == "PyTorch (Current)" else "",
                        disabled=True
                    )
                
                with col2:
                    st.write("### Framework Capabilities")
                    capabilities = {
                        "PyTorch (Current)": [
                            "Dynamic Computational Graphs",
                            "GPU Acceleration",
                            "Automatic Differentiation",
                            "Distributed Training"
                        ],
                        "TensorFlow": [
                            "Static Graphs (Coming Soon)",
                            "TPU Support (Coming Soon)",
                            "TF.js Export (Coming Soon)",
                            "TensorBoard Integration"
                        ],
                        "Scikit-learn": [
                            "Traditional ML Algorithms",
                            "Pipeline Support",
                            "Cross-validation",
                            "Model Selection"
                        ],
                        "Custom Framework": [
                            "Custom Implementation",
                            "Flexible Architecture",
                            "Framework Interoperability",
                            "Custom Optimizations"
                        ]
                    }
                    
                    for capability in capabilities.get(selected_framework, []):
                        st.markdown(f"- {capability}")
                
                st.info("Framework switching functionality coming soon! Currently using PyTorch.")
            
            with tabs[2]:  # Training Configuration
                st.subheader("Training Configuration")
                
                if 'prepared_data' not in st.session_state:
                    st.warning("Please initialize the model first")
                    return
                
                col1, col2 = st.columns(2)
                with col1:
                    num_epochs = st.number_input("Number of Epochs", min_value=1, max_value=1000, value=100)
                    batch_size = st.number_input("Batch Size", min_value=1, max_value=512, value=32)
                
                with col2:
                    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%f")
                    patience = st.number_input("Early Stopping Patience", min_value=1, max_value=50, value=10)
                
                if st.button("Start Training"):
                    try:
                        with st.spinner("Training model..."):
                            metrics = st.session_state['model_trainer'].train(
                                model=st.session_state['model_manager'].model,
                                dataloaders=st.session_state['model_manager'].create_dataloaders(
                                    st.session_state['prepared_data'], batch_size
                                ),
                                problem_type=st.session_state['model_manager'].problem_type,
                                num_epochs=num_epochs,
                                learning_rate=learning_rate,
                                early_stopping_patience=patience
                            )
                            
                            st.session_state['training_metrics'] = metrics
                            st.success("Training completed successfully!")
                    
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")

def render_training():
    st.header("Training")
    
    tabs = st.tabs(["Training Progress", "TensorBoard", "Training History"])
    
    with tabs[0]:  # Training Progress
        if 'training_metrics' in st.session_state:
            metrics = st.session_state['training_metrics']
            
            # Plot training curves
            fig_loss = px.line(
                {'epoch': range(len(metrics['train_loss'])),
                 'Training Loss': metrics['train_loss'],
                 'Validation Loss': metrics['val_loss']},
                x='epoch',
                y=['Training Loss', 'Validation Loss'],
                title='Training and Validation Loss'
            )
            st.plotly_chart(fig_loss)
            
            if 'train_accuracy' in metrics and len(metrics['train_accuracy']) > 0:
                fig_acc = px.line(
                    {'epoch': range(len(metrics['train_accuracy'])),
                     'Training Accuracy': metrics['train_accuracy'],
                     'Validation Accuracy': metrics['val_accuracy']},
                    x='epoch',
                    y=['Training Accuracy', 'Validation Accuracy'],
                    title='Training and Validation Accuracy'
                )
                st.plotly_chart(fig_acc)
        else:
            st.info("No training data available. Please train the model first.")
    
    with tabs[1]:  # TensorBoard
        st.write("### TensorBoard Integration")
        if st.button("Launch TensorBoard"):
            st.info("TensorBoard will be launched in a new window")
            # Placeholder for TensorBoard launch functionality
            st.code("tensorboard --logdir=logs", language="bash")
    
    with tabs[2]:  # Training History
        st.write("### Training History")
        st.info("Training history and model versioning coming soon!")

def render_results():
    st.header("Results")
    
    # Check if model and evaluation data are available
    if ('model_manager' not in st.session_state or 
        st.session_state['model_manager'].model is None or
        'prepared_data' not in st.session_state):
        st.warning("No trained model available. Please train a model first.")
        return
    
    tabs = st.tabs(["Model Evaluation", "Model Interpretability", "Model Testing", "Deployment"])
    
    with tabs[0]:  # Model Evaluation
        st.write("### Model Evaluation")
        
        # Get problem type
        problem_type = st.session_state['model_manager'].problem_type
        
        # Create dataloaders if not already in session state
        if 'evaluation_results' not in st.session_state:
            with st.spinner("Evaluating model..."):
                # Create test dataloader
                test_loader = st.session_state['model_manager'].create_dataloaders(
                    st.session_state['prepared_data'], 
                    batch_size=32
                )['test']
                
                # Evaluate model
                evaluation_results = ModelEvaluator.evaluate_model(
                    st.session_state['model_manager'].model,
                    test_loader,
                    problem_type,
                    st.session_state['model_manager'].device
                )
                
                st.session_state['evaluation_results'] = evaluation_results
        
        # Display evaluation metrics based on problem type
        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            # Classification metrics
            st.write("#### Classification Metrics")
            
            # Display confusion matrix
            conf_matrix = np.array(st.session_state['evaluation_results']['confusion_matrix'])
            fig_conf = DataVisualizer.generate_confusion_matrix(conf_matrix)
            st.plotly_chart(fig_conf, use_container_width=True)
            
            # Display classification report
            report = st.session_state['evaluation_results']['classification_report']
            report_df = pd.DataFrame(report).transpose()
            st.write("Classification Report:")
            st.dataframe(report_df.style.format("{:.3f}"))
            
            # For binary classification, show ROC and PR curves
            if problem_type == ProblemType.BINARY_CLASSIFICATION and 'roc' in st.session_state['evaluation_results']:
                col1, col2 = st.columns(2)
                
                with col1:
                    # ROC curve
                    roc_data = st.session_state['evaluation_results']['roc']
                    fig_roc = DataVisualizer.generate_roc_curve(
                        np.array(roc_data['fpr']),
                        np.array(roc_data['tpr']),
                        roc_data['auc']
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                
                with col2:
                    # PR curve
                    pr_data = st.session_state['evaluation_results']['pr_curve']
                    fig_pr = DataVisualizer.generate_pr_curve(
                        np.array(pr_data['precision']),
                        np.array(pr_data['recall']),
                        pr_data['avg_precision']
                    )
                    st.plotly_chart(fig_pr, use_container_width=True)
        
        else:  # Regression metrics
            st.write("#### Regression Metrics")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"{st.session_state['evaluation_results']['mae']:.4f}")
            with col2:
                st.metric("MSE", f"{st.session_state['evaluation_results']['mse']:.4f}")
            with col3:
                st.metric("RMSE", f"{st.session_state['evaluation_results']['rmse']:.4f}")
            with col4:
                st.metric("R²", f"{st.session_state['evaluation_results']['r2']:.4f}")
            
            # Display plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Residual plot
                residuals = np.array(st.session_state['evaluation_results']['residuals'])
                y_true = np.array(st.session_state['prepared_data']['test']['targets'].numpy())
                y_pred = y_true - residuals
                
                fig_residual = DataVisualizer.generate_residual_plot(y_true, y_pred)
                st.plotly_chart(fig_residual, use_container_width=True)
            
            with col2:
                # Actual vs Predicted plot
                fig_actual_pred = DataVisualizer.generate_actual_vs_predicted_plot(y_true, y_pred)
                st.plotly_chart(fig_actual_pred, use_container_width=True)
    
    with tabs[1]:  # Model Interpretability
        st.write("### Model Interpretability")
        
        # Feature importance visualization
        st.write("#### Feature Importance")
        
        # Get feature importance if available
        feature_importance = st.session_state['model_manager'].get_feature_importance()
        
        if feature_importance:
            # Display feature importance plot
            fig = DataVisualizer.generate_feature_importance_plot(
                feature_importance,
                title=f"Feature Importance for {st.session_state['model_manager'].model_type.value.replace('_', ' ').title()} Model"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display feature importance as a table
            importance_df = pd.DataFrame({
                'Feature': list(feature_importance.keys()),
                'Importance': list(feature_importance.values())
            }).sort_values('Importance', ascending=False)
            
            st.write("Feature Importance Table:")
            st.dataframe(importance_df, use_container_width=True)
        else:
            if st.session_state['model_manager'].model_type.value == "neural_network":
                st.info("Feature importance is not directly available for neural network models. Consider using SHAP values for model interpretation.")
            else:
                st.info("Feature importance not available for this model type.")
        
        # SHAP values
        st.write("#### SHAP Values")
        st.info("SHAP value analysis coming soon!")
    
    with tabs[2]:  # Model Testing
        st.write("### Model Testing")
        
        test_tabs = st.tabs(["Upload Test Data", "Manual Input", "Batch Testing"])
        
        with test_tabs[0]:  # Upload Test Data
            st.write("#### Upload New Test Data")
            uploaded_file = st.file_uploader("Upload test data", type=['csv', 'xlsx'])
            
            if uploaded_file:
                try:
                    # Load test data
                    if uploaded_file.name.endswith('.csv'):
                        test_df = pd.read_csv(uploaded_file)
                    else:
                        test_df = pd.read_excel(uploaded_file)
                    
                    st.success(f"Successfully loaded test data with shape: {test_df.shape}")
                    st.dataframe(test_df.head())
                    
                    # Test data preprocessing button
                    if st.button("Preprocess and Predict"):
                        st.info("Test data preprocessing and prediction functionality coming soon!")
                except Exception as e:
                    st.error(f"Error loading test file: {str(e)}")
        
        with test_tabs[1]:  # Manual Input
            st.write("#### Manual Input Testing")
            
            # Get feature names
            if 'prepared_data' in st.session_state:
                # This is a simplified approach - in a real implementation, we would need to 
                # track the original feature names and their transformations
                st.info("Manual input testing functionality coming soon!")
            else:
                st.warning("Model must be trained before manual testing is available")
        
        with test_tabs[2]:  # Batch Testing
            st.write("#### Batch Testing")
            st.info("Batch testing functionality coming soon!")
    
    with tabs[3]:  # Deployment
        st.write("### Model Deployment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Export Options")
            
            export_format = st.selectbox(
                "Export Format",
                ["PyTorch Model (.pth)", "ONNX Format (.onnx)", "TorchScript (.pt)"]
            )
            
            if st.button("Export Model"):
                st.info(f"Model export in {export_format} coming soon!")
        
        with col2:
            st.write("#### Deployment Status")
            st.info("Deployment tracking and management coming soon!")

if __name__ == "__main__":
    main()
