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