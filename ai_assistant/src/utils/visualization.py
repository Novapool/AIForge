import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import seaborn as sns
from scipy import stats

class DataVisualizer:
    """Handles all data visualization functionality for data analysis and insights"""
    
    @staticmethod
    def generate_confusion_matrix(confusion_matrix: np.ndarray, class_names: Optional[List[str]] = None) -> go.Figure:
        """
        Generate confusion matrix visualization
        
        Args:
            confusion_matrix: Confusion matrix as numpy array
            class_names: Optional list of class names
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(confusion_matrix))]
        
        fig = px.imshow(
            confusion_matrix,
            x=class_names,
            y=class_names,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            title="Confusion Matrix"
        )
        
        # Add text annotations
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix)):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(confusion_matrix[i, j]),
                    showarrow=False,
                    font=dict(color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black")
                )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def generate_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> go.Figure:
        """
        Generate ROC curve visualization
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc: Area under curve value
        """
        fig = go.Figure()
        
        # Add ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Add diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        return fig

    @staticmethod
    def generate_pr_curve(precision: np.ndarray, recall: np.ndarray, avg_precision: float) -> go.Figure:
        """
        Generate Precision-Recall curve visualization
        
        Args:
            precision: Precision values
            recall: Recall values
            avg_precision: Average precision value
        """
        fig = go.Figure()
        
        # Add PR curve
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AP = {avg_precision:.3f})',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        return fig

    @staticmethod
    def generate_residual_plot(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """
        Generate residual plot for regression evaluation
        
        Args:
            y_true: True values
            y_pred: Predicted values
        """
        residuals = y_true - y_pred
        
        fig = go.Figure()
        
        # Add scatter plot of residuals
        fig.add_trace(go.Scatter(
            x=y_pred, y=residuals,
            mode='markers',
            marker=dict(color='blue', size=8, opacity=0.6),
            name='Residuals'
        ))
        
        # Add horizontal line at y=0
        fig.add_trace(go.Scatter(
            x=[min(y_pred), max(y_pred)], y=[0, 0],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Zero Line'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Residual Plot",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals (Actual - Predicted)"
        )
        return fig

    @staticmethod
    def generate_actual_vs_predicted_plot(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """
        Generate actual vs predicted plot for regression evaluation
        
        Args:
            y_true: True values
            y_pred: Predicted values
        """
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred,
            mode='markers',
            marker=dict(color='blue', size=8, opacity=0.6),
            name='Predictions'
        ))
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Perfect Prediction'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Actual vs Predicted Values",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values"
        )
        return fig
    
    @staticmethod
    def generate_distribution_plot(
        df: pd.DataFrame, 
        column: str, 
        plot_type: str = "histogram",
        kde: bool = True
    ) -> go.Figure:
        """
        Generate distribution plot for numerical data with multiple visualization options
        
        Args:
            df: Input DataFrame
            column: Column to visualize
            plot_type: Type of plot ('histogram', 'box', 'violin')
            kde: Whether to show KDE curve for histogram
        """
        if plot_type == "histogram":
            fig = px.histogram(
                df, 
                x=column, 
                marginal="box" if kde else None,
                histnorm='probability density' if kde else None
            )
            if kde:
                # Add KDE curve
                kde_x, kde_y = DataVisualizer._calculate_kde(df[column].dropna())
                fig.add_scatter(x=kde_x, y=kde_y, name='KDE', line=dict(color='red'))
        elif plot_type == "box":
            fig = px.box(df, y=column)
        else:  # violin
            fig = px.violin(df, y=column, box=True)
            
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=f"{plot_type.title()} Plot: {column}"
        )
        return fig
    
    @staticmethod
    def _calculate_kde(data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate KDE for smooth density estimation"""
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        return x_range, kde(x_range)

    @staticmethod
    def generate_correlation_matrix(
        df: pd.DataFrame, 
        numerical_columns: List[str],
        method: str = 'pearson'
    ) -> go.Figure:
        """
        Generate correlation matrix heatmap with multiple correlation methods
        
        Args:
            df: Input DataFrame
            numerical_columns: List of numerical columns
            method: Correlation method ('pearson', 'spearman', 'kendall')
        """
        corr_matrix = df[numerical_columns].corr(method=method)
        mask = np.triu(np.ones_like(corr_matrix))
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu",
            aspect="auto",
            title=f"Correlation Matrix ({method.title()})"
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    
    @staticmethod
    def generate_scatter_matrix(
        df: pd.DataFrame, 
        numerical_columns: List[str], 
        n_samples: Optional[int] = None
    ) -> go.Figure:
        """
        Generate scatter plot matrix for selected numerical columns
        
        Args:
            df: Input DataFrame
            numerical_columns: List of numerical columns to plot
            n_samples: Number of samples to plot (for large datasets)
        """
        if n_samples and len(df) > n_samples:
            df_sample = df.sample(n_samples, random_state=42)
        else:
            df_sample = df
            
        fig = px.scatter_matrix(
            df_sample,
            dimensions=numerical_columns,
            title="Scatter Plot Matrix"
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def generate_missing_values_chart(df: pd.DataFrame) -> go.Figure:
        """Generate comprehensive missing values visualization"""
        # Calculate missing values and percentages
        missing_values = df.isnull().sum()
        missing_percentages = (missing_values / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Count': missing_values.values,
            'Missing Percentage': missing_percentages.values
        }).sort_values('Missing Percentage', ascending=False)
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name='Missing Count',
                x=missing_df['Column'],
                y=missing_df['Missing Count'],
                yaxis='y'
            )
        )
        fig.add_trace(
            go.Scatter(
                name='Missing Percentage',
                x=missing_df['Column'],
                y=missing_df['Missing Percentage'],
                yaxis='y2',
                line=dict(color='red')
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Missing Values Analysis",
            yaxis=dict(title="Missing Count"),
            yaxis2=dict(title="Missing Percentage (%)", overlaying='y', side='right'),
            showlegend=True,
            hovermode='x unified'
        )
        return fig
    
    @staticmethod
    def generate_categorical_plot(
        df: pd.DataFrame, 
        column: str,
        plot_type: str = 'bar'
    ) -> go.Figure:
        """
        Generate visualization for categorical data
        
        Args:
            df: Input DataFrame
            column: Categorical column to visualize
            plot_type: Type of plot ('bar', 'pie')
        """
        value_counts = df[column].value_counts()
        
        if plot_type == 'bar':
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {column}",
                labels={'x': column, 'y': 'Count'}
            )
        else:  # pie
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {column}"
            )
            
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    @staticmethod
    def generate_outlier_plot(
        df: pd.DataFrame,
        column: str,
        method: str = 'zscore',
        threshold: float = 3.0
    ) -> go.Figure:
        """
        Generate outlier detection visualization with box plot and highlighted outliers
        
        Args:
            df: Input DataFrame
            column: Column to analyze for outliers
            method: Detection method ('zscore', 'iqr')
            threshold: Threshold for outlier detection
        """
        try:
            data = df[column].dropna()
            
            # Detect outliers
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                is_outlier = z_scores > threshold
            else:  # IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                is_outlier = (data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR))
            
            # Create figure
            fig = go.Figure()
            
            # Add box plot with basic styling
            fig.add_trace(go.Box(
                y=data,
                name=column,
                boxpoints=False,  # Hide all points
                marker_color='blue'
            ))
            
            # Add scatter plot for non-outliers
            fig.add_trace(go.Scatter(
                y=data[~is_outlier],
                mode='markers',
                name='Normal',
                marker=dict(
                    color='blue',
                    size=4
                ),
                showlegend=True
            ))
            
            # Add scatter plot for outliers
            if any(is_outlier):
                fig.add_trace(go.Scatter(
                    y=data[is_outlier],
                    mode='markers',
                    name='Outliers',
                    marker=dict(
                        color='red',
                        size=8
                    ),
                    showlegend=True
                ))
        
            # Update layout
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=f"Outlier Detection: {column} ({method})",
                yaxis_title=column,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            return fig
            
        except Exception as e:
            # Create an error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error generating plot: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red")
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
    def generate_feature_importance_plot(
        feature_importance: Dict[str, float],
        title: str = "Feature Importance"
    ) -> go.Figure:
        """
        Generate feature importance visualization
        
        Args:
            feature_importance: Dictionary mapping feature names to importance values
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Convert to DataFrame and sort by importance
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values('Importance', ascending=False)
        
        # Create horizontal bar chart
        fig = px.bar(
            importance_df,
            y='Feature',
            x='Importance',
            orientation='h',
            title=title
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title=""),
            xaxis=dict(title="Importance"),
            height=max(400, len(feature_importance) * 25)  # Dynamic height based on number of features
        )
        
        return fig
    
    @staticmethod
    def generate_summary_stats(df: pd.DataFrame) -> Dict:
        """Generate comprehensive summary statistics for the dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "missing_values": df.isnull().sum().sum(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # in MB
            "numeric_stats": df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
            "categorical_stats": {
                col: df[col].value_counts().to_dict() 
                for col in categorical_cols
            } if len(categorical_cols) > 0 else {},
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_by_column": df.memory_usage(deep=True).to_dict()
        }
        return summary
