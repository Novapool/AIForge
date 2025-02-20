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

- **Data Preprocessing**
  - State management for preprocessing operations
  - Categorical variable encoding (Label, One-Hot)
  - Data normalization (Standard, MinMax, Robust)
  - Missing value handling with multiple strategies
  - Outlier detection and removal
  - Transformation rule persistence
  - Batch processing support
  - Preprocessing state tracking and history

### Model Development
- **Tabular Data Support**
  - Neural network architecture for tabular data
  - Support for multiple problem types:
    - Binary classification
    - Multiclass classification
    - Regression
  - Configurable architecture
    - Customizable hidden layers
    - Dropout regularization
    - Batch normalization
  - Model state management
  - Automated data preparation and splitting

- **Training Infrastructure**
  - Automated train/validation/test splitting
  - Early stopping support
  - Model checkpointing
  - Training metrics tracking
    - Loss monitoring
    - Accuracy metrics
  - Device-agnostic training (CPU/GPU)
  - Batch processing capabilities

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
- Training progress monitoring

## 🛠 Technical Stack

- **Frontend:** Streamlit
- **Data Processing:** 
  - Pandas (Data manipulation)
  - NumPy (Numerical operations)
- **Machine Learning:** 
  - PyTorch (Neural Networks)
  - scikit-learn (Data Splitting)
- **Visualization:** Plotly
- **State Management:** JSON
- **Development Tools:** 
  - Black (formatting)
  - Flake8 (linting)
  - Pytest (testing)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AIForge.git
cd AIForge
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

1. Activate your virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Run the application:
```bash
streamlit run ai_assistant/src/app.py
```

3. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## 🗂 Project Structure

```
ai_assistant/
├── assets/
│   └── AIForge_Logo.png
├── configs/
│   └── config.yaml
├── models/
│   └── checkpoints/      # Model checkpoint storage
├── notebooks/
│   └── examples.ipynb
├── preprocessing_states/ # Preprocessing state storage
├── src/
│   ├── app.py           # Main Streamlit application
│   ├── data/            # Data handling modules
│   │   ├── data_loader.py
│   │   ├── data_preprocessor.py
│   │   └── preprocessing_manager.py
│   ├── models/          # ML model implementations
│   │   ├── base_model.py
│   │   ├── model_manager.py
│   │   ├── tabular_models.py
│   │   └── trainer.py
│   └── utils/           # Utility functions
│       ├── directory_handler.py
│       └── visualization.py
└── tests/               # Test suite
```

## 🔜 Upcoming Features

- **Advanced Model Development**
  - Additional model architectures
  - Hyperparameter optimization
  - Cross-validation support
  - Model versioning and comparison

- **Enhanced Training Visualization**
  - Interactive training progress visualization
  - Advanced metric plotting
  - Model comparison tools
  - Performance analysis dashboard

- **Extended Data Processing**
  - Advanced feature engineering
  - Automated feature selection
  - Custom preprocessing pipelines
  - Data augmentation strategies

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📫 Contact

For questions and feedback, please open an issue on GitHub.
