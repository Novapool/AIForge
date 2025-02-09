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
```

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
