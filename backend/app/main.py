# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import io
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import uvicorn

# Import your existing data preprocessor
# We'll copy its content and simplify it for the MVP
class DataPreprocessor:
    """Simplified version of your data preprocessor for MVP"""
    
    def encode_categorical(self, df: pd.DataFrame, method: str = 'label', columns: Optional[List[str]] = None):
        """Simple categorical encoding"""
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if method == 'label':
            from sklearn.preprocessing import LabelEncoder
            for col in columns:
                if col in df_encoded.columns:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        elif method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=columns)
        
        return df_encoded
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'standard', columns: Optional[List[str]] = None):
        """Simple normalization"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        df_normalized = df.copy()
        
        if columns is None:
            columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return df_normalized
        
        df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
        return df_normalized
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None):
        """Simple missing value handling"""
        from sklearn.impute import SimpleImputer
        
        df_imputed = df.copy()
        
        if columns is None:
            columns = df_imputed.columns[df_imputed.isnull().any()].tolist()
        
        if not columns:
            return df_imputed
        
        # Only handle numeric columns with mean/median
        if strategy in ['mean', 'median']:
            numeric_columns = df_imputed[columns].select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                imputer = SimpleImputer(strategy=strategy)
                df_imputed[numeric_columns] = imputer.fit_transform(df_imputed[numeric_columns])
        else:
            # For other strategies, handle all columns
            imputer = SimpleImputer(strategy=strategy)
            df_imputed[columns] = imputer.fit_transform(df_imputed[columns])
        
        return df_imputed

# Create FastAPI app
app = FastAPI(title="AI Assistant MVP - Data Preprocessing")

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data preprocessor
preprocessor = DataPreprocessor()

# Store uploaded files temporarily
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory storage for simple state management
uploaded_files: Dict[str, pd.DataFrame] = {}

@app.get("/")
async def root():
    return {"message": "AI Assistant MVP - Data Preprocessing API"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read the file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Store in memory
        file_id = file.filename
        uploaded_files[file_id] = df
        
        # Save to disk for later use
        file_path = UPLOAD_DIR / file.filename
        df.to_csv(file_path, index=False)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "preview": df.head(5).to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files")
async def list_files():
    """List uploaded files"""
    return [
        {
            "file_id": file_id,
            "shape": df.shape,
            "columns": df.columns.tolist()
        }
        for file_id, df in uploaded_files.items()
    ]

@app.get("/api/files/{file_id}")
async def get_file_info(file_id: str):
    """Get detailed file information"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    df = uploaded_files[file_id]
    
    # Calculate basic statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return {
        "file_id": file_id,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "preview": df.head(10).to_dict('records')
    }

@app.post("/api/preprocess/{file_id}")
async def preprocess_file(file_id: str, operations: List[Dict]):
    """Apply preprocessing operations to a file"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    df = uploaded_files[file_id].copy()
    
    try:
        for operation in operations:
            op_type = operation.get("type")
            params = operation.get("params", {})
            
            if op_type == "encode_categorical":
                df = preprocessor.encode_categorical(
                    df,
                    method=params.get("method", "label"),
                    columns=params.get("columns")
                )
            elif op_type == "normalize":
                df = preprocessor.normalize_data(
                    df,
                    method=params.get("method", "standard"),
                    columns=params.get("columns")
                )
            elif op_type == "handle_missing":
                df = preprocessor.handle_missing_values(
                    df,
                    strategy=params.get("strategy", "mean"),
                    columns=params.get("columns")
                )
        
        # Save preprocessed file
        output_filename = f"preprocessed_{file_id}"
        uploaded_files[output_filename] = df
        
        # Save to disk
        output_path = UPLOAD_DIR / f"preprocessed_{file_id}.csv"
        df.to_csv(output_path, index=False)
        
        return {
            "file_id": output_filename,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "preview": df.head(5).to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{file_id}")
async def download_file(file_id: str):
    """Download a processed file"""
    file_path = UPLOAD_DIR / f"{file_id}.csv"
    
    if not file_path.exists():
        # Try with .csv extension
        file_path = UPLOAD_DIR / f"{file_id}"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=f"{file_id}.csv",
        media_type="text/csv"
    )

if __name__ == "__main__":
    # This is for development. PyInstaller will use a different entry point
    uvicorn.run(app, host="0.0.0.0", port=8000)