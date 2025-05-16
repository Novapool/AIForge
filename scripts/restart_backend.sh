#!/bin/bash
# restart_backend.sh - Script to restart the backend

# Kill any existing python processes running the backend
pkill -f "python.*backend/app/main.py"
pkill -f "uvicorn.*main:app"

# Navigate to the backend directory
cd backend

# Activate virtual environment from the root directory
if [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

# Install requirements if needed
pip install -r requirements.txt

# Start the backend
python app/main.py