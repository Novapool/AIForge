#!/bin/bash
# build.sh - Build script for macOS/Linux

echo "Building AI Assistant MVP..."

# Clean previous builds
rm -rf dist/
rm -rf backend/dist/
rm -rf backend/build/

# Build Backend
echo "Building Python backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
pip install pyinstaller

# Build with PyInstaller
pyinstaller main.spec --clean

# Return to root
cd ..

# Build Electron app
echo "Building Electron app..."

# Install npm dependencies
npm install

# Build the Electron app
npm run dist

echo "Build complete! Check the dist/ folder for your packaged application."