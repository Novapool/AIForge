#!/bin/bash
# build.sh - Build script for macOS/Linux

echo "Building AI Assistant MVP..."

# Clean previous builds
rm -rf dist/
rm -rf backend/dist/
rm -rf backend/build/

# Create virtual environment if it doesn't exist at root level
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate virtual environment from root
source .venv/bin/activate

# Build Backend
echo "Building Python backend..."
cd backend

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