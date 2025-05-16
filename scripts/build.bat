@echo off
REM build.bat - Build script for Windows

echo Building AI Assistant MVP...

REM Clean previous builds
rmdir /s /q dist
rmdir /s /q backend\dist
rmdir /s /q backend\build

REM Create virtual environment if it doesn't exist at root level
if not exist ".venv" (
    python -m venv .venv
)

REM Activate virtual environment from root
call .venv\Scripts\activate

REM Build Backend
echo Building Python backend...
cd backend

REM Install requirements
pip install -r requirements.txt
pip install pyinstaller

REM Build with PyInstaller
pyinstaller main.spec --clean

REM Return to root
cd ..

REM Build Electron app
echo Building Electron app...

REM Install npm dependencies
npm install

REM Build the Electron app
npm run dist

echo Build complete! Check the dist\ folder for your packaged application.