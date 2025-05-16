# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

block_cipher = None

# Determine the root directory
try:
    # When run directly
    root_dir = os.path.abspath(os.path.dirname(__file__))
except NameError:
    # When run with pyinstaller command
    import sys
    root_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
uploads_dir = os.path.join(root_dir, 'uploads')

# Create uploads directory if it doesn't exist
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Make sure uploads directory gets included
added_files = [
    ('uploads', 'uploads'),  # Include uploads folder
]

a = Analysis(
    ['app/main.py'],  # Entry point script
    pathex=[root_dir],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        # Uvicorn components
        'uvicorn.logging',
        'uvicorn.protocols',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'uvicorn.lifespan.off',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        
        # FastAPI and dependencies
        'fastapi',
        'starlette',
        'pydantic',
        
        # File handling
        'python-multipart',  # For file uploads
        'aiofiles',          # For async file operations
        
        # Data processing
        'pandas',
        'numpy',
        'scikit-learn',
        'sklearn.preprocessing',  # Explicit import for preprocessing modules
        'sklearn.impute',         # Explicit import for imputers
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)