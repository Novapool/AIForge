import streamlit as st
import os
import platform
from tkinter import Tk, filedialog
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List

class DirectoryHandler:
    """Handles directory selection, structure display, and file processing"""
    
    SUPPORTED_FORMATS = {
        'data': ('.csv', '.xlsx'),
        'image': ('.png', '.jpg', '.jpeg')
    }
    
    @staticmethod
    def select_directory() -> Optional[str]:
        """Opens native file explorer for directory selection based on OS"""
        if platform.system() == "Darwin":  # macOS
            try:
                # Use AppleScript to open native folder picker
                cmd = (
                    'osascript -e \'tell application "System Events"\' '
                    '-e \'activate\' '
                    '-e \'return POSIX path of (choose folder with prompt "Select a folder:")\' '
                    '-e \'end tell\''
                )
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
                return None
            except Exception:
                return DirectoryHandler._fallback_tkinter_picker()
        else:  # Windows and other platforms
            return DirectoryHandler._fallback_tkinter_picker()
    
    @staticmethod
    def _fallback_tkinter_picker() -> Optional[str]:
        """Fallback directory picker using tkinter"""
        root = Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        
        directory = filedialog.askdirectory()
        root.destroy()
        
        return directory if directory else None
    
    @staticmethod
    def get_directory_structure(path: str) -> Dict:
        """Returns a dictionary representing the directory structure"""
        structure = {}
        
        try:
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                # Skip hidden files and directories
                if item.startswith('.'):
                    continue
                    
                if os.path.isdir(full_path):
                    sub_structure = DirectoryHandler.get_directory_structure(full_path)
                    if sub_structure:  # Only add if not empty
                        structure[item] = {
                            'type': 'directory',
                            'contents': sub_structure
                        }
                else:
                    # Only add files with supported extensions
                    ext = os.path.splitext(item)[1].lower()
                    if any(ext in formats for formats in DirectoryHandler.SUPPORTED_FORMATS.values()):
                        structure[item] = {
                            'type': 'file',
                            'path': full_path,
                            'format': 'data' if ext in DirectoryHandler.SUPPORTED_FORMATS['data'] else 'image'
                        }
        except PermissionError:
            st.error(f"Permission denied for {path}")
        except Exception as e:
            st.error(f"Error accessing {path}: {str(e)}")
        
        return structure
    
    @staticmethod
    def display_structure(structure: Dict, indent: int = 0) -> None:
        """Displays the directory structure in Streamlit"""
        for name, info in sorted(structure.items()):
            if info['type'] == 'directory':
                st.markdown(
                    f"{'&nbsp;' * (indent * 4)}📁 **{name}**",
                    unsafe_allow_html=True
                )
                DirectoryHandler.display_structure(info['contents'], indent + 1)
            else:
                icon = "📊" if info['format'] == 'data' else "🖼️"
                st.markdown(
                    f"{'&nbsp;' * (indent * 4)}{icon} {name}",
                    unsafe_allow_html=True
                )
    
    @staticmethod
    def load_data_files(structure: Dict, base_path: str = "") -> Dict[str, pd.DataFrame]:
        """Recursively loads all data files from the directory structure"""
        data_files = {}
        
        for name, info in structure.items():
            if info['type'] == 'directory':
                # Recursively process subdirectories
                sub_path = os.path.join(base_path, name) if base_path else name
                sub_files = DirectoryHandler.load_data_files(info['contents'], sub_path)
                data_files.update(sub_files)
            elif info['format'] == 'data':
                try:
                    if name.endswith('.csv'):
                        df = pd.read_csv(info['path'])
                    else:  # .xlsx
                        df = pd.read_excel(info['path'])
                    
                    # Use relative path as key
                    rel_path = os.path.join(base_path, name) if base_path else name
                    data_files[rel_path] = df
                except Exception as e:
                    st.error(f"Error loading {name}: {str(e)}")
        
        return data_files