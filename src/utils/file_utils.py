import os
import shutil
from pathlib import Path
from typing import List

def ensure_directory_exists(directory: str):
    """Creates a directory if it does not exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def list_files(directory: str, extensions: List[str] = None) -> List[str]:
    """Lists all files in a directory with optional filtering by extensions."""
    path = Path(directory)
    if not path.exists() or not path.is_dir():
        return []
    
    if extensions:
        return [str(f) for f in path.glob("*") if f.suffix in extensions]
    return [str(f) for f in path.iterdir() if f.is_file()]

def copy_file(src: str, dest: str):
    """Copies a file from src to dest."""
    shutil.copy(src, dest)

def move_file(src: str, dest: str):
    """Moves a file from src to dest."""
    shutil.move(src, dest)

def delete_file(filepath: str):
    """Deletes a file if it exists."""
    path = Path(filepath)
    if path.exists():
        path.unlink()

def delete_directory(directory: str):
    """Deletes a directory and all its contents."""
    path = Path(directory)
    if path.exists():
        shutil.rmtree(directory)
