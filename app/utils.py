# app/utils.py
import os
from pathlib import Path
from fastapi import UploadFile
import shutil

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

def allowed_file(filename: str) -> bool:
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def save_uploaded_file(file: UploadFile, upload_folder: str) -> str:
    """Save uploaded file to the specified folder and return the file path."""
    try:
        # Ensure the upload folder exists
        Path(upload_folder).mkdir(exist_ok=True)
        
        # Create a safe filename
        filename = file.filename
        file_path = os.path.join(upload_folder, filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return file_path
    except Exception as e:
        raise Exception(f"Error saving file: {str(e)}")
    finally:
        file.file.close()