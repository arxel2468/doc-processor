# app/utils.py
import os
import uuid
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'tiff', 'bmp'}

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file, upload_folder):
    """Save uploaded file with a secure filename"""
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
        
    # Generate a secure filename with UUID to avoid collisions
    original_filename = secure_filename(file.filename)
    extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
    new_filename = f"{uuid.uuid4().hex}.{extension}"
    
    file_path = os.path.join(upload_folder, new_filename)
    file.save(file_path)
    
    return file_path