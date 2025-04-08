# app/routes.py
import os
from flask import Blueprint, render_template, request, jsonify, current_app, flash, redirect, url_for
from app.processor import DocumentProcessor
from app.utils import allowed_file, save_uploaded_file

main = Blueprint('main', __name__)
processor = DocumentProcessor()

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_file():
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save the file
        file_path = save_uploaded_file(file, current_app.config['UPLOAD_FOLDER'])
        
        # Process the document
        result = processor.process_document(file_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500