# app/routes.py
import os
from flask import Blueprint, render_template, request, jsonify, current_app
from app.processor import DocumentProcessor
from app.utils import allowed_file, save_uploaded_file

main = Blueprint('main', __name__)
processor = DocumentProcessor()

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        file_path = save_uploaded_file(file, current_app.config['UPLOAD_FOLDER'])
        result = processor.process_document(file_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@main.route('/test')
def test_processing():
    test_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'sample_invoice.jpg')
    result = processor.process_document(test_file)
    return jsonify(result)