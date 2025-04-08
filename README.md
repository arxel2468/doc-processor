# Document Processing Solution

An AI-powered document processing solution that extracts key information from invoices and receipts.

## Features

- Upload documents (images or PDFs)
- Extract dates, amounts, vendor information, and line items
- View and verify extracted information
- Simple and intuitive user interface

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Install Tesseract OCR:
   - Linux: `sudo apt install tesseract-ocr libtesseract-dev`
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Mac: `brew install tesseract`
6. Create a `.env` file with your configuration
7. Run the application: `python run.py`

## Usage

1. Navigate to http://localhost:5000
2. Upload an invoice or receipt
3. View the extracted information
4. Process additional documents as needed

## License

MIT