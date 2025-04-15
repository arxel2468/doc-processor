from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.processor import DocumentProcessor
from app.utils import allowed_file, save_uploaded_file
import os
import uvicorn
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(title="Document Processor API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor
processor = DocumentProcessor()

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = "uploads"
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Document Processor API</title>
        </head>
        <body>
            <h1>Document Processor API is running</h1>
            <p>Use the frontend application to interact with this API.</p>
        </body>
    </html>
    """

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    try:
        file_path = await save_uploaded_file(file, UPLOAD_FOLDER)
        result = processor.process_document(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test_processing():
    test_file = os.path.join(UPLOAD_FOLDER, "sample_invoice.jpg")
    if not os.path.exists(test_file):
        raise HTTPException(status_code=404, detail="Test file not found")
    
    result = processor.process_document(test_file)
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 