# app/processor.py
import os
import pytesseract
from PIL import Image
import re
from datetime import datetime
import cv2
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

class DocumentProcessor:
    def __init__(self):
        # Initialize LayoutLM model and tokenizer
        model_path = "fine_tuned_layoutlm"  # Replace with your actual model path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    def preprocess_image(self, image_path):
        """Preprocess image to improve OCR quality"""
        try:
            # Read the image
            img = cv2.imread(image_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get black and white image
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # Save preprocessed image temporarily
            temp_path = os.path.join(os.path.dirname(image_path), "temp_processed.jpg")
            cv2.imwrite(temp_path, thresh)
            
            return temp_path
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def extract_text(self, image_path):
        """Extract text from image using OCR"""
        # Preprocess the image
        processed_image_path = self.preprocess_image(image_path)
        
        # Open the processed image
        img = Image.open(processed_image_path)
        
        # Extract text using Tesseract
        raw_text = pytesseract.image_to_string(img)
        
        # Clean up temporary file
        if os.path.exists(processed_image_path):
            os.remove(processed_image_path)
            
        return raw_text
    
    def extract_date(self, text):
        """Extract date from text"""
        # Common date patterns
        date_patterns = [
            r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b',  # DD/MM/YYYY or MM/DD/YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return None
    
    def extract_amount(self, text):
        """Extract total amount from text"""
        # Look for common amount patterns near keywords
        amount_patterns = [
            r'(?:total|amount|sum|due):?\s*[$€£]?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))',
            r'(?:total|amount|sum|due)[^$€£\d]*[$€£]?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))',
            r'[$€£]\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))'
        ]
        
        for pattern in amount_patterns:            
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Return the last match, as total is usually at the bottom
                return matches[-1]
        
        return None
    
    def extract_vendor(self, text, tokens, ner_tags):
        """Extract vendor information"""
        vendor = ""
        for i, tag in enumerate(ner_tags):
            if tag == "B-VENDOR":  # Assuming 'VENDOR' tag for vendor name
                vendor += tokens[i]
            elif tag == "I-VENDOR":
                vendor += " " + tokens[i]
        
        if vendor:
            return vendor
        
        # Otherwise look for common patterns
        vendor_patterns = [
            r'(?:vendor|supplier|from|by):\s*([A-Z][A-Za-z0-9\s&.,]+)',
            r'^([A-Z][A-Za-z0-9\s&.,]+)(?:\n|$)'
        ]
        
        for pattern in vendor_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0].strip()
        
        return None
    
    def extract_line_items(self, text):
        """Extract line items from text"""
        # This is simplified - in reality, you'd need more sophisticated parsing
        lines = text.split('\n')
        items = []
        
        for line in lines:
            # Look for lines with quantity, description, and price
            if re.search(r'\d+\s+[A-Za-z0-9\s]+\s+\$?\d+\.\d{2}', line):
                items.append(line.strip())
        
        return items
    
    def process_document(self, image_path):
        """Process document and extract structured information"""
        # Extract text
        raw_text = self.extract_text(image_path)
        
        # Tokenize with LayoutLM tokenizer
        inputs = self.tokenizer(raw_text, return_tensors="pt")
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract predicted entities
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Convert token IDs to labels
        labels = [self.model.config.id2label[i] for i in predictions[0].tolist()]
        
        # Map tokens to text
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Remove special tokens
        tokens = [
            token
            for token in tokens
            if token not in self.tokenizer.all_special_tokens
        ]

        # Remove special labels
        labels = labels[1 : len(tokens) + 1]

        # Extract words from subtokens
        words = [self.tokenizer.convert_tokens_to_string([tok]) for tok in tokens]
        
        # Extract structured data
        # Here you need to replace entities with the new labels and words from LayoutLM
        result = {
            "date": self.extract_date(raw_text),
            "total_amount": self.extract_amount(raw_text),
            "vendor": self.extract_vendor(raw_text, entities),
            "items": self.extract_line_items(raw_text),
            "raw_text": raw_text
        }
        
        return result