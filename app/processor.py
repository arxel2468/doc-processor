# app/processor.py
import os
os.environ['TRANSFORMERS_CACHE'] = 'D:\\huggingface_cache'
import re
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor
from pdf2image import convert_from_path
import cv2
import numpy as np
from transformers import pipeline

class DocumentProcessor:
    def __init__(self):
        # Initialize NER pipeline
        self.ner = pipeline("ner")
    
    def preprocess_image(self, image_path):
        """Preprocess image to improve OCR quality"""
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
    
    def extract_text(self, image_path):
        """Run the Donut model to extract text."""
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        decoder_input_ids = self.processor.tokenizer(
            self.task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=768,
            num_beams=3,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id
        )

        result = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print("🔍 Donut model raw output:\n", result)
        return result

    def clean_text(self, text):
        """Clean and normalize text."""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Fix common number formatting issues
        text = re.sub(r'(\d+)\s+(\d+)', r'\1\2', text)  # Join split numbers
        text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)  # Fix decimal points
        return text.strip()

    def extract_tag_value(self, text, tag):
        """Extract the first value from a given tag."""
        # First try exact tag match
        match = re.search(fr"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
        if match:
            value = self.clean_text(match.group(1))
            return value

        # Try to find the value in any tag if it's a number
        if tag in ['s_price', 's_unitprice', 's_total_price']:
            matches = re.findall(r'<s_\w+>\s*([$\d.,]+)\s*</s_\w+>', text)
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
    
    def extract_vendor(self, text, entities):
        """Extract vendor information"""
        # First try to find company name from NER
        company_entities = [entity for entity in entities if entity['entity'] == 'I-ORG' or entity['entity'] == 'B-ORG']
        if company_entities:
            return company_entities[0]['word']
        
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
        items = []
        # Try structured tag parsing
        menu_match = re.search(r"<s_menu>(.*?)</s_menu>", text, re.DOTALL)
        if menu_match:
            menu_content = menu_match.group(1)
            item_chunks = re.split(r"<sep\s*/?>", menu_content)

            for chunk in item_chunks:
                # Clean up any mismatched tags
                chunk = re.sub(r'<s_price>(.*?)</s_nm>', r'<s_nm>\1</s_nm>', chunk)
                chunk = re.sub(r'<s_num>(.*?)</s_nm>', r'<s_nm>\1</s_nm>', chunk)

                name = self.extract_tag_value(chunk, "s_nm")
                unit_price = self.extract_tag_value(chunk, "s_unitprice")
                quantity = self.extract_tag_value(chunk, "s_cnt")
                total = self.extract_tag_value(chunk, "s_price")

                # Skip if it's not a valid item (e.g., header or footer text)
                if not name or name.lower() in ['invoice', 'date', 'due', 'bill to', 'rate qty', 'amount']:
                    continue

                # Clean up the values
                name = self.clean_text(name)
                unit_price = self.clean_text(unit_price)
                quantity = self.clean_text(quantity)
                total = self.clean_text(total)

                if name and any([unit_price, quantity, total]):
                    items.append({
                        "name": name,
                        "unit_price": unit_price,
                        "quantity": quantity,
                        "total": total
                    })

        return items
    
    def process_document(self, image_path):
        """Process document and extract structured information"""
        # Extract text
        raw_text = self.extract_text(image_path)
        
        # Use NER to identify entities
        entities = self.ner(raw_text)
        
        # Extract structured data
        result = {
            "date": self.extract_date(raw_text),
            "total_amount": self.extract_amount(raw_text),
            "vendor": self.extract_vendor(raw_text, entities),
            "items": self.extract_line_items(raw_text),
            "raw_text": raw_text
        }
        
        return result