# app/processor.py
import os
os.environ['HF_HOME'] = 'D:\\huggingface_cache'
import re
import torch
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import cv2
from transformers import DonutProcessor, VisionEncoderDecoderModel
import json
from typing import Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize Donut model for document processing
        self.model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"
        try:
            self.processor = DonutProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
            self.task_prompt = "<s_cord-v2>"
            logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Set Tesseract path if needed (uncomment and adjust if Tesseract isn't in PATH)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def convert_pdf_to_image(self, pdf_path):
        """Convert first page of PDF to image."""
        try:
            images = convert_from_path(pdf_path, dpi=300)
            image_path = os.path.splitext(pdf_path)[0] + "_page1.jpg"
            images[0].save(image_path, "JPEG")
            logger.info(f"Converted PDF to image: {image_path}")
            return image_path
        except Exception as e:
            logger.error(f"Error converting PDF to image: {str(e)}")
            raise

    def preprocess_image(self, image_path):
        """Enhance image for better OCR quality."""
        try:
            img = cv2.imread(image_path)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 2)
            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh)
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            # Save processed image
            temp_path = os.path.splitext(image_path)[0] + "_processed.jpg"
            cv2.imwrite(temp_path, enhanced)
            logger.info(f"Preprocessed image saved to: {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def get_ocr_text(self, image_path):
        """Get OCR text from the image using Tesseract."""
        try:
            image = Image.open(image_path)
            ocr_text = pytesseract.image_to_string(image)
            return ocr_text
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            return ""

    def process_with_donut(self, image_path):
        """Process the image with Donut model."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Generate pixel values
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            # Generate decoder input ids
            decoder_input_ids = self.processor.tokenizer(
                self.task_prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(self.device)
            
            # Generate output
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=768,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                num_beams=4,
            )
            
            # Decode output
            sequence = self.processor.batch_decode(outputs)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            
            logger.info(f"Donut model output generated (truncated): {sequence[:100]}...")
            return sequence
            
        except Exception as e:
            logger.error(f"Error processing with Donut: {str(e)}")
            return None

    def find_labeled_section(self, text, label, lines_after=3):
        """Find a section of text that follows a label."""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if re.search(r'\b' + re.escape(label) + r'\b', line, re.IGNORECASE):
                # Return the next few lines
                return '\n'.join(lines[i:i+lines_after+1])
        return ""

    def extract_currency_symbol(self, text):
        """Extract currency symbol from text."""
        # Look for common currency symbols
        currency_match = re.search(r'[$€£₹]', text)
        if currency_match:
            return currency_match.group(0)
        
        # Look for currency codes
        currency_codes = ["USD", "EUR", "GBP", "INR", "CAD", "AUD", "JPY", "CNY"]
        for code in currency_codes:
            if re.search(r'\b' + re.escape(code) + r'\b', text, re.IGNORECASE):
                return code
        
        return ""

    def extract_from_context(self, text, field_labels, exclude_words=None):
        """
        Extract field value using context labels that commonly appear near the field.
        Returns the most likely value based on proximity to labels.
        """
        if exclude_words is None:
            exclude_words = []
            
        candidates = []
        lines = text.split('\n')
        
        # First pass: look for labels and extract adjacent text
        for i, line in enumerate(lines):
            for label in field_labels:
                if re.search(r'\b' + re.escape(label) + r'\b', line, re.IGNORECASE):
                    # Check current line for value after the label
                    after_label = re.sub(r'.*\b' + re.escape(label) + r'\b[:\s]*', '', line, flags=re.IGNORECASE).strip()
                    if after_label and not any(re.search(r'\b' + re.escape(word) + r'\b', after_label, re.IGNORECASE) for word in exclude_words):
                        candidates.append((after_label, 1))  # Higher confidence for same-line matches
                    
                    # Check next line if current line only has the label
                    if i < len(lines) - 1 and (not after_label or len(after_label) < 3):
                        next_line = lines[i+1].strip()
                        if next_line and not any(re.search(r'\b' + re.escape(word) + r'\b', next_line, re.IGNORECASE) for word in exclude_words):
                            candidates.append((next_line, 0.8))
        
        # Sort by confidence and return best match
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return ""

    def extract_date(self, text, donut_output=None):
        """Extract date from document text using multiple strategies."""
        # Strategy 1: Try to extract from Donut output
        if donut_output and "<s_date>" in donut_output:
            match = re.search(r'<s_date>(.*?)</s_date>', donut_output)
            if match:
                date_text = match.group(1).strip()
                if date_text and re.search(r'\d', date_text):  # Make sure it contains digits
                    return date_text
        
        # Strategy 2: Look for text near date-related labels
        date_labels = ["DATE", "INVOICE DATE", "ISSUE DATE", "ISSUED", "TRANSACTION DATE"]
        date_section = self.extract_from_context(text, date_labels, ["DUE", "PAYMENT", "EXPIRY"])
        if date_section:
            # Try to extract date from the section using regex
            date_patterns = [
                r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b',  # DD/MM/YYYY
                r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
                r'\b\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # DD Month YYYY
                r'\b\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b',  # YYYY/MM/DD
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, date_section, re.IGNORECASE)
                if match:
                    return match.group(0)
            
            # If we have a section but no pattern match, return the section if it's short
            if len(date_section) < 15 and re.search(r'\d', date_section):
                return date_section
        
        # Strategy 3: Just find any date-like pattern in the text
        date_patterns = [
            r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b',  # DD/MM/YYYY
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # DD Month YYYY
            r'\b\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b',  # YYYY/MM/DD
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
                
        return "Not found"

    def extract_vendor(self, text, donut_output=None):
        """Extract vendor information."""
        # Strategy 1: Try to extract from Donut output
        if donut_output and "<s_nm>" in donut_output:
            match = re.search(r'<s_nm>(.*?)</s_nm>', donut_output)
            if match:
                vendor = match.group(1).strip()
                # Filter out common non-vendor text
                if (vendor.lower() not in ["invoice", "date", "due", "bill to", "receipt", "payment"] and
                    len(vendor) > 3 and not re.match(r'^[0-9\s]+$', vendor)):
                    return vendor
        
        # Strategy 2: Check for vendor/merchant labels
        vendor_labels = ["VENDOR", "MERCHANT", "SELLER", "STORE", "BUSINESS", "COMPANY", "FROM"]
        vendor_section = self.extract_from_context(text, vendor_labels)
        if vendor_section and len(vendor_section) < 50:  # Avoid long paragraphs
            return vendor_section
        
        # Strategy 3: Often the vendor is at the top of the document
        lines = text.strip().split('\n')
        for i in range(min(3, len(lines))):
            line = lines[i].strip()
            # Skip if line is just a heading or very short
            if line and len(line) > 3 and not re.match(r'^\s*(INVOICE|INV|#|FAX|TEL|PHONE|DATE|RECEIPT)\s*$', line, re.IGNORECASE):
                # Skip lines that are likely just numbers or dates
                if not re.match(r'^[\d\s./:-]+$', line):
                    return line
                
        return "Not found"

    def extract_total(self, text, donut_output=None):
        """Extract total amount from document."""
        # Get currency symbol if available
        currency = self.extract_currency_symbol(text) or ""
        
        # Strategy 1: Try to extract from Donut output
        if donut_output and "<s_total_price>" in donut_output:
            match = re.search(r'<s_total_price>(.*?)</s_total_price>', donut_output)
            if match:
                total = match.group(1).strip()
                # Add currency if not present
                if currency and not re.search(r'[$€£₹]', total):
                    total = currency + total
                return total
        
        # Strategy 2: Look for context near total-related labels
        total_labels = ["TOTAL", "AMOUNT DUE", "BALANCE DUE", "TOTAL DUE", "GRAND TOTAL", "PAYMENT DUE"]
        
        # Sort text sections by proximity to total labels
        candidates = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            for label in total_labels:
                if re.search(r'\b' + re.escape(label) + r'\b', line, re.IGNORECASE):
                    # Extract numbers from the current line
                    amount_match = re.search(r'[$€£₹]?\s*(\d+(?:[.,]\d+)*)', line)
                    if amount_match:
                        amount = amount_match.group(0).strip()
                        # Higher confidence for lines with "total" keywords
                        confidence = 2 if re.search(r'\b(TOTAL|BALANCE|AMOUNT)\b', line, re.IGNORECASE) else 1
                        candidates.append((amount, confidence))
                    
                    # Check next line if needed
                    if (i < len(lines) - 1 and 
                        not amount_match or line.strip().lower().endswith(label.lower())):
                        next_line = lines[i+1].strip()
                        amount_match = re.search(r'[$€£₹]?\s*(\d+(?:[.,]\d+)*)', next_line)
                        if amount_match:
                            candidates.append((amount_match.group(0).strip(), 1.5))
        
        # Find the best candidate (highest confidence, likely to be the total)
        if candidates:
            candidates.sort(key=lambda x: (x[1], float(re.sub(r'[^0-9.]', '', x[0] or '0'))), reverse=True)
            best_match = candidates[0][0]
            # Add currency if not present
            if currency and not re.search(r'[$€£₹]', best_match):
                best_match = currency + best_match
            return best_match
        
        # Strategy 3: Look for any number after keywords
        total_patterns = [
            r'(?:total|amount|balance|due|sum).*?[$€£₹]?\s*(\d+(?:[.,]\d+)*)',
            r'[$€£₹]\s*(\d+(?:[.,]\d+)*)'
        ]
        
        for pattern in total_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                best_match = max(matches, key=lambda x: float(re.sub(r'[^0-9.]', '', x or '0')))
                # Add currency if not present
                if currency and not re.search(r'[$€£₹]', best_match):
                    best_match = currency + best_match
                return best_match
                
        return "Not found"

    def extract_invoice_number(self, text, donut_output=None):
        """Extract invoice/receipt number."""
        # Define confidence level for results
        confidence = 0.0
        best_match = "Not found"
        
        # First Strategy: Direct pattern match for invoice numbers 
        inv_direct_patterns = [
            r'\b(INV\s*0*1)\b',                     # Matches INV01, INV1, INV 1, etc.
            r'\b(INV[0-9]{4,})\b',                  # Matches INV0001, INV12345, etc.
            r'\bINVOICE\s+(?:NO\.?|NUMBER|#)?\s*([A-Za-z0-9-]+)', # Invoice NO. 12345
            r'\bINV\s*(?:NO\.?|NUMBER|#)?\s*([A-Za-z0-9-]+)',    # INV NO. 12345
        ]
        
        for pattern in inv_direct_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Use the first capture group if available, otherwise the whole match
                candidate = match.group(1) if len(match.groups()) > 0 else match.group(0)
                # Filter out false matches (words like INVOICE, BALANCE, etc.)
                if not re.match(r'^\s*(INVOICE|BALANCE|DUE|TOTAL|DATE)\s*$', candidate, re.IGNORECASE):
                    best_match = candidate
                    confidence = 0.9
                    logger.info(f"Found invoice number (direct pattern): {best_match} with confidence {confidence}")
                    return best_match
        
        # Strategy 1: Try to extract from Donut output
        if donut_output and "<s_invoice>" in donut_output:
            match = re.search(r'<s_invoice>(.*?)</s_invoice>', donut_output)
            if match:
                candidate = match.group(1).strip()
                # Filter out false matches
                if not re.match(r'^\s*(INVOICE|BALANCE|DUE|TOTAL|DATE)\s*$', candidate, re.IGNORECASE):
                    if confidence < 0.8:
                        best_match = candidate
                        confidence = 0.8
                        logger.info(f"Found invoice number (Donut): {best_match} with confidence {confidence}")
        
        # Strategy 2: Look for context near invoice labels
        invoice_labels = ["INVOICE", "INVOICE #", "INVOICE NO", "INVOICE NUMBER", "ORDER", "ORDER #", "RECEIPT", "RECEIPT #"]
        invoice_section = self.extract_from_context(text, invoice_labels, ["DATE", "TOTAL", "AMOUNT", "BALANCE", "DUE"])
        
        if invoice_section:
            # Try to extract invoice number patterns from the section
            inv_patterns = [
                r'[A-Z0-9]{3,}[-/#]?[A-Z0-9]{2,}',  # Common invoice number format
                r'(?:INV|INVOICE|RECEIPT)[-/#]?[A-Z0-9]+',  # INV-12345 format
                r'#\s*([A-Z0-9-]+)',  # #12345 format
                r'(\d{4,})'  # Just digits
            ]
            
            for pattern in inv_patterns:
                match = re.search(pattern, invoice_section, re.IGNORECASE)
                if match:
                    # Return the match, or the capture group if available
                    candidate = match.group(1) if len(match.groups()) > 0 else match.group(0)
                    # Filter out false matches
                    if not re.match(r'^\s*(INVOICE|BALANCE|DUE|TOTAL|DATE)\s*$', candidate, re.IGNORECASE):
                        if confidence < 0.7:
                            best_match = candidate
                            confidence = 0.7
                            logger.info(f"Found invoice number (context): {best_match} with confidence {confidence}")
            
            # If we have a section but no pattern match, return the section if it's short
            if confidence < 0.5 and len(invoice_section) < 15 and not re.match(r'^\s*(INVOICE|BALANCE|DUE|TOTAL|DATE)\s*$', invoice_section, re.IGNORECASE):
                best_match = invoice_section
                confidence = 0.5
                logger.info(f"Found invoice number (short section): {best_match} with confidence {confidence}")
        
        # Strategy 3: Look for patterns in the full text
        inv_patterns = [
            r'(?:INV|INVOICE|RECEIPT)[-/#]?\s*([A-Z0-9-]+)',
            r'(?:ORDER|TRANSACTION)[-/#]?\s*([A-Z0-9-]+)',
            r'#\s*([A-Z0-9-]+)',
        ]
        
        for pattern in inv_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1) if len(match.groups()) > 0 else match.group(0)
                # Filter out false matches
                if not re.match(r'^\s*(INVOICE|BALANCE|DUE|TOTAL|DATE)\s*$', candidate, re.IGNORECASE):
                    if confidence < 0.6:
                        best_match = candidate
                        confidence = 0.6
                        logger.info(f"Found invoice number (full text): {best_match} with confidence {confidence}")
                
        return best_match

    def extract_line_items(self, text, donut_output=None):
        """Extract line items from the document."""
        items = []
        
        # Strategy 1: Try to extract from Donut output (CORD format)
        donut_items = []
        if donut_output and "<s_menu>" in donut_output:
            menu_match = re.search(r"<s_menu>(.*?)</s_menu>", donut_output, re.DOTALL)
            if menu_match:
                menu_content = menu_match.group(1)
                item_chunks = re.split(r"<sep\s*/?>", menu_content)
                
                for chunk in item_chunks:
                    name_match = re.search(r"<s_nm>(.*?)</s_nm>", chunk)
                    price_match = re.search(r"<s_price>(.*?)</s_price>", chunk)
                    unit_price_match = re.search(r"<s_unitprice>(.*?)</s_unitprice>", chunk)
                    qty_match = re.search(r"<s_cnt>(.*?)</s_cnt>", chunk)
                    
                    if name_match:
                        name = name_match.group(1).strip()
                        # Skip headers and non-items
                        if name.lower() in ["invoice", "date", "due", "bill to", "description"]:
                            continue
                            
                        donut_items.append({
                            "name": name,
                            "quantity": qty_match.group(1).strip() if qty_match else "1",
                            "unit_price": unit_price_match.group(1).strip() if unit_price_match else (price_match.group(1).strip() if price_match else "Not found"),
                            "total": price_match.group(1).strip() if price_match else "Not found"
                        })
        
        # Strategy 2: Use OCR text to identify tabular data
        ocr_items = []
        currency = self.extract_currency_symbol(text)
        
        # Find potential item section by looking for headers
        headers_pattern = r'(description|item|product|service).*?(qty|quantity).*?(price|rate|cost).*?(amount|total)'
        item_section_start = -1
        item_section_end = -1
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if re.search(headers_pattern, line, re.IGNORECASE):
                item_section_start = i + 1
                break
                
        # Look for the end of the section (subtotal, total, etc.)
        if item_section_start > 0:
            for i in range(item_section_start, len(lines)):
                if re.search(r'\b(subtotal|total|sum|amount due)\b', lines[i], re.IGNORECASE):
                    item_section_end = i
                    break
            
            # Default to the end of the document if we can't find a proper ending
            if item_section_end < 0:
                item_section_end = len(lines)
                
            # Extract items from the identified section
            item_lines = lines[item_section_start:item_section_end]
            
            # Look for line item patterns: text followed by 2-3 numbers
            for line in item_lines:
                # Skip lines that are likely not items
                if len(line.strip()) < 5 or re.match(r'^\s*\d+\s*$', line):
                    continue
                    
                # Pattern: Product name, then 2-3 numbers (possibly with currency symbols)
                items_pattern = r'([A-Za-z][\w\s\-&.,]+)\s+([\d.,]+)\s+([\d.,]+)(?:\s+([\d.,]+))?'
                match = re.search(items_pattern, line)
                
                if match:
                    name = match.group(1).strip()
                    # Skip if the line appears to be a header or footer
                    if any(word in name.lower() for word in ['total', 'subtotal', 'tax', 'discount', 'shipping']):
                        continue
                        
                    # Extract the numbers (quantity, unit price, total)
                    numbers = [g for g in match.groups()[1:] if g]
                    
                    if len(numbers) >= 2:
                        item = {
                            "name": name,
                            "quantity": numbers[0]
                        }
                        
                        # Handle different number formats
                        if len(numbers) == 2:  # qty and total only
                            item["unit_price"] = "Not found"
                            item["total"] = currency + numbers[1] if currency else numbers[1]
                        else:  # qty, unit price, and total
                            item["unit_price"] = currency + numbers[1] if currency else numbers[1]
                            item["total"] = currency + numbers[2] if currency else numbers[2]
                            
                        ocr_items.append(item)
        
        # Strategy 3: Fallback to looking for any product/price pattern
        fallback_items = []
        if not ocr_items:
            # Simple pattern for product followed by price
            simple_pattern = r'([A-Za-z][\w\s\-&.,]+)\s+[\$€£₹]?\s*([\d.,]+)'
            
            # Skip common headers and sections
            skip_words = ['invoice', 'date', 'due', 'bill', 'total', 'subtotal', 'discount', 'tax', 'shipping']
            
            for match in re.finditer(simple_pattern, text):
                name = match.group(1).strip()
                price = match.group(2)
                
                # Skip non-items
                if any(word in name.lower() for word in skip_words):
                    continue
                if len(name) < 3 or len(price) < 1:
                    continue
                    
                fallback_items.append({
                    "name": name,
                    "quantity": "1",  # Default
                    "unit_price": currency + price if currency else price,
                    "total": currency + price if currency else price
                })
        
        # Combine results, prioritizing more structured data
        if ocr_items:
            items = ocr_items
        elif donut_items:
            items = donut_items
        elif fallback_items:
            items = fallback_items[:5]  # Limit to avoid noise
        
        return items

    def extract_customer(self, text, donut_output=None):
        """Extract customer information."""
        # Strategy 1: Look for common customer/client labels
        customer_labels = ["BILL TO", "SOLD TO", "CUSTOMER", "CLIENT", "RECIPIENT", "SHIP TO", "BUYER"]
        customer_section = self.extract_from_context(text, customer_labels, ["INVOICE", "TOTAL", "VENDOR"])
        
        if customer_section:
            # Clean up the section
            lines = customer_section.strip().split('\n')
            # First line is usually the name
            if lines:
                return lines[0].strip()
        
        # Strategy 2: Look for email pattern
        email_match = re.search(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', text)
        if email_match:
            # Get the text near the email
            email_index = text.find(email_match.group(0))
            if email_index > 0:
                # Look for a name above the email
                email_context = text[max(0, email_index - 100):email_index].strip()
                # Last line before email might be the name
                lines = email_context.split('\n')
                if lines:
                    return lines[-1].strip()
        
        return "Not found"

    def get_confidence_score(self, field, value, raw_text):
        """Calculate confidence score for extracted fields."""
        if value == "Not found":
            return 0.0
            
        # Basic confidence based on field type and content
        if field == "date":
            # Higher confidence for well-formatted dates
            if re.match(r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}\b', value) or \
               re.match(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', value, re.IGNORECASE):
                return 0.9
            elif re.search(r'\d{4}', value):  # Has a year
                return 0.7
            return 0.5
            
        elif field == "total_amount":
            # Higher confidence for amounts with currency symbols
            if re.match(r'[$€£₹]\s*\d+([.,]\d+)?', value):
                return 0.9
            elif re.match(r'\d+([.,]\d+)?', value):  # Just numbers
                return 0.7
            return 0.6
            
        elif field == "vendor":
            # Higher confidence for vendors found at top of document
            first_few_lines = '\n'.join(raw_text.split('\n')[:5])
            if value in first_few_lines:
                return 0.8
            return 0.6
            
        elif field == "invoice_number":
            # Higher confidence for invoice numbers with INV prefix
            if re.match(r'INV[0-9-]+', value, re.IGNORECASE):
                return 0.9
            elif re.match(r'[A-Z0-9-]+', value):  # Alphanumeric
                return 0.7
            return 0.5
            
        # Default confidence
        return 0.5
        
    def process_document(self, file_path):
        """Process document and extract structured information."""
        try:
            # Handle PDF files by converting to image
            if file_path.lower().endswith('.pdf'):
                file_path = self.convert_pdf_to_image(file_path)
            
            # Preprocess image for better quality
            processed_path = self.preprocess_image(file_path)
            
            # Get OCR text
            raw_text = self.get_ocr_text(processed_path)
            logger.info(f"OCR Text extracted: {len(raw_text)} characters")
            
            # Process with Donut model
            donut_output = self.process_with_donut(processed_path)
            
            # Extract invoice information
            vendor = self.extract_vendor(raw_text, donut_output)
            date = self.extract_date(raw_text, donut_output)
            invoice_number = self.extract_invoice_number(raw_text, donut_output)
            total_amount = self.extract_total(raw_text, donut_output)
            items = self.extract_line_items(raw_text, donut_output)
            
            # Additional data extraction
            customer = self.extract_customer(raw_text, donut_output)
            
            # Calculate confidence scores
            vendor_confidence = self.get_confidence_score("vendor", vendor, raw_text)
            date_confidence = self.get_confidence_score("date", date, raw_text)
            invoice_confidence = self.get_confidence_score("invoice_number", invoice_number, raw_text)
            total_confidence = self.get_confidence_score("total_amount", total_amount, raw_text)
            
            # Build result with confidence scores
            result = {
                "vendor": vendor,
                "vendor_confidence": vendor_confidence,
                "date": date,
                "date_confidence": date_confidence,
                "invoice_number": invoice_number,
                "invoice_number_confidence": invoice_confidence,
                "total_amount": total_amount,
                "total_amount_confidence": total_confidence,
                "customer": customer, 
                "items": items,
                "raw_text": raw_text
            }
            
            logger.info("Extracted information:")
            for key, value in result.items():
                if key not in ["raw_text", "items"] and not key.endswith("_confidence"):
                    logger.info(f"  {key}: {value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "raw_text": self.get_ocr_text(file_path) if os.path.exists(file_path) else "No text extracted"
            }