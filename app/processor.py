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
import time
import traceback
from datetime import datetime

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
            return [image_path]
        except Exception as e:
            logger.error(f"Error converting PDF to image: {str(e)}")
            return None

    def preprocess_image(self, image_path):
        """Enhance image for better OCR quality."""
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 2
            )
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Ensure the image is in the correct format
            if enhanced.dtype != np.uint8:
                enhanced = enhanced.astype(np.uint8)
            
            # Save processed image
            temp_path = os.path.splitext(image_path)[0] + "_processed.jpg"
            success = cv2.imwrite(temp_path, enhanced)
            
            if not success:
                raise ValueError(f"Failed to save processed image to {temp_path}")
            
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
            logger.info(f"Loaded image from {image_path}")
            
            # Generate pixel values
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            logger.info("Generated pixel values")

            # Generate decoder input ids
            decoder_input_ids = self.processor.tokenizer(
                self.task_prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(self.device)
            logger.info("Generated decoder input ids")

            # Generate output
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=768,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                num_beams=4,
            )
            logger.info("Generated model output")
            
            # Decode output
            sequence = self.processor.batch_decode(outputs)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            logger.info(f"Full Donut model output: {sequence}")
            
            # Parse the CORD-v2 format output
            result = {}
            
            # Extract date with improved pattern
            date_match = re.search(r'<s_nm>\s*DATE\s*</s_nm>\s*<s_price>\s*([^<]+)\s*</s_price>', sequence)
            if date_match:
                date_str = date_match.group(1).strip()
                # Try to parse the date
                try:
                    # Handle various date formats
                    date_formats = [
                        '%b %d, %Y',  # Apr 11, 2025
                        '%d %b, %Y',  # 11 Apr, 2025
                        '%Y-%m-%d',   # 2025-04-11
                        '%d/%m/%Y',   # 11/04/2025
                        '%m/%d/%Y'    # 04/11/2025
                    ]
                    for fmt in date_formats:
                        try:
                            date_obj = datetime.strptime(date_str, fmt)
                            result['date'] = date_obj.strftime('%b %d, %Y')
                            break
                        except ValueError:
                            continue
                    if 'date' not in result:
                        result['date'] = date_str
                except ValueError:
                    result['date'] = date_str
                logger.info(f"Found date: {result['date']}")
            
            # Extract vendor with improved pattern
            vendor_match = re.search(r'<s_nm>\s*([^<]+)\s*</s_nm>\s*<s_price>\s*INVOICE\s*</s_price>', sequence)
            if vendor_match:
                result['vendor'] = vendor_match.group(1).strip()
                logger.info(f"Found vendor: {result['vendor']}")
            
            # Extract invoice number with improved pattern
            invoice_match = re.search(r'INV[#\s]*([A-Z0-9-]+)', sequence)
            if not invoice_match:
                # Try alternative patterns
                invoice_match = re.search(r'Invoice[#\s]*([A-Z0-9-]+)', sequence, re.IGNORECASE)
            if invoice_match:
                result['invoice_number'] = invoice_match.group(1).strip()
                logger.info(f"Found invoice number: {result['invoice_number']}")
            
            # Extract total with improved pattern
            total_match = re.search(r'<s_total_price>\s*([^<]+)\s*</s_total_price>', sequence)
            if total_match:
                total_str = total_match.group(1).strip()
                # Try to parse the total amount
                try:
                    # Remove currency symbols and other non-numeric characters
                    total_str = re.sub(r'[^\d.]', '', total_str)
                    total = float(total_str)
                    result['total'] = f"{total:.2f}"
                except ValueError:
                    result['total'] = total_str
                logger.info(f"Found total: {result['total']}")
            
            # Extract line items with improved pattern
            line_items = []
            item_matches = re.finditer(r'<s_nm>\s*([^<]+)\s*</s_nm>\s*<s_unitprice>\s*([^<]+)\s*</s_unitprice>\s*<s_cnt>\s*([^<]+)\s*</s_cnt>', sequence)
            for match in item_matches:
                description = match.group(1).strip()
                rate = match.group(2).strip()
                quantity = match.group(3).strip()
                
                # Try to parse the amounts
                try:
                    # Clean up rate and quantity
                    rate_float = float(re.sub(r'[^\d.]', '', rate))
                    quantity_int = int(re.sub(r'[^\d]', '', quantity))
                    amount = rate_float * quantity_int
                    line_items.append({
                        'description': description,
                        'quantity': str(quantity_int),
                        'rate': f"{rate_float:.2f}",
                        'amount': f"{amount:.2f}"
                    })
                    logger.info(f"Found line item: {description} - {quantity_int} x {rate_float} = {amount}")
                except ValueError:
                    continue
            
            if line_items:
                result['line_items'] = line_items
            
            # Calculate confidence scores based on extraction success
            confidence_scores = {
                'date': 0.9 if result.get('date') and result['date'] != 'DATE' else 0.0,
                'vendor': 0.9 if result.get('vendor') else 0.0,
                'total': 0.9 if result.get('total') else 0.0,
                'invoice_number': 0.9 if result.get('invoice_number') and result['invoice_number'] != 'OICE' else 0.0,
                'line_items': 0.9 if result.get('line_items') else 0.0,
            }
            
            # Calculate overall confidence
            valid_scores = [score for score in confidence_scores.values() if score > 0]
            confidence_scores['overall'] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            
            result['confidence_scores'] = confidence_scores
            
            logger.info(f"Final parsed result: {json.dumps(result, indent=2)}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing with Donut: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {}

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

    def extract_date(self, text: str, donut_output: dict) -> str:
        """Extract date from text using regex patterns and Donut output."""
        date = None
        
        # Try to get date from Donut output first
        if 'date' in donut_output:
            date = donut_output['date']
        
        # If not found or invalid, try regex patterns
        if not date or date == 'DATE':
            date_patterns = [
                r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}',
                r'\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}',
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{2}-\d{2}-\d{4}'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    date = match.group(0)
                    break
        
        # If still not found, look for date-like patterns in the text
        if not date or date == 'DATE':
            lines = text.split('\n')
            for line in lines:
                if any(month in line.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                    date = line.strip()
                    break
        
        return date if date else 'Not found'

    def extract_vendor(self, text, donut_output=None):
        """Extract vendor name from the text content."""
        # Try to get vendor from donut output first
        if donut_output and 'supplier' in donut_output and donut_output['supplier']:
            return donut_output['supplier'], 0.9
        
        # Common words to ignore in potential vendor names
        ignore_words = ['invoice', 'bill', 'payment', 'receipt', 'total', 'date', 'due', 'amount', 'tax', 'paid']
        
        # Extract potential vendor names (capitalize words that are not common ignore words)
        lines = text.split('\n')
        potential_vendors = []
        
        # Look for lines that might contain company names (before any other field markers)
        for line in lines[:10]:  # Check first 10 lines as vendor usually appears at the top
            line = line.strip()
            if not line or len(line) < 3 or any(word.lower() in line.lower() for word in ignore_words):
                continue
            
            # Check if line has email format
            if '@' in line and '.' in line.split('@')[1]:
                # This might be an email, extract the domain as potential vendor
                email_parts = line.split('@')
                if len(email_parts) > 1:
                    domain = email_parts[1].split('.')[0]
                    if len(domain) > 2 and domain.lower() not in ignore_words:
                        potential_vendors.append((domain.capitalize(), 0.7))
            else:
                # Consider lines with capitalized words as potential vendors
                words = line.split()
                if len(words) <= 5 and any(word[0].isupper() for word in words if len(word) > 1):
                    confidence = 0.6 if any(word[0].isupper() for word in words if len(word) > 1) else 0.3
                    potential_vendors.append((line, confidence))
        
        # Return the most likely vendor name with the highest confidence
        if potential_vendors:
            potential_vendors.sort(key=lambda x: x[1], reverse=True)
            return potential_vendors[0]
        
        return None, 0.0

    def extract_total(self, text, donut_output=None):
        """Extract total amount from the text content."""
        # Try to get total from donut output first
        if isinstance(donut_output, dict) and 'total' in donut_output:
            total = donut_output['total']
            # Clean the total amount
            total = re.sub(r'[^\d.]', '', total)
            try:
                return float(total), 0.9
            except ValueError:
                pass
        
        # Currency symbols and their regex patterns
        currency_patterns = {
            'USD': r'\$\s*[\d,]+\.\d{2}',
            'EUR': r'€\s*[\d,]+\.\d{2}',
            'GBP': r'£\s*[\d,]+\.\d{2}',
            'INR': r'₹\s*[\d,]+(?:\.\d{2})?|(?:Rs|INR)\s*[\d,]+(?:\.\d{2})?',
            'Generic': r'(?:total|amount|balance)[^\d]*?([\d,]+\.\d{2})',
            'Number': r'\b(?:total|amount|balance|due)[^\d]*?([\d,]+(?:\.\d{2})?)\b'
        }
        
        results = []
        
        # Look for formatted currency amounts
        for currency, pattern in currency_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if currency == 'Generic' or currency == 'Number':
                    for match in matches:
                        # For generic patterns, we need to include some context
                        context_match = re.search(f"(?:total|amount|balance)[^\\d]*?{re.escape(match)}", text, re.IGNORECASE)
                        if context_match:
                            results.append((match, 0.8))
                else:
                    # Extract the highest amount as it's likely to be the total
                    amounts = []
                    for match in matches:
                        # Remove currency symbol and commas, then convert to float
                        clean_amount = re.sub(r'[^\d.]', '', match)
                        try:
                            amounts.append((match, float(clean_amount)))
                        except ValueError:
                            continue
                    
                    if amounts:
                        # Sort by amount value (descending) and take the highest
                        amounts.sort(key=lambda x: x[1], reverse=True)
                        results.append((amounts[0][0], 0.9))
        
        # Search for "total" or "due" followed by an amount
        total_patterns = [
            r'(?:total|amount due|balance due|due amount)[^\d]*?([\d,]+\.\d{2})',
            r'(?:total|amount due|balance due|due amount)[^\d]*?([A-Z]{3}\s*[\d,]+\.\d{2})',
            r'(?:total|amount due|balance due|due amount)[^\d]*?([₹$€£]\s*[\d,]+\.\d{2})',
        ]
        
        for pattern in total_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    results.append((match, 0.95))  # High confidence for amounts with "total" context
        
        # If we have results, return the one with highest confidence
        if results:
            results.sort(key=lambda x: x[1], reverse=True)
            total = results[0][0]
            # Clean the total amount
            total = re.sub(r'[^\d.]', '', total)
            try:
                return float(total), results[0][1]
            except ValueError:
                pass
        
        return None, 0.0

    def extract_invoice_number(self, text: str, donut_output: dict) -> str:
        """Extract invoice number from text using regex patterns and Donut output."""
        invoice_number = None
        
        # Try to get invoice number from Donut output first
        if 'invoice_number' in donut_output:
            invoice_number = donut_output['invoice_number']
        
        # If not found or invalid, try regex patterns
        if not invoice_number or invoice_number == 'Niteswift':
            patterns = [
                r'INV[#\s]*[\w-]+',  # Matches INV followed by any characters
                r'Invoice[#\s]*[\w-]+',  # Matches Invoice followed by any characters
                r'INV\d+',  # Matches INV followed by numbers
                r'Invoice\s+\d+',  # Matches Invoice followed by numbers
                r'INV-\d+',  # Matches INV- followed by numbers
                r'INV/\d+',  # Matches INV/ followed by numbers
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    invoice_number = match.group(0)
                    break
        
        # If still not found, look for invoice-like patterns in the text
        if not invoice_number or invoice_number == 'Niteswift':
            lines = text.split('\n')
            for line in lines:
                if any(prefix in line.upper() for prefix in ['INV', 'INVOICE']):
                    invoice_number = line.strip()
                    break
        
        return invoice_number if invoice_number else 'Not found'

    def extract_line_items(self, text: str, donut_output: dict) -> list:
        """Extract line items from text using regex patterns and Donut output."""
        line_items = []
        
        # Try to get line items from Donut output first
        if 'line_items' in donut_output:
            line_items = donut_output['line_items']
        
        # If not found or empty, try regex patterns
        if not line_items:
            # Pattern to match line items with description, quantity, rate, and amount
            pattern = r'([A-Za-z\s]+)\s+(\d+\.?\d*)\s+(\d+)\s+(\d+\.?\d*)'
            matches = re.finditer(pattern, text)
            
            for match in matches:
                description = match.group(1).strip()
                rate = match.group(2)
                quantity = match.group(3)
                amount = match.group(4)
                
                # Validate the amounts
                try:
                    rate_float = float(rate)
                    quantity_int = int(quantity)
                    amount_float = float(amount)
                    
                    # Check if the calculation makes sense
                    if abs(rate_float * quantity_int - amount_float) < 0.01:
                        line_items.append({
                            'description': description,
                            'quantity': quantity,
                            'rate': rate,
                            'amount': amount
                        })
                except ValueError:
                    continue
        
        # If still not found, look for line items in the text
        if not line_items:
            lines = text.split('\n')
            current_item = None
            
            for line in lines:
                # Look for lines that might contain item information
                if any(keyword in line.lower() for keyword in ['description', 'item', 'product']):
                    continue
                    
                # Try to parse the line as a line item
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        # Try to identify which parts are numbers
                        numbers = []
                        description_parts = []
                        for part in parts:
                            try:
                                float(part)
                                numbers.append(part)
                            except ValueError:
                                description_parts.append(part)
                        
                        if len(numbers) >= 3:
                            description = ' '.join(description_parts)
                            rate = numbers[0]
                            quantity = numbers[1]
                            amount = numbers[2]
                            
                            line_items.append({
                                'description': description,
                                'quantity': quantity,
                                'rate': rate,
                                'amount': amount
                            })
                    except (ValueError, IndexError):
                        continue
        
        return line_items

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
        """Process the document and extract all relevant information."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting document processing for {file_path}")
            
            # Check if the file is a PDF
            is_pdf = file_path.lower().endswith('.pdf')
            logger.info(f"File type: {'PDF' if is_pdf else 'Image'}")
            
            if is_pdf:
                # Convert PDF to images
                image_paths = self.convert_pdf_to_image(file_path)
                if not image_paths:
                    logger.error("Failed to convert PDF to images")
                    return {'error': 'Failed to convert PDF to images'}
                
                # Process the first page
                image_path = image_paths[0]
                logger.info(f"Using first page of PDF: {image_path}")
            else:
                # Assume it's an image
                image_path = file_path
                logger.info(f"Processing image: {image_path}")
            
            # Preprocess the image
            processed_image_path = self.preprocess_image(image_path)
            logger.info(f"Preprocessed image saved to: {processed_image_path}")
            
            # Use hybrid extraction
            result = self.hybrid_extract(processed_image_path)
            
            # Add processing time and file information
            processing_time = time.time() - start_time
            result['processing_time'] = round(processing_time, 2)
            result['processed_image_path'] = processed_image_path
            
            logger.info(f"Total processing time: {processing_time:.2f} seconds")
            logger.info(f"Final result: {json.dumps(result, indent=2)}")
            return result
        
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {'error': str(e)}

    def hybrid_extract(self, image_path):
        """Extract information using both Tesseract and Donut, then combine results."""
        try:
            # Get OCR text
            ocr_text = self.get_ocr_text(image_path)
            logger.info(f"Extracted OCR text: {ocr_text[:200]}...")
            
            # Process with Donut
            donut_result = self.process_with_donut(image_path)
            logger.info(f"Donut model output: {json.dumps(donut_result, indent=2)}")
            
            # Extract date from OCR text
            date_patterns = [
                r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}',
                r'\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}',
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{2}-\d{2}-\d{4}'
            ]
            
            date = None
            for pattern in date_patterns:
                match = re.search(pattern, ocr_text, re.IGNORECASE)
                if match:
                    date = match.group(0)
                    break
            
            # Extract invoice number from OCR text
            invoice_patterns = [
                r'INV[#\s]*([A-Z0-9-]+)',
                r'Invoice[#\s]*([A-Z0-9-]+)',
                r'INV\d+',
                r'Invoice\s+\d+',
                r'INV-\d+',
                r'INV/\d+'
            ]
            
            invoice_number = None
            for pattern in invoice_patterns:
                match = re.search(pattern, ocr_text, re.IGNORECASE)
                if match:
                    invoice_number = match.group(1)
                    break
            
            # Extract line items from OCR text
            line_items = []
            item_pattern = r'([A-Za-z\s]+)\s+(\d+\.?\d*)\s+(\d+)\s+(\d+\.?\d*)'
            matches = re.finditer(item_pattern, ocr_text)
            
            for match in matches:
                description = match.group(1).strip()
                rate = match.group(2)
                quantity = match.group(3)
                amount = match.group(4)
                
                try:
                    rate_float = float(rate)
                    quantity_int = int(quantity)
                    amount_float = float(amount)
                    
                    if abs(rate_float * quantity_int - amount_float) < 0.01:
                        line_items.append({
                            'description': description,
                            'quantity': str(quantity_int),
                            'rate': f"{rate_float:.2f}",
                            'amount': f"{amount_float:.2f}"
                        })
                except ValueError:
                    continue
            
            # Combine results
            result = {
                'date': date or donut_result.get('date', 'Not found'),
                'vendor': donut_result.get('vendor', 'Not found'),
                'invoice_number': invoice_number or donut_result.get('invoice_number', 'Not found'),
                'total': donut_result.get('total', 'Not found'),
                'line_items': line_items or donut_result.get('line_items', [])
            }
            
            # Calculate confidence scores
            confidence_scores = {
                'date': 0.9 if result['date'] != 'Not found' and result['date'] != 'DATE' else 0.0,
                'vendor': 0.9 if result['vendor'] != 'Not found' else 0.0,
                'invoice_number': 0.9 if result['invoice_number'] != 'Not found' and result['invoice_number'] != 'OICE' else 0.0,
                'total': 0.9 if result['total'] != 'Not found' else 0.0,
                'line_items': 0.9 if result['line_items'] else 0.0
            }
            
            # Calculate overall confidence
            valid_scores = [score for score in confidence_scores.values() if score > 0]
            confidence_scores['overall'] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            
            result['confidence_scores'] = confidence_scores
            
            logger.info(f"Hybrid extraction result: {json.dumps(result, indent=2)}")
            return result
            
        except Exception as e:
            logger.error(f"Error in hybrid extraction: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {}