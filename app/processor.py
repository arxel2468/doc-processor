# app/processor.py
import os
os.environ['TRANSFORMERS_CACHE'] = 'D:\\huggingface_cache'
import re
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor
from pdf2image import convert_from_path
import cv2

class DocumentProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2").to(self.device)
        self.task_prompt = "<s_cord-v2>"

    def preprocess_image(self, image_path):
        """Enhanced image preprocessing."""
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
        return temp_path

    def convert_pdf_to_image(self, pdf_path):
        """Convert first page of PDF to image."""
        images = convert_from_path(pdf_path, dpi=300)
        image_path = os.path.splitext(pdf_path)[0] + "_page1.jpg"
        images[0].save(image_path, "JPEG")
        return image_path

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
                return self.clean_text(matches[-1])  # Return the last number found

        return "Not found"

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

    def format_raw_text(self, text):
        # Replace tags with readable format
        text = re.sub(r"<s_(\w+)>", r"\n\1: ", text)
        text = re.sub(r"</s_\w+>", "", text)
        text = re.sub(r"<sep\s*/?>", "\n---", text)
        return text.strip()

    def process_document(self, file_path):
        try:
            if file_path.lower().endswith('.pdf'):
                file_path = self.convert_pdf_to_image(file_path)

            # Preprocess image
            processed_path = self.preprocess_image(file_path)
            
            # Extract text
            raw_text = self.extract_text(processed_path)
            formatted_text = self.format_raw_text(raw_text)
            items = self.extract_line_items(raw_text)

            # Extract vendor name (try to find the first non-item name)
            vendor = self.extract_tag_value(raw_text, "s_nm")
            if vendor.lower() in ['invoice', 'date', 'due', 'bill to', 'rate qty', 'amount']:
                vendor = "Not found"

            # Extract date (try to find a date pattern)
            date = self.extract_tag_value(raw_text, "s_date")
            if date == "Not found":
                # Try to find date in any tag
                date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}', raw_text)
                if date_match:
                    date = date_match.group(0)

            return {
                "vendor": vendor,
                "date": date,
                "total_amount": self.extract_tag_value(raw_text, "s_total_price"),
                "items": items,
                "raw_text": formatted_text
            }

        except Exception as e:
            return {"error": str(e)}