import sys
import os
from app.processor import DocumentProcessor

def test_processing():
    # Initialize processor
    processor = DocumentProcessor()
    
    # Check if a file path was provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Use default sample file
        file_path = os.path.join('uploads', 'sample_invoice.jpg')
    
    print(f"Processing file: {file_path}")
    
    # Process the document
    result = processor.process_document(file_path)
    
    # Print results
    print("\n=== PROCESSING RESULTS ===")
    print(f"Vendor: {result.get('vendor', 'Not found')}")
    print(f"Date: {result.get('date', 'Not found')}")
    print(f"Invoice Number: {result.get('invoice_number', 'Not found')}")
    print(f"Total Amount: {result.get('total_amount', 'Not found')}")
    
    print("\nLine Items:")
    for i, item in enumerate(result.get('items', []), 1):
        print(f"  {i}. {item.get('name')} - {item.get('quantity')} x {item.get('unit_price')} = {item.get('total')}")
    
    print("\nRaw Text (first 300 chars):")
    print(result.get('raw_text', '')[:300] + "...")
    
    return result

if __name__ == "__main__":
    test_processing() 