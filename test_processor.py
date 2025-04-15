import sys
import os
from app.processor import DocumentProcessor
import argparse
import json

def test_processing():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test the document processor')
    parser.add_argument('--file', '-f', help='Path to the invoice file to process')
    parser.add_argument('--output', '-o', help='Path to save JSON output')
    args = parser.parse_args()
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Get file path - use command-line arg, env var, or default
    file_path = args.file
    if not file_path:
        file_path = os.environ.get('TEST_INVOICE_FILE')
    if not file_path:
        # Use default sample file
        file_path = os.path.join('uploads', 'sample_invoice.jpg')
    
    print(f"Processing file: {file_path}")
    
    # Process the document
    result = processor.process_document(file_path)
    
    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            # Remove raw text to make output more readable
            output_result = {k: v for k, v in result.items() if k != 'raw_text'}
            json.dump(output_result, f, indent=2)
        print(f"Results saved to {args.output}")
    
    # Print results
    print("\n=== PROCESSING RESULTS ===")
    print(f"Vendor: {result.get('vendor', 'Not found')} (confidence: {result.get('vendor_confidence', 'N/A')})")
    print(f"Date: {result.get('date', 'Not found')} (confidence: {result.get('date_confidence', 'N/A')})")
    print(f"Invoice Number: {result.get('invoice_number', 'Not found')} (confidence: {result.get('invoice_number_confidence', 'N/A')})")
    print(f"Total Amount: {result.get('total_amount', 'Not found')} (confidence: {result.get('total_amount_confidence', 'N/A')})")
    print(f"Customer: {result.get('customer', 'Not found')}")
    
    print("\nLine Items:")
    for i, item in enumerate(result.get('items', []), 1):
        print(f"  {i}. {item.get('name')} - {item.get('quantity')} x {item.get('unit_price')} = {item.get('total')}")
    
    print("\nRaw Text (first 300 chars):")
    print(result.get('raw_text', '')[:300] + "...")
    
    return result

if __name__ == "__main__":
    test_processing() 