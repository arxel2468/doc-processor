import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from processor import DocumentProcessor
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Benchmark:
    def __init__(self, test_data_dir: str = "test_data"):
        """Initialize the benchmark with test data directory."""
        self.processor = DocumentProcessor()
        self.test_data_dir = test_data_dir
        self.results = []
        logger.info(f"Initialized Benchmark with test data directory: {test_data_dir}")

    def load_test_cases(self) -> List[Dict]:
        """Load test cases from JSON files in the test data directory."""
        test_cases = []
        try:
            for filename in os.listdir(self.test_data_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(self.test_data_dir, filename), 'r') as f:
                        test_case = json.load(f)
                        test_cases.append(test_case)
            logger.info(f"Loaded {len(test_cases)} test cases")
            return test_cases
        except Exception as e:
            logger.error(f"Error loading test cases: {str(e)}")
            raise

    def calculate_accuracy(self, extracted: Dict, expected: Dict) -> Dict[str, float]:
        """Calculate accuracy metrics for each field."""
        metrics = {}
        total_fields = 0
        correct_fields = 0

        for field in ['date', 'vendor', 'total', 'invoice_number']:
            total_fields += 1
            if field in extracted and field in expected:
                if field == 'total':
                    # Compare total amounts with tolerance for floating point differences
                    if abs(float(extracted[field]) - float(expected[field])) < 0.01:
                        correct_fields += 1
                else:
                    if str(extracted[field]).lower() == str(expected[field]).lower():
                        correct_fields += 1

        # Calculate line items accuracy
        if 'line_items' in extracted and 'line_items' in expected:
            total_items = len(expected['line_items'])
            if total_items > 0:
                correct_items = 0
                for expected_item in expected['line_items']:
                    for extracted_item in extracted['line_items']:
                        if (abs(float(extracted_item['amount']) - float(expected_item['amount'])) < 0.01 and
                            abs(float(extracted_item['quantity']) - float(expected_item['quantity'])) < 0.01 and
                            str(extracted_item['description']).lower() == str(expected_item['description']).lower()):
                            correct_items += 1
                            break
                metrics['line_items_accuracy'] = correct_items / total_items

        metrics['overall_accuracy'] = correct_fields / total_fields if total_fields > 0 else 0
        return metrics

    def run_benchmark(self) -> List[Dict]:
        """Run the benchmark on all test cases."""
        try:
            test_cases = self.load_test_cases()
            results = []

            for test_case in test_cases:
                try:
                    # Process the document
                    extracted = self.processor.process_document(test_case['document_path'])
                    
                    # Calculate accuracy
                    metrics = self.calculate_accuracy(extracted, test_case['expected'])
                    
                    # Store results
                    result = {
                        'document': test_case['document_path'],
                        'extracted': extracted,
                        'expected': test_case['expected'],
                        'metrics': metrics,
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result)
                    
                    logger.info(f"Processed {test_case['document_path']} with accuracy: {metrics['overall_accuracy']:.2%}")
                except Exception as e:
                    logger.error(f"Error processing {test_case['document_path']}: {str(e)}")
                    continue

            self.results = results
            return results
        except Exception as e:
            logger.error(f"Error running benchmark: {str(e)}")
            raise

    def generate_report(self, output_file: str = "benchmark_report.json") -> None:
        """Generate a detailed benchmark report."""
        try:
            if not self.results:
                self.run_benchmark()

            report = {
                'timestamp': datetime.now().isoformat(),
                'total_test_cases': len(self.results),
                'results': self.results,
                'summary': {
                    'average_accuracy': sum(r['metrics']['overall_accuracy'] for r in self.results) / len(self.results),
                    'successful_extractions': len([r for r in self.results if r['metrics']['overall_accuracy'] > 0]),
                    'failed_extractions': len([r for r in self.results if r['metrics']['overall_accuracy'] == 0])
                }
            }

            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Generated benchmark report: {output_file}")
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    benchmark = Benchmark()
    benchmark.run_benchmark()
    benchmark.generate_report() 