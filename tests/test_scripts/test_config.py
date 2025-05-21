"""
Centralized configuration file for test paths
"""
import os

# Base paths
TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_DIR = os.path.join(TEST_DIR, 'test_data')
TEST_RESULTS_DIR = os.path.join(TEST_DIR, 'test_results')
TEST_SCRIPTS_DIR = os.path.join(TEST_DIR, 'test_scripts')

# Test data files
PDF_TEST_FILE = os.path.join(TEST_DATA_DIR, 'large-doc.pdf')
DOCX_TEST_FILE = os.path.join(TEST_DATA_DIR, 'large-doc.docx')
PPTX_TEST_FILE = os.path.join(TEST_DATA_DIR, 'example.pptx')
EXAMPLE_PDF = os.path.join(TEST_DATA_DIR, 'example.pdf')
EXAMPLE_LARGE_PDF = os.path.join(TEST_DATA_DIR, 'example-large.pdf')

# Test result files
STANDARD_RESULT_FILE = os.path.join(TEST_RESULTS_DIR, 'standard_result.json')
STREAMING_RESULT_FILE = os.path.join(TEST_RESULTS_DIR, 'streaming_result.json')
OPTIMIZED_RESULT_FILE = os.path.join(TEST_RESULTS_DIR, 'optimized_result.json')
STANDARD_TIME_FILE = os.path.join(TEST_RESULTS_DIR, 'standard_time.txt')
STREAMING_TIME_FILE = os.path.join(TEST_RESULTS_DIR, 'streaming_time.txt')
OPTIMIZED_TIME_FILE = os.path.join(TEST_RESULTS_DIR, 'optimized_time.txt')

# Make sure test results directory exists
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Print debug info
print(f"Test data directory: {TEST_DATA_DIR}")
print(f"Test results directory: {TEST_RESULTS_DIR}")
print(f"PDF test file path: {PDF_TEST_FILE}")
print(f"DOCX test file path: {DOCX_TEST_FILE}")
print(f"PPTX test file path: {PPTX_TEST_FILE}")
