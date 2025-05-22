# Unsiloed Chunker Tests

This directory contains tests for the Unsiloed-chunker project.

## Directory Structure

- `test_scripts/` - Python test scripts
- `test_data/` - Test documents (PDF, DOCX, PPTX)
- `test_results/` - Test result files (JSON, TXT)

## Test Files

- `test_scripts/test_standard.py` - Tests the standard document processing
- `test_scripts/test_streaming.py` - Tests the streaming document processing approach
- `test_scripts/test_optimized_docx.py` - Tests the enhanced streaming document processing with DOCX files
- `test_scripts/test_performance_pdf.py` - Tests performance with different PDF processing approaches
- `test_scripts/test_performance_pptx.py` - Tests performance with different PPTX processing approaches
- `test_scripts/compare_results.py` - Utility script to compare results from multiple test runs
- `test_scripts/test_config.py` - Central configuration for test file paths

## Running Tests

You can run individual tests:

```bash
cd tests
python ./test_scripts/test_standard.py
python ./test_scripts/test_streaming.py
python ./test_scripts/test_optimized_docx.py
python ./test_scripts/test_performance_pdf.py
python ./test_scripts/test_performance_pptx.py
```

Or run all tests in sequence using the isolated test script:

```bash
cd tests
./run_isolated_tests.sh
```

## Test Documents

The test documents should be placed in the `test_data/` directory:

- `large-doc.pdf` - Sample PDF document for testing
- `large-doc.docx` - Sample DOCX document for testing
- `example.pdf` - Another sample PDF for testing
- `example-large.pdf` - A larger sample PDF for testing
- `example.pptx` - Sample PPTX document for testing

## Results

The test results are stored in the `test_results/` directory:

- `standard_time.txt` - Performance metrics for standard processing
- `streaming_time.txt` - Performance metrics for streaming processing
- `optimized_time.txt` - Performance metrics for optimized processing
- `standard_result.json` - Detailed results from standard processing
- `streaming_result.json` - Detailed results from streaming processing
- `optimized_result.json` - Detailed results from optimized processing
