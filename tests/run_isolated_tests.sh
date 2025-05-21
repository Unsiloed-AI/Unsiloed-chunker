#!/bin/bash

# Run tests separately to avoid any interference between the methods

# Set up environment - make sure we're in the tests directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Create directories if they don't exist
mkdir -p test_data
mkdir -p test_results

# Check if the test documents exist
if [ ! -f "./test_data/large-doc.pdf" ]; then
  echo "Error: Test document test_data/large-doc.pdf not found"
  echo "Please add test files to the test_data directory"
  exit 1
fi

if [ ! -f "./test_data/large-doc.docx" ]; then
  echo "Error: Test document test_data/large-doc.docx not found"
  echo "Please add test files to the test_data directory"
  exit 1
fi

if [ ! -f "./test_data/example.pptx" ]; then
  echo "Error: Test document test_data/example.pptx not found"
  echo "Please add test files to the test_data directory"
  exit 1
fi

echo "Running isolated performance tests for each processing method and document type"
echo "==============================================================================="

# Clear any previous results
rm -f ./test_results/*.txt ./test_results/*.json

# Run PDF performance test in isolation
echo -e "\n=== Running PDF Performance Tests ==="
python ./test_scripts/test_performance_pdf.py
if [ $? -ne 0 ]; then
  echo "Error running PDF performance test"
else
  echo "PDF performance tests completed successfully"
fi

# Wait a moment to clear any resources
sleep 2

# Run DOCX performance test in isolation
echo -e "\n=== Running DOCX Performance Tests ==="
python ./test_scripts/test_optimized_docx.py
if [ $? -ne 0 ]; then
  echo "Error running DOCX performance test"
else
  echo "DOCX performance tests completed successfully"
fi

# Wait a moment to clear any resources
sleep 2

# Run PPTX performance test in isolation
echo -e "\n=== Running PPTX Performance Tests ==="
python ./test_scripts/test_performance_pptx.py
if [ $? -ne 0 ]; then
  echo "Error running PPTX performance test"
else
  echo "PPTX performance tests completed successfully"
fi

# Wait a moment before combining results
sleep 1

# Compare all results
echo -e "\n=== Comprehensive Performance Comparison ==="
python ./test_scripts/compare_results.py
