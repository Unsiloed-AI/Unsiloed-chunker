import os
import time
import statistics
import json
from dotenv import load_dotenv
import Unsiloed
from test_config import PPTX_TEST_FILE, TEST_RESULTS_DIR

# Load environment variables
load_dotenv()

# Number of times to run each test
NUM_RUNS = 1  # Set to 1 for consistency

def test_standard_processing():
    print("Testing standard processing for PPTX...")
    times = []
    total_chunks = 0
    result = None
    
    for i in range(NUM_RUNS):
        print(f"  Run {i+1}/{NUM_RUNS}")
        start_time = time.time()
        result = Unsiloed.process_sync({
            "filePath": PPTX_TEST_FILE,
            "credentials": {
                "apiKey": os.environ.get("OPENAI_API_KEY")
            },
            "strategy": "paragraph",  # Using paragraph to avoid OpenAI API costs
        })
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        total_chunks = result['total_chunks']
    
    avg_time = statistics.mean(times)
    print(f"Standard processing completed in {avg_time:.2f} seconds (average)")
    print(f"Total chunks: {total_chunks}")
    
    # Save results to file for PPTX standard processing
    with open(os.path.join(TEST_RESULTS_DIR, 'pptx_standard_time.txt'), 'w') as f:
        f.write(f"{avg_time}\n{total_chunks}")
    
    with open(os.path.join(TEST_RESULTS_DIR, 'pptx_standard_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
        
    return avg_time, total_chunks

def test_streaming_processing():
    print("\nTesting streaming processing for PPTX...")
    times = []
    total_chunks = 0
    result = None
    
    for i in range(NUM_RUNS):
        print(f"  Run {i+1}/{NUM_RUNS}")
        start_time = time.time()
        result = Unsiloed.process_streaming_sync({
            "filePath": PPTX_TEST_FILE,
            "credentials": {
                "apiKey": os.environ.get("OPENAI_API_KEY")
            },
            "strategy": "paragraph",  # Using paragraph to avoid OpenAI API costs
        })
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        total_chunks = result['total_chunks']
    
    avg_time = statistics.mean(times)
    print(f"Streaming processing completed in {avg_time:.2f} seconds (average)")
    print(f"Total chunks: {total_chunks}")
    
    # Save results to file for PPTX streaming processing
    with open(os.path.join(TEST_RESULTS_DIR, 'pptx_streaming_time.txt'), 'w') as f:
        f.write(f"{avg_time}\n{total_chunks}")
        
    with open(os.path.join(TEST_RESULTS_DIR, 'pptx_streaming_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
        
    return avg_time, total_chunks

if __name__ == "__main__":
    # Check if document exists
    if not os.path.exists(PPTX_TEST_FILE):
        print(f"Error: Test document {PPTX_TEST_FILE} not found.")
        print("Please ensure the PPTX file exists in the test_data directory.")
        exit(1)

    print(f"Running PPTX performance tests on {PPTX_TEST_FILE}")

    # Run tests - run standard first, then streaming
    standard_time, standard_chunks = test_standard_processing()
    streaming_time, streaming_chunks = test_streaming_processing()
    
    # Print comparison
    print("\n=== PPTX Performance Comparison ===")
    print(f"Standard processing: {standard_time:.2f} seconds")
    print(f"Streaming processing: {streaming_time:.2f} seconds")
    
    # Calculate improvements
    if standard_time > 0:
        improvement = (standard_time - streaming_time) / standard_time * 100
        print(f"Streaming improvement: {improvement:.1f}%")
        
    # Calculate chunk difference
    if standard_chunks > 0:
        chunk_diff = streaming_chunks - standard_chunks
        chunk_percent = (chunk_diff / standard_chunks) * 100
        print(f"Chunk difference: {chunk_diff} ({chunk_percent:.1f}%)")
