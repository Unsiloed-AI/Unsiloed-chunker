"""
Test document cache performance with different file types.

This script measures the performance improvement from the document cache
by processing the same documents multiple times and comparing execution times.
"""
import os
import sys
import time
import json
import logging
from typing import Dict, Any, List
import argparse

# Add the parent directory to sys.path to import the Unsiloed package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Unsiloed.services.chunking import process_document_chunking_streaming
from Unsiloed.utils.document_cache import document_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
STRATEGIES = ["semantic", "paragraph", "fixed", "heading"]
ITERATIONS = 3  # Number of times to process each document


def time_execution(func, *args, **kwargs):
    """Measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    return result, execution_time


def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def test_document(file_path: str, file_type: str, skip_strategies=None) -> Dict[str, Any]:
    """
    Test document processing with caching.
    
    Args:
        file_path: Path to the document file
        file_type: Type of the document (pdf, docx, pptx)
        skip_strategies: List of strategies to skip (e.g., if API quota is exceeded)
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing {file_type.upper()} document: {os.path.basename(file_path)}")
    
    if skip_strategies is None:
        skip_strategies = []
    
    results = {}
    max_execution_time = 60  # Maximum 60 seconds per iteration
    
    # Test each chunking strategy
    for strategy in STRATEGIES:
        if strategy in skip_strategies:
            logger.info(f"Skipping strategy: {strategy} (as requested)")
            continue
            
        logger.info(f"Testing strategy: {strategy}")
        
        strategy_results = {
            "iterations": [],
            "cache_stats": {}
        }
        
        # Clear the cache before testing this strategy
        document_cache.clear()
        
        # Process document multiple times
        for i in range(ITERATIONS):
            logger.info(f"Iteration {i+1}/{ITERATIONS}")
            
            try:
                # Time the execution with a timeout
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Processing timed out after {max_execution_time} seconds")
                
                # Set the timeout
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(max_execution_time)
                
                try:
                    # Time the execution
                    _, execution_time = time_execution(
                        process_document_chunking_streaming,
                        file_path,
                        file_type,
                        strategy
                    )
                    
                    # Cancel the timeout
                    signal.alarm(0)
                finally:
                    # Restore the old signal handler
                    signal.signal(signal.SIGALRM, old_handler)
            
            except TimeoutError as e:
                logger.warning(f"Timeout: {str(e)}")
                execution_time = max_execution_time
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                execution_time = -1  # Mark as error
            
            # Record results
            strategy_results["iterations"].append({
                "iteration": i + 1,
                "time": execution_time,
                "time_formatted": format_time(execution_time)
            })
            
            # Get cache stats after this iteration
            cache_stats = document_cache.get_stats()
            strategy_results["cache_stats"][f"iteration_{i+1}"] = cache_stats
            
        # Calculate improvement percentage between first and second iteration
        if len(strategy_results["iterations"]) >= 2:
            first_time = strategy_results["iterations"][0]["time"]
            second_time = strategy_results["iterations"][1]["time"]
            
            if first_time > 0:  # Avoid division by zero
                improvement = ((first_time - second_time) / first_time) * 100
                strategy_results["cache_improvement_percentage"] = improvement
                
        results[strategy] = strategy_results
    
    return results


def main():
    """Run document cache performance tests."""
    parser = argparse.ArgumentParser(description='Test document cache performance')
    parser.add_argument('--pdf', default='tests/test_data/example.pdf', help='Path to test PDF file')
    parser.add_argument('--docx', default='tests/test_data/large-doc.docx', help='Path to test DOCX file')
    parser.add_argument('--pptx', default='tests/test_data/example.pptx', help='Path to test PPTX file')
    parser.add_argument('--output', default='tests/test_results/cache_performance.json', help='Output JSON file')
    parser.add_argument('--skip-semantic', action='store_true', help='Skip semantic chunking (e.g., if OpenAI API is unavailable)')
    args = parser.parse_args()
    
    # Configure document cache for testing
    document_cache.configure(ttl_seconds=3600, ttl_max_size=20, lru_max_size=10)
    
    # Check if files exist
    files_to_test = []
    if os.path.exists(args.pdf):
        files_to_test.append((args.pdf, "pdf"))
    if os.path.exists(args.docx):
        files_to_test.append((args.docx, "docx"))
    if os.path.exists(args.pptx):
        files_to_test.append((args.pptx, "pptx"))
    
    if not files_to_test:
        logger.error("No test files found")
        return
    
    # Determine strategies to skip
    skip_strategies = []
    if args.skip_semantic:
        skip_strategies.append("semantic")
        logger.info("Skipping semantic chunking as requested")
    
    # Run tests
    all_results = {}
    for file_path, file_type in files_to_test:
        try:
            results = test_document(file_path, file_type, skip_strategies=skip_strategies)
            all_results[file_type] = {
                "file": os.path.basename(file_path),
                "results": results
            }
        except Exception as e:
            logger.error(f"Error testing {file_type} document {file_path}: {str(e)}")
            all_results[file_type] = {
                "file": os.path.basename(file_path),
                "error": str(e)
            }
    
    # Print summary
    print("\n===== DOCUMENT CACHE PERFORMANCE SUMMARY =====")
    for file_type, data in all_results.items():
        print(f"\n{file_type.upper()}: {data['file']}")
        for strategy, results in data["results"].items():
            if "cache_improvement_percentage" in results:
                improvement = results["cache_improvement_percentage"]
                print(f"  {strategy}: {improvement:.2f}% improvement with cache")
                
                first_time = results["iterations"][0]["time_formatted"]
                second_time = results["iterations"][1]["time_formatted"]
                print(f"    First run: {first_time}, Second run: {second_time}")
    
    # Save results to JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
