"""
Comprehensive performance test for document processing optimizations.

This script tests all optimizations together:
1. Document caching
2. Memory-optimized PDF extraction
3. Optimized DOCX processing
4. Enhanced chunking strategies
5. Parallel processing
"""
import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, Any, List, Optional
import concurrent.futures
import gc
import tracemalloc

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Unsiloed.utils.document_cache import document_cache
from Unsiloed.utils.memory_profiling import MemoryProfiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define test configurations
ITERATIONS = 3
STRATEGIES = ["semantic", "paragraph", "fixed", "heading"]
TEST_TYPES = ["standard", "streaming", "enhanced", "cached"]
AVAILABLE_FILE_TYPES = ["pdf", "docx", "pptx"]

# Test file paths (change these to your actual test files)
TEST_FILES = {
    "pdf": {
        "small": "tests/test_data/example.pdf",
        "large": "tests/test_data/example-large.pdf"  # Larger PDF for memory testing
    },
    "docx": {
        "small": "tests/test_data/example.docx",
        "large": "tests/test_data/large-doc.docx"
    },
    "pptx": {
        "small": "tests/test_data/example.pptx",
        "large": "tests/test_data/example-large.pptx"  # May need to be created
    }
}

def time_execution(func, *args, **kwargs):
    """Measure execution time and memory usage of a function."""
    # Track memory before and after
    tracemalloc.start()
    start_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
    
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    current, peak = tracemalloc.get_traced_memory()
    current_mb = current / 1024 / 1024  # MB
    peak_mb = peak / 1024 / 1024  # MB
    memory_increase = current_mb - start_memory
    
    tracemalloc.stop()
    
    return result, execution_time, {
        "peak_memory_mb": peak_mb,
        "memory_increase_mb": memory_increase
    }

def format_time(seconds):
    """Format time in seconds to a readable string."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"

def compare_results(result1, result2, tolerance=0.8):
    """
    Compare two chunking results to ensure they're similar.
    Returns similarity score (0-1) and whether they're considered similar.
    """
    # Check if both results contain chunks
    if ("chunks" not in result1 or "chunks" not in result2 or
            not result1["chunks"] or not result2["chunks"]):
        return 0, False
    
    # Compare chunk counts first
    chunk_count1 = len(result1["chunks"])
    chunk_count2 = len(result2["chunks"])
    count_ratio = min(chunk_count1, chunk_count2) / max(chunk_count1, chunk_count2)
    
    # If chunk counts are too different, results are not similar
    if count_ratio < tolerance:
        return count_ratio, False
    
    # Compare total text content length
    text1 = " ".join(chunk["text"] for chunk in result1["chunks"])
    text2 = " ".join(chunk["text"] for chunk in result2["chunks"])
    
    len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
    
    # Results are similar if text lengths are within tolerance
    return len_ratio, len_ratio >= tolerance

def clear_cache():
    """Clear document cache to ensure fresh tests."""
    cache = document_cache
    cache.clear()
    logger.info("Document cache cleared")

def test_document_processing(
    file_path: str, 
    file_type: str,
    strategies: Optional[List[str]] = None,
    test_types: Optional[List[str]] = None,
    iterations: int = ITERATIONS
) -> Dict[str, Any]:
    """
    Test document processing with different strategies and processing types.
    
    Args:
        file_path: Path to the test document
        file_type: File type (pdf, docx, pptx)
        strategies: Chunking strategies to test
        test_types: Processing types to test (standard, streaming, enhanced, cached)
        iterations: Number of iterations for each test
        
    Returns:
        Dictionary with test results
    """
    if not os.path.exists(file_path):
        logger.error(f"Test file not found: {file_path}")
        return {"error": f"File not found: {file_path}"}
    
    strategies = strategies or STRATEGIES
    test_types = test_types or TEST_TYPES
    
    results = {}
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    logger.info(f"Testing {file_path} ({file_size_mb:.2f} MB) with {len(strategies)} strategies")
    
    from Unsiloed.services.chunking import (
        process_document_chunking,
        process_document_chunking_streaming
    )
    
    # Try to import enhanced processing if available
    try:
        from Unsiloed.utils.enhanced_chunking import (
            process_document_chunking_enhanced,
            process_document_chunking_auto_optimize
        )
        HAS_ENHANCED = True
    except ImportError:
        logger.warning("Enhanced chunking not available")
        HAS_ENHANCED = False
        # Remove enhanced from test types if not available
        if "enhanced" in test_types:
            test_types.remove("enhanced")
    
    # Test each strategy
    for strategy in strategies:
        strategy_results = {}
        reference_result = None
        
        # First, get standard result as reference
        logger.info(f"Getting reference result with {strategy} strategy")
        try:
            reference_result, _, _ = time_execution(
                process_document_chunking,
                file_path, file_type, strategy
            )
        except Exception as e:
            logger.error(f"Error getting reference result: {str(e)}")
            reference_result = None
        
        # Test each processing type
        for test_type in test_types:
            logger.info(f"Testing {test_type} processing with {strategy} strategy")
            
            # Skip unavailable types
            if test_type == "enhanced" and not HAS_ENHANCED:
                continue
                
            type_timings = []
            memory_stats = []
            
            # Warm up (important for fair comparison)
            if test_type == "standard":
                func = process_document_chunking
            elif test_type == "streaming":
                func = process_document_chunking_streaming
            elif test_type == "enhanced" and HAS_ENHANCED:
                func = process_document_chunking_enhanced
            elif test_type == "cached" and HAS_ENHANCED:
                # First run to populate cache
                process_document_chunking_auto_optimize(file_path, file_type, strategy)
                func = process_document_chunking_auto_optimize
            else:
                continue
                
            # Force garbage collection before tests
            gc.collect()
            
            # Run multiple iterations
            for i in range(iterations):
                # Clear cache between non-cached runs to ensure fair testing
                if test_type != "cached":
                    clear_cache()
                
                try:
                    result, time_taken, memory_usage = time_execution(
                        func, file_path, file_type, strategy
                    )
                    
                    # Store timing and memory stats
                    type_timings.append(time_taken)
                    memory_stats.append(memory_usage)
                    
                    # Compare with reference result for validation
                    if reference_result:
                        similarity, is_similar = compare_results(reference_result, result)
                        logger.info(f"Iteration {i+1}: {format_time(time_taken)}, "
                                  f"similarity: {similarity:.3f}, similar: {is_similar}")
                    else:
                        logger.info(f"Iteration {i+1}: {format_time(time_taken)}")
                        
                except Exception as e:
                    logger.error(f"Error testing {test_type} processing: {str(e)}")
                    
            # Calculate statistics
            if type_timings:
                avg_time = sum(type_timings) / len(type_timings)
                min_time = min(type_timings)
                max_time = max(type_timings)
                
                # Calculate memory statistics
                avg_peak_memory = sum(s["peak_memory_mb"] for s in memory_stats) / len(memory_stats)
                max_peak_memory = max(s["peak_memory_mb"] for s in memory_stats)
                
                strategy_results[test_type] = {
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "formatted_avg": format_time(avg_time),
                    "memory": {
                        "avg_peak_mb": avg_peak_memory,
                        "max_peak_mb": max_peak_memory
                    }
                }
                
        results[strategy] = strategy_results
    
    # Add document metadata
    results["metadata"] = {
        "file_path": file_path,
        "file_size_mb": file_size_mb,
        "file_type": file_type
    }
    
    return results

def run_all_tests(
    file_types: List[str] = None,
    strategies: List[str] = None,
    test_types: List[str] = None,
    iterations: int = ITERATIONS,
    size: str = "both"
):
    """
    Run tests for all specified file types.
    """
    file_types = file_types or AVAILABLE_FILE_TYPES
    results = {}
    
    # Run tests for each file type and size
    for file_type in file_types:
        file_type_results = {}
        
        if file_type not in TEST_FILES:
            logger.warning(f"No test files configured for {file_type}")
            continue
            
        # Test small files if requested
        if size in ["both", "small"] and "small" in TEST_FILES[file_type]:
            file_path = TEST_FILES[file_type]["small"]
            if os.path.exists(file_path):
                logger.info(f"Testing small {file_type}: {file_path}")
                file_type_results["small"] = test_document_processing(
                    file_path, file_type, strategies, test_types, iterations
                )
            else:
                logger.warning(f"Small {file_type} test file not found: {file_path}")
                
        # Test large files if requested
        if size in ["both", "large"] and "large" in TEST_FILES[file_type]:
            file_path = TEST_FILES[file_type]["large"]
            if os.path.exists(file_path):
                logger.info(f"Testing large {file_type}: {file_path}")
                file_type_results["large"] = test_document_processing(
                    file_path, file_type, strategies, test_types, iterations
                )
            else:
                logger.warning(f"Large {file_type} test file not found: {file_path}")
                
        results[file_type] = file_type_results
    
    # Save results to JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"tests/test_results/comprehensive_test_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # Generate performance summary
    generate_summary(results, timestamp)
    
    return results

def generate_summary(results, timestamp):
    """Generate a summary of test results."""
    summary = {
        "timestamp": timestamp,
        "performance_improvements": {}
    }
    
    for file_type, size_results in results.items():
        file_type_summary = {}
        
        for size, test_results in size_results.items():
            if "metadata" not in test_results or "error" in test_results:
                continue
                
            file_size = test_results["metadata"]["file_size_mb"]
            strategy_improvements = {}
            
            for strategy, strategy_result in test_results.items():
                if strategy == "metadata":
                    continue
                    
                # Calculate improvements
                improvements = {}
                if "standard" in strategy_result:
                    standard_time = strategy_result["standard"]["avg_time"]
                    standard_memory = strategy_result["standard"]["memory"]["avg_peak_mb"]
                    
                    for test_type in ["streaming", "enhanced", "cached"]:
                        if test_type in strategy_result:
                            test_time = strategy_result[test_type]["avg_time"]
                            test_memory = strategy_result[test_type]["memory"]["avg_peak_mb"]
                            
                            time_improvement = (standard_time - test_time) / standard_time * 100
                            memory_improvement = (standard_memory - test_memory) / standard_memory * 100
                            
                            improvements[test_type] = {
                                "time_pct_improved": time_improvement,
                                "memory_pct_improved": memory_improvement,
                                "time_factor": standard_time / test_time if test_time > 0 else float("inf"),
                                "formatted": f"{time_improvement:.1f}% faster, "
                                           f"{memory_improvement:.1f}% less memory"
                            }
                
                strategy_improvements[strategy] = improvements
            
            file_type_summary[size] = {
                "file_size_mb": file_size,
                "improvements": strategy_improvements
            }
        
        summary["performance_improvements"][file_type] = file_type_summary
    
    # Save summary to JSON
    output_file = f"tests/test_results/summary_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print key findings
    print("\n===== PERFORMANCE SUMMARY =====")
    for file_type, size_results in summary["performance_improvements"].items():
        print(f"\n{file_type.upper()} RESULTS:")
        
        for size, size_data in size_results.items():
            print(f"  {size} file ({size_data['file_size_mb']:.2f} MB):")
            
            for strategy, improvements in size_data["improvements"].items():
                print(f"    {strategy} strategy:")
                
                for test_type, data in improvements.items():
                    if data["time_pct_improved"] > 0:
                        speed_color = "\033[92m"  # Green for improvement
                    else:
                        speed_color = "\033[91m"  # Red for degradation
                        
                    if data["memory_pct_improved"] > 0:
                        memory_color = "\033[92m"  # Green for improvement
                    else:
                        memory_color = "\033[91m"  # Red for degradation
                        
                    reset = "\033[0m"
                    
                    print(f"      {test_type}: {speed_color}{data['time_pct_improved']:.1f}%{reset} faster, "
                          f"{memory_color}{data['memory_pct_improved']:.1f}%{reset} less memory "
                          f"(time factor: {data['time_factor']:.2f}x)")
                    
    print("\n================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test document processing optimizations")
    parser.add_argument("--file-types", nargs="+", choices=AVAILABLE_FILE_TYPES,
                        help="File types to test")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                        help="Chunking strategies to test")
    parser.add_argument("--test-types", nargs="+", choices=TEST_TYPES,
                        help="Processing types to test")
    parser.add_argument("--iterations", type=int, default=ITERATIONS,
                        help="Number of iterations for each test")
    parser.add_argument("--size", choices=["small", "large", "both"], default="both",
                        help="Size of test files to use")
    
    args = parser.parse_args()
    
    run_all_tests(
        file_types=args.file_types,
        strategies=args.strategies,
        test_types=args.test_types,
        iterations=args.iterations,
        size=args.size
    )
