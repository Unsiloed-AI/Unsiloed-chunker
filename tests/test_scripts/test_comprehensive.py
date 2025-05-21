"""
Comprehensive test script to verify all optimizations.

This script tests all major optimizations together to ensure they work correctly:
1. PyMuPDF integration
2. Memory mapping for large files
3. Document caching with TTL/LRU strategies
4. Parallel processing
5. Optimized chunking strategies
"""
import os
import sys
import time
import json
import logging
import argparse
import psutil
import tracemalloc
from typing import Dict, Any, List, Tuple

# Add the parent directory to sys.path to import the Unsiloed package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Unsiloed.services.chunking import process_document_chunking_streaming
from Unsiloed.utils.document_cache import document_cache
from Unsiloed.utils.pdf_extraction import extract_text_pymupdf, extract_text_with_mmap
from Unsiloed.utils.parallel_extraction import extract_text_with_pymupdf_parallel
from Unsiloed.utils.optimized_chunking import paragraph_chunking_optimized

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Memory snapshots for memory profiling
memory_snapshots = {}

def take_memory_snapshot(label: str):
    """Take a snapshot of current memory usage."""
    tracemalloc.start()
    snapshot = tracemalloc.take_snapshot()
    memory_snapshots[label] = snapshot
    
    # Also record current process memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage ({label}): {memory_info.rss / 1024 / 1024:.2f} MB")
    return memory_info.rss

def compare_memory_snapshots(before_label: str, after_label: str):
    """Compare two memory snapshots to find leaks or excessive usage."""
    if before_label not in memory_snapshots or after_label not in memory_snapshots:
        logger.error(f"Missing memory snapshot for comparison: {before_label} or {after_label}")
        return
        
    snapshot1 = memory_snapshots[before_label]
    snapshot2 = memory_snapshots[after_label]
    
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    logger.info(f"\nMemory comparison: {before_label} -> {after_label}")
    logger.info("Top 10 memory differences:")
    
    for stat in top_stats[:10]:
        logger.info(f"{stat}")

def time_execution(func, *args, **kwargs):
    """Measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    return result, execution_time

def test_pdf_extraction_methods(pdf_path: str) -> Dict[str, float]:
    """Test different PDF extraction methods and return execution times."""
    results = {}
    
    # 1. Test PyMuPDF extraction
    logger.info("Testing PyMuPDF extraction...")
    take_memory_snapshot("before_pymupdf")
    start_time = time.time()
    try:
        text_pymupdf = extract_text_pymupdf(pdf_path)
        results["pymupdf"] = time.time() - start_time
        logger.info(f"PyMuPDF extraction: {results['pymupdf']:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in PyMuPDF extraction: {str(e)}")
        results["pymupdf"] = -1
    take_memory_snapshot("after_pymupdf")
    
    # 2. Test memory-mapped extraction
    logger.info("Testing memory-mapped extraction...")
    take_memory_snapshot("before_mmap")
    start_time = time.time()
    try:
        text_mmap = extract_text_with_mmap(pdf_path)
        results["mmap"] = time.time() - start_time
        logger.info(f"Memory-mapped extraction: {results['mmap']:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in memory-mapped extraction: {str(e)}")
        results["mmap"] = -1
    take_memory_snapshot("after_mmap")
    
    # 3. Test parallel PyMuPDF extraction
    logger.info("Testing parallel PyMuPDF extraction...")
    take_memory_snapshot("before_parallel")
    start_time = time.time()
    try:
        text_parallel = extract_text_with_pymupdf_parallel(pdf_path)
        results["parallel"] = time.time() - start_time
        logger.info(f"Parallel PyMuPDF extraction: {results['parallel']:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in parallel PyMuPDF extraction: {str(e)}")
        results["parallel"] = -1
    take_memory_snapshot("after_parallel")
    
    # Compare memory usage
    compare_memory_snapshots("before_pymupdf", "after_pymupdf")
    compare_memory_snapshots("before_mmap", "after_mmap")
    compare_memory_snapshots("before_parallel", "after_parallel")
    
    return results

def test_document_caching(file_path: str, file_type: str) -> Dict[str, Any]:
    """Test document caching with multiple strategies."""
    strategies = ["paragraph", "fixed", "heading"]
    results = {}
    
    # Configure document cache
    document_cache.configure(ttl_seconds=3600, ttl_max_size=20, lru_max_size=10)
    document_cache.clear()
    
    for strategy in strategies:
        strategy_results = []
        
        logger.info(f"Testing {strategy} strategy with caching...")
        
        # First run (uncached)
        take_memory_snapshot(f"before_{strategy}_uncached")
        _, first_time = time_execution(
            process_document_chunking_streaming,
            file_path,
            file_type,
            strategy
        )
        take_memory_snapshot(f"after_{strategy}_uncached")
        strategy_results.append(first_time)
        
        # Second run (should be cached)
        take_memory_snapshot(f"before_{strategy}_cached")
        _, second_time = time_execution(
            process_document_chunking_streaming,
            file_path,
            file_type,
            strategy
        )
        take_memory_snapshot(f"after_{strategy}_cached")
        strategy_results.append(second_time)
        
        # Calculate improvement
        if first_time > 0:
            improvement = ((first_time - second_time) / first_time) * 100
            logger.info(f"{strategy}: {improvement:.2f}% faster with cache")
            logger.info(f"  First run: {first_time:.4f}s, Second run: {second_time:.4f}s")
        
        results[strategy] = strategy_results
        
        # Compare memory usage
        compare_memory_snapshots(f"before_{strategy}_uncached", f"after_{strategy}_uncached")
        compare_memory_snapshots(f"before_{strategy}_cached", f"after_{strategy}_cached")
    
    return results

def test_chunking_methods(text: str) -> Dict[str, float]:
    """Test different chunking methods and return execution times."""
    results = {}
    
    # Test optimized paragraph chunking
    logger.info("Testing optimized paragraph chunking...")
    start_time = time.time()
    try:
        chunks = paragraph_chunking_optimized(text)
        results["optimized_paragraph"] = time.time() - start_time
        logger.info(f"Optimized paragraph chunking: {results['optimized_paragraph']:.2f} seconds")
        logger.info(f"Generated {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Error in optimized paragraph chunking: {str(e)}")
        results["optimized_paragraph"] = -1
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test all optimizations together')
    parser.add_argument('--pdf', default='tests/test_data/example.pdf', help='Path to test PDF file')
    parser.add_argument('--large-pdf', default='tests/test_data/large-doc.pdf', help='Path to large PDF file')
    parser.add_argument('--docx', default='tests/test_data/large-doc.docx', help='Path to test DOCX file')
    parser.add_argument('--pptx', default='tests/test_data/example.pptx', help='Path to test PPTX file')
    parser.add_argument('--output', default='tests/test_results/comprehensive_test.json', help='Output JSON file')
    args = parser.parse_args()
    
    all_results = {}
    
    # 1. Test PDF extraction methods with large PDF
    if os.path.exists(args.large_pdf):
        logger.info(f"Testing PDF extraction methods with large PDF: {args.large_pdf}")
        all_results["large_pdf_extraction"] = test_pdf_extraction_methods(args.large_pdf)
    else:
        logger.warning(f"Large PDF file not found: {args.large_pdf}")
    
    # 2. Test document caching with different file types
    all_results["caching"] = {}
    
    if os.path.exists(args.pdf):
        logger.info(f"Testing document caching with PDF: {args.pdf}")
        all_results["caching"]["pdf"] = test_document_caching(args.pdf, "pdf")
    else:
        logger.warning(f"PDF file not found: {args.pdf}")
    
    if os.path.exists(args.docx):
        logger.info(f"Testing document caching with DOCX: {args.docx}")
        all_results["caching"]["docx"] = test_document_caching(args.docx, "docx")
    else:
        logger.warning(f"DOCX file not found: {args.docx}")
    
    if os.path.exists(args.pptx):
        logger.info(f"Testing document caching with PPTX: {args.pptx}")
        all_results["caching"]["pptx"] = test_document_caching(args.pptx, "pptx")
    else:
        logger.warning(f"PPTX file not found: {args.pptx}")
    
    # 3. Test chunking with text from PDF
    if os.path.exists(args.pdf):
        logger.info(f"Testing chunking methods with text from PDF: {args.pdf}")
        try:
            pdf_text = extract_text_pymupdf(args.pdf)
            all_results["chunking"] = test_chunking_methods(pdf_text)
        except Exception as e:
            logger.error(f"Error extracting text from PDF for chunking tests: {str(e)}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n===== COMPREHENSIVE TEST SUMMARY =====")
    
    # PDF extraction summary
    if "large_pdf_extraction" in all_results:
        print("\nLarge PDF Extraction Methods:")
        methods = all_results["large_pdf_extraction"]
        for method, time_taken in methods.items():
            if time_taken > 0:
                print(f"  {method}: {time_taken:.2f} seconds")
    
    # Caching summary
    if "caching" in all_results:
        print("\nDocument Cache Performance:")
        for file_type, strategies in all_results["caching"].items():
            print(f"\n  {file_type.upper()}:")
            for strategy, times in strategies.items():
                if len(times) >= 2 and times[0] > 0:
                    improvement = ((times[0] - times[1]) / times[0]) * 100
                    print(f"    {strategy}: {improvement:.2f}% improvement")
                    print(f"      First: {times[0]:.4f}s, Cached: {times[1]:.4f}s")
    
    # Chunking summary
    if "chunking" in all_results:
        print("\nChunking Methods:")
        for method, time_taken in all_results["chunking"].items():
            if time_taken > 0:
                print(f"  {method}: {time_taken:.2f} seconds")
    
    print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main()
