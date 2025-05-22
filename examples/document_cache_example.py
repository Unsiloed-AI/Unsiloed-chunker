"""
Example demonstrating usage of the document cache in custom applications.

This script shows how to leverage the document cache API
for custom document processing applications.
"""
import os
import sys
import time
import logging
from typing import Dict, Any

# Add the parent directory to sys.path to import the Unsiloed package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Unsiloed.utils.document_cache import document_cache
from Unsiloed.utils.extraction import extract_text_streaming_pdf
from Unsiloed.utils.chunking import paragraph_chunking

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_document(file_path: str, strategy: str = "paragraph") -> Dict[str, Any]:
    """
    Process a document with caching.
    
    Args:
        file_path: Path to the document file
        strategy: Processing strategy
        
    Returns:
        Processing result
    """
    # First check if the result is in cache
    cached_result = document_cache.get(file_path, strategy)
    if cached_result:
        logger.info(f"Using cached result for {file_path} with strategy {strategy}")
        return cached_result
        
    logger.info(f"Processing document {file_path} with strategy {strategy}")
    
    # Process the document
    start_time = time.time()
    
    # Extract text (this example works with PDF files)
    all_text = []
    for segment in extract_text_streaming_pdf(file_path):
        if segment.get("text", "").strip():
            all_text.append(segment["text"])
    
    # Combine all text segments
    combined_text = "\n\n".join(all_text)
    
    # Apply chunking strategy
    chunks = paragraph_chunking(combined_text)
    
    # Create result
    result = {
        "file_path": file_path,
        "strategy": strategy,
        "chunks": chunks,
        "processing_time": time.time() - start_time,
        "timestamp": time.time()
    }
    
    # Cache the result
    document_cache.set(file_path, strategy, result)
    logger.info(f"Cached result for {file_path} with strategy {strategy}")
    
    return result

def main():
    # Configure the document cache
    document_cache.configure(
        ttl_seconds=3600,  # Cache entries expire after 1 hour
        ttl_max_size=100,  # Maximum 100 entries in TTL cache
        lru_max_size=50    # Maximum 50 entries in LRU cache
    )
    
    # Process a PDF file multiple times to demonstrate caching
    pdf_file = "tests/test_data/example.pdf"
    if not os.path.exists(pdf_file):
        logger.error(f"File not found: {pdf_file}")
        return
    
    # First run (not cached)
    logger.info("First run - processing document...")
    start_time = time.time()
    result1 = process_document(pdf_file, "paragraph")
    logger.info(f"First run completed in {time.time() - start_time:.2f} seconds")
    
    # Get cache stats
    stats = document_cache.get_stats()
    logger.info(f"Cache stats after first run: {stats}")
    
    # Second run (should be cached)
    logger.info("Second run - should use cached result...")
    start_time = time.time()
    result2 = process_document(pdf_file, "paragraph")
    logger.info(f"Second run completed in {time.time() - start_time:.2f} seconds")
    
    # Compare results to verify cache is working
    chunks_match = len(result1["chunks"]) == len(result2["chunks"])
    logger.info(f"Results match: {chunks_match}")
    
    # Process with a different strategy (should not be cached)
    logger.info("Processing with different strategy...")
    start_time = time.time()
    result3 = process_document(pdf_file, "heading")
    logger.info(f"Different strategy run completed in {time.time() - start_time:.2f} seconds")
    
    # Get final cache stats
    stats = document_cache.get_stats()
    logger.info(f"Final cache stats: {stats}")
    
    # Example of removing items from cache
    document_cache.remove(pdf_file, "paragraph")
    logger.info("Removed 'paragraph' strategy from cache")
    
    # Example of clearing the cache
    document_cache.clear()
    logger.info("Cache cleared")

if __name__ == "__main__":
    main()
