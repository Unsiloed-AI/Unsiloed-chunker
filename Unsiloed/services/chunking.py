from Unsiloed.utils.chunking import (
    fixed_size_chunking,
    page_based_chunking,
    paragraph_chunking,
    heading_chunking,
    semantic_chunking,
    check_memory_usage,
    log_memory_usage,
    adjust_batch_size,
    get_memory_threshold,
    INITIAL_BATCH_SIZE,
)
from Unsiloed.utils.openai import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_pptx,
)

import logging
import concurrent.futures
from typing import Dict, Any, List, Optional
import os
import hashlib
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Cache for storing processed documents with LRU eviction
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Dict[str, Any]) -> None:
        if key in self.cache:
            # Remove existing key
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)
        self.cache[key] = value

# Initialize cache with 100 document capacity
_document_cache = LRUCache(100)

def process_document_chunking(
    file_path: str,
    file_type: str,
    strategy: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    batch_size: Optional[int] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Process a document file (PDF, DOCX, PPTX) with the specified chunking strategy.
    Optimized for performance with caching, parallel processing, and batch support.

    Args:
        file_path: Path to the document file
        file_type: Type of document (pdf, docx, pptx)
        strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy
        overlap: Overlap size for fixed strategy
        batch_size: Optional batch size for processing large documents
        progress_callback: Optional callback function to track progress

    Returns:
        Dictionary with chunking results
    """
    logger.info(f"Processing {file_type.upper()} document with {strategy} chunking strategy")
    log_memory_usage("start")

    # Generate cache key using file hash for better cache hits
    cache_key = _generate_cache_key(file_path, strategy, chunk_size, overlap)
    cached_result = _document_cache.get(cache_key)
    if cached_result:
        logger.info("Retrieved result from cache")
        if progress_callback:
            progress_callback(100)  # 100% complete from cache
        return cached_result

    try:
        # Handle page-based chunking for PDFs only
        if strategy == "page" and file_type == "pdf":
            chunks = page_based_chunking(file_path, progress_callback)
        else:
            # Extract text based on file type with optimized extraction
            if progress_callback:
                progress_callback(10)  # 10% complete - starting extraction
            text = extract_document_text(file_path, file_type)
            
            if progress_callback:
                progress_callback(30)  # 30% complete - extraction done

            # Check memory usage before processing
            if not check_memory_usage():
                logger.warning("Memory usage high before processing, adjusting batch size")
                batch_size = INITIAL_BATCH_SIZE // 2

            # Apply batch processing for large documents
            if batch_size is None:
                batch_size = min(INITIAL_BATCH_SIZE, len(text))
            
            if len(text) > batch_size:
                chunks = process_in_batches(text, strategy, chunk_size, overlap, batch_size, progress_callback)
            else:
                # Apply the selected chunking strategy with optimized parameters
                chunks = apply_chunking_strategy(text, strategy, chunk_size, overlap)
                if progress_callback:
                    progress_callback(90)  # 90% complete - chunking done

        # Calculate statistics efficiently
        total_chunks = len(chunks)
        avg_chunk_size = sum(len(chunk["text"]) for chunk in chunks) / total_chunks if total_chunks > 0 else 0

        result = {
            "file_type": file_type,
            "strategy": strategy,
            "total_chunks": total_chunks,
            "avg_chunk_size": avg_chunk_size,
            "chunks": chunks,
        }

        # Cache the result
        _document_cache.put(cache_key, result)

        if progress_callback:
            progress_callback(100)  # 100% complete

        log_memory_usage("end")
        return result

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

def _generate_cache_key(file_path: str, strategy: str, chunk_size: int, overlap: int) -> str:
    """
    Generate a cache key using file hash and parameters.
    
    Args:
        file_path: Path to the document
        strategy: Chunking strategy
        chunk_size: Chunk size
        overlap: Overlap size
        
    Returns:
        Cache key string
    """
    # Generate file hash
    file_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            file_hash.update(chunk)
    
    # Combine parameters into cache key
    return f"{file_hash.hexdigest()}:{strategy}:{chunk_size}:{overlap}"

def process_in_batches(
    text: str,
    strategy: str,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    progress_callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """
    Process large text in batches for better memory management.
    
    Args:
        text: Text to process
        strategy: Chunking strategy
        chunk_size: Chunk size
        overlap: Overlap size
        batch_size: Size of each batch
        progress_callback: Optional callback function to track progress
        
    Returns:
        List of chunks
    """
    chunks = []
    text_length = len(text)
    start = 0
    total_batches = (text_length + batch_size - 1) // batch_size
    current_batch = 0
    
    while start < text_length:
        # Calculate batch end with overlap
        end = min(start + batch_size + overlap, text_length)
        batch_text = text[start:end]
        
        # Process batch
        batch_chunks = apply_chunking_strategy(batch_text, strategy, chunk_size, overlap)
        
        # Adjust positions for global text
        for chunk in batch_chunks:
            chunk["metadata"]["start_char"] += start
            chunk["metadata"]["end_char"] += start
        
        chunks.extend(batch_chunks)
        start = end - overlap if end < text_length else text_length
        
        # Update progress
        current_batch += 1
        if progress_callback:
            progress = 30 + (current_batch / total_batches * 60)  # 30-90% range
            progress_callback(int(progress))
    
    return chunks

def extract_document_text(file_path: str, file_type: str) -> str:
    """
    Extract text from document with optimized extraction method.
    
    Args:
        file_path: Path to the document
        file_type: Type of document
        
    Returns:
        Extracted text
    """
    if file_type == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_type == "docx":
        return extract_text_from_docx(file_path)
    elif file_type == "pptx":
        return extract_text_from_pptx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def apply_chunking_strategy(
    text: str,
    strategy: str,
    chunk_size: int,
    overlap: int
) -> List[Dict[str, Any]]:
    """
    Apply the specified chunking strategy with optimized parameters.
    
    Args:
        text: Text to chunk
        strategy: Chunking strategy
        chunk_size: Size of chunks
        overlap: Overlap size
        
    Returns:
        List of chunks
    """
    if strategy == "fixed":
        return fixed_size_chunking(text, chunk_size, overlap)
    elif strategy == "semantic":
        return semantic_chunking(text)
    elif strategy == "paragraph":
        return paragraph_chunking(text)
    elif strategy == "heading":
        return heading_chunking(text)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
