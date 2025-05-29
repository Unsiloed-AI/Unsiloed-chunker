import concurrent.futures
from typing import Literal, List, Dict, Any, Optional
import logging
import PyPDF2
import re
import psutil
import os
from Unsiloed.utils.openai import (
    semantic_chunk_with_structured_output,
)

logger = logging.getLogger(__name__)

# Memory management configuration
MEMORY_THRESHOLD_PERCENT = int(os.environ.get('MEMORY_THRESHOLD_PERCENT', 80))  # Default 80% of available memory
MIN_BATCH_SIZE = 1000  # Minimum batch size in characters
MAX_BATCH_SIZE = 100000  # Maximum batch size in characters
INITIAL_BATCH_SIZE = 50000  # Initial batch size in characters

def get_available_memory() -> int:
    """
    Get available system memory in bytes.
    
    Returns:
        int: Available memory in bytes
    """
    return psutil.virtual_memory().available

def get_memory_threshold() -> int:
    """
    Calculate dynamic memory threshold based on available system memory.
    
    Returns:
        int: Memory threshold in bytes
    """
    total_memory = psutil.virtual_memory().total
    return int(total_memory * (MEMORY_THRESHOLD_PERCENT / 100))

def check_memory_usage() -> bool:
    """
    Check if current memory usage is within acceptable limits.
    
    Returns:
        bool: True if memory usage is acceptable, False otherwise
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    threshold = get_memory_threshold()
    return memory_info.rss < threshold

def log_memory_usage(stage: str) -> None:
    """
    Log current memory usage for monitoring.
    
    Args:
        stage: Current processing stage
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()
    
    logger.debug(
        f"Memory usage at {stage}: "
        f"Process: {memory_info.rss / 1024 / 1024:.2f} MB, "
        f"System: {system_memory.percent}% used, "
        f"Available: {system_memory.available / 1024 / 1024:.2f} MB"
    )

def adjust_batch_size(current_batch_size: int, memory_usage_percent: float) -> int:
    """
    Dynamically adjust batch size based on memory usage.
    
    Args:
        current_batch_size: Current batch size
        memory_usage_percent: Current memory usage percentage
        
    Returns:
        int: Adjusted batch size
    """
    if memory_usage_percent > 90:
        return max(MIN_BATCH_SIZE, current_batch_size // 4)
    elif memory_usage_percent > 80:
        return max(MIN_BATCH_SIZE, current_batch_size // 2)
    elif memory_usage_percent < 50:
        return min(MAX_BATCH_SIZE, current_batch_size * 2)
    return current_batch_size

ChunkingStrategy = Literal["fixed", "page", "semantic", "paragraph", "heading"]

# Optimized regex patterns for heading detection
HEADING_PATTERNS = [
    r"^#{1,6}\s+.+$",  # Markdown headings
    r"^[A-Z][A-Za-z\s]+$",  # All caps or title case single line
    r"^\d+\.\s+[A-Z]",  # Numbered headings (1. Title)
    r"^[IVXLCDMivxlcdm]+\.\s+[A-Z]",  # Roman numeral headings (IV. Title)
]
HEADING_REGEX = re.compile("|".join(f"({pattern})" for pattern in HEADING_PATTERNS))

def fixed_size_chunking(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Split text into fixed-size chunks with optional overlap.
    Optimized for performance.

    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of chunks with metadata
    """
    chunks = []
    text_length = len(text)
    start = 0

    # Pre-allocate list size for better performance
    estimated_chunks = (text_length // (chunk_size - overlap)) + 1
    chunks = [None] * estimated_chunks
    chunk_index = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end]

        chunks[chunk_index] = {
                "text": chunk_text,
            "metadata": {
                "start_char": start,
                "end_char": end,
                "strategy": "fixed"
            }
        }

        chunk_index += 1
        start = end - overlap if end < text_length else text_length

    # Trim any unused pre-allocated space
    return chunks[:chunk_index]

def page_based_chunking(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Split PDF by pages, with each page as a separate chunk.
    Optimized for parallel processing.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of chunks with metadata
    """
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            
            # Function to process a single page with optimized extraction
            def process_page(page_idx: int) -> Dict[str, Any]:
                try:
                    page = reader.pages[page_idx]
                    text = page.extract_text(extraction_mode="layout") or ""
                    return {
                        "text": text.strip(),
                        "metadata": {
                            "page": page_idx + 1,
                            "strategy": "page"
                        }
                    }
                except Exception as e:
                    logger.warning(f"Error processing page {page_idx}: {str(e)}")
                    return {
                        "text": "",
                        "metadata": {
                            "page": page_idx + 1,
                            "strategy": "page",
                            "error": str(e)
                        }
                    }

            # Determine optimal chunk size for parallel processing
            optimal_chunk_size = min(10, max(1, total_pages // 4))
            
            # Process pages in parallel with optimized chunking
            chunks = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, total_pages)) as executor:
                # Create chunks of pages for better memory management
                page_chunks = [range(i, min(i + optimal_chunk_size, total_pages)) 
                             for i in range(0, total_pages, optimal_chunk_size)]
                
                # Process each chunk of pages
                for chunk in page_chunks:
                    futures = [executor.submit(process_page, page_idx) for page_idx in chunk]
                    chunk_results = [f.result() for f in concurrent.futures.as_completed(futures)]
                    chunks.extend([r for r in chunk_results if r["text"]])

        return chunks

    except Exception as e:
        logger.error(f"Error in page-based chunking: {str(e)}")
        raise

def paragraph_chunking(text: str) -> List[Dict[str, Any]]:
    """
    Split text by paragraphs.
    Optimized for performance.

    Args:
        text: The text to chunk

    Returns:
        List of chunks with metadata
    """
    # Split text by double newlines and filter empty paragraphs efficiently
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Pre-allocate list size for better performance
    chunks = [None] * len(paragraphs)
    current_position = 0

    for i, paragraph in enumerate(paragraphs):
        start_position = text.find(paragraph, current_position)
        end_position = start_position + len(paragraph)

        chunks[i] = {
                "text": paragraph,
                "metadata": {
                    "start_char": start_position,
                    "end_char": end_position,
                "strategy": "paragraph"
            }
        }

        current_position = end_position

    return chunks

def heading_chunking(text: str) -> List[Dict[str, Any]]:
    """
    Split text by headings (identified by heuristics).
    Optimized for performance with pre-compiled regex.

    Args:
        text: The text to chunk

    Returns:
        List of chunks with metadata
    """
    # Split by lines and process efficiently
    lines = text.split("\n")
    chunks = []
    current_heading = "Introduction"
    current_text = []
    current_start = 0

    for line in lines:
        if HEADING_REGEX.match(line.strip()):
            # If we have accumulated text, save it as a chunk
            if current_text:
                chunk_text = "\n".join(current_text)
                chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "heading": current_heading,
                            "start_char": current_start,
                            "end_char": current_start + len(chunk_text),
                        "strategy": "heading"
                    }
                })

            # Start a new chunk with this heading
            current_heading = line.strip()
            current_text = []
            current_start = text.find(line, current_start)
        else:
            current_text.append(line)

    # Add the last chunk
    if current_text:
        chunk_text = "\n".join(current_text)
        chunks.append({
                "text": chunk_text,
                "metadata": {
                    "heading": current_heading,
                    "start_char": current_start,
                    "end_char": current_start + len(chunk_text),
                "strategy": "heading"
            }
        })

    return chunks

def semantic_chunking(text: str) -> List[Dict[str, Any]]:
    """
    Use OpenAI to identify semantic chunks in the text.
    Delegates to optimized semantic chunking implementation.

    Args:
        text: The text to chunk

    Returns:
        List of chunks with metadata
    """
    return semantic_chunk_with_structured_output(text)

def process_in_batches(
    text: str,
    strategy: str,
    chunk_size: int,
    overlap: int,
    batch_size: Optional[int] = None,
    progress_callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """
    Process large text in batches for better memory management.
    
    Args:
        text: Text to process
        strategy: Chunking strategy
        chunk_size: Chunk size
        overlap: Overlap size
        batch_size: Optional initial batch size
        progress_callback: Optional callback function to track progress
        
    Returns:
        List of chunks
    """
    chunks = []
    text_length = len(text)
    start = 0
    
    # Initialize batch size if not provided
    if batch_size is None:
        batch_size = min(INITIAL_BATCH_SIZE, text_length)
    
    total_batches = (text_length + batch_size - 1) // batch_size
    current_batch = 0
    
    while start < text_length:
        # Get current memory usage
        memory_usage = psutil.virtual_memory().percent
        
        # Adjust batch size based on memory usage
        batch_size = adjust_batch_size(batch_size, memory_usage)
        
        # Check if we have enough memory to proceed
        if not check_memory_usage():
            logger.warning(f"Memory usage high ({memory_usage}%), reducing batch size to {batch_size}")
            if batch_size < MIN_BATCH_SIZE:
                raise MemoryError("Insufficient memory to process document")
        
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
        
        # Log memory usage
        log_memory_usage(f"batch {current_batch}/{total_batches}")
    
    return chunks
