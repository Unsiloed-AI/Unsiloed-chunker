from Unsiloed.utils.chunking import (
    fixed_size_chunking,
    page_based_chunking,
    paragraph_chunking,
    heading_chunking,
    semantic_chunking,
    ChunkingStrategy
)
from Unsiloed.utils.openai import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_pptx,
)
from Unsiloed.utils.extraction import extract_text_streaming
from Unsiloed.utils.optimized_chunking import (
    semantic_chunk_optimized,
    paragraph_chunking_optimized,
    fixed_size_chunking_optimized,
    heading_chunking_optimized
)
from Unsiloed.utils.document_cache import document_cache
import concurrent.futures
import asyncio
import contextlib
import functools
import logging
import time
import os
import gc

# Try to import our enhanced processing modules
try:
    from Unsiloed.utils.enhanced_chunking import (
        process_document_chunking_enhanced,
        process_document_chunking_auto_optimize,
        batch_process_documents
    )
    ENHANCED_CHUNKING_AVAILABLE = True
except ImportError:
    ENHANCED_CHUNKING_AVAILABLE = False

logger = logging.getLogger(__name__)


def process_document_chunking_streaming(
    file_path,
    file_type,
    strategy,
    chunk_size=1000,
    overlap=100,
):
    """
    Process a document file (PDF, DOCX, PPTX) with the specified chunking strategy using
    streaming extraction for better performance with large documents.

    Args:
        file_path: Path to the document file
        file_type: Type of document (pdf, docx, pptx)
        strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy
        overlap: Overlap size for fixed strategy

    Returns:
        Dictionary with chunking results
    """
    start_time = time.time()
    logger.info(
        f"Processing {file_type.upper()} document with {strategy} chunking strategy (streaming)"
    )
    
    # Check if we have a cached result for this file and strategy
    try:
        from Unsiloed.utils.document_cache import document_cache
        cached_result = document_cache.get(file_path, strategy)
        if cached_result:
            logger.info(f"Using cached result for {file_path} with strategy {strategy}")
            return cached_result
    except Exception as e:
        logger.debug(f"Document cache not available or failed: {str(e)}")
        cached_result = None

    # For PDF files with paragraph or page strategy, use the original functions
    # which are better optimized for these cases
    if file_type == "pdf" and strategy in ["page", "paragraph"]:
        if strategy == "page":
            chunks = page_based_chunking(file_path)
        else:  # paragraph
            # For PDFs with paragraph strategy, use better extraction
            try:
                # Try to use improved extraction if available
                from Unsiloed.utils.pdf_extraction import extract_text_robust, extract_text_robust_with_chunking, normalize_text
                from Unsiloed.utils.parallel_extraction import process_pdf_in_parallel
                
                # First try parallel extraction for better performance
                try:
                    text = process_pdf_in_parallel(file_path)
                except Exception as e:
                    logger.warning(f"Parallel extraction failed: {str(e)}, falling back to standard methods")
                    # Fall back to improved extraction
                    try:
                        text = extract_text_robust_with_chunking(file_path)
                    except Exception as e:
                        logger.warning(f"Improved extraction failed: {str(e)}, falling back to standard extraction")
                        # Fall back to standard extraction
                        text = extract_text_from_pdf(file_path)
                        
                # Try to normalize the text anyway
                try:
                    text = normalize_text(text)
                except Exception as e:
                    logger.debug(f"Text normalization failed: {str(e)}")
            except ImportError:
                # Fall back to original extraction
                text = extract_text_from_pdf(file_path)
                
            # Use optimized paragraph chunking for better performance
            chunks = paragraph_chunking_optimized(text)
    else:
        # For non-PDF files or other strategies, use parallel streaming extraction
        # Collect all segments first, using a thread pool for better I/O performance
        all_text = []
        
        # Use a generator to collect text segments
        for segment in extract_text_streaming(file_path):
            if segment["text"].strip():
                all_text.append(segment["text"])
        
        # Combine all text segments efficiently
        combined_text = "\n\n".join(all_text)
        
        # Apply the selected chunking strategy
        if strategy == "fixed":
            chunks = fixed_size_chunking_optimized(combined_text, chunk_size, overlap)
        elif strategy == "semantic":
            # We can't use semantic_chunk_optimized directly because it's async
            # Use standard semantic chunking for now
            chunks = semantic_chunking(combined_text)
        elif strategy == "heading":
            chunks = heading_chunking(combined_text)
        elif strategy == "paragraph":
            chunks = paragraph_chunking_optimized(combined_text)
        elif strategy == "page" and file_type != "pdf":
            logger.warning(
                f"Page-based chunking not supported for {file_type}, falling back to paragraph chunking"
            )
            chunks = paragraph_chunking_optimized(combined_text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    # Calculate statistics
    total_chunks = len(chunks)
    avg_chunk_size = (
        sum(len(chunk["text"]) for chunk in chunks) / total_chunks
        if total_chunks > 0
        else 0
    )
    
    # Record processing time
    processing_time = time.time() - start_time
    
    result = {
        "file_type": file_type,
        "strategy": strategy,
        "total_chunks": total_chunks,
        "avg_chunk_size": avg_chunk_size,
        "processing_time_seconds": processing_time,
        "chunks": chunks,
    }
    
    # Cache the result for future use
    try:
        from Unsiloed.utils.document_cache import document_cache
        document_cache.set(file_path, strategy, result)
        logger.debug(f"Cached processing result for {file_path} with strategy {strategy}")
    except Exception as e:
        logger.debug(f"Failed to cache result: {str(e)}")
    
    return result


def process_document_chunking(
    file_path,
    file_type,
    strategy,
    chunk_size=1000,
    overlap=100,
):
    """
    Process a document file (PDF, DOCX, PPTX) with the specified chunking strategy.

    Args:
        file_path: Path to the document file
        file_type: Type of document (pdf, docx, pptx)
        strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy
        overlap: Overlap size for fixed strategy

    Returns:
        Dictionary with chunking results
    """
    logger.info(
        f"Processing {file_type.upper()} document with {strategy} chunking strategy"
    )

    # Handle page-based chunking for PDFs only
    if strategy == "page" and file_type == "pdf":
        chunks = page_based_chunking(file_path)
    else:
        # Extract text based on file type
        if file_type == "pdf":
            text = extract_text_from_pdf(file_path)
        elif file_type == "docx":
            text = extract_text_from_docx(file_path)
        elif file_type == "pptx":
            text = extract_text_from_pptx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Apply the selected chunking strategy
        if strategy == "fixed":
            chunks = fixed_size_chunking(text, chunk_size, overlap)
        elif strategy == "semantic":
            chunks = semantic_chunking(text)
        elif strategy == "paragraph":
            chunks = paragraph_chunking(text)
        elif strategy == "heading":
            chunks = heading_chunking(text)
        elif strategy == "page" and file_type != "pdf":
            # For non-PDF files, fall back to paragraph chunking for page strategy
            logger.warning(
                f"Page-based chunking not supported for {file_type}, falling back to paragraph chunking"
            )
            chunks = paragraph_chunking(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    # Calculate statistics
    total_chunks = len(chunks)
    avg_chunk_size = (
        sum(len(chunk["text"]) for chunk in chunks) / total_chunks
        if total_chunks > 0
        else 0
    )

    return {
        "file_type": file_type,
        "strategy": strategy,
        "total_chunks": total_chunks,
        "avg_chunk_size": avg_chunk_size,
        "chunks": chunks,
    }


def process_document_optimized(
    file_path,
    file_type,
    strategy: ChunkingStrategy,
    chunk_size=1000,
    overlap=100,
):
    """
    Process a document with automatic optimization selection.
    This function automatically selects the best processing approach 
    (standard, streaming, or enhanced) based on document characteristics.
    
    Args:
        file_path: Path to the document file
        file_type: Type of document (pdf, docx, pptx)
        strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy
        overlap: Overlap size for fixed strategy
        
    Returns:
        Dictionary with chunking results
    """
    start_time = time.time()

    # Check cache first
    cache_strategy = f"chunking_{strategy}_{chunk_size}_{overlap}"
    if cached_result := document_cache.get(file_path, cache_strategy):
        elapsed = time.time() - start_time
        logger.info(f"Retrieved cached result in {elapsed:.4f} seconds")
        return cached_result

    # If enhanced chunking is available, use its auto-optimize function
    if ENHANCED_CHUNKING_AVAILABLE:
        from Unsiloed.utils.enhanced_chunking import process_document_chunking_auto_optimize
        return process_document_chunking_auto_optimize(
            file_path, file_type, strategy, chunk_size, overlap
        )
    # Otherwise, choose between standard and streaming based on file size
    file_size_mb = None
    if isinstance(file_path, str):
        with contextlib.suppress(OSError, IOError):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
    # Ensure cache strategy is available in this scope for the later caching
    cache_strategy = f"chunking_{strategy}_{chunk_size}_{overlap}"

    # For large files, use streaming
    if file_size_mb and file_size_mb > 50:  # 50MB threshold
        logger.info(f"Large document detected ({file_size_mb:.2f} MB), using streaming extraction")
        result = process_document_chunking_streaming(
            file_path, file_type, strategy, chunk_size, overlap
        )
    else:
        # For smaller files, use standard processing
        result = process_document_chunking(
            file_path, file_type, strategy, chunk_size, overlap
        )

    # Cache the result
    try:
        document_cache.set(file_path, cache_strategy, result)
    except Exception as e:
        logger.warning(f"Error caching result: {str(e)}")

    return result
