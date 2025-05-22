"""
Enhanced document chunking service with optimized extraction and processing.

This module integrates a    # Check cache first
    cache_strategy = f"chunking_{strategy}_{chunk_size}_{overlap}"
    cached_result = document_cache.get(file_path, cache_strategy)
    if cached_result:
        elapsed = time.time() - start_time
        logger.info(f"Retrieved cached result in {elapsed:.4f} seconds")
        return cached_resulte optimized document processing components into
a high-performance document chunking system with intelligent resource management.
"""
import logging
import time
import os
import gc
import traceback
import concurrent.futures
from typing import Dict, Any, List, Optional, Union, BinaryIO

from Unsiloed.utils.document_cache import document_cache
from Unsiloed.utils.memory_profiling import MemoryProfiler

# Import custom exceptions if available
try:
    from Unsiloed.utils.exceptions import (
        DocumentProcessingError,
        ExtractionError,
        ChunkingError,
        MemoryError,
        CacheError,
        FileFormatError
    )
    CUSTOM_EXCEPTIONS_AVAILABLE = True
except ImportError:
    CUSTOM_EXCEPTIONS_AVAILABLE = False

# Import original methods for backward compatibility
from Unsiloed.utils.chunking import (
    fixed_size_chunking,
    page_based_chunking, 
    paragraph_chunking,
    heading_chunking,
    semantic_chunking,
    ChunkingStrategy
)

# Import optimized methods
from Unsiloed.utils.optimized_chunking import (
    semantic_chunk_optimized,
    paragraph_chunking_optimized,
    fixed_size_chunking_optimized,
    heading_chunking_optimized
)

# Import optimized document extraction modules
try:
    from Unsiloed.utils.optimized_pdf import (
        extract_text_streaming_optimized as extract_pdf_streaming,
        extract_text_mmap_pdf,
        extract_pdf_with_images,
        extract_pdf_page_count,
        get_optimal_pdf_extractor,
        repair_pdf
    )
    OPTIMIZED_PDF_AVAILABLE = True
except ImportError:
    OPTIMIZED_PDF_AVAILABLE = False

try:
    from Unsiloed.utils.optimized_docx import (
        extract_text_streaming_docx,
        extract_text_docx_with_structure,
        extract_docx_with_images,
        get_optimal_docx_extractor
    )
    OPTIMIZED_DOCX_AVAILABLE = True
except ImportError:
    OPTIMIZED_DOCX_AVAILABLE = False

# Import fallback methods for backward compatibility
from Unsiloed.utils.extraction import extract_text_streaming
from Unsiloed.utils.openai import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_pptx,
)

logger = logging.getLogger(__name__)


def process_document_chunking_enhanced(
    file_path: Union[str, BinaryIO],
    file_type: str,
    strategy: ChunkingStrategy,
    chunk_size: int = 1000,
    overlap: int = 100,
    use_optimized: bool = True,
    extraction_method: Optional[str] = None,
    memory_limit_mb: Optional[int] = None,
):
    """
    Process a document file (PDF, DOCX, PPTX) with the specified chunking strategy
    using enhanced extraction and processing for better memory efficiency and performance.

    This function intelligently selects the best extraction and processing methods
    based on document type, size, and system capabilities, adapting to available
    resources to prevent memory issues with large documents.
    
    Args:
        file_path: Path to the document file or file-like object
        file_type: Type of document (pdf, docx, pptx)
        strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy
        overlap: Overlap size for fixed strategy
        use_optimized: Whether to use optimized processing when available
        extraction_method: Force a specific extraction method
        memory_limit_mb: Optional memory limit in MB to adapt processing approach
        
    Returns:
        Dictionary with chunking results
    """
    start_time = time.time()
    logger.info(
        f"Processing {file_type.upper()} document with {strategy} chunking strategy (enhanced)"
    )
    
    # Check cache first
    cache_strategy = f"chunking_{strategy}_{chunk_size}_{overlap}"
    cached_result = document_cache.get(file_path, cache_strategy)
    if cached_result:
        elapsed = time.time() - start_time
        logger.info(f"Retrieved cached result in {elapsed:.4f} seconds")
        return cached_result
    
    # Start memory profiling for this operation
    with MemoryProfiler(f"{file_type}_processing_{os.path.basename(str(file_path))}") as profiler:
        # Extract text based on file type and size, using optimized methods when available
        if file_type.lower() == "pdf":
            # For PDFs, use optimized extraction when available
            if use_optimized and OPTIMIZED_PDF_AVAILABLE:
                try:
                    # Get file size if possible to determine best approach
                    file_size_mb = None
                    if isinstance(file_path, str):
                        try:
                            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        except (OSError, IOError):
                            pass
                    
                    # For very large PDFs, use streaming approach
                    if file_size_mb and file_size_mb > 50:  # 50MB threshold
                        logger.info(f"Large PDF detected ({file_size_mb:.2f} MB), using streaming extraction")
                        text_chunks = []
                        for page_data in extract_pdf_streaming(file_path, extraction_method):
                            text_chunks.append(page_data["text"])
                        text = "\n\n".join(text_chunks)
                    else:
                        # For smaller PDFs, use memory-mapped extraction
                        text = extract_text_mmap_pdf(file_path, extraction_method=extraction_method)
                except Exception as e:
                    logger.warning(f"Optimized PDF extraction failed: {str(e)}, falling back to standard method")
                    # Fall back to standard extraction
                    if isinstance(file_path, str):
                        text = extract_text_from_pdf(file_path)
                    else:
                        text = extract_text_streaming(file_path, "pdf")
            else:
                # Use standard extraction
                if isinstance(file_path, str):
                    text = extract_text_from_pdf(file_path)
                else:
                    text = extract_text_streaming(file_path, "pdf")
                    
        elif file_type.lower() == "docx":
            # For DOCX, use optimized extraction when available
            if use_optimized and OPTIMIZED_DOCX_AVAILABLE:
                try:
                    # Get structured extraction with better formatting
                    result = extract_text_docx_with_structure(file_path, extraction_method)
                    text = result["text"]
                except Exception as e:
                    logger.warning(f"Optimized DOCX extraction failed: {str(e)}, falling back to standard method")
                    # Fall back to standard extraction
                    if isinstance(file_path, str):
                        text = extract_text_from_docx(file_path)
                    else:
                        text = extract_text_streaming(file_path, "docx")
            else:
                # Use standard extraction
                if isinstance(file_path, str):
                    text = extract_text_from_docx(file_path)
                else:
                    text = extract_text_streaming(file_path, "docx")
                    
        elif file_type.lower() == "pptx":
            # Standard extraction for PPTX
            if isinstance(file_path, str):
                text = extract_text_from_pptx(file_path)
            else:
                text = extract_text_streaming(file_path, "pptx")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Apply chunking strategy
        chunks = []
        
        # Do a simple memory check before chunking
        current_memory = profiler.get_current_memory_mb()
        memory_warning = ""
        
        if memory_limit_mb and current_memory > memory_limit_mb * 0.8:
            # We're approaching memory limit, enforce more garbage collection
            gc.collect()
            memory_warning = f"Memory usage high ({current_memory:.2f}MB), enforcing additional GC"
            logger.warning(memory_warning)
        
        try:
            if strategy == "fixed":
                if use_optimized:
                    chunks = fixed_size_chunking_optimized(
                        text, chunk_size=chunk_size, overlap=overlap
                    )
                else:
                    chunks = fixed_size_chunking(
                        text, chunk_size=chunk_size, overlap=overlap
                    )
                    
            elif strategy == "page" and file_type.lower() == "pdf":
                chunks = page_based_chunking(text)
                
            elif strategy == "semantic":
                if use_optimized:
                    # Use the optimized implementation which handles large texts better
                    import asyncio
                    try:
                        # Check if we can run in asyncio context
                        if asyncio.get_event_loop().is_running():
                            chunks = asyncio.run(semantic_chunk_optimized(text))
                        else:
                            chunks = asyncio.run(semantic_chunk_optimized(text))
                    except:
                        # Fallback to sync version
                        chunks = semantic_chunking(text)
                else:
                    chunks = semantic_chunking(text)
                    
            elif strategy == "paragraph":
                if use_optimized:
                    chunks = paragraph_chunking_optimized(text)
                else:
                    chunks = paragraph_chunking(text)
                    
            elif strategy == "heading":
                if use_optimized:
                    chunks = heading_chunking_optimized(text)
                else:
                    chunks = heading_chunking(text)
                    
            else:
                raise ValueError(f"Unsupported chunking strategy: {strategy}")
                
        except Exception as e:
            error_msg = f"Error applying {strategy} chunking strategy: {str(e)}"
            logger.error(error_msg)
            
            if CUSTOM_EXCEPTIONS_AVAILABLE:
                # Record the original error but continue with fallback
                logger.warning(f"Falling back to fixed size chunking due to error in {strategy} strategy")
                try:
                    chunks = fixed_size_chunking(text, chunk_size=chunk_size, overlap=overlap)
                except Exception as fallback_error:
                    # If even the fallback fails, raise a proper error
                    raise ChunkingError(
                        f"Chunking failed with {strategy} and fallback method: {str(fallback_error)}",
                        strategy=strategy,
                        text_length=len(text) if text else 0,
                        details={
                            "original_error": str(e),
                            "fallback_error": str(fallback_error),
                            "chunk_size": chunk_size,
                            "overlap": overlap
                        }
                    ) from fallback_error
            else:
                # Fall back to simple fixed size chunking which is most robust
                chunks = fixed_size_chunking(text, chunk_size=chunk_size, overlap=overlap)
        
        # Collect all results
        result = {
            "document_type": file_type,
            "chunking_strategy": strategy,
            "chunks": chunks,
            "metadata": {
                "chunk_count": len(chunks),
                "processing_time": time.time() - start_time,
                "memory_usage": profiler.get_current_memory_mb(),
                "memory_warning": memory_warning
            },
        }
        
        # Only cache if the result is not too large
        try:
            # Rough size estimation
            import sys
            import json
            result_size = sys.getsizeof(json.dumps(result)) / (1024 * 1024)  # size in MB
            if result_size < 10:  # 10MB threshold for caching
                cache_strategy = f"chunking_{strategy}_{chunk_size}_{overlap}"
                document_cache.set(file_path, cache_strategy, result)
            else:
                logger.info(f"Result too large ({result_size:.2f}MB) to cache")
        except Exception as e:
            logger.warning(f"Error estimating result size for caching: {str(e)}")
        
        elapsed = time.time() - start_time
        logger.info(f"Document processing completed in {elapsed:.4f} seconds")
        
        return result


def process_document_chunking_auto_optimize(
    file_path: Union[str, BinaryIO],
    file_type: str,
    strategy: ChunkingStrategy,
    chunk_size: int = 1000,
    overlap: int = 100,
):
    """
    Process a document with automatic optimization selection based on
    document characteristics and system capabilities.
    
    This function automatically selects between standard, streaming, or enhanced processing
    based on document size, available memory, and optimization availability.
    
    Args:
        file_path: Path to the document file or file-like object
        file_type: Type of document (pdf, docx, pptx)
        strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy
        overlap: Overlap size for fixed strategy
        
    Returns:
        Dictionary with chunking results
    """
    from Unsiloed.services.chunking import (
        process_document_chunking_streaming,
        process_document_chunking
    )
    
    # Check file size if possible
    file_size_mb = None
    if isinstance(file_path, str):
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        except (OSError, IOError):
            pass
    
    # Check system memory
    try:
        import psutil
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        available_memory_mb = 4000  # default to assuming 4GB if psutil not available
    
    # Decide on processing strategy
    optimized_available = (
        (file_type.lower() == "pdf" and OPTIMIZED_PDF_AVAILABLE) or
        (file_type.lower() == "docx" and OPTIMIZED_DOCX_AVAILABLE)
    )
    
    # For very large files or low memory, use streaming
    if (file_size_mb and file_size_mb > 100) or available_memory_mb < 1000:
        logger.info(f"Large document or low memory detected, using streaming processing")
        return process_document_chunking_streaming(
            file_path, file_type, strategy, chunk_size, overlap
        )
    
    # For medium files with optimization available, use enhanced processing
    elif optimized_available:
        logger.info(f"Using enhanced optimized processing")
        return process_document_chunking_enhanced(
            file_path, file_type, strategy, chunk_size, overlap
        )
    
    # For smaller files or when optimization not available, use standard processing
    else:
        logger.info(f"Using standard processing")
        return process_document_chunking(
            file_path, file_type, strategy, chunk_size, overlap
        )


def batch_process_documents(
    file_paths: List[str],
    file_types: List[str],
    strategy: ChunkingStrategy,
    chunk_size: int = 1000,
    overlap: int = 100,
    max_workers: int = 3
) -> Dict[str, Any]:
    """
    Process multiple documents in parallel with optimized resource usage.
    
    Args:
        file_paths: List of paths to document files
        file_types: List of document types corresponding to file_paths
        strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy
        overlap: Overlap size for fixed strategy
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary with results for each document
    """
    if len(file_paths) != len(file_types):
        raise ValueError("file_paths and file_types must have the same length")
    
    results = {}
    start_time = time.time()
    
    # Process in parallel with executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(
                process_document_chunking_auto_optimize,
                file_path, file_type, strategy, chunk_size, overlap
            ): (file_path, file_type)
            for file_path, file_type in zip(file_paths, file_types)
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file_path, file_type = future_to_file[future]
            try:
                result = future.result()
                results[file_path] = result
                logger.info(f"Completed processing {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results[file_path] = {"error": str(e)}
    
    elapsed = time.time() - start_time
    logger.info(f"Batch processing of {len(file_paths)} documents completed in {elapsed:.4f} seconds")
    
    return {
        "results": results,
        "metadata": {
            "document_count": len(file_paths),
            "processing_time": elapsed,
        }
    }
