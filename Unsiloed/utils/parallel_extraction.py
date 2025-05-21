"""
Enhanced text extraction module for improved PDF processing speed.

This module provides optimized extraction capabilities with multi-processing
for better performance with large PDFs.
"""
import os
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional
import multiprocessing
from functools import partial
from Unsiloed.utils.pdf_extraction import extract_text_robust

logger = logging.getLogger(__name__)

def process_pdf_in_parallel(file_path: str, max_workers: Optional[int] = None) -> str:
    """
    Process a PDF file using parallel execution for better performance.
    Divides the PDF into sections that are processed concurrently,
    then combines the results.
    
    Args:
        file_path: Path to the PDF file
        max_workers: Maximum number of worker processes to use
                    (default: number of CPU cores)
                    
    Returns:
        Extracted text as a string
    """
    # Check if PyMuPDF is available for better performance
    try:
        from Unsiloed.utils.pdf_extraction import extract_text_pymupdf, PYMUPDF_AVAILABLE
        if PYMUPDF_AVAILABLE:
            logger.debug("Using PyMuPDF for faster parallel extraction")
            return extract_text_with_pymupdf_parallel(file_path, max_workers)
    except Exception as e:
        logger.debug(f"PyMuPDF parallel extraction not available: {str(e)}")
    
    # Determine optimal number of workers based on CPU cores and file size
    if max_workers is None:
        try:
            # Get file size to determine optimal workers
            file_size = os.path.getsize(file_path)
            # For very small files, don't use parallelization
            if file_size < 1_000_000:  # 1MB
                logger.debug(f"PDF file {file_path} is small, using direct extraction")
                return extract_text_robust(file_path)

            # For larger files, scale workers with file size, but not more than CPU cores - 1
            cpu_count = multiprocessing.cpu_count()
            max_workers = min(cpu_count - 1, max(1, file_size // 5_000_000) + 1)
            logger.debug(f"Using {max_workers} workers for PDF extraction of {file_path}")
        except Exception as e:
            logger.warning(f"Error determining optimal workers: {str(e)}, using default")
            max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free

    try:
        import PyPDF2

        # Open the PDF and get total page count
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)

        # If PDF has very few pages, just extract directly
        if total_pages <= 5:
            return extract_text_robust(file_path)

        # Calculate pages per worker
        pages_per_worker = max(1, total_pages // max_workers)

        # Create temporary directory for page segments
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_files = []

        try:
            # Split PDF into segments for parallel processing
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                page_ranges = []
                for i in range(0, total_pages, pages_per_worker):
                    end_page = min(i + pages_per_worker, total_pages)
                    page_ranges.append((i, end_page))

                # Process each segment in parallel
                segment_texts = []

                # Create a partial function for worker execution
                process_page_range = partial(extract_page_range, pdf_path=file_path)

                # Use ProcessPoolExecutor for true parallelism
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_page_range, start=start, end=end) 
                              for start, end in page_ranges]

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            if segment_text := future.result():
                                segment_texts.append(segment_text)
                        except Exception as e:
                            logger.warning(f"Error processing PDF segment: {str(e)}")

                # Combine all segments
                return "\n\n".join(segment_texts)

        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        logger.warning(f"Error in parallel PDF processing, falling back to standard extraction: {str(e)}")
        # Fall back to standard extraction
        return extract_text_robust(file_path)

def extract_page_range(start: int, end: int, pdf_path: str) -> str:
    """
    Extract text from a range of pages in a PDF.
    
    Args:
        start: Starting page number (0-based)
        end: Ending page number (exclusive)
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the page range
    """
    try:
        import PyPDF2
        import tempfile
        
        # Create a new PDF with just these pages
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pdf_writer = PyPDF2.PdfWriter()
            
            for page_num in range(start, end):
                if page_num < len(pdf_reader.pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        pdf_writer.add_page(page)
                    except Exception as e:
                        logger.warning(f"Error adding page {page_num}: {str(e)}")
            
            # Write the segment to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                pdf_writer.write(temp_file)
                temp_path = temp_file.name
        
        # Extract text from the segment
        try:
            text = extract_text_robust(temp_path)
            return text
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.warning(f"Error extracting page range {start}-{end}: {str(e)}")
        return ""

def extract_text_with_pymupdf_parallel(file_path: str, max_workers: Optional[int] = None) -> str:
    """
    Extract text from PDF using PyMuPDF with parallel processing for best performance.
    
    Args:
        file_path: Path to the PDF file
        max_workers: Maximum number of worker processes to use
        
    Returns:
        Extracted text as a string
    """
    try:
        import fitz  # PyMuPDF
        
        # Open the PDF document
        doc = fitz.open(file_path)
        page_count = len(doc)
        
        # For small documents, just process sequentially
        if page_count <= 5:
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n\n".join(text_parts)
        
        # For larger documents, use parallel processing
        if max_workers is None:
            # Use half the available cores for text extraction to avoid CPU overload
            max_workers = max(1, multiprocessing.cpu_count() // 2)
        
        # Process chunks of pages in parallel for better performance
        def extract_page_range(doc_path, start_page, end_page):
            """Extract text from a range of pages"""
            local_doc = fitz.open(doc_path)
            texts = []
            for i in range(start_page, min(end_page, len(local_doc))):
                try:
                    page = local_doc[i]
                    texts.append(page.get_text())
                except Exception as e:
                    logger.warning(f"Error extracting page {i}: {str(e)}")
                    texts.append("")
            local_doc.close()
            return "\n\n".join(texts)
            
        # Determine optimal chunk size based on document size
        chunk_size = max(5, page_count // max_workers)
        ranges = [(i, min(i + chunk_size, page_count)) 
                 for i in range(0, page_count, chunk_size)]
        
        # Close original document before spawning new processes to avoid file handle issues
        doc.close()
        
        # Process page ranges in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(extract_page_range, file_path, start, end)
                for start, end in ranges
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Reorder results if they completed out of order
        ordered_results = [""] * len(ranges)
        for i, future in enumerate(futures):
            if future in concurrent.futures.as_completed(futures):
                range_idx = i
                ordered_results[range_idx] = results[list(concurrent.futures.as_completed(futures)).index(future)]
                
        return "\n\n".join(ordered_results)
        
    except ImportError:
        logger.warning("PyMuPDF not available for parallel extraction")
        return extract_text_robust(file_path)
    except Exception as e:
        logger.warning(f"Error in PyMuPDF parallel extraction: {str(e)}")
        return extract_text_robust(file_path)
