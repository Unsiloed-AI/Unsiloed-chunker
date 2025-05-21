"""
Optimized PDF processing module with streaming and memory-efficient operations.

This module provides advanced PDF processing capabilities with:
1. Memory-mapped file access for large PDFs
2. Lazy loading of PDF pages
3. Incremental processing for reduced memory footprint
4. Advanced text extraction algorithms
"""
import os
import io
import re
import tempfile
import mmap
import gc
import logging
import traceback
from pathlib import Path
from typing import Iterator, Dict, Union, BinaryIO, List, Any, Optional, Tuple, Generator
import concurrent.futures

from Unsiloed.utils.document_cache import document_cache
from Unsiloed.utils.memory_profiling import MemoryProfiler

# Import custom exceptions if available
try:
    from Unsiloed.utils.exceptions import (
        PDFExtractionError, 
        DependencyError,
        UnsupportedOperationError,
        FileFormatError
    )
    CUSTOM_EXCEPTIONS_AVAILABLE = True
except ImportError:
    CUSTOM_EXCEPTIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Try to import specialized PDF libraries in order of preference
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available, will use alternative extraction methods")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available, will use alternative extraction methods")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 not available, will use alternative extraction methods")

# Import lazily if available for better image processing
try:
    from pdf2image import convert_from_path, convert_from_bytes
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


def get_optimal_pdf_extractor(file_path: Union[str, BinaryIO], 
                             force_method: Optional[str] = None) -> str:
    """
    Determine the optimal PDF extraction method based on file characteristics
    and available libraries.
    
    Args:
        file_path: Path to the PDF file or file-like object
        force_method: Force a specific extraction method ("pymupdf", "pdfplumber", "pypdf2")
        
    Returns:
        Best method to use for extraction ("pymupdf", "pdfplumber", "pypdf2", "ocr")
    """
    # If method is forced, use it if available
    if force_method:
        if force_method == "pymupdf" and PYMUPDF_AVAILABLE:
            return "pymupdf"
        if force_method == "pdfplumber" and PDFPLUMBER_AVAILABLE:
            return "pdfplumber"
        if force_method == "pypdf2" and PYPDF2_AVAILABLE:
            return "pypdf2"
        if force_method == "ocr" and TESSERACT_AVAILABLE:
            return "ocr"
    
    # Otherwise, determine the best method
    if PYMUPDF_AVAILABLE:
        return "pymupdf"  # Fastest and most robust method
    elif PDFPLUMBER_AVAILABLE:
        return "pdfplumber"  # Good for text-heavy PDFs
    elif PYPDF2_AVAILABLE:
        return "pypdf2"  # Basic fallback
    elif TESSERACT_AVAILABLE:
        return "ocr"  # Last resort for image-based PDFs
    else:
        raise RuntimeError("No PDF extraction methods available")


def extract_text_mmap_pdf(file_path: str, 
                         page_ranges: Optional[List[Tuple[int, int]]] = None,
                         extraction_method: Optional[str] = None) -> str:
    """
    Extract text from PDF using memory-mapped file access for better performance
    with large documents.
    
    Args:
        file_path: Path to PDF file
        page_ranges: Optional list of (start_page, end_page) tuples to extract
        extraction_method: Force a specific extraction method
        
    Returns:
        Extracted text as string
    """
    # Check cache first
    cache_strategy = f"mmapextract_{str(page_ranges or 'full')}"
    cached_result = document_cache.get(file_path, cache_strategy)
    if cached_result:
        return cached_result
    
    # Determine best extraction method
    method = extraction_method or get_optimal_pdf_extractor(file_path)
    
    # Profile memory usage during extraction
    with MemoryProfiler(f"pdf_mmap_extract_{os.path.basename(file_path)}") as profiler:
        result = ""
        
        if method == "pymupdf":
            # PyMuPDF with optimized memory usage
            with open(file_path, 'rb') as file:
                # Create memory-mapped file for more efficient access
                with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    # Create document from memory-mapped file
                    doc = fitz.open("pdf", mmapped_file)
                    
                    try:
                        # Process only requested pages or all if not specified
                        if page_ranges:
                            pages = []
                            for start, end in page_ranges:
                                pages.extend(range(start, min(end+1, doc.page_count)))
                        else:
                            pages = range(doc.page_count)
                            
                        # Process pages in batches to control memory usage
                        batch_size = 10  # Adjust based on document size
                        texts = []
                        
                        for i in range(0, len(pages), batch_size):
                            batch_pages = pages[i:i+batch_size]
                            batch_text = []
                            
                            for page_num in batch_pages:
                                page = doc.load_page(page_num)
                                page_text = page.get_text("text")
                                batch_text.append(page_text)
                                # Force page deallocation to free memory
                                page = None
                                
                            texts.extend(batch_text)
                            
                            # Explicit garbage collection after each batch
                            gc.collect()
                            
                        result = "\n\n".join(texts)
                    finally:
                        # Ensure document is closed and memory is freed
                        doc.close()
                        doc = None
                        gc.collect()
        
        elif method == "pdfplumber":
            # PDFPlumber with optimized memory usage
            with pdfplumber.open(file_path) as pdf:
                texts = []
                
                # Process only requested pages or all if not specified
                if page_ranges:
                    pages = []
                    for start, end in page_ranges:
                        pages.extend(range(start, min(end+1, len(pdf.pages))))
                else:
                    pages = range(len(pdf.pages))
                
                # Process pages in batches
                batch_size = 5  # Adjust based on document size
                for i in range(0, len(pages), batch_size):
                    batch_pages = pages[i:i+batch_size]
                    batch_text = []
                    
                    for page_num in batch_pages:
                        page = pdf.pages[page_num]
                        page_text = page.extract_text() or ""
                        batch_text.append(page_text)
                        # Force page deallocation
                        page = None
                        
                    texts.extend(batch_text)
                    
                    # Explicit garbage collection after each batch
                    gc.collect()
                    
                result = "\n\n".join(texts)
                
        elif method == "pypdf2":
            # PyPDF2 with memory optimization
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                texts = []
                
                # Process only requested pages or all if not specified
                if page_ranges:
                    pages = []
                    for start, end in page_ranges:
                        pages.extend(range(start, min(end+1, len(reader.pages))))
                else:
                    pages = range(len(reader.pages))
                
                # Process pages in batches
                batch_size = 10
                for i in range(0, len(pages), batch_size):
                    batch_pages = pages[i:i+batch_size]
                    batch_text = []
                    
                    for page_num in batch_pages:
                        page_text = reader.pages[page_num].extract_text() or ""
                        batch_text.append(page_text)
                        
                    texts.extend(batch_text)
                    
                    # Explicit garbage collection after each batch
                    gc.collect()
                    
                result = "\n\n".join(texts)
                
        elif method == "ocr" and TESSERACT_AVAILABLE:
            # OCR-based extraction for problematic PDFs
            from pdf2image import convert_from_path
            import pytesseract
            
            # Create temporary directory for images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert PDF to images
                images = convert_from_path(
                    file_path,
                    dpi=300,
                    output_folder=temp_dir,
                    fmt="jpeg",
                    grayscale=True,
                    use_pdftocairo=True,
                )
                
                texts = []
                # Process only requested pages or all if not specified
                if page_ranges:
                    page_indices = []
                    for start, end in page_ranges:
                        page_indices.extend(range(start, min(end+1, len(images))))
                else:
                    page_indices = range(len(images))
                
                # Process images in batches
                batch_size = 3  # OCR is memory-intensive
                for i in range(0, len(page_indices), batch_size):
                    batch_indices = page_indices[i:i+batch_size]
                    batch_text = []
                    
                    for idx in batch_indices:
                        img = images[idx]
                        page_text = pytesseract.image_to_string(img, lang='eng')
                        batch_text.append(page_text)
                        # Release image from memory
                        img = None
                        
                    texts.extend(batch_text)
                    
                    # Explicit garbage collection after each batch
                    gc.collect()
                    
                result = "\n\n".join(texts)
        
        else:
            raise RuntimeError(f"No suitable PDF extraction method available")
    
    # Store in cache for future use
    cache_strategy = f"mmapextract_{str(page_ranges or 'full')}"
    document_cache.set(file_path, cache_strategy, result)
    
    return result


def extract_text_streaming_optimized(
    file_path: Union[str, BinaryIO], 
    extraction_method: Optional[str] = None,
    batch_size: int = 5
) -> Generator[Dict[str, Union[str, int]], None, None]:
    """
    Stream text extraction from PDF with optimized memory usage.
    
    Args:
        file_path: Path to the PDF file or file-like object
        extraction_method: Force a specific extraction method
        batch_size: Number of pages to process in each batch
        
    Yields:
        Dictionary with page number and extracted text
    """
    # Determine best extraction method
    method = extraction_method or get_optimal_pdf_extractor(file_path)
    
    # Use a memory profiler to track usage during streaming extraction
    profiler = MemoryProfiler(f"pdf_stream_extract_{os.path.basename(str(file_path))}")
    profiler.start()
    
    try:
        if method == "pymupdf":
            # Convert BytesIO to file if needed
            if isinstance(file_path, io.BytesIO):
                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                    temp_path = temp_file.name
                    file_path.seek(0)
                    temp_file.write(file_path.read())
                
                try:
                    doc = fitz.open(temp_path)
                    
                    # Process pages in batches
                    for i in range(0, doc.page_count, batch_size):
                        batch_end = min(i + batch_size, doc.page_count)
                        
                        for page_num in range(i, batch_end):
                            page = doc.load_page(page_num)
                            text = page.get_text("text")
                            yield {"page": page_num + 1, "text": text}
                            
                            # Free page from memory
                            page = None
                        
                        # Explicit garbage collection after each batch
                        gc.collect()
                        
                    # Close document and clean up
                    doc.close()
                    doc = None
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            else:
                # Direct file path processing
                doc = fitz.open(file_path)
                
                try:
                    # Process pages in batches
                    for i in range(0, doc.page_count, batch_size):
                        batch_end = min(i + batch_size, doc.page_count)
                        
                        for page_num in range(i, batch_end):
                            page = doc.load_page(page_num)
                            text = page.get_text("text")
                            yield {"page": page_num + 1, "text": text}
                            
                            # Free page from memory
                            page = None
                        
                        # Explicit garbage collection after each batch
                        gc.collect()
                finally:
                    # Close document and clean up
                    doc.close()
                    doc = None
                    
        elif method == "pdfplumber":
            # Handle BytesIO similarly
            if isinstance(file_path, io.BytesIO):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                    temp_path = temp_file.name
                    file_path.seek(0)
                    temp_file.write(file_path.read())
                
                try:
                    with pdfplumber.open(temp_path) as pdf:
                        # Process pages in batches
                        for i in range(0, len(pdf.pages), batch_size):
                            batch_end = min(i + batch_size, len(pdf.pages))
                            
                            for page_num in range(i, batch_end):
                                page = pdf.pages[page_num]
                                text = page.extract_text() or ""
                                yield {"page": page_num + 1, "text": text}
                                
                                # Free page from memory
                                page = None
                            
                            # Explicit garbage collection after each batch
                            gc.collect()
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            else:
                with pdfplumber.open(file_path) as pdf:
                    # Process pages in batches
                    for i in range(0, len(pdf.pages), batch_size):
                        batch_end = min(i + batch_size, len(pdf.pages))
                        
                        for page_num in range(i, batch_end):
                            page = pdf.pages[page_num]
                            text = page.extract_text() or ""
                            yield {"page": page_num + 1, "text": text}
                            
                            # Free page from memory
                            page = None
                        
                        # Explicit garbage collection after each batch
                        gc.collect()
                        
        elif method == "pypdf2":
            # Similar pattern for PyPDF2
            if isinstance(file_path, io.BytesIO):
                file_obj = file_path
                file_obj.seek(0)
                reader = PyPDF2.PdfReader(file_obj)
            else:
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
            
            # Process pages in batches
            for i in range(0, len(reader.pages), batch_size):
                batch_end = min(i + batch_size, len(reader.pages))
                
                for page_num in range(i, batch_end):
                    text = reader.pages[page_num].extract_text() or ""
                    yield {"page": page_num + 1, "text": text}
                
                # Explicit garbage collection after each batch
                gc.collect()
                
        elif method == "ocr" and TESSERACT_AVAILABLE:
            # OCR-based extraction for problematic PDFs
            if isinstance(file_path, io.BytesIO):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                    temp_path = temp_file.name
                    file_path.seek(0)
                    temp_file.write(file_path.read())
                file_to_use = temp_path
            else:
                file_to_use = file_path
                
            try:
                # Create temporary directory for images
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Convert PDF to images
                    images = convert_from_path(
                        file_to_use,
                        dpi=300,
                        output_folder=temp_dir,
                        fmt="jpeg",
                        grayscale=True,
                        use_pdftocairo=True,
                    )
                    
                    # Process images in batches
                    for i in range(0, len(images), batch_size):
                        batch_end = min(i + batch_size, len(images))
                        
                        for page_num in range(i, batch_end):
                            img = images[page_num]
                            text = pytesseract.image_to_string(img, lang='eng')
                            yield {"page": page_num + 1, "text": text}
                            
                            # Free image from memory
                            img = None
                        
                        # Explicit garbage collection after each batch
                        gc.collect()
            finally:
                # Clean up temporary file if created
                if isinstance(file_path, io.BytesIO):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
        else:
            raise RuntimeError(f"No suitable PDF extraction method available")
    finally:
        # Stop profiling and log memory usage
        memory_stats = profiler.stop()
        logger.debug(f"Memory usage during PDF extraction: {memory_stats}")


def extract_pdf_page_count(file_path: Union[str, BinaryIO]) -> int:
    """
    Efficiently get the page count of a PDF without loading the entire document.
    
    Args:
        file_path: Path to the PDF file or file-like object
        
    Returns:
        Number of pages in the PDF
    """
    # Try different methods in order of efficiency
    if PYMUPDF_AVAILABLE:
        try:
            if isinstance(file_path, io.BytesIO):
                file_path.seek(0)
                doc = fitz.open(stream=file_path.read(), filetype="pdf")
            else:
                doc = fitz.open(file_path)
            count = doc.page_count
            doc.close()
            return count
        except Exception as e:
            logger.debug(f"Error getting page count with PyMuPDF: {str(e)}")
    
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(file_path) as pdf:
                return len(pdf.pages)
        except Exception as e:
            logger.debug(f"Error getting page count with pdfplumber: {str(e)}")
    
    # Fall back to PyPDF2
    try:
        if isinstance(file_path, io.BytesIO):
            file_path.seek(0)
            reader = PyPDF2.PdfReader(file_path)
        else:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
        return len(reader.pages)
    except Exception as e:
        logger.debug(f"Error getting page count with PyPDF2: {str(e)}")
    
    # If all methods failed
    raise RuntimeError("Failed to determine PDF page count")


def extract_pdf_with_images(file_path: Union[str, BinaryIO]) -> Dict[str, Any]:
    """
    Extract both text and images from a PDF document.
    
    Args:
        file_path: Path to the PDF file or file-like object
        
    Returns:
        Dictionary with extracted text and image data
    """
    # Ensure PyMuPDF is available for image extraction
    if not PYMUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF is required for image extraction")
    
    result = {
        "text": "",
        "images": [],
        "pages": []
    }
    
    try:
        # Open the document
        if isinstance(file_path, io.BytesIO):
            file_path.seek(0)
            doc = fitz.open(stream=file_path.read(), filetype="pdf")
        else:
            doc = fitz.open(file_path)
        
        # Process pages in batches for better memory management
        batch_size = 5
        page_texts = []
        
        for i in range(0, doc.page_count, batch_size):
            batch_end = min(i + batch_size, doc.page_count)
            
            for page_num in range(i, batch_end):
                page = doc.load_page(page_num)
                page_text = page.get_text("text")
                page_images = []
                
                # Extract images from page
                image_list = page.get_images(full=True)
                
                for img_idx, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        
                        if base_image:
                            image_data = {
                                "index": img_idx,
                                "width": base_image["width"],
                                "height": base_image["height"],
                                "format": base_image["ext"],
                                "size": len(base_image["image"]),
                                "page": page_num + 1
                            }
                            
                            # Only store the binary data if specifically requested
                            # to avoid excessive memory usage
                            # image_data["data"] = base_image["image"]
                            
                            page_images.append(image_data)
                            result["images"].append(image_data)
                    except Exception as e:
                        logger.warning(f"Error extracting image: {str(e)}")
                
                # Collect page information
                result["pages"].append({
                    "number": page_num + 1,
                    "text": page_text,
                    "image_count": len(page_images)
                })
                
                page_texts.append(page_text)
                
                # Free page from memory
                page = None
            
            # Explicit garbage collection after each batch
            gc.collect()
        
        # Combine all text
        result["text"] = "\n\n".join(page_texts)
        
    finally:
        # Ensure document is closed and memory is freed
        if 'doc' in locals():
            doc.close()
        
        gc.collect()
    
    return result


def repair_pdf(file_path: Union[str, BinaryIO], output_path: str) -> bool:
    """
    Attempt to repair a corrupted PDF file.
    
    Args:
        file_path: Path to the corrupted PDF file or file-like object
        output_path: Path to save the repaired PDF
        
    Returns:
        True if repair was successful, False otherwise
    """
    try:
        # Try using PyMuPDF for repair first
        if PYMUPDF_AVAILABLE:
            try:
                if isinstance(file_path, io.BytesIO):
                    file_path.seek(0)
                    doc = fitz.open(stream=file_path.read(), filetype="pdf")
                else:
                    doc = fitz.open(file_path)
                
                # Clean up and sanitize the document
                doc.save(output_path, garbage=4, clean=True, deflate=True)
                doc.close()
                return True
            except Exception as e:
                logger.debug(f"PyMuPDF repair failed: {str(e)}")
        
        # Fall back to PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                if isinstance(file_path, io.BytesIO):
                    file_path.seek(0)
                    reader = PyPDF2.PdfReader(file_path, strict=False)
                else:
                    with open(file_path, "rb") as file:
                        reader = PyPDF2.PdfReader(file, strict=False)
                
                writer = PyPDF2.PdfWriter()
                
                # Copy pages to a new document
                for page in reader.pages:
                    writer.add_page(page)
                
                # Save the repaired document
                with open(output_path, "wb") as output_file:
                    writer.write(output_file)
                
                return True
            except Exception as e:
                logger.debug(f"PyPDF2 repair failed: {str(e)}")
        
        return False
    except Exception as e:
        logger.error(f"PDF repair failed: {str(e)}")
        return False
