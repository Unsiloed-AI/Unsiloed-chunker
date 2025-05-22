"""
More robust PDF extraction utilities to improve performance and reliability.

This module provides alternative PDF extraction methods using different PDF libraries
to address the extraction errors found in PyPDF2, with graceful fallbacks if
dependencies are not available.
"""
import os
import io
import re
import logging
import concurrent.futures
import mmap  # Added for memory mapping
import warnings
import time
from typing import Iterator, Dict, Union, BinaryIO, List, Any, Optional, Tuple
import PyPDF2  # Keep original as fallback

# Dictionary to track dependency errors for better messaging
DEPENDENCY_ERRORS = {}

# Try to import pdfplumber which is more reliable for text extraction
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError as e:
    PDFPLUMBER_AVAILABLE = False
    DEPENDENCY_ERRORS["pdfplumber"] = str(e)

# Try to import PyMuPDF which is faster and more robust than PyPDF2
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    # Check if this is a version with known issues
    PYMUPDF_VERSION = tuple(int(x) for x in fitz.version[0].split('.'))
    if PYMUPDF_VERSION < (1, 18, 0):
        warnings.warn(f"PyMuPDF version {fitz.version[0]} is older than recommended (1.18.0+)")
except ImportError as e:
    PYMUPDF_AVAILABLE = False
    DEPENDENCY_ERRORS["pymupdf"] = str(e)
except Exception as e:
    PYMUPDF_AVAILABLE = False
    DEPENDENCY_ERRORS["pymupdf"] = f"Unexpected error loading PyMuPDF: {str(e)}"

# Try to import pdf2image for image-based extraction as fallback
try:
    from pdf2image import convert_from_path, convert_from_bytes
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError as e:
    TESSERACT_AVAILABLE = False
    DEPENDENCY_ERRORS["pdf2image_tesseract"] = str(e)

# Precompile regular expressions for better performance
RE_WINDOWS_NEWLINES = re.compile(r'\r\n')
RE_TABS = re.compile(r'\t')
RE_MULTIPLE_SPACES = re.compile(r' {2,}')
RE_MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
RE_HYPHENATED_WORDS = re.compile(r'(\w)-\n(\w)')
RE_SENTENCE_LINEBREAK = re.compile(r'([.!?])\s*\n\s*([A-Z])')
RE_SENTENCE_SPACE = re.compile(r'([.!?])\s+([A-Z])')
RE_PARAGRAPHS = re.compile(r'\n\n+')

logger = logging.getLogger(__name__)

def extract_text_robust(file_path: Union[str, BinaryIO]) -> str:
    """
    Extract text from PDF using the most robust available method.
    Simplified for better performance.
    
    Args:
        file_path: Path to the PDF file or file-like object
        
    Returns:
        Extracted text as a string
    """
    # First check if we can use memory mapping for best performance with large files
    if isinstance(file_path, str) and os.path.isfile(file_path) and os.path.getsize(file_path) > 10 * 1024 * 1024:  # > 10MB
        try:
            return extract_text_with_mmap(file_path)
        except Exception as e:
            logger.warning(f"Memory-mapped extraction failed: {str(e)}")
    
    # Next try PyMuPDF which is generally faster and more robust
    if PYMUPDF_AVAILABLE:
        try:
            return extract_text_pymupdf(file_path)
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {str(e)}")
    
    # Try pdfplumber if available
    if PDFPLUMBER_AVAILABLE:
        try:
            # Simple sequential extraction for better reliability
            if isinstance(file_path, str):
                with pdfplumber.open(file_path) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        try:
                            text = page.extract_text(x_tolerance=3) or ""
                            if text.strip():
                                text_parts.append(text)
                        except Exception as e:
                            logger.warning(f"Error extracting page with pdfplumber: {str(e)}")
                    
                    if text_parts:
                        return "\n\n".join(text_parts)
            else:
                # Handle file-like objects
                if hasattr(file_path, 'seek'):
                    file_path.seek(0)
                
                with pdfplumber.open(file_path) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        try:
                            text = page.extract_text(x_tolerance=3) or ""
                            if text.strip():
                                text_parts.append(text)
                        except Exception as e:
                            logger.warning(f"Error extracting page with pdfplumber: {str(e)}")
                    
                    if text_parts:
                        return "\n\n".join(text_parts)
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")
    
    # Fall back to PyPDF2
    try:
        if isinstance(file_path, str):
            with open(file_path, 'rb') as file:
                text = extract_text_pypdf2(file)
                if text.strip():
                    return text
        else:
            # Ensure we're at the start of the file
            if hasattr(file_path, 'seek'):
                file_path.seek(0)
            text = extract_text_pypdf2(file_path)
            if text.strip():
                return text
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {str(e)}")
    
    # Last resort: OCR if available
    if TESSERACT_AVAILABLE:
        try:
            text = extract_text_ocr(file_path)
            if text.strip():
                return text
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
    
    # If we get here, we couldn't extract text - return empty string
    logger.error(f"All PDF extraction methods failed for {file_path}")
    return ""
        
def extract_text_pdfplumber_parallel(file_path: Union[str, BinaryIO]) -> str:
    """
    Extract text from PDF using pdfplumber with parallel processing for better performance.
    
    Args:
        file_path: Path to the PDF file or file-like object
        
    Returns:
        Extracted text as a string
    """
    if not PDFPLUMBER_AVAILABLE:
        raise ImportError("pdfplumber is not available. Install with 'pip install pdfplumber'")
    
    # Handle file-like objects
    if not isinstance(file_path, str):
        if hasattr(file_path, 'read') and hasattr(file_path, 'seek'):
            file_path.seek(0)
            pdf_content = file_path.read()
            file_obj = io.BytesIO(pdf_content)
        else:
            file_obj = io.BytesIO(file_path.getvalue())
    else:
        file_obj = file_path
    
    # Open the PDF and get the number of pages
    with pdfplumber.open(file_obj) as pdf:
        total_pages = len(pdf.pages)
        
        # For small PDFs, use the normal sequential extraction
        if total_pages <= 5:
            text_parts = []
            for page in pdf.pages:
                try:
                    text = page.extract_text(x_tolerance=3) or ""
                    if text.strip():
                        text_parts.append(text)
                except Exception as e:
                    logger.warning(f"Error extracting text from page: {str(e)}")
            return "\n\n".join(text_parts)
        
        # For larger PDFs, use parallel processing
        # Extract page numbers first to avoid keeping the PDF open during thread execution
        page_numbers = list(range(total_pages))
        
        # Define extraction function for a single page
        def extract_page(page_num):
            try:
                # Reopen the PDF for each page to avoid thread safety issues
                if isinstance(file_path, str):
                    with pdfplumber.open(file_path) as pdf:
                        page = pdf.pages[page_num]
                        text = page.extract_text(x_tolerance=3) or ""
                        return text.strip()
                else:
                    # For file objects, we need to create a new bytes object
                    if hasattr(file_path, 'seek') and hasattr(file_path, 'read'):
                        file_path.seek(0)
                        pdf_bytes = file_path.read()
                    else:
                        pdf_bytes = file_path.getvalue()
                        
                    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                        page = pdf.pages[page_num]
                        text = page.extract_text(x_tolerance=3) or ""
                        return text.strip()
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                return ""
        
        # Process pages in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as executor:
            text_parts = list(executor.map(extract_page, page_numbers))
        
        return "\n\n".join([t for t in text_parts if t])

def extract_text_pdfplumber(file_path: Union[str, BinaryIO]) -> str:
    """
    Extract text from PDF using pdfplumber which is more reliable than PyPDF2.
    
    Args:
        file_path: Path to the PDF file or file-like object
        
    Returns:
        Extracted text as a string
    """
    if not PDFPLUMBER_AVAILABLE:
        raise ImportError("pdfplumber is not available. Install with 'pip install pdfplumber'")
    
    # Handle file-like objects
    if not isinstance(file_path, str):
        if hasattr(file_path, 'read') and hasattr(file_path, 'seek'):
            file_path.seek(0)
            pdf_content = file_path.read()
            file_obj = io.BytesIO(pdf_content)
        else:
            file_obj = io.BytesIO(file_path.getvalue())
    else:
        file_obj = file_path
    
    # Extract text
    text_parts = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text(x_tolerance=3) or ""
                if text.strip():
                    text_parts.append(text)
            except Exception as e:
                logger.warning(f"Error extracting text from page: {str(e)}")
    
    return "\n\n".join(text_parts)

def extract_text_pypdf2(file_obj: BinaryIO) -> str:
    """
    Extract text from PDF using PyPDF2 with better error handling.
    
    Args:
        file_obj: File-like object containing the PDF
        
    Returns:
        Extracted text as a string
    """
    text_parts = []
    
    try:
        reader = PyPDF2.PdfReader(file_obj)
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                if text.strip():
                    text_parts.append(text)
            except Exception as e:
                logger.warning(f"Error extracting page {page_num}: {str(e)}")
    except Exception as e:
        logger.error(f"Error opening PDF: {str(e)}")
        raise
    
    return "\n\n".join(text_parts)

def extract_text_pypdf2_parallel(file_obj: BinaryIO) -> str:
    """
    Extract text from PDF using PyPDF2 with parallel processing for better performance.
    
    Args:
        file_obj: File-like object containing the PDF
        
    Returns:
        Extracted text as a string
    """
    try:
        # Create a reader
        reader = PyPDF2.PdfReader(file_obj)
        total_pages = len(reader.pages)
        
        # For small PDFs, use the normal sequential extraction
        if total_pages <= 5:
            return extract_text_pypdf2(file_obj)
        
        # For larger PDFs, use parallel processing
        # Store page numbers to process
        page_numbers = list(range(total_pages))
        
        # Define extraction function for a single page
        def extract_page(page_num):
            try:
                # Need to reopen the PDF for each thread to avoid thread safety issues
                if isinstance(file_obj, io.BytesIO):
                    # For BytesIO objects, we need to work with the bytes directly
                    pdf_bytes = file_obj.getvalue()
                    with io.BytesIO(pdf_bytes) as temp_file:
                        reader = PyPDF2.PdfReader(temp_file)
                        page = reader.pages[page_num]
                        text = page.extract_text() or ""
                        return text.strip()
                else:
                    # For file paths, reopen the file
                    file_path = getattr(file_obj, 'name', None)
                    if file_path:
                        with open(file_path, 'rb') as file:
                            reader = PyPDF2.PdfReader(file)
                            page = reader.pages[page_num]
                            text = page.extract_text() or ""
                            return text.strip()
                    else:
                        # Fall back to single-threaded if we can't reopen
                        return ""
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                return ""
        
        # Process pages in parallel, using fewer threads for PyPDF2 which can be memory-intensive
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as executor:
            text_parts = list(executor.map(extract_page, page_numbers))
        
        # Filter out empty results and join
        return "\n\n".join([t for t in text_parts if t])
    except Exception as e:
        logger.error(f"Error in parallel PDF extraction: {str(e)}")
        # Try the non-parallel version as fallback
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
        return extract_text_pypdf2(file_obj)

def extract_text_ocr(file_path: Union[str, BinaryIO]) -> str:
    """
    Extract text from PDF using OCR as a last resort.
    
    Args:
        file_path: Path to the PDF file or file-like object
        
    Returns:
        Extracted text as a string
    """
    if not TESSERACT_AVAILABLE:
        raise ImportError("pdf2image and pytesseract are required for OCR")
    
    # Convert PDF to images
    if isinstance(file_path, str):
        images = convert_from_path(file_path, dpi=300)
    else:
        # Handle file-like objects
        if hasattr(file_path, 'read') and hasattr(file_path, 'seek'):
            file_path.seek(0)
            pdf_bytes = file_path.read()
        else:
            pdf_bytes = file_path.getvalue()
        
        images = convert_from_bytes(pdf_bytes, dpi=300)
    
    # Process images with OCR in parallel
    text_parts = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as executor:
        text_parts = list(executor.map(lambda img: pytesseract.image_to_string(img), images))
    
    return "\n\n".join(text_parts)

def extract_text_streaming_robust(file_path: Union[str, BinaryIO]) -> Iterator[Dict[str, Any]]:
    """
    Extract text from PDF in a streaming fashion with fallback mechanisms.
    Yields one page at a time, with better error handling.
    
    Args:
        file_path: Path to the PDF file or file-like object
        
    Yields:
        Dictionary with text and metadata for each page
    """
    # First try pdfplumber for streaming extraction
    if PDFPLUMBER_AVAILABLE:
        try:
            if not isinstance(file_path, str):
                if hasattr(file_path, 'read') and hasattr(file_path, 'seek'):
                    file_path.seek(0)
                    pdf_content = file_path.read()
                    file_obj = io.BytesIO(pdf_content)
                else:
                    file_obj = io.BytesIO(file_path.getvalue())
            else:
                file_obj = file_path
            
            with pdfplumber.open(file_obj) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text(x_tolerance=3) or ""
                        if text.strip():
                            yield {
                                "text": text,
                                "metadata": {"page": page_num + 1}
                            }
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                        yield {
                            "text": "",
                            "metadata": {"page": page_num + 1, "error": str(e)}
                        }
            
            # Exit after successful extraction
            return
        except Exception as e:
            logger.warning(f"pdfplumber streaming extraction failed: {str(e)}, falling back to PyPDF2")
    
    # Fall back to PyPDF2 with improved error handling
    try:
        if isinstance(file_path, str):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                # For larger PDFs, use smaller batches to avoid memory issues
                batch_size = min(5, total_pages)
                
                for batch_start in range(0, total_pages, batch_size):
                    batch_end = min(batch_start + batch_size, total_pages)
                    page_numbers = list(range(batch_start, batch_end))
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(batch_size, os.cpu_count() or 2)) as executor:
                        # Define the extraction function with better error handling
                        def extract_page(page_idx):
                            try:
                                page = reader.pages[page_idx]
                                text = page.extract_text() or ""
                                if text and text.strip():
                                    return {
                                        "text": text,
                                        "metadata": {"page": page_idx + 1}
                                    }
                                return {
                                    "text": "",
                                    "metadata": {"page": page_idx + 1, "empty": True}
                                }
                            except Exception as e:
                                logger.warning(f"Error extracting text from page {page_idx}: {str(e)}")
                                return {
                                    "text": "",
                                    "metadata": {"page": page_idx + 1, "error": str(e)}
                                }
                        
                        # Execute in parallel
                        results = executor.map(extract_page, page_numbers)
                        
                        # Yield results
                        for result in results:
                            yield result
        else:
            # Similar processing for file-like objects
            if hasattr(file_path, 'seek'):
                file_path.seek(0)
            
            reader = PyPDF2.PdfReader(file_path)
            
            # Process single pages to save memory
            for page_num in range(len(reader.pages)):
                try:
                    text = reader.pages[page_num].extract_text() or ""
                    if text and text.strip():
                        yield {
                            "text": text,
                            "metadata": {"page": page_num + 1}
                        }
                    else:
                        yield {
                            "text": "",
                            "metadata": {"page": page_num + 1, "empty": True}
                        }
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                    yield {
                        "text": "",
                        "metadata": {"page": page_num + 1, "error": str(e)}
                    }
    except Exception as e:
        logger.error(f"Error opening PDF file: {str(e)}")
        
        # Last resort: Try OCR if available
        if TESSERACT_AVAILABLE:
            try:
                # Process with OCR
                if isinstance(file_path, str):
                    images = convert_from_path(file_path, dpi=300)
                else:
                    # Handle file-like objects
                    if hasattr(file_path, 'seek'):
                        file_path.seek(0)
                    
                    if hasattr(file_path, 'read'):
                        pdf_bytes = file_path.read()
                    else:
                        pdf_bytes = file_path.getvalue()
                    
                    images = convert_from_bytes(pdf_bytes, dpi=300)
                
                # Process each page
                for page_num, image in enumerate(images):
                    try:
                        text = pytesseract.image_to_string(image)
                        if text and text.strip():
                            yield {
                                "text": text,
                                "metadata": {"page": page_num + 1, "source": "ocr"}
                            }
                        else:
                            yield {
                                "text": "",
                                "metadata": {"page": page_num + 1, "empty": True, "source": "ocr"}
                            }
                    except Exception as e:
                        logger.warning(f"OCR error on page {page_num}: {str(e)}")
                        yield {
                            "text": "",
                            "metadata": {"page": page_num + 1, "error": str(e), "source": "ocr"}
                        }
            except Exception as e:
                logger.error(f"OCR extraction failed: {str(e)}")
                # Yield an empty result to prevent pipeline from breaking
                yield {
                    "text": "",
                    "metadata": {"error": f"All extraction methods failed: {str(e)}"}
                }
        else:
            # If all else fails, yield an empty result to prevent pipeline from breaking
            yield {
                "text": "",
                "metadata": {"error": f"PDF extraction failed: {str(e)}"}
            }

def normalize_text(text: str) -> str:
    """
    Normalize text by fixing common PDF extraction issues.
    
    Args:
        text: The raw text extracted from PDF
        
    Returns:
        Normalized text with proper paragraph breaks
    """
    # Fix various whitespace issues
    text = RE_WINDOWS_NEWLINES.sub('\n', text)                # Convert Windows to Unix line endings
    text = RE_TABS.sub('    ', text)                # Convert tabs to spaces
    text = RE_MULTIPLE_SPACES.sub(' ', text)                # Normalize multiple spaces
    text = RE_MULTIPLE_NEWLINES.sub('\n\n', text)            # Normalize multiple line breaks
    
    # Fix common PDF extraction issues
    text = RE_HYPHENATED_WORDS.sub(r'\1\2', text)      # Fix hyphenated words across lines
    
    # Make sure sentences and paragraphs are properly separated
    text = RE_SENTENCE_LINEBREAK.sub(r'\1\n\n\2', text)  # Add paragraph break after sentences at line breaks
    
    # Ensure text doesn't end up as one giant paragraph
    # If there are no paragraph breaks but the text is long, add some based on semantic clues
    if len(text) > 5000 and text.count('\n\n') < 5:
        # Try to break at logical points (e.g., after sentences)
        text = RE_SENTENCE_SPACE.sub(r'\1\n\n\2', text)
    
    return text

def extract_text_robust_with_chunking(file_path: Union[str, BinaryIO]) -> str:
    """
    Extract text from PDF with improved paragraph detection and chunking.
    
    This version enhances the base PDF extraction by adding better paragraph
    detection and processing, which helps prevent the issue where PDFs are
    processed as a single large chunk.
    
    Args:
        file_path: Path to the PDF file or file-like object
        
    Returns:
        Extracted text with properly formatted paragraphs
    """
    # Get the raw text from the PDF
    raw_text = extract_text_robust(file_path)
    
    # Return empty string if extraction failed
    if not raw_text or not raw_text.strip():
        logger.warning("PDF extraction returned empty text")
        return ""
    
    # Normalize text - fixes line breaks and whitespace
    text = normalize_text(raw_text)
    
    # Return the normalized text
    return text

def split_into_semantic_paragraphs(text: str, max_size: int = 5000) -> List[Dict[str, Any]]:
    """
    Split text into semantic paragraphs with size limit.
    
    This helps prevent the issue of having a single giant chunk.
    
    Args:
        text: The text to split
        max_size: Maximum size of each paragraph
        
    Returns:
        List of paragraphs with metadata
    """
    # First try to split by double newlines (standard paragraph breaks)
    paragraphs = RE_PARAGRAPHS.split(text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # If we have very few paragraphs and text is long, try more aggressive splitting
    if len(paragraphs) < 3 and len(text) > max_size:
        # Try to split by sentences
        sentence_pattern = r'(?<=[.!?])\s+'
        new_paragraphs = []
        
        for para in paragraphs:
            if len(para) > max_size:
                # Split long paragraph by sentences
                sentences = re.split(sentence_pattern, para)
                
                # Combine sentences into chunks under max_size
                current_chunk = []
                current_size = 0
                
                for sentence in sentences:
                    if current_size + len(sentence) > max_size and current_chunk:
                        # Start a new chunk if adding this sentence would exceed max_size
                        new_paragraphs.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_size = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_size += len(sentence)
                
                # Add the last chunk if it's not empty
                if current_chunk:
                    new_paragraphs.append(' '.join(current_chunk))
            else:
                # Keep paragraphs that are already under max_size
                new_paragraphs.append(para)
        
        paragraphs = new_paragraphs
    
    # Convert to the standard chunk format
    chunks = []
    current_position = 0
    
    for i, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue
            
        # Find the exact position in the original text
        start_position = text.find(paragraph, current_position)
        if start_position == -1:
            # If not found, use approximate position
            start_position = current_position
            
        end_position = start_position + len(paragraph)
        
        chunks.append({
            "text": paragraph,
            "metadata": {
                "start_char": start_position,
                "end_char": end_position,
                "strategy": "paragraph",
                "paragraph_number": i + 1,
                "title": f"Paragraph {i + 1}"
            }
        })
        
        current_position = end_position
    
    return chunks

def extract_text_pymupdf(file_path: Union[str, BinaryIO], fallback_on_error: bool = True) -> str:
    """
    Extract text from PDF using PyMuPDF (fitz), which is significantly faster than PyPDF2.
    
    Args:
        file_path: Path to the PDF file or file-like object
        fallback_on_error: Whether to fall back to robust extraction on error
        
    Returns:
        Extracted text from the PDF
        
    Raises:
        ImportError: If PyMuPDF is not available and fallback_on_error is False
        RuntimeError: If text extraction fails and fallback_on_error is False
    """
    if not PYMUPDF_AVAILABLE:
        error_msg = f"PyMuPDF not available: {DEPENDENCY_ERRORS.get('pymupdf', 'Not installed')}"
        logger.warning(error_msg)
        
        if fallback_on_error:
            logger.info("Falling back to robust extraction method")
            return extract_text_robust(file_path)
        else:
            raise ImportError(error_msg)
    
    try:
        start_time = time.time()
        
        # For file-like objects
        if not isinstance(file_path, str):
            # We need to get the bytes from the file-like object
            try:
                if hasattr(file_path, 'read'):
                    # Preserve the position if it's a seekable file
                    if hasattr(file_path, 'tell') and hasattr(file_path, 'seek'):
                        pos = file_path.tell()
                        file_content = file_path.read()
                        file_path.seek(pos)  # Reset position
                    else:
                        file_content = file_path.read()
                else:
                    file_content = file_path.getvalue()
                    
                # Open the PDF from memory buffer
                doc = fitz.open(stream=file_content, filetype="pdf")
            except Exception as e:
                # Specific handling for file-like object errors
                logger.error(f"Error reading file-like object with PyMuPDF: {str(e)}")
                if fallback_on_error:
                    return extract_text_robust(file_path)
                else:
                    raise RuntimeError(f"PyMuPDF failed to read file-like object: {str(e)}")
        else:
            # Check if file exists when path is provided as string
            if not os.path.exists(file_path):
                error_msg = f"PDF file not found: {file_path}"
                logger.error(error_msg)
                if fallback_on_error:
                    return ""
                else:
                    raise FileNotFoundError(error_msg)
                
            # Open the PDF from file path
            try:
                doc = fitz.open(file_path)
            except Exception as e:
                logger.error(f"PyMuPDF error opening {file_path}: {str(e)}")
                if fallback_on_error:
                    return extract_text_robust(file_path)
                else:
                    raise RuntimeError(f"PyMuPDF failed to open {file_path}: {str(e)}")
        
        # Extract text from all pages
        text_parts = []
        for page in doc:
            text = page.get_text()
            text_parts.append(text)
        
        # Close the document to free resources
        doc.close()
        
        return "\n\n".join(text_parts)
    
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {str(e)}")
        # Fall back to robust extraction
        return extract_text_robust(file_path)

def extract_text_with_mmap(file_path: str) -> str:
    """
    Extract text from PDF using memory-mapped file access for better performance with very large files.
    This method avoids loading the entire file into memory at once.
    
    Args:
        file_path: Path to the PDF file (must be a string path, not a file-like object)
        
    Returns:
        Extracted text from the PDF
    """
    if not isinstance(file_path, str) or not os.path.isfile(file_path):
        logger.warning("Memory mapping requires a file path, falling back to standard extraction")
        return extract_text_pymupdf(file_path) if PYMUPDF_AVAILABLE else extract_text_robust(file_path)
        
    try:
        # Try to use PyMuPDF with memory mapping for best performance
        if PYMUPDF_AVAILABLE:
            with open(file_path, 'rb') as f:
                # Use mmap to avoid loading the entire file
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mapped_file:
                    doc = fitz.open("pdf", mapped_file)
                    
                    # Extract text from all pages in parallel for large documents
                    if len(doc) > 20:  # For PDFs with many pages
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            # Extract text from pages in parallel
                            future_to_page = {executor.submit(lambda p: p.get_text(), doc[i]): i 
                                            for i in range(len(doc))}
                            
                            # Collect results in page order
                            text_parts = [""] * len(doc)
                            for future in concurrent.futures.as_completed(future_to_page):
                                page_idx = future_to_page[future]
                                try:
                                    text_parts[page_idx] = future.result()
                                except Exception as e:
                                    logger.warning(f"Error extracting text from page {page_idx}: {str(e)}")
                    else:
                        # For smaller PDFs, sequential is fine
                        text_parts = [page.get_text() for page in doc]
                    
                    doc.close()
                    return "\n\n".join(text_parts)
        
        # If PyMuPDF is not available, try alternative methods
        else:
            # Try pdfplumber with memory mapping
            if PDFPLUMBER_AVAILABLE:
                with open(file_path, 'rb') as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mapped_file:
                        with pdfplumber.open(mapped_file) as pdf:
                            text_parts = []
                            for page in pdf.pages:
                                try:
                                    text = page.extract_text() or ""
                                    text_parts.append(text)
                                except Exception as e:
                                    logger.warning(f"Error extracting text from page: {str(e)}")
                            return "\n\n".join(text_parts)
            
            # Fall back to PyPDF2 with memory mapping
            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mapped_file:
                    reader = PyPDF2.PdfReader(mapped_file)
                    text_parts = []
                    for i in range(len(reader.pages)):
                        try:
                            page = reader.pages[i]
                            text = page.extract_text() or ""
                            text_parts.append(text)
                        except Exception as e:
                            logger.warning(f"Error extracting text from page {i}: {str(e)}")
                    return "\n\n".join(text_parts)
                    
    except Exception as e:
        logger.warning(f"Memory-mapped extraction failed: {str(e)}")
        # Fall back to non-memory-mapped extraction
        return extract_text_pymupdf(file_path) if PYMUPDF_AVAILABLE else extract_text_robust(file_path)

def extract_text_streaming_pymupdf(file_path: Union[str, BinaryIO]) -> Iterator[Dict[str, Any]]:
    """
    Extract text from PDF in a streaming fashion using PyMuPDF, yielding one page at a time.
    This is significantly faster than PyPDF2-based extraction.
    
    Args:
        file_path: Path to the PDF file or file-like object
        
    Yields:
        Dictionary with page text and metadata
    """
    if not PYMUPDF_AVAILABLE:
        logger.warning("PyMuPDF not available, falling back to robust streaming extraction")
        yield from extract_text_streaming_robust(file_path)
        return
    
    try:
        # Handle file-like objects
        if not isinstance(file_path, str):
            if hasattr(file_path, 'read'):
                # Save position if seekable
                if hasattr(file_path, 'tell') and hasattr(file_path, 'seek'):
                    pos = file_path.tell()
                    file_content = file_path.read()
                    file_path.seek(pos)
                else:
                    file_content = file_path.read()
            else:
                file_content = file_path.getvalue()
                
            # Open document from memory buffer
            doc = fitz.open(stream=file_content, filetype="pdf")
        else:
            # Open from file path
            doc = fitz.open(file_path)
        
        try:
            page_count = len(doc)
            
            # Process each page
            for page_idx in range(page_count):
                try:
                    page = doc[page_idx]
                    
                    # Extract text
                    text = page.get_text()
                    
                    # Extract metadata
                    metadata = {
                        "page": page_idx + 1,
                        "total_pages": page_count,
                        "width": page.rect.width,
                        "height": page.rect.height,
                    }
                    
                    # Add image count if available
                    image_list = page.get_images(full=True)
                    if image_list:
                        metadata["image_count"] = len(image_list)
                    
                    # Check if page has meaningful content
                    if text.strip():
                        yield {
                            "text": text,
                            "metadata": metadata
                        }
                    else:
                        # Still yield empty pages with metadata
                        yield {
                            "text": "",
                            "metadata": {**metadata, "empty": True}
                        }
                        
                except Exception as e:
                    logger.warning(f"Error extracting page {page_idx}: {str(e)}")
                    yield {
                        "text": "",
                        "metadata": {
                            "page": page_idx + 1,
                            "total_pages": page_count,
                            "error": str(e)
                        }
                    }
        finally:
            # Make sure to close the document
            doc.close()
    
    except Exception as e:
        logger.warning(f"PyMuPDF streaming extraction failed: {str(e)}")
        # Fall back to robust extraction
        yield from extract_text_streaming_robust(file_path)
