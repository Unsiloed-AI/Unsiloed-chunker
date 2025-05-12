import concurrent.futures
from typing import Literal, List, Dict, Any
import logging
import PyPDF2
import re
from Unsiloed.utils.openai import (
    semantic_chunk_with_structured_output,
)

logger = logging.getLogger(__name__)

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
