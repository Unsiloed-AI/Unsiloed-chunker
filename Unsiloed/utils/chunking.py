import concurrent.futures
from typing import Literal
import logging
import PyPDF2
from Unsiloed.utils.openai import (
    semantic_chunk_with_structured_output,
)

logger = logging.getLogger(__name__)

ChunkingStrategy = Literal["fixed", "page", "semantic", "paragraph", "heading"]


def fixed_size_chunking(text, chunk_size=1000, overlap=100):
    """
    Split text into fixed-size chunks with optional overlap.
    Optimized for performance with minimal memory usage.
    """
    # Pre-calculate total chunks to avoid list resizing
    text_length = len(text)
    total_chunks = (text_length + chunk_size - 1) // chunk_size
    chunks = [None] * total_chunks
    
    # Process chunks
    chunk_idx = 0
    start = 0
    
    while start < text_length:
        # Calculate end position for current chunk
        end = min(start + chunk_size, text_length)
        
        # Extract chunk using string slicing (more efficient than find)
        chunk_text = text[start:end]
        
        # Add chunk to result
        chunks[chunk_idx] = {
            "text": chunk_text,
            "metadata": {
                "start_char": start,
                "end_char": end,
                "strategy": "fixed"
            }
        }
        
        # Move start position for next chunk, considering overlap
        start = end - overlap if end < text_length else text_length
        chunk_idx += 1
    
    return chunks


def page_based_chunking(pdf_path):
    """
    Split PDF by pages, with each page as a separate chunk.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of chunks with metadata
    """
    try:
        chunks = []
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            # Use ThreadPoolExecutor to process pages in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Function to process a single page
                def process_page(page_idx):
                    page = reader.pages[page_idx]
                    text = page.extract_text()
                    return {
                        "text": text,
                        "metadata": {"page": page_idx + 1, "strategy": "page"},
                    }

                # Process all pages in parallel
                chunks = list(executor.map(process_page, range(len(reader.pages))))

        return chunks
    except Exception as e:
        logger.error(f"Error in page-based chunking: {str(e)}")
        raise


def paragraph_chunking(text):
    """
    Split text by paragraphs.
    Optimized for performance with efficient string operations.
    """
    # Use a more efficient paragraph splitting approach
    paragraphs = []
    current_para = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if line:
            current_para.append(line)
        elif current_para:
            paragraphs.append(' '.join(current_para))
            current_para = []
    
    # Add the last paragraph if exists
    if current_para:
        paragraphs.append(' '.join(current_para))
    
    # Pre-allocate chunks list
    chunks = [None] * len(paragraphs)
    current_position = 0
    
    for i, paragraph in enumerate(paragraphs):
        # Use string slicing for position tracking
        start_position = current_position
        end_position = start_position + len(paragraph)
        
        chunks[i] = {
            "text": paragraph,
            "metadata": {
                "start_char": start_position,
                "end_char": end_position,
                "strategy": "paragraph"
            }
        }
        
        current_position = end_position + 2  # +2 for the "\n\n" separator
    
    return chunks


def heading_chunking(text):
    """
    Split text by headings (identified by heuristics).
    Optimized for performance with compiled regex patterns.
    """
    import re
    
    # Compile regex patterns once
    heading_patterns = [
        re.compile(r"^#{1,6}\s+.+$"),  # Markdown headings
        re.compile(r"^[A-Z][A-Za-z\s]+$"),  # All caps or title case single line
        re.compile(r"^\d+\.\s+[A-Z]"),  # Numbered headings (1. Title)
        re.compile(r"^[IVXLCDMivxlcdm]+\.\s+[A-Z]")  # Roman numeral headings (IV. Title)
    ]
    
    # Split by lines and process in one pass
    lines = text.split("\n")
    chunks = []
    current_heading = "Introduction"
    current_text = []
    current_start = 0
    
    # Pre-allocate chunks list with estimated size
    estimated_chunks = len(lines) // 10  # Rough estimate: 10 lines per chunk
    chunks = [None] * estimated_chunks
    chunk_idx = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line matches any heading pattern
        is_heading = any(pattern.match(line) for pattern in heading_patterns)
        
        if is_heading:
            # Save current chunk if exists
            if current_text:
                chunk_text = "\n".join(current_text)
                if chunk_idx >= len(chunks):
                    chunks.append(None)  # Extend list if needed
                chunks[chunk_idx] = {
                    "text": chunk_text,
                    "metadata": {
                        "heading": current_heading,
                        "start_char": current_start,
                        "end_char": current_start + len(chunk_text),
                        "strategy": "heading"
                    }
                }
                chunk_idx += 1
            
            # Start new chunk
            current_heading = line
            current_text = []
            current_start = text.find(line, current_start)
        else:
            current_text.append(line)
    
    # Add the last chunk
    if current_text:
        chunk_text = "\n".join(current_text)
        if chunk_idx >= len(chunks):
            chunks.append(None)
        chunks[chunk_idx] = {
            "text": chunk_text,
            "metadata": {
                "heading": current_heading,
                "start_char": current_start,
                "end_char": current_start + len(chunk_text),
                "strategy": "heading"
            }
        }
        chunk_idx += 1
    
    # Trim the list to actual size
    return chunks[:chunk_idx]


def semantic_chunking(text):
    """
    Use OpenAI to identify semantic chunks in the text.

    Args:
        text: The text to chunk

    Returns:
        List of chunks with metadata
    """
    # Use the optimized semantic chunking with Structured Outputs
    return semantic_chunk_with_structured_output(text)
