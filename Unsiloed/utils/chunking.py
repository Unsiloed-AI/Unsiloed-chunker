import concurrent.futures
from typing import Literal, List, Dict, Any
import logging
import PyPDF2
from Unsiloed.utils.openai import (
    semantic_chunk_with_structured_output,
)
from Unsiloed.utils.text_utils import paragraph_chunking

logger = logging.getLogger(__name__)

ChunkingStrategy = Literal["fixed", "page", "semantic", "paragraph", "heading"]


def fixed_size_chunking(text, chunk_size=1000, overlap=100):
    """
    Split text into fixed-size chunks with optional overlap.

    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of chunks with metadata
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position for current chunk
        end = min(start + chunk_size, text_length)

        # Extract chunk
        chunk_text = text[start:end]

        # Add chunk to result
        chunks.append(
            {
                "text": chunk_text,
                "metadata": {"start_char": start, "end_char": end, "strategy": "fixed"},
            }
        )

        # Move start position for next chunk, considering overlap
        start = end - overlap if end < text_length else text_length

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


def heading_chunking(text):
    """
    Split text by headings (identified by heuristics).

    Args:
        text: The text to chunk

    Returns:
        List of chunks with metadata
    """
    import re

    # Define patterns for common heading formats
    heading_patterns = [
        r"^#{1,6}\s+.+$",  # Markdown headings
        r"^[A-Z][A-Za-z\s]+$",  # All caps or title case single line
        r"^\d+\.\s+[A-Z]",  # Numbered headings (1. Title)
        r"^[IVXLCDMivxlcdm]+\.\s+[A-Z]",  # Roman numeral headings (IV. Title)
    ]

    # Combine patterns
    combined_pattern = "|".join(f"({pattern})" for pattern in heading_patterns)

    # Split by lines first
    lines = text.split("\n")

    chunks = []
    current_heading = "Introduction"
    current_text = []
    current_start = 0

    for line in lines:
        if re.match(combined_pattern, line.strip()):
            # If we have accumulated text, save it as a chunk
            if current_text:
                chunk_text = "\n".join(current_text)
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            "heading": current_heading,
                            "start_char": current_start,
                            "end_char": current_start + len(chunk_text),
                            "strategy": "heading",
                        },
                    }
                )

            # Start a new chunk with this heading
            current_heading = line.strip()
            current_text = []
            current_start = text.find(line, current_start)
        else:
            current_text.append(line)

    # Add the last chunk
    if current_text:
        chunk_text = "\n".join(current_text)
        chunks.append(
            {
                "text": chunk_text,
                "metadata": {
                    "heading": current_heading,
                    "start_char": current_start,
                    "end_char": current_start + len(chunk_text),
                    "strategy": "heading",
                },
            }
        )

    return chunks


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
