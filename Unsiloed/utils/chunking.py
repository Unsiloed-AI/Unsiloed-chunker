import concurrent.futures
from typing import Literal
import logging
import PyPDF2
from Unsiloed.utils.openai import (
    semantic_chunk_with_structured_output,
)
from Unsiloed.text_cleaning.cleaning_pipeline import TextCleaningPipeline

logger = logging.getLogger(__name__)

ChunkingStrategy = Literal["fixed", "page", "semantic", "paragraph", "heading"]

cleaner = TextCleaningPipeline()

def fixed_size_chunking(text, chunk_size=1000, overlap=100):
    """
    Split text into fixed-size chunks with optional overlap.
    """
    text = cleaner.clean(text)

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end]

        chunks.append(
            {
                "text": chunk_text,
                "metadata": {"start_char": start, "end_char": end, "strategy": "fixed"},
            }
        )

        start = end - overlap if end < text_length else text_length

    return chunks


def page_based_chunking(pdf_path):
    """
    Split PDF by pages.
    """
    try:
        chunks = []
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                def process_page(page_idx):
                    page = reader.pages[page_idx]
                    text = page.extract_text()
                    return {
                        "text": text,
                        "metadata": {"page": page_idx + 1, "strategy": "page"},
                    }

                chunks = list(executor.map(process_page, range(len(reader.pages))))

        return chunks
    except Exception as e:
        logger.error(f"Error in page-based chunking: {str(e)}")
        raise


def paragraph_chunking(text):
    """
    Split text by paragraphs.
    """
    text = cleaner.clean(text)

    paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    chunks = []
    current_position = 0

    for paragraph in paragraphs:
        start_position = text.find(paragraph, current_position)
        end_position = start_position + len(paragraph)

        chunks.append(
            {
                "text": paragraph,
                "metadata": {
                    "start_char": start_position,
                    "end_char": end_position,
                    "strategy": "paragraph",
                },
            }
        )

        current_position = end_position

    return chunks


def heading_chunking(text):
    """
    Split text by headings.
    """
    import re
    text = cleaner.clean(text)

    heading_patterns = [
        r"^#{1,6}\s+.+$",
        r"^[A-Z][A-Za-z\s]+$",
        r"^\d+\.\s+[A-Z]",
        r"^[IVXLCDMivxlcdm]+\.\s+[A-Z]",
    ]

    combined_pattern = "|".join(f"({pattern})" for pattern in heading_patterns)
    lines = text.split("\n")

    chunks = []
    current_heading = "Introduction"
    current_text = []
    current_start = 0

    for line in lines:
        if re.match(combined_pattern, line.strip()):
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

            current_heading = line.strip()
            current_text = []
            current_start = text.find(line, current_start)
        else:
            current_text.append(line)

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
    Use OpenAI to identify semantic chunks.
    """
    text = cleaner.clean(text)
    return semantic_chunk_with_structured_output(text)
