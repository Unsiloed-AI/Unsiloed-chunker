import concurrent.futures
from typing import Literal, Dict, Any, List
import logging
import PyPDF2
import json
from Unsiloed.utils.model_providers import get_model_provider

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


def paragraph_chunking(text):
    """
    Split text by paragraphs.

    Args:
        text: The text to chunk

    Returns:
        List of chunks with metadata
    """
    # Split text by double newlines to identify paragraphs
    paragraphs = text.split("\n\n")

    # Remove empty paragraphs
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


def semantic_chunking(text: str, provider_type: str = "openai", **provider_kwargs) -> List[Dict[str, Any]]:
    """
    Use LLM to identify semantic chunks in the text.

    Args:
        text: The text to chunk
        provider_type: Type of model provider to use (openai, anthropic, huggingface)
        provider_kwargs: Additional arguments for the model provider

    Returns:
        List of chunks with metadata
    """
    try:
        # Get the model provider
        provider = get_model_provider(provider_type, **provider_kwargs)
        
        # Create a prompt for semantic chunking
        system_prompt = """You are an expert at analyzing and dividing text into meaningful semantic chunks. 
        Your output should be valid JSON."""
        
        prompt = f"""Please analyze the following text and divide it into logical semantic chunks. 
        Each chunk should represent a cohesive unit of information or a distinct section.
        
        Return your results as a JSON object with this structure:
        {{
            "chunks": [
                {{
                    "text": "the text of the chunk",
                    "title": "a descriptive title for this chunk",
                    "position": "beginning/middle/end"
                }},
                ...
            ]
        }}
        
        Text to chunk:
        
        {text}"""
        
        # Get structured completion from the model
        result = provider.get_structured_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=4000,
            temperature=0.1
        )
        
        # Parse the response
        result = json.loads(result)
        
        # Convert the response to our standard chunk format
        chunks = []
        current_position = 0
        
        for i, chunk_data in enumerate(result.get("chunks", [])):
            chunk_text = chunk_data.get("text", "")
            # Find the chunk in the original text to get accurate character positions
            start_position = text.find(chunk_text, current_position)
            if start_position == -1:
                # If exact match not found, use approximate position
                start_position = current_position
                
            end_position = start_position + len(chunk_text)
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "title": chunk_data.get("title", f"Chunk {i + 1}"),
                    "position": chunk_data.get("position", "unknown"),
                    "start_char": start_position,
                    "end_char": end_position,
                    "strategy": "semantic",
                }
            })
            
            current_position = end_position
            
        return chunks
        
    except Exception as e:
        logger.error(f"Error in semantic chunking: {str(e)}")
        # Fall back to paragraph chunking if semantic chunking fails
        logger.info("Falling back to paragraph chunking")
        return paragraph_chunking(text)
