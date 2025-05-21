import json
import concurrent.futures
import logging
import re
from typing import List, Dict, Any
import asyncio
from .optimized_openai import chat_completion_with_cache

logger = logging.getLogger(__name__)

async def semantic_chunk_optimized(text: str) -> List[Dict[str, Any]]:
    """
    Chunk text semantically using OpenAI with optimized client and caching.
    Better performance for identical or similar documents.
    
    Args:
        text: The text to chunk
        
    Returns:
        List of semantic chunks with metadata
    """
    # For shorter texts, process in a single request
    if len(text) <= 25000:
        return await _process_text_segment(text)
    
    # For longer texts, split and process in parallel
    return await process_long_text_semantically_optimized(text)


async def _process_text_segment(text: str) -> List[Dict[str, Any]]:
    """
    Process a single text segment with semantic chunking.
    
    Args:
        text: The text segment to process
        
    Returns:
        List of semantic chunks with metadata
    """
    try:
        # Use the cached version of chat completion
        response = chat_completion_with_cache(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing and dividing text into meaningful semantic chunks. Your output should be valid JSON.",
                },
                {
                    "role": "user",
                    "content": f"""Please analyze the following text and divide it into logical semantic chunks. 
                    Each chunk should represent a cohesive unit of information or a distinct section.
                    
                    Return your results as a JSON object with this structure:
                    {{
                        "chunks": [
                            {{
                                "text": "the text of the chunk",
                                "title": "a descriptive title for this chunk",
                                "summary": "a brief 1-2 sentence summary of the key information",
                                "position": "beginning/middle/end"
                            }},
                            ...
                        ]
                    }}
                    
                    Text to chunk:
                    
                    {text}""",
                },
            ],
            max_tokens=4000,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        # Parse the response
        result = json.loads(response.choices[0].message.content)

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

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "title": chunk_data.get("title", f"Chunk {i + 1}"),
                        "summary": chunk_data.get("summary", ""),
                        "position": chunk_data.get("position", "unknown"),
                        "start_char": start_position,
                        "end_char": end_position,
                        "strategy": "semantic",
                    },
                }
            )

            current_position = end_position

        return chunks
    except Exception as e:
        logger.error(f"Error in semantic chunking: {str(e)}")
        # Instead of returning a single chunk, fallback to paragraph chunking
        logger.info("Falling back to paragraph chunking due to semantic chunking failure")
        return paragraph_chunking_optimized(text)


async def process_long_text_semantically_optimized(text: str) -> List[Dict[str, Any]]:
    """
    Process a long text by breaking it into smaller pieces and chunking each piece semantically.
    Uses more efficient concurrent processing and improved caching.
    
    Args:
        text: The long text to process
        
    Returns:
        List of semantic chunks
    """
    # Create chunks of 25000 characters with 500 character overlap
    text_chunks = []
    chunk_size = 25000
    overlap = 500
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        text_chunks.append(text[start:end])
        start = end - overlap if end < text_length else text_length

    # Process chunks concurrently with asyncio for better control
    tasks = [_process_text_segment(chunk) for chunk in text_chunks]
    chunk_results = await asyncio.gather(*tasks)
    
    # Flatten results and adjust positions
    all_chunks = []
    position_offset = 0
    
    for i, chunks in enumerate(chunk_results):
        # Skip empty results
        if not chunks:
            continue
            
        # Adjust character positions for all but the first chunk
        if i > 0:
            overlap_size = overlap if i < len(chunk_results) - 1 else 0
            for chunk in chunks:
                # Only add if it's not mostly duplicate content from overlap
                if chunk["metadata"]["start_char"] >= overlap_size:
                    chunk["metadata"]["start_char"] += position_offset - overlap_size
                    chunk["metadata"]["end_char"] += position_offset - overlap_size
                    all_chunks.append(chunk)
        else:
            # Add all chunks from the first segment
            all_chunks.extend(chunks)
            
        # Update position offset for next chunk
        chunk_text = text_chunks[i]
        position_offset += len(chunk_text) - (overlap if i < len(chunk_results) - 1 else 0)

    return all_chunks

def fixed_size_chunking_optimized(text: str, chunk_size=1000, overlap=100) -> List[Dict[str, Any]]:
    """
    Split text into fixed-size chunks with optional overlap.
    Optimized version that tries to break at natural boundaries.
    
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
        
        # Try to find a natural breaking point (period, question mark, or exclamation followed by space)
        if end < text_length:
            # Look for the last sentence break within the last 20% of the chunk
            search_start = max(start, end - int(chunk_size * 0.2))
            
            # Find last sentence break in this range
            last_break = -1
            for match in re.finditer(r'[.!?]\s+', text[search_start:end]):
                last_break = search_start + match.end()
            
            # If we found a good breaking point, use it
            if last_break != -1:
                end = last_break
        
        # Extract chunk
        chunk_text = text[start:end].strip()
        
        # Add chunk to result
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "start_char": start,
                    "end_char": end,
                    "strategy": "fixed",
                    "title": f"Chunk {len(chunks) + 1}",
                    "size": len(chunk_text)
                }
            })
        
        # Move start position for next chunk, considering overlap
        start = end - overlap if end < text_length else text_length
    
    return chunks


def paragraph_chunking_optimized(text: str) -> List[Dict[str, Any]]:
    """
    Split text by paragraphs with optimized processing.
    Creates meaningful, appropriately sized chunks for better performance.
    
    Args:
        text: The text to chunk
        
    Returns:
        List of chunks with metadata
    """
    # Normalize text - removes excessive whitespace and normalizes line endings
    text = re.sub(r'\r\n', '\n', text)  # Convert Windows to Unix line endings
    text = re.sub(r'\n{3,}', '\n\n', text)  # Convert multiple newlines to double newlines
    
    # Split text by double newlines to identify paragraphs
    paragraphs = re.split(r'\n\n+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # If there are too many small paragraphs, combine them into larger chunks
    max_chunk_size = 5000  # Target max chunk size
    min_chunk_size = 1000  # Minimum size for a paragraph to stand alone
    
    chunks = []
    current_position = 0
    current_chunk = []
    current_chunk_size = 0
    
    for paragraph in paragraphs:
        para_size = len(paragraph)
        
        # Find position in the original text
        start_position = text.find(paragraph, current_position)
        if start_position == -1:
            start_position = current_position
            
        current_position = start_position + para_size
        
        # If this paragraph is substantial on its own, or adding it would exceed our target size,
        # finish the current chunk and start a new one
        if (para_size >= min_chunk_size or current_chunk_size + para_size > max_chunk_size) and current_chunk:
            # Combine current chunk and add it to the results
            chunk_text = "\n\n".join(current_chunk)
            chunk_start = text.find(current_chunk[0], 0)
            chunk_end = chunk_start + len(chunk_text)
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "start_char": chunk_start,
                    "end_char": chunk_end,
                    "strategy": "paragraph",
                    "title": f"Section {len(chunks) + 1}",
                    "summary": f"Contains {len(current_chunk)} paragraphs"
                }
            })
            
            # Start a new chunk
            current_chunk = []
            current_chunk_size = 0
        
        # Add current paragraph to the chunk
        current_chunk.append(paragraph)
        current_chunk_size += para_size
    
    # Add the last chunk if there's anything left
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunk_start = text.find(current_chunk[0], 0)
        chunk_end = chunk_start + len(chunk_text)
        
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "start_char": chunk_start,
                "end_char": chunk_end,
                "strategy": "paragraph",
                "title": f"Section {len(chunks) + 1}",
                "summary": f"Contains {len(current_chunk)} paragraphs"
            }
        })
    
    # If we still have no chunks (maybe the text was empty or had no paragraphs),
    # then create chunks of maximum 5000 characters
    if not chunks and text.strip():
        return fixed_size_chunking_optimized(text, 5000, 100)
        
    return chunks

def heading_chunking_optimized(text: str) -> List[Dict[str, Any]]:
    """
    Optimized version of heading-based chunking with better regex performance
    and memory efficiency.
    
    Args:
        text: The text to chunk
        
    Returns:
        List of chunks with metadata
    """
    if not text.strip():
        return []
    
    # Precompile regex patterns for better performance
    heading_pattern = re.compile(
        r'^(#+|\d+\.+|\w+\:|\*\*|\[.+\])\s+(.+?)$', 
        re.MULTILINE
    )
    
    # Find all headings in the text
    headings = [
        (match.start(), match.group())
        for match in heading_pattern.finditer(text)
    ]
    
    if not headings:
        # If no headings found, treat the entire text as one chunk
        return [{
            "text": text,
            "metadata": {
                "start_char": 0,
                "end_char": len(text),
                "strategy": "heading",
                "heading": None,
            }
        }]
    
    # Process the text in chunks based on heading positions
    chunks = []
    for i, (start_pos, heading) in enumerate(headings):
        # Get the end position (start of next heading or end of text)
        end_pos = headings[i+1][0] if i < len(headings) - 1 else len(text)
        
        # Get the chunk text from heading to next heading
        chunk_text = text[start_pos:end_pos]
        
        # Add the chunk with metadata
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "start_char": start_pos,
                "end_char": end_pos,
                "strategy": "heading",
                "heading": heading,
            }
        })
    
    return chunks
