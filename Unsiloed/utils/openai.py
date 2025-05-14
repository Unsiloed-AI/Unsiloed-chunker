import os
import base64
import json
from typing import List, Dict, Any
from openai import OpenAI
import logging
import concurrent.futures
import PyPDF2
from dotenv import load_dotenv
import numpy as np
import cv2
import re
from Unsiloed.utils.chunking import (
    check_memory_usage,
    log_memory_usage,
    adjust_batch_size,
    INITIAL_BATCH_SIZE,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

# Instead of initializing at import, create a function to get the client
client = None


def get_openai_client():
    """Get an OpenAI client with proper configuration"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        logger.debug("Attempting to create OpenAI client...")

        # Create client with explicit parameters
        client = OpenAI(api_key=api_key, timeout=60.0, max_retries=3)
        logger.debug(
            "OpenAI client created, now testing..."
        )  # Log after client creation

        # Test the client by listing available models
        models = client.models.list()
        if models and hasattr(models, "data") and len(models.data) > 0:
            logger.debug(
                f"OpenAI client initialized successfully, available models: {len(models.data)}"
            )
            return client
        else:
            logger.error("OpenAI client initialized but returned no models.")
            return None

    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {str(e)}")
        return None


def encode_image_to_base64(image_path):
    """
    Encode an image to base64.

    Args:
        image_path: Path to the image file or numpy array

    Returns:
        Base64 encoded string of the image
    """
    logger.debug("Encoding image to base64")

    # Handle numpy array (from CV2)
    if isinstance(image_path, np.ndarray):
        success, buffer = cv2.imencode(".jpg", image_path)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer).decode("utf-8")

    # Handle file path
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_extraction_prompt(schema: Dict[str, Any], page_count: int) -> str:
    """
    Create a prompt instructing the model how to extract data according to the schema.

    Args:
        schema: JSON schema defining the structure
        page_count: Number of pages in the document

    Returns:
        Prompt string for the model
    """
    # Create a compact JSON representation of the schema
    schema_str = json.dumps(schema, indent=2)

    prompt = f"""
    You are an expert at extracting structured data from documents.
    
    I have a document with {page_count} pages that has been converted to images. I need you to extract specific information from these images according to the following JSON schema:
    
    {schema_str}
    
    Please follow these instructions carefully:
    
    1. Examine all {page_count} images thoroughly to find the requested information.
    2. Extract the exact text from the document that matches each field in the schema.
    3. If you cannot find information for a specific field in any of the pages, return an empty string or null value for that field.
    4. For array fields, include all instances found throughout the document.
    5. Maintain the structure defined in the schema exactly.
    6. Return only the extracted data as a valid JSON object, matching the structure of the schema.
    7. Do not add any explanatory text or notes outside the JSON structure.
    8. Be precise and accurate in your extraction.
    9. If text is unclear or ambiguous, make your best guess based on context.
    10. For dates, numbers, and other formatted data, maintain the format as shown in the document.
    11. IMPORTANT: Your response MUST be a valid JSON object that exactly matches the structure of the provided schema.
    12. IMPORTANT: Do not include any explanations, just return the JSON object.
    
    Your response should be a valid JSON object containing only the extracted data.
    """

    return prompt


def semantic_chunk_with_structured_output(text: str) -> List[Dict[str, Any]]:
    """
    Use OpenAI API to create semantic chunks from text using JSON mode with optimized performance.
    Falls back to local semantic chunking for better performance.

    Args:
        text: The text to chunk

    Returns:
        List of chunks with metadata
    """
    # Optimize text length threshold and chunk size
    MAX_DIRECT_TEXT_LENGTH = 15000  # Reduced from 25000 for better performance
    CHUNK_SIZE = 15000  # Optimized chunk size
    OVERLAP = 300  # Reduced overlap for better performance

    # If text is too long, split it first using a simpler method
    if len(text) > MAX_DIRECT_TEXT_LENGTH:
        logger.info("Text too long for direct semantic chunking, applying parallel processing")
        return process_long_text_semantically(text, CHUNK_SIZE, OVERLAP)

    try:
        # First try local semantic chunking for better performance
        local_chunks = local_semantic_chunking(text)
        if local_chunks:
            return local_chunks

        # Fall back to OpenAI if local chunking fails
        openai_client = get_openai_client()
        if not openai_client:
            return paragraph_chunking(text)  # Fallback to paragraph chunking if OpenAI is not available

        # Optimize the prompt for faster processing
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing and dividing text into meaningful semantic chunks. Focus on natural breaks in the content. Your output must be valid JSON.",
                },
                {
                    "role": "user",
                    "content": f"""Analyze this text and divide it into logical semantic chunks. Each chunk should be a cohesive unit.
                    Return JSON with this structure:
                    {{
                        "chunks": [
                            {{
                                "text": "chunk text",
                                "title": "descriptive title",
                                "position": "beginning/middle/end"
                            }}
                        ]
                    }}
                    
                    Text: {text}""",
                },
            ],
            max_tokens=4000,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        chunks = []
        current_position = 0

        for i, chunk_data in enumerate(result.get("chunks", [])):
            chunk_text = chunk_data.get("text", "")
            start_position = text.find(chunk_text, current_position)
            if start_position == -1:
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
                    },
            })
            current_position = end_position

        return chunks

    except Exception as e:
        logger.error(f"Error in semantic chunking: {str(e)}")
        return local_semantic_chunking(text)  # Fallback to local semantic chunking


def process_long_text_semantically(text: str, chunk_size: int = 15000, overlap: int = 300) -> List[Dict[str, Any]]:
    """
    Process a long text by breaking it into smaller pieces and chunking each piece semantically.
    Uses optimized parallel processing for better performance.

    Args:
        text: The long text to process
        chunk_size: Size of each text chunk
        overlap: Overlap between chunks

    Returns:
        List of semantic chunks
    """
    # Create optimized chunks
    text_chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        text_chunks.append(text[start:end])
        start = end - overlap if end < text_length else text_length

    # Process chunks in parallel with optimized worker count
    all_semantic_chunks = []
    max_workers = min(8, len(text_chunks))  # Limit parallel workers
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(semantic_chunk_with_structured_output, chunk) for chunk in text_chunks]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                chunks = future.result()
                all_semantic_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing semantic chunk: {str(e)}")
                continue

    return all_semantic_chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file with optimized performance.
    Uses parallel processing and improved memory management.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text from the PDF
    """
    try:
        log_memory_usage("pdf_extraction_start")
        
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)

            # Function to extract text from a page with improved error handling
            def extract_page_text(page_idx):
                try:
                    # Check memory usage before processing each page
                    if not check_memory_usage():
                        logger.warning(f"Memory usage high before processing page {page_idx}")
                        return ""
                        
                    page = reader.pages[page_idx]
                    # Use a more efficient text extraction method with layout preservation
                    text = page.extract_text(extraction_mode="layout") or ""
                    return text.strip()
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_idx}: {str(e)}")
                    return ""

            # For small PDFs, use sequential processing
            if total_pages <= 3:
                return "\n\n".join(extract_page_text(i) for i in range(total_pages))

            # For larger PDFs, use parallel processing with optimized chunking
            optimal_chunk_size = min(10, max(1, total_pages // 4))
            max_workers = min(8, total_pages)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create chunks of pages for better memory management
                page_chunks = [range(i, min(i + optimal_chunk_size, total_pages)) 
                             for i in range(0, total_pages, optimal_chunk_size)]
                
                # Process each chunk of pages
                chunk_results = []
                for chunk in page_chunks:
                    # Check memory usage before processing each chunk
                    if not check_memory_usage():
                        logger.warning("Memory usage high before processing chunk")
                        # Reduce chunk size if memory is low
                        optimal_chunk_size = max(1, optimal_chunk_size // 2)
                        continue
                        
                    futures = [executor.submit(extract_page_text, page_idx) for page_idx in chunk]
                    chunk_texts = [f.result() for f in concurrent.futures.as_completed(futures)]
                    chunk_results.extend(chunk_texts)
                    
                    # Log memory usage after each chunk
                    log_memory_usage(f"pdf_chunk_{chunk[0]}-{chunk[-1]}")

            # Join results with minimal memory overhead
            result = "\n\n".join(filter(None, chunk_results))
            log_memory_usage("pdf_extraction_end")
            return result

    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise


def extract_text_from_docx(docx_path: str) -> str:
    """
    Extract text from a DOCX file.

    Args:
        docx_path: Path to the DOCX file

    Returns:
        Extracted text from the DOCX
    """
    try:
        import docx  # python-docx package

        doc = docx.Document(docx_path)
        full_text = []

        # Extract text from paragraphs
        for para in doc.paragraphs:
            full_text.append(para.text)

        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    full_text.append(cell.text)

        return "\n\n".join(full_text)
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise


def extract_text_from_pptx(pptx_path: str) -> str:
    """
    Extract text from a PPTX file.

    Args:
        pptx_path: Path to the PPTX file

    Returns:
        Extracted text from the PPTX
    """
    try:
        from pptx import Presentation  # python-pptx package

        presentation = Presentation(pptx_path)
        full_text = []

        # Loop through slides
        for slide_number, slide in enumerate(presentation.slides, 1):
            slide_text = []
            slide_text.append(f"Slide {slide_number}")

            # Extract text from shapes (including text boxes)
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_text.append(shape.text)

                # Extract text from tables
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = []
                        for cell in row.cells:
                            if cell.text:
                                row_text.append(cell.text)
                        if row_text:
                            slide_text.append(" | ".join(row_text))

            full_text.append("\n".join(slide_text))

        return "\n\n".join(full_text)
    except Exception as e:
        logger.error(f"Error extracting text from PPTX: {str(e)}")
        raise


def local_semantic_chunking(text: str) -> List[Dict[str, Any]]:
    """
    Perform semantic chunking locally without using OpenAI API.
    Uses heuristics to identify semantic boundaries.

    Args:
        text: The text to chunk

    Returns:
        List of chunks with metadata
    """
    # Define semantic boundary patterns
    boundary_patterns = [
        r"\n\s*\n",  # Double newline
        r"[.!?]\s+(?=[A-Z])",  # Sentence end followed by capital letter
        r"\n(?=[A-Z][a-z])",  # Newline followed by title case
        r"\n(?=\d+\.\s)",  # Newline followed by numbered list
        r"\n(?=[IVXLCDM]+\.\s)",  # Newline followed by roman numeral
    ]
    
    # Combine patterns
    boundary_regex = re.compile("|".join(f"({pattern})" for pattern in boundary_patterns))
    
    # Split text into potential chunks
    potential_chunks = boundary_regex.split(text)
    chunks = []
    current_position = 0
    
    # Process each potential chunk
    for chunk in potential_chunks:
        if not chunk or chunk.isspace():
            continue
            
        # Clean up the chunk
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # Find position in original text
        start_position = text.find(chunk, current_position)
        if start_position == -1:
            start_position = current_position
            
        end_position = start_position + len(chunk)
        
        # Only create chunk if it's not too small
        if len(chunk) >= 50:  # Minimum chunk size
            chunks.append({
                "text": chunk,
                "metadata": {
                    "title": f"Chunk {len(chunks) + 1}",
                    "position": "middle" if len(chunks) > 0 and len(chunks) < len(potential_chunks) - 1 else "beginning" if len(chunks) == 0 else "end",
                    "start_char": start_position,
                    "end_char": end_position,
                    "strategy": "semantic",
                },
            })
            
        current_position = end_position
    
    return chunks
