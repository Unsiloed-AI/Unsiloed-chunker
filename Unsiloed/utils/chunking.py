"""
Advanced Document Chunking Utilities

This module provides comprehensive document chunking capabilities with multiple strategies:
- Fixed-size chunking with configurable overlap
- Page-based chunking for PDF documents
- Paragraph-based chunking using text structure
- Heading-based chunking using pattern recognition
- Semantic chunking using YOLO + OpenAI for intelligent document understanding

Key Features:
- Parallel processing with semaphore-controlled concurrency
- OpenAI integration for semantic boundary detection
- YOLO-based document layout analysis
- Configurable parameters through ChunkingConfig class
- Comprehensive error handling and validation
- Production-ready logging and monitoring

Dependencies:
- OpenAI API for semantic analysis
- YOLO model for document layout detection
- Tesseract OCR for text extraction
- PIL for image processing
- PyPDF2 for PDF handling

Author: Unsiloed Team
Version: 2.1.0
"""

import javatools
from unstructured.partition.auto import partition
import fitz  # PyMuPDF
import filetype
import pytesseract
from numba import njit
from docx import Document
import random
import xgboost as xgb
import dask.dataframe as dd
from dask.distributed import LocalCluster
from multiprocessing import Pool
import asyncio
import concurrent.futures
import time
import os
import json
import re
import numpy as np
from typing import List, Literal, Union, Dict, Any
from PIL import Image
import logging
import PyPDF2
from Unsiloed.utils.extractionutils import _extract_image_with_openai_async, _extract_table_with_openai_async, _extract_text_with_ocr
from Unsiloed.utils.fileutils import pdf_to_images
from Unsiloed.utils.openai import (
    semantic_chunk_with_structured_output,
    get_openai_client
)
from Unsiloed.utils.yolo_model_utils import run_yolo_inference

logger = logging.getLogger(__name__)

# Configuration class for centralized settings
class ChunkingConfig:
    """Centralized configuration for chunking operations."""
    
    DEFAULT_OPENAI_CONFIDENCE_THRESHOLD = 0.7
    OPENAI_MODEL = "gpt-4o"
    OPENAI_MAX_TOKENS = 300
    OPENAI_TEMPERATURE = 0.1
    OPENAI_TIMEOUT = 60.0
    OPENAI_MAX_RETRIES = 2
    
    DEFAULT_MAX_CONCURRENT_CALLS = 15
    DEFAULT_EXTRACTION_CONCURRENT_CALLS = 5
    DEFAULT_GROUPING_CONCURRENT_CALLS = 5
    
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_OVERLAP = 100
    MAX_TEXT_PREVIEW_LENGTH = 80
    LARGE_TEXT_THRESHOLD = 25000
    
    READING_ORDER_TOLERANCE = 0.02
    
    MAX_ELEMENTS_PER_GROUP = 5
    HEURISTIC_CONFIDENCE = 0.8

DEFAULT_OPENAI_CONFIDENCE_THRESHOLD = ChunkingConfig.DEFAULT_OPENAI_CONFIDENCE_THRESHOLD

ChunkingStrategy = Literal["fixed", "page", "semantic", "paragraph", "heading"]

# Custom Exceptions for better error handling
class ChunkingError(Exception):
    """Base exception for chunking operations."""
    pass

class OpenAIServiceError(ChunkingError):
    """Exception raised when OpenAI service is unavailable or fails."""
    pass

class YOLOProcessingError(ChunkingError):
    """Exception raised when YOLO processing fails."""
    pass

class DocumentProcessingError(ChunkingError):
    """Exception raised when document processing fails."""
    pass

class InvalidConfigurationError(ChunkingError):
    """Exception raised when configuration is invalid."""
    pass


def validate_chunk_size(chunk_size: int) -> int:
    """Validate and return chunk size parameter."""
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise InvalidConfigurationError(f"chunk_size must be a positive integer, got {chunk_size}")
    return chunk_size

def validate_overlap(overlap: int, chunk_size: int) -> int:
    """Validate and return overlap parameter."""
    if not isinstance(overlap, int) or overlap < 0:
        raise InvalidConfigurationError(f"overlap must be a non-negative integer, got {overlap}")
    if overlap >= chunk_size:
        raise InvalidConfigurationError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    return overlap

def validate_file_path(file_path: str) -> str:
    """Validate file path exists and is readable."""
    if not isinstance(file_path, str):
        raise InvalidConfigurationError(f"file_path must be a string, got {type(file_path)}")
    if not os.path.exists(file_path):
        raise DocumentProcessingError(f"File does not exist: {file_path}")
    if not os.path.isfile(file_path):
        raise DocumentProcessingError(f"Path is not a file: {file_path}")
    return file_path

class OpenAISemanticBoundaryDetector:
    """
    OpenAI-based semantic boundary detector that processes text sequentially
    and determines where to insert semantic boundaries.
    """
    
    def __init__(self, confidence_threshold: float = None, debug: bool = False):
        """
        Initialize the OpenAI-based boundary detector.
        
        Args:
            confidence_threshold: Confidence threshold for boundary detection (0.0 to 1.0)
            debug: Enable debug output
        """
        self.client = get_openai_client()
        if not self.client:
            raise OpenAIServiceError("OpenAI client not available. Please check your API key.")
        
        # Use configuration defaults if not provided
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None 
            else ChunkingConfig.DEFAULT_OPENAI_CONFIDENCE_THRESHOLD
        )
        
        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise InvalidConfigurationError(
                f"Confidence threshold must be between 0.0 and 1.0, got {self.confidence_threshold}"
            )
        
        self.debug = debug
        
        self.system_prompt = """You are an expert document analyzer specialized in semantic boundary detection. Your task is to determine if there should be a semantic boundary (split) between the current accumulated text and the next piece of text.

You will receive:
1. Current accumulated text (what we have so far)
2. Next text element to potentially add
3. Metadata about the next element (type, position, etc.)

Your job is to analyze whether adding the next element would maintain semantic coherence or if it should start a new semantic group.

Respond with a JSON object containing:
{
  "should_split": boolean,
  "confidence": float (0.0 to 1.0),
  "reasoning": "Brief explanation of your decision",
  "boundary_type": "major" | "minor" | "none"
}

Guidelines:
- "major" split: Completely different topics, new sections, different document structure
- "minor" split: Related but distinct subtopics, paragraph breaks, list transitions
- "none": Continuation of the same semantic context
- Consider element types: titles/headers usually start new groups
- Consider content flow: does the next element logically continue the current context?
- Be conservative: prefer keeping related content together"""

    def detect_boundary(self, current_text: str, next_text: str, next_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect if there should be a semantic boundary between current and next text.
        
        Args:
            current_text: Currently accumulated text
            next_text: Next text element to potentially add
            next_metadata: Metadata about the next element
            
        Returns:
            Dictionary with boundary decision and confidence
        """
        try:
            # Prepare the analysis prompt
            user_prompt = f"""Analyze whether to insert a semantic boundary:

CURRENT ACCUMULATED TEXT:
{current_text[:1000]}{"..." if len(current_text) > 1000 else ""}

NEXT TEXT ELEMENT:
{next_text}

NEXT ELEMENT METADATA:
- Element Type: {next_metadata.get('element_type', 'Unknown')}
- Content Type: {next_metadata.get('content_type', 'text')}
- Page Number: {next_metadata.get('page_number', 'Unknown')}
- Reading Order Index: {next_metadata.get('reading_order_index', 'Unknown')}
- Confidence: {next_metadata.get('confidence', 'Unknown')}

Should there be a semantic boundary before adding this next element?"""

            response = self.client.chat.completions.create(
                model=ChunkingConfig.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=ChunkingConfig.OPENAI_MAX_TOKENS,
                temperature=ChunkingConfig.OPENAI_TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate and normalize the response
            should_split = bool(result.get('should_split', False))
            confidence = float(result.get('confidence', 0.0))
            reasoning = str(result.get('reasoning', 'No reasoning provided'))
            boundary_type = str(result.get('boundary_type', 'none'))
            
            # Ensure confidence is within valid range
            confidence = max(0.0, min(1.0, confidence))
            
            if self.debug:
                logger.debug(f"Boundary decision: should_split={should_split}, confidence={confidence:.3f}, type={boundary_type}")
                logger.debug(f"Reasoning: {reasoning}")
            
            return {
                'should_split': should_split,
                'confidence': confidence,
                'reasoning': reasoning,
                'boundary_type': boundary_type,
                'meets_threshold': confidence >= self.confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error in OpenAI boundary detection: {e}")
            # Fallback decision based on heuristics
            return self._fallback_boundary_detection(current_text, next_text, next_metadata)
    
    def _fallback_boundary_detection(self, current_text: str, next_text: str, next_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback boundary detection using heuristics when OpenAI fails.
        """
        element_type = next_metadata.get('element_type', 'Text')
        
        # Heuristic rules
        should_split = False
        confidence = 0.5
        boundary_type = 'none'
        reasoning = 'Fallback heuristic decision'
        
        # Strong indicators for splitting
        if element_type in ['Title', 'Section-header']:
            should_split = True
            confidence = 0.9
            boundary_type = 'major'
            reasoning = 'New heading detected'
        elif element_type == 'Table' and len(current_text) > 0:
            should_split = True
            confidence = 0.8
            boundary_type = 'minor'
            reasoning = 'Table element detected'
        elif element_type in ['Picture', 'Formula'] and len(current_text) > 500:
            should_split = True
            confidence = 0.7
            boundary_type = 'minor'
            reasoning = 'Non-text element with substantial prior content'
        
        return {
            'should_split': should_split,
            'confidence': confidence,
            'reasoning': reasoning,
            'boundary_type': boundary_type,
            'meets_threshold': confidence >= self.confidence_threshold
        }

    async def detect_boundary_async(self, current_text: str, next_text: str, next_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async version of detect_boundary for parallel processing.
        
        Args:
            current_text: Currently accumulated text
            next_text: Next text element to potentially add
            next_metadata: Metadata about the next element
            
        Returns:
            Dictionary with boundary decision and confidence
        """
        try:
            # Create async OpenAI client
            from openai import AsyncOpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not available, using fallback")
                return self._fallback_boundary_detection(current_text, next_text, next_metadata)
            
            async_client = AsyncOpenAI(
                api_key=api_key, 
                timeout=ChunkingConfig.OPENAI_TIMEOUT, 
                max_retries=ChunkingConfig.OPENAI_MAX_RETRIES
            )
            
            # Prepare the analysis prompt
            user_prompt = f"""Analyze whether to insert a semantic boundary:

CURRENT ACCUMULATED TEXT:
{current_text[:1000]}{"..." if len(current_text) > 1000 else ""}

NEXT TEXT ELEMENT:
{next_text}

NEXT ELEMENT METADATA:
- Element Type: {next_metadata.get('element_type', 'Unknown')}
- Content Type: {next_metadata.get('content_type', 'text')}
- Page Number: {next_metadata.get('page_number', 'Unknown')}
- Reading Order Index: {next_metadata.get('reading_order_index', 'Unknown')}
- Confidence: {next_metadata.get('confidence', 'Unknown')}

Should there be a semantic boundary before adding this next element?"""

            response = await async_client.chat.completions.create(
                model=ChunkingConfig.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=ChunkingConfig.OPENAI_MAX_TOKENS,
                temperature=ChunkingConfig.OPENAI_TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate and normalize the response
            should_split = bool(result.get('should_split', False))
            confidence = float(result.get('confidence', 0.0))
            reasoning = str(result.get('reasoning', 'No reasoning provided'))
            boundary_type = str(result.get('boundary_type', 'none'))
            
            # Ensure confidence is within valid range
            confidence = max(0.0, min(1.0, confidence))
            
            if self.debug:
                logger.debug(f"Async boundary decision: should_split={should_split}, confidence={confidence:.3f}, type={boundary_type}")
                logger.debug(f"Reasoning: {reasoning}")
            
            return {
                'should_split': should_split,
                'confidence': confidence,
                'reasoning': reasoning,
                'boundary_type': boundary_type,
                'meets_threshold': confidence >= self.confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error in async OpenAI boundary detection: {e}")
            # Fallback decision based on heuristics
            return self._fallback_boundary_detection(current_text, next_text, next_metadata)


def fixed_size_chunking(text: str, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
    """
    Split text into fixed-size chunks with optional overlap.

    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk in characters. 
                   Defaults to ChunkingConfig.DEFAULT_CHUNK_SIZE
        overlap: Number of characters to overlap between chunks.
                Defaults to ChunkingConfig.DEFAULT_OVERLAP

    Returns:
        List of chunks with metadata
        
    Raises:
        InvalidConfigurationError: When parameters are invalid
    """
    if not isinstance(text, str):
        raise InvalidConfigurationError(f"text must be a string, got {type(text)}")
    
    # Set defaults from configuration
    if chunk_size is None:
        chunk_size = ChunkingConfig.DEFAULT_CHUNK_SIZE
    if overlap is None:
        overlap = ChunkingConfig.DEFAULT_OVERLAP
    
    # Validate parameters
    chunk_size = validate_chunk_size(chunk_size)
    overlap = validate_overlap(overlap, chunk_size)
    
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


def page_based_chunking(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Split PDF by pages, with each page as a separate chunk.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of chunks with metadata
        
    Raises:
        DocumentProcessingError: When PDF processing fails
        InvalidConfigurationError: When file path is invalid
    """
    # Validate file path
    pdf_path = validate_file_path(pdf_path)
    
    if not pdf_path.lower().endswith('.pdf'):
        raise InvalidConfigurationError(f"File must be a PDF, got: {pdf_path}")
    
    try:
        chunks = []
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            
            if len(reader.pages) == 0:
                logger.warning(f"PDF file has no pages: {pdf_path}")
                return []

            # Use ThreadPoolExecutor to process pages in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                def process_page(page_idx):
                    try:
                        page = reader.pages[page_idx]
                        text = page.extract_text()
                        return {
                            "text": text,
                            "metadata": {"page": page_idx + 1, "strategy": "page"},
                        }
                    except Exception as e:
                        logger.error(f"Error processing page {page_idx + 1}: {str(e)}")
                        return {
                            "text": "",
                            "metadata": {"page": page_idx + 1, "strategy": "page", "error": str(e)},
                        }

                # Process all pages in parallel
                chunks = list(executor.map(process_page, range(len(reader.pages))))

        logger.info(f"Successfully processed {len(chunks)} pages from PDF: {pdf_path}")
        return chunks
        
    except Exception as e:
        logger.error(f"Error in page-based chunking for {pdf_path}: {str(e)}")
        raise DocumentProcessingError(f"Failed to process PDF: {str(e)}") from e


def paragraph_chunking(text: str) -> List[Dict[str, Any]]:
    """
    Split text by paragraphs.

    Args:
        text: The text to chunk

    Returns:
        List of chunks with metadata
        
    Raises:
        InvalidConfigurationError: When text parameter is invalid
    """
    if not isinstance(text, str):
        raise InvalidConfigurationError(f"text must be a string, got {type(text)}")
    
    # Split text by double newlines to identify paragraphs
    paragraphs = text.split("\n\n")

    # Remove empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        logger.warning("No paragraphs found in text")
        return []

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

    logger.info(f"Created {len(chunks)} paragraph chunks")
    return chunks


def heading_chunking(text: str) -> List[Dict[str, Any]]:
    """
    Split text by headings (identified by heuristics).

    Args:
        text: The text to chunk

    Returns:
        List of chunks with metadata
        
    Raises:
        InvalidConfigurationError: When text parameter is invalid
    """
    if not isinstance(text, str):
        raise InvalidConfigurationError(f"text must be a string, got {type(text)}")
    
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

    logger.info(f"Created {len(chunks)} heading-based chunks")
    return chunks


def semantic_chunking(text_or_file_path: Union[str, List[Image.Image]], max_concurrent_calls: int = None):
    """
    Advanced semantic chunking using YOLO for page segmentation with PARALLEL OpenAI-based semantic grouping.
    
    This function provides intelligent document chunking by:
    1. Using YOLO for page segmentation to detect text, images, tables, headings
    2. Applying Tesseract OCR for text and headings extraction
    3. Using OpenAI for table and image summarization (with semaphore-controlled parallelism)
    4. Employing OpenAI to determine semantic boundaries during processing
    5. Grouping bbox results based on OpenAI confidence scores
    
    All content extraction and boundary detection operations run in parallel with 
    semaphore control for optimal performance while respecting API rate limits.

    Args:
        text_or_file_path: Either text string (fallback to legacy method) or file path 
                          for document processing, or list of PIL Images
        max_concurrent_calls: Maximum number of concurrent OpenAI API calls. 
                             Defaults to ChunkingConfig.DEFAULT_MAX_CONCURRENT_CALLS

    Returns:
        Dictionary containing:
        - 'chunks': List of semantically grouped chunks with metadata in reading order
        - 'image_dimensions': List of dictionaries with 'width' and 'height' for each page
        
    Raises:
        DocumentProcessingError: When document processing fails
        OpenAIServiceError: When OpenAI service is unavailable
        InvalidConfigurationError: When configuration parameters are invalid
    """
    
    # Validate and set default max_concurrent_calls
    if max_concurrent_calls is None:
        max_concurrent_calls = ChunkingConfig.DEFAULT_MAX_CONCURRENT_CALLS
    elif not isinstance(max_concurrent_calls, int) or max_concurrent_calls <= 0:
        raise InvalidConfigurationError(
            f"max_concurrent_calls must be a positive integer, got {max_concurrent_calls}"
        )
    
    logger.info(f"🚀 Using YOLO-based semantic segmentation with PARALLEL OpenAI calls (max {max_concurrent_calls} concurrent)")
    
    try:
        # For text-only input, fall back to legacy processing
        if isinstance(text_or_file_path, str) and not text_or_file_path.endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            logger.warning("Text input detected, falling back to legacy semantic chunking")
            legacy_chunks = _legacy_semantic_chunking(text_or_file_path)
            return {
                'chunks': legacy_chunks,
                'image_dimensions': []
            }
        
        # Get image dimensions first if processing images/PDF
        if isinstance(text_or_file_path, str):
            images = pdf_to_images(text_or_file_path)
        else:
            images = text_or_file_path
            
        # Capture image dimensions
        image_dimensions = []
        for page_idx, image in enumerate(images):
            width, height = image.size
            image_dimensions.append({
                'page_number': page_idx + 1,
                'width': width,
                'height': height
            })
        
        # Run the async semaphore-controlled version from sync context
        chunks = run_semantic_chunking_with_semaphore(text_or_file_path, max_concurrent_calls)
        
        # Return in the expected dictionary format
        return {
            'chunks': chunks,
            'image_dimensions': image_dimensions
        }
        
    except Exception as e:
        logger.error(f"Error in parallel semantic chunking: {str(e)}")
        # Fallback to text-based processing if it's a string
        if isinstance(text_or_file_path, str) and not text_or_file_path.endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            logger.warning("Falling back to legacy semantic chunking")
            legacy_chunks = _legacy_semantic_chunking(text_or_file_path)
            return {
                'chunks': legacy_chunks,
                'image_dimensions': []
            }
        else:
            raise DocumentProcessingError(f"Failed to process document: {str(e)}") from e

def run_semantic_chunking_with_semaphore(
    text_or_file_path: Union[str, List[Image.Image]], 
    max_concurrent_calls: int = None
):
    """
    Wrapper function to run semantic chunking with semaphore control.
    Handles event loop management and provides fallback mechanisms.
    
    Args:
        text_or_file_path: File path or list of images to process
        max_concurrent_calls: Maximum number of concurrent OpenAI API calls
        
    Returns:
        List of semantic chunks with metadata
        
    Raises:
        DocumentProcessingError: When processing fails
    """
    if max_concurrent_calls is None:
        max_concurrent_calls = ChunkingConfig.DEFAULT_MAX_CONCURRENT_CALLS
        
    logger.info(f"Starting PARALLEL semantic chunking with max {max_concurrent_calls} concurrent OpenAI calls")
    
    try:
        # Check if an event loop is already running
        try:
            loop = asyncio.get_running_loop()
            logger.warning("Event loop already running, using thread executor for async operations")
            
            # Use thread executor to run the async function
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run, 
                    semantic_chunking_with_semaphore(text_or_file_path, max_concurrent_calls)
                )
                result = future.result()
                return result
                
        except RuntimeError:
            # No event loop running, we can create one
            logger.info(f"Creating new event loop for PARALLEL processing with {max_concurrent_calls} concurrent calls")
            return asyncio.run(semantic_chunking_with_semaphore(text_or_file_path, max_concurrent_calls))
            
    except Exception as e:
        logger.error(f"Error in semaphore-controlled semantic chunking: {e}")
        logger.warning("Falling back to synchronous semantic chunking")
        return semantic_chunking_legacy_fallback(text_or_file_path)

async def semantic_chunking_with_semaphore(
    text_or_file_path: Union[str, List[Image.Image]], 
    max_concurrent_calls: int = None
):
    """
    Core async function for semantic chunking with semaphore-controlled concurrency.
    
    Args:
        text_or_file_path: PDF file path or list of PIL Images
        max_concurrent_calls: Maximum concurrent OpenAI API calls
        
    Returns:
        List[Dict[str, Any]]: Semantic chunks with metadata
        
    Raises:
        DocumentProcessingError: When processing fails
        OpenAIServiceError: When OpenAI service is unavailable
    """
    if max_concurrent_calls is None:
        max_concurrent_calls = ChunkingConfig.DEFAULT_MAX_CONCURRENT_CALLS
        
    start_time = time.time()
    logger.info(f"🔄 Starting SEMAPHORE-CONTROLLED semantic chunking with {max_concurrent_calls} concurrent calls")
    
    client = None
    try:
        # Convert to images if needed
        if isinstance(text_or_file_path, str):
            images = pdf_to_images(text_or_file_path)
            logger.info(f"Loaded {len(images)} pages from PDF")
        else:
            images = text_or_file_path
            logger.info(f"Processing {len(images)} provided images")

        if not images:
            logger.warning("No images to process")
            return []

        # Log dimensions for debugging
        for i, image in enumerate(images, 1):
            logger.debug(f"Page {i}: {image.width}x{image.height}")

        client = get_openai_client()
        if not client:
            raise OpenAIServiceError("OpenAI client unavailable for semantic chunking")

        # Process all pages in parallel with semaphore control
        semaphore = asyncio.Semaphore(max_concurrent_calls)
        
        async def process_single_page_with_semaphore(page_idx: int, image: Image.Image):
            async with semaphore:
                try:
                    logger.info(f"Processing page {page_idx + 1}/{len(images)} with YOLO detection")
                    
                    # Run YOLO inference on the image
                    yolo_results = run_yolo_inference([image])
                    yolo_result = yolo_results[0] if yolo_results else []
                    
                    if not yolo_result:
                        logger.info(f"Page {page_idx + 1}: No YOLO detections found")
                        return []
                    
                    logger.info(f"Page {page_idx + 1}: Found {len(yolo_result)} YOLO detections")
                    
                    # Extract bbox results for semantic grouping
                    bbox_results = await _extract_bbox_results_for_grouping_with_semaphore(
                        image, yolo_result, page_idx + 1, max_concurrent_calls
                    )
                    
                    return bbox_results
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_idx + 1}: {str(e)}")
                    return []

        # Execute all page processing tasks in parallel
        logger.info(f"⚡ Processing {len(images)} pages in parallel with semaphore control")
        page_tasks = [
            process_single_page_with_semaphore(i, image) 
            for i, image in enumerate(images)
        ]
        
        page_results = await asyncio.gather(*page_tasks, return_exceptions=True)
        
        all_bbox_results = []
        for i, result in enumerate(page_results):
            if isinstance(result, Exception):
                logger.error(f"Page {i + 1} processing failed: {result}")
            else:
                all_bbox_results.extend(result)
        
        if not all_bbox_results:
            logger.warning(" No content extracted from any pages")
            return []

        logger.info(f"Extracted {len(all_bbox_results)} content elements across all pages")
        
        # Perform semantic grouping with semaphore control
        logger.info(f"Starting semaphore-controlled semantic grouping for {len(all_bbox_results)} elements")
        semantic_chunks = await _perform_openai_semantic_grouping_with_semaphore(
            all_bbox_results, max_concurrent_calls=max_concurrent_calls
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed in {elapsed_time:.2f}s")
        logger.info(f"Created {len(semantic_chunks)} semantic chunks with {max_concurrent_calls} concurrent calls")
        
        return semantic_chunks
        
    except Exception as e:
        logger.error(f" Error in semaphore-controlled semantic chunking: {str(e)}")
        return semantic_chunking_legacy_fallback(text_or_file_path)
    
    finally:
        # Proper cleanup to prevent "Event loop is closed" errors
        if client and hasattr(client, 'close'):
            try:
                await client.close()
            except Exception:
                pass  # Ignore cleanup errors

async def _extract_bbox_results_for_grouping_with_semaphore(
    image: Image.Image, 
    yolo_result, 
    page_number: int, 
    max_concurrent_calls: int = 5
) -> List[Dict[str, Any]]:
    """
    Process YOLO detection results and extract bounding box results with content for semantic grouping.
    Uses semaphore to control concurrent OpenAI API calls.
    
    Args:
        image: PIL Image of the page
        yolo_result: YOLO detection results
        page_number: Page number for logging
        max_concurrent_calls: Maximum number of concurrent OpenAI calls
        
    Returns:
        List of bbox results with extracted content, sorted in reading order
    """
    if not yolo_result or not hasattr(yolo_result, 'boxes') or len(yolo_result.boxes) == 0:
        logger.warning(f"Page {page_number}: No YOLO detections found")
        return []
    
    image_np = np.array(image)
    img_height, img_width = image_np.shape[:2]
    
    logger.info(f"Page {page_number}: Processing {len(yolo_result.boxes)} initial detections")
    
    # Extract detections with bbox coordinates and confidence
    detections = []
    for detection_index, (box, conf, cls) in enumerate(zip(yolo_result.boxes.xyxy, yolo_result.boxes.conf, yolo_result.boxes.cls)):
        x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
        confidence = float(conf)
        class_id = int(cls)
        class_name = yolo_result.names[class_id]
        
        center_x = float((x1 + x2) / 2)
        center_y = float((y1 + y2) / 2)
        area = int((x2 - x1) * (y2 - y1))
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': confidence,
            'class': class_name,
            'detection_index': detection_index,
            'center_x': center_x,
            'center_y': center_y,
            'area': area
        })
    
    # Sort detections in reading order
    detections = improve_reading_order(detections, img_width, img_height)
    
    logger.info(f"Page {page_number}: Starting semaphore-controlled content extraction for {len(detections)} detections (max_concurrent: {max_concurrent_calls})")
    
    openai_semaphore = asyncio.Semaphore(max_concurrent_calls)
    
    async def extract_content_for_detection_with_semaphore(i: int, detection: dict):
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']
        
        # Crop the region
        cropped_region = image_np[y1:y2, x1:x2]
        cropped_image = Image.fromarray(cropped_region)
        
        # Extract content based on element type using semaphore for OpenAI calls
        if class_name in ['Text', 'List-item', 'Caption', 'Footnote', 'Title', 'Section-header', 'Page-header']:
            content = _extract_text_with_ocr(cropped_image)
            content_type = 'heading' if class_name in ['Title', 'Section-header', 'Page-header'] else 'text'
        elif class_name == 'Table':
            async with openai_semaphore:
                logger.debug(f"Page {page_number}, Detection {i}: Acquiring semaphore for Table extraction")
                content = await _extract_table_with_openai_async(cropped_image)
                logger.debug(f"Page {page_number}, Detection {i}: Released semaphore for Table extraction")
            content_type = 'table'
        elif class_name in ['Picture', 'Formula']:
            async with openai_semaphore:
                logger.debug(f"Page {page_number}, Detection {i}: Acquiring semaphore for {class_name} extraction")
                content = await _extract_image_with_openai_async(cropped_image, class_name)
                logger.debug(f"Page {page_number}, Detection {i}: Released semaphore for {class_name} extraction")
            content_type = 'image' if class_name == 'Picture' else 'formula'
        else:
            content = _extract_text_with_ocr(cropped_image)
            content_type = 'text'
        
        return {
            'index': i,
            'detection': detection,
            'content': content,
            'content_type': content_type,
            'confidence': confidence
        }
    
    tasks = [extract_content_for_detection_with_semaphore(i, detection) for i, detection in enumerate(detections)]
    extraction_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    bbox_results = []
    
    for result in extraction_results:
        if isinstance(result, Exception):
            logger.error(f"Page {page_number}: Content extraction failed: {result}")
            continue
            
        i = result['index']
        detection = result['detection']
        content = result['content']
        content_type = result['content_type']
        confidence = result['confidence']
        
        if not content or content.strip() in ['[No text detected]', '[OCR failed]', '']:
            bbox_str = f"[{detection['bbox'][0]},{detection['bbox'][1]},{detection['bbox'][2]},{detection['bbox'][3]}]"
            logger.warning(f"Page {page_number}, ReadingOrder {i}: Skipping {detection['class']} at bbox {bbox_str} - no content extracted")
            continue
        
        bbox_result = {
            'content': content.strip(),
            'metadata': {
                'page_number': int(page_number),
                'reading_order_index': int(i),
                'element_type': str(detection['class']),
                'content_type': str(content_type),
                'confidence': float(confidence),
                'bbox': [int(x) for x in detection['bbox']],
                'reading_order': f"page_{page_number}_element_{i}",
                'original_detection_index': int(detection['detection_index'])
            }
        }
        
        bbox_results.append(bbox_result)
        
        content_preview = content.strip()[:ChunkingConfig.MAX_TEXT_PREVIEW_LENGTH] + (
            "..." if len(content.strip()) > ChunkingConfig.MAX_TEXT_PREVIEW_LENGTH else ""
        )
        bbox_str = f"[{detection['bbox'][0]},{detection['bbox'][1]},{detection['bbox'][2]},{detection['bbox'][3]}]"
        logger.info(f"Page {page_number}, ReadingOrder {i}: {detection['class']} at bbox {bbox_str} "
                   f"-> '{content_preview}' (confidence: {confidence:.3f}) [SEMAPHORE]")
    
    bbox_results.sort(key=lambda x: x['metadata']['reading_order_index'])
    
    logger.info(f"Page {page_number}: Extracted {len(bbox_results)} valid bbox results in reading order (SEMAPHORE-CONTROLLED)")
    
    return bbox_results


def semantic_chunking_legacy_fallback(text_or_file_path: Union[str, List[Image.Image]]):
    """Legacy fallback that doesn't take max_concurrent_calls parameter"""
    if isinstance(text_or_file_path, str) and not text_or_file_path.endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        legacy_chunks = _legacy_semantic_chunking(text_or_file_path)
        return {
            'chunks': legacy_chunks,
            'image_dimensions': []
        }
    else:
        return {
            'chunks': [],
            'image_dimensions': []
        }

async def _perform_openai_semantic_grouping_with_semaphore(
    bbox_results: List[Dict[str, Any]], 
    confidence_threshold: float = DEFAULT_OPENAI_CONFIDENCE_THRESHOLD,
    max_concurrent_calls: int = 5
) -> List[Dict[str, Any]]:
    """
    SEMAPHORE-CONTROLLED OpenAI semantic boundary detection with parallel processing.
    
    Args:
        bbox_results: List of content parts with text and metadata
        confidence_threshold: OpenAI confidence threshold for boundary decisions
        max_concurrent_calls: Maximum concurrent OpenAI API calls
        
    Returns:
        List of semantic chunks with grouped content
    """
    logger.info(f"Starting semaphore-controlled semantic grouping with max {max_concurrent_calls} concurrent calls")
    
    if not bbox_results:
        logger.warning(" No bbox results provided for semantic grouping")
        return []
    
    client = None
    try:
        boundary_detector = OpenAISemanticBoundaryDetector(confidence_threshold=confidence_threshold)
        
        client = get_openai_client()
        
        semaphore = asyncio.Semaphore(max_concurrent_calls)
        
        boundary_tasks = []
        
        async def detect_boundary_for_pair_with_semaphore(i: int, current_result: dict, next_result: dict):
            async with semaphore:
                try:
                    current_text = current_result.get('content', '')
                    next_text = next_result.get('content', '')
                    next_metadata = next_result.get('metadata', {})
                    
                    boundary_result = await boundary_detector.detect_boundary_async(
                        current_text, next_text, next_metadata
                    )
                    
                    return i, boundary_result
                except Exception as e:
                    logger.error(f" Boundary detection failed for pair {i}: {str(e)}")
                    return i, {'should_split': False, 'confidence': 0.0, 'reasoning': f'Error: {str(e)}'}
        
        # Create boundary detection tasks for adjacent pairs
        for i in range(len(bbox_results) - 1):
            current_result = bbox_results[i]
            next_result = bbox_results[i + 1]
            
            task = detect_boundary_for_pair_with_semaphore(i, current_result, next_result)
            boundary_tasks.append(task)
        
        if boundary_tasks:
            logger.info(f"Executing {len(boundary_tasks)} boundary detection tasks with semaphore control")
            
            # Execute all boundary detection tasks in parallel
            boundary_results = await asyncio.gather(*boundary_tasks, return_exceptions=True)
            
            # Process boundary results
            boundary_decisions = {}
            for result in boundary_results:
                if isinstance(result, Exception):
                    logger.error(f"Boundary detection task failed: {result}")
                    continue
                    
                pair_index, decision = result
                boundary_decisions[pair_index] = decision
        else:
            boundary_decisions = {}
        
        semantic_chunks = []
        current_group = []
        group_index = 0
        
        for i, bbox_result in enumerate(bbox_results):
            current_group.append(bbox_result)
            
            should_split = False
            split_confidence = 0.0
            
            if i < len(bbox_results) - 1:
                boundary_decision = boundary_decisions.get(i, {})
                should_split = boundary_decision.get('should_split', False)
                split_confidence = boundary_decision.get('confidence', 0.0)
            else:
                should_split = True
                split_confidence = 1.0
            
            if should_split:
                if current_group:
                    group_dict = {
                        'content_parts': [result['content'] for result in current_group],
                        'metadata_parts': [result['metadata'] for result in current_group],
                        'start_page': current_group[0]['metadata']['page_number'],
                        'start_reading_order': current_group[0]['metadata']['reading_order_index'],
                        'boundary_decisions': []
                    }
                    chunk = _create_semantic_chunk_from_group(group_dict, group_index, split_confidence)
                    semantic_chunks.append(chunk)
                    group_index += 1
                    current_group = []
        
        if current_group:
            group_dict = {
                'content_parts': [result['content'] for result in current_group],
                'metadata_parts': [result['metadata'] for result in current_group],
                'start_page': current_group[0]['metadata']['page_number'],
                'start_reading_order': current_group[0]['metadata']['reading_order_index'],
                'boundary_decisions': []
            }
            chunk = _create_semantic_chunk_from_group(group_dict, group_index, 1.0)
            semantic_chunks.append(chunk)
        
        logger.info(f"semantic grouping completed")
        logger.info(f"boundary decisions: {len(boundary_decisions)}")
        logger.info(f"Semantic chunks created: {len(semantic_chunks)}")
        
        return semantic_chunks
        
    except Exception as e:
        logger.error(f" Error in semaphore-controlled semantic grouping: {str(e)}")
        # Fallback to heuristic grouping
        logger.warning("Falling back to heuristic grouping")
        return _fallback_heuristic_grouping(bbox_results)
    
    finally:
        # Proper cleanup to prevent "Event loop is closed" errors
        if client and hasattr(client, 'close'):
            try:
                await client.close()
            except Exception:
                pass  # Ignore cleanup errors

def improve_reading_order(detections: List[Dict[str, Any]], image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """
    Improve reading order detection using more sophisticated algorithms.
    
    Args:
        detections: List of detection dictionaries
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        Detections sorted in improved reading order
    """
    if not detections:
        return []
    
    logger.debug(f"Reading order improvement: Processing {len(detections)} detections")
    logger.debug(f"Image dimensions: {image_width}x{image_height}")
    
    # Sort primarily by Y coordinate (top to bottom), then by X coordinate (left to right)
    sorted_detections = sorted(detections, key=lambda d: (d['center_y'], d['center_x']))
    
    logger.debug("Sorted detections by Y-coordinate (top to bottom), then X-coordinate (left to right):")
    for i, detection in enumerate(sorted_detections):
        bbox_str = f"[{detection['bbox'][0]},{detection['bbox'][1]},{detection['bbox'][2]},{detection['bbox'][3]}]"
        logger.debug(f"  Position {i}: {detection['class']} at bbox {bbox_str} "
                    f"(center: X:{detection['center_x']:.1f}, Y:{detection['center_y']:.1f})")
    
    # For more sophisticated grouping, we can still group into rows but with stricter Y-coordinate ordering
    rows = []
    tolerance = ChunkingConfig.READING_ORDER_TOLERANCE
    tolerance_pixels = tolerance * image_height
    
    logger.debug(f"Using tolerance: {tolerance:.3f} ({tolerance_pixels:.1f} pixels)")
    
    for i, detection in enumerate(sorted_detections):
        placed = False
        det_y_min = detection['bbox'][1]
        det_y_max = detection['bbox'][3]
        det_center_y = detection['center_y']
        bbox_str = f"[{detection['bbox'][0]},{detection['bbox'][1]},{detection['bbox'][2]},{detection['bbox'][3]}]"
        
        logger.debug(f"  Processing detection {i}: {detection['class']} "
                    f"at bbox {bbox_str} (Y center: {det_center_y:.1f})")
        
        # Try to place in existing row only if Y-centers are very close
        for row_idx, row in enumerate(rows):
            row_y_centers = [d['center_y'] for d in row]
            row_y_center_avg = sum(row_y_centers) / len(row_y_centers)
            
            # More strict condition: only group if Y-centers are within tolerance
            if abs(det_center_y - row_y_center_avg) <= tolerance_pixels:
                row.append(detection)
                placed = True
                logger.debug(f"    ✓ Added to existing row {row_idx} "
                           f"(row Y center avg: {row_y_center_avg:.1f})")
                break
        
        if not placed:
            rows.append([detection])
            logger.debug(f"    ✓ Created new row {len(rows)-1} for this detection")
    
    logger.debug(f"Created {len(rows)} rows:")
    for i, row in enumerate(rows):
        row_y_centers = [d['center_y'] for d in row]
        row_y_center_avg = sum(row_y_centers) / len(row_y_centers)
        row_elements = []
        for d in row:
            bbox_str = f"[{d['bbox'][0]},{d['bbox'][1]},{d['bbox'][2]},{d['bbox'][3]}]"
            row_elements.append(f"{d['class']}@{bbox_str}")
        logger.debug(f"  Row {i} (Y center avg: {row_y_center_avg:.1f}): {len(row)} elements")
        logger.debug(f"    Elements: {row_elements}")
    
    reading_order_detections = []
    logger.debug("Sorting rows left-to-right and combining:")
    
    for row_idx, row in enumerate(rows):
        row_before = []
        for d in row:
            bbox_str = f"[{d['bbox'][0]},{d['bbox'][1]},{d['bbox'][2]},{d['bbox'][3]}]"
            row_before.append(f"{d['class']}@{bbox_str}")
        
        row.sort(key=lambda d: d['center_x'])
        
        row_after = []
        for d in row:
            bbox_str = f"[{d['bbox'][0]},{d['bbox'][1]},{d['bbox'][2]},{d['bbox'][3]}]"
            row_after.append(f"{d['class']}@{bbox_str}")
        
        logger.debug(f"  Row {row_idx} before sort: {row_before}")
        logger.debug(f"  Row {row_idx} after sort:  {row_after}")
        
        # Add to final order
        start_idx = len(reading_order_detections)
        reading_order_detections.extend(row)
        end_idx = len(reading_order_detections) - 1
        
        logger.debug(f"    Added to reading order positions {start_idx}-{end_idx}")
    
    logger.debug(f"Final reading order ({len(reading_order_detections)} elements):")
    for i, detection in enumerate(reading_order_detections):
        bbox_str = f"[{detection['bbox'][0]},{detection['bbox'][1]},{detection['bbox'][2]},{detection['bbox'][3]}]"
        logger.debug(f"  Position {i}: {detection['class']} at bbox {bbox_str} "
                    f"(center: X:{detection['center_x']:.1f}, Y:{detection['center_y']:.1f})")
    
    # Final validation: ensure strict top-to-bottom order
    # If any element has a Y-center significantly higher than a later element, we need to fix it
    needs_final_sort = False
    for i in range(len(reading_order_detections) - 1):
        current_y = reading_order_detections[i]['center_y']
        next_y = reading_order_detections[i + 1]['center_y']
        if current_y > next_y + tolerance_pixels: 
            needs_final_sort = True
            logger.warning(f"Reading order issue detected: element {i} (Y:{current_y:.1f}) is below element {i+1} (Y:{next_y:.1f})")
            break
    
    if needs_final_sort:
        logger.info("Applying final sort to fix reading order issues")
        reading_order_detections = sorted(reading_order_detections, key=lambda d: (d['center_y'], d['center_x']))
        
        logger.debug("Final corrected reading order:")
        for i, detection in enumerate(reading_order_detections):
            bbox_str = f"[{detection['bbox'][0]},{detection['bbox'][1]},{detection['bbox'][2]},{detection['bbox'][3]}]"
            logger.debug(f"  Position {i}: {detection['class']} at bbox {bbox_str} "
                        f"(center: X:{detection['center_x']:.1f}, Y:{detection['center_y']:.1f})")
    
    return reading_order_detections

def _create_semantic_chunk_from_group(group: Dict[str, Any], group_index: int, split_confidence: float) -> Dict[str, Any]:
    """
    Create a semantic chunk from a group of bbox results.
    
    Args:
        group: Group dictionary containing content_parts, metadata_parts, etc.
        group_index: Index of this semantic group
        split_confidence: Confidence score for the boundary decision
        
    Returns:
        Semantic chunk dictionary
    """
    group_text = ' '.join(group['content_parts'])
    
    # Determine primary content type and element type
    element_types = [meta['element_type'] for meta in group['metadata_parts']]
    content_types = [meta['content_type'] for meta in group['metadata_parts']]
    
    primary_element_type = max(set(element_types), key=element_types.count)
    primary_content_type = max(set(content_types), key=content_types.count)
    
    # Calculate combined bbox (bounding box that encompasses all parts)
    all_bboxes = [meta['bbox'] for meta in group['metadata_parts']]
    combined_bbox = [
        min(bbox[0] for bbox in all_bboxes),  # min x1
        min(bbox[1] for bbox in all_bboxes),  # min y1
        max(bbox[2] for bbox in all_bboxes),  # max x2
        max(bbox[3] for bbox in all_bboxes)   # max y2
    ]
    
    # Calculate average confidence
    avg_confidence = sum(meta['confidence'] for meta in group['metadata_parts']) / len(group['metadata_parts'])
    
    return {
        'text': group_text,
        'metadata': {
            'page_number': int(group['start_page']),
            'semantic_group_index': int(group_index),
            'element_count': int(len(group['content_parts'])),
            'primary_element_type': str(primary_element_type),
            'primary_content_type': str(primary_content_type),
            'element_types': element_types,
            'content_types': content_types,
            'avg_confidence': float(avg_confidence),
            'combined_bbox': [int(x) for x in combined_bbox],
            'strategy': 'semantic_openai_boundary_detection',
            'split_confidence': float(split_confidence),
            'reading_order_start': int(group['start_reading_order']),
            'reading_order_end': int(group['metadata_parts'][-1]['reading_order_index']),
            'boundary_decisions': group.get('boundary_decisions', []),
            'constituent_elements': [
                {
                    'element_type': part_meta['element_type'],
                    'bbox': part_meta['bbox'],
                    'reading_order': part_meta['reading_order']
                }
                for part_content, part_meta in zip(group['content_parts'], group['metadata_parts'])
            ]
        }
    }
def _fallback_heuristic_grouping(bbox_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fallback grouping using simple heuristics when OpenAI is not available.
    
    Args:
        bbox_results: List of bbox results with extracted content in reading order
        
    Returns:
        List of semantically grouped chunks using heuristic rules
    """
    if not bbox_results:
        return []
    
    semantic_chunks = []
    current_group = {
        'content_parts': [],
        'metadata_parts': [],
        'start_page': bbox_results[0]['metadata']['page_number'],
        'start_reading_order': bbox_results[0]['metadata']['reading_order_index']
    }
    
    logger.info(f"Using heuristic-based grouping for {len(bbox_results)} bbox results")
    
    for i, bbox_result in enumerate(bbox_results):
        content = bbox_result['content']
        metadata = bbox_result['metadata']
        element_type = metadata.get('element_type', 'Text')
        
        # Simple heuristic rules for splitting
        should_split = False
        
        if i > 0:
            if element_type in ['Title', 'Section-header']:
                should_split = True
            elif element_type in ['Table', 'Picture', 'Formula'] and len(current_group['content_parts']) > 0:
                prev_element_type = current_group['metadata_parts'][-1].get('element_type', 'Text')
                if prev_element_type not in ['Table', 'Picture', 'Formula']:
                    should_split = True
            elif len(current_group['content_parts']) >= ChunkingConfig.MAX_ELEMENTS_PER_GROUP:
                should_split = True
        
        if should_split:
            semantic_chunk = _create_semantic_chunk_from_group(current_group, len(semantic_chunks), ChunkingConfig.HEURISTIC_CONFIDENCE)
            semantic_chunks.append(semantic_chunk)
            
            current_group = {
                'content_parts': [content],
                'metadata_parts': [metadata],
                'start_page': metadata['page_number'],
                'start_reading_order': metadata['reading_order_index']
            }
        else:
            current_group['content_parts'].append(content)
            current_group['metadata_parts'].append(metadata)
    
    if current_group['content_parts']:
        semantic_chunk = _create_semantic_chunk_from_group(current_group, len(semantic_chunks), ChunkingConfig.HEURISTIC_CONFIDENCE)
        semantic_chunks.append(semantic_chunk)
    
    logger.info(f"Completed heuristic-based grouping: {len(semantic_chunks)} semantic groups created")
    return semantic_chunks


def _legacy_semantic_chunking(text: str) -> List[Dict[str, Any]]:
    """
    Legacy semantic chunking using OpenAI text analysis (for backward compatibility).
    """
    return semantic_chunk_with_structured_output(text)

def extract_jar(filepath):
    with javatools.unpack.unpack_class(filepath) as unpacker:
        return "\n".join(unpacker.get_method_names())

def extract_ico(file_path):
    with Image.open(file_path) as img:
        return f"ICO image: {img.size[0]}x{img.size[1]}, {len(img.info)} metadata entries"

def extract_pdf(file_path):
    with fitz.open(file_path) as doc:
        return "\n".join(page.get_text() for page in doc)

def detect_file_type(file_path):
    kind = filetype.guess(file_path)
    return kind.mime if kind else "unknown"

def perform_ocr(file_path):
    return pytesseract.image_to_string(Image.open(file_path))

def get_all_files(directory):
    text_file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            absolute_path = os.path.abspath(os.path.join(root, file))
            text_file_paths.append(absolute_path)
    return text_file_paths

def extract_with_unstructured(file_path):
    elements = partition(filename=file_path)
    content = "\n".join([str(element) for element in elements])
    return content

def extract_data(file_path):
    try:
        file_type = detect_file_type(file_path)
        if file_type == "application/java-archive":
            return extract_jar(file_path)
        elif file_type == "application/pdf":
            return extract_pdf(file_path)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return extract_docx(file_path)
        elif file_type.startswith("image/"):
            return perform_ocr(file_path)
        else:
            try:
                return extract_with_unstructured(file_path)
            except:
                with open(file_path, 'r') as f:
                    content = f.read()
                return content
    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"

directory = ''  # Replace with the path to your folder
files = get_all_files(directory)

results = {}
working = []
n_working = []

for file_path in files:
    content = extract_data(file_path)
    results[file_path] = content

    if content.startswith("Error"):
        n_working.append(os.path.splitext(file_path)[1])
    else:
        working.append(os.path.splitext(file_path)[1])

@njit
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

df = dd.read_parquet("s3://my-data/")
dtrain = xgb.dask.DaskDMatrix(df)

model = xgb.dask.train(
    dtrain,
    {"tree_method": "hist", },
    ...
)

df = dask.datasets.timeseries()  # Randomly generated data
# df = dd.read_parquet(...)      # In practice, you would probably read data though

train, test = df.random_split([0.80, 0.20])
X_train, y_train, X_test, y_test = ...

with LocalCluster() as cluster:
    with cluster.get_client() as client:
        d_train = xgb.dask.DaskDMatrix(client, X_train, y_train, enable_categorical=True)
        model = xgb.dask.train(...d_train,)
        predictions = xgb.dask.predict(client, model, X_test)

df = dd.read_parquet("/path/to/my/data.parquet")

model = load_model("/path/to/my/model")

# pandas code
# predictions = model.predict(df)
# predictions.to_parquet("/path/to/results.parquet")

# Dask code
predictions = df.map_partitions(model.predict)
predictions.to_parquet("/path/to/results.parquet")

print("Non-working file extensions:")
print(list(set(n_working)))
print("Working file extensions:")
print(list(set(working)))