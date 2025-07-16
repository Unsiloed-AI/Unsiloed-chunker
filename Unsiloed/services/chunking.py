from Unsiloed.utils.chunking import (
    fixed_size_chunking,
    page_based_chunking,
    paragraph_chunking,
    heading_chunking,
    semantic_chunking,
)
from Unsiloed.utils.openai import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_pptx,
    extract_text_from_html, 
    extract_text_from_markdown_file,
    extract_text_from_url,
)
from Unsiloed.utils.web_utils import get_content_type_from_url, validate_url

import logging

logger = logging.getLogger(__name__)


def process_document_chunking(
    file_path,
    file_type,
    strategy,
    chunk_size=1000,
    overlap=100,
):
    """
    Process a document file (PDF, DOCX, PPTX, HTML, Markdown) or URL with the specified chunking strategy.

    Args:
        file_path: Path to the document file or URL
        file_type: Type of document (pdf, docx, pptx, html, markdown, url)
        strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy
        overlap: Overlap size for fixed strategy

    Returns:
        Dictionary with chunking results
    """
    logger.info(
        f"Processing {file_type.upper()} document with {strategy} chunking strategy"
    )

    # Handle page-based chunking for PDFs only
    if strategy == "page" and file_type == "pdf":
        chunks = page_based_chunking(file_path)
    elif strategy == "semantic":
        # For semantic chunking, pass the file path directly to enable YOLO segmentation for PDFs
        # For other file types, extract text first
        if file_type == "pdf":
            semantic_result = semantic_chunking(file_path)
            chunks = semantic_result.get('chunks', []) if isinstance(semantic_result, dict) else semantic_result
        else:
            # Extract text first for non-PDF files
            text = _extract_text_by_type(file_path, file_type)
            semantic_result = semantic_chunking(text)
            chunks = semantic_result.get('chunks', []) if isinstance(semantic_result, dict) else semantic_result
    else:
        # Extract text based on file type for other strategies
        text = _extract_text_by_type(file_path, file_type)

        # Apply the selected chunking strategy
        if strategy == "fixed":
            chunks = fixed_size_chunking(text, chunk_size, overlap)
        elif strategy == "paragraph":
            chunks = paragraph_chunking(text)
        elif strategy == "heading":
            chunks = heading_chunking(text)
        elif strategy == "page" and file_type != "pdf":
            # For non-PDF files, fall back to paragraph chunking for page strategy
            logger.warning(
                f"Page-based chunking not supported for {file_type}, falling back to paragraph chunking"
            )
            chunks = paragraph_chunking(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    # Calculate statistics
    total_chunks = len(chunks)
    avg_chunk_size = (
        sum(len(chunk["text"]) for chunk in chunks) / total_chunks
        if total_chunks > 0
        else 0
    )

    result = {
        "file_type": file_type,
        "strategy": strategy,
        "total_chunks": total_chunks,
        "avg_chunk_size": avg_chunk_size,
        "chunks": chunks,
    }

    return result


def _extract_text_by_type(file_path: str, file_type: str) -> str:
    """
    Extract text based on file type.
    
    Args:
        file_path: Path to the document file or URL
        file_type: Type of document
        
    Returns:
        Extracted text content
    """
    if file_type == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_type == "docx":
        return extract_text_from_docx(file_path)
    elif file_type == "pptx":
        return extract_text_from_pptx(file_path)
    elif file_type == "html":
        return extract_text_from_html(file_path)
    elif file_type == "markdown":
        return extract_text_from_markdown_file(file_path)
    elif file_type == "url":
        return extract_text_from_url(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def determine_file_type_from_url(url: str) -> str:
    """
    Determine the file type from a URL.
    
    Args:
        url: URL to analyze
        
    Returns:
        File type string
    """
    try:
        content_type = get_content_type_from_url(url)
        if content_type == 'html':
            return 'url'  # Treat HTML URLs as web pages
        elif content_type == 'markdown':
            return 'url'  # Treat markdown URLs as web pages
        else:
            return 'url'  # Default to URL processing
    except Exception as e:
        logger.warning(f"Could not determine content type for URL {url}: {str(e)}")
        return 'url'  # Default to URL processing
