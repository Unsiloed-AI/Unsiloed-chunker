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
from Unsiloed.agentic_rag import AgenticRAG

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


def process_document_chunking_with_agentic_rag(
    file_path,
    file_type,
    strategy,
    query=None,
    chunk_size=1000,
    overlap=100,
    top_k=3,
):
    """
    Process a document and (optionally) run Agentic RAG for a query.
    Returns both chunking result and, if query is provided, a RAG answer.
    """
    result = process_document_chunking(
        file_path,
        file_type,
        strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    # Only run AgenticRAG if semantic chunking and query provided
    rag_answer = None
    if strategy == "semantic" and query:
        # Group chunks by page and section if possible
        # Here, we assume each chunk has 'metadata' with 'page_number' and 'section' if available
        chunks = result.get("chunks", [])
        pages = {}
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            page_id = f"page_{meta.get('page_number', 1)}"
            section_id = meta.get('heading') or meta.get('section') or f"section_{page_id}_0"
            # Build page
            if page_id not in pages:
                pages[page_id] = {"page_id": page_id, "text": "", "sections": {}}
            # Build section
            if section_id not in pages[page_id]["sections"]:
                pages[page_id]["sections"][section_id] = {"section_id": section_id, "text": "", "semantic_chunks": []}
            # Add semantic chunk
            pages[page_id]["sections"][section_id]["semantic_chunks"].append({
                "chunk_id": chunk.get("id") or chunk.get("chunk_id") or f"chunk_{len(pages[page_id]["sections"][section_id]["semantic_chunks"])}",
                "text": chunk["text"]
            })
        # Convert to list structure
        page_list = []
        for page in pages.values():
            page_obj = {"page_id": page["page_id"], "text": page["text"], "sections": []}
            for section in page["sections"].values():
                page_obj["sections"].append(section)
            page_list.append(page_obj)
        rag = AgenticRAG()
        rag_answer = rag.run_hierarchical(query, page_list, top_k=top_k)
    return {**result, "rag_answer": rag_answer}
