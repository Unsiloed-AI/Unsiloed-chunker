import os
import requests
import tempfile
import logging
from urllib.parse import urlparse

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
)

logger = logging.getLogger(__name__)


def process_document_chunking(
    file_path,
    file_type,
    strategy,
    chunk_size=1000,
    overlap=100,
):
    """
    Process a document file (PDF, DOCX, PPTX) with the specified chunking strategy.
    """
    logger.info(
        f"Processing {file_type.upper()} document with {strategy} chunking strategy"
    )

    if strategy == "page" and file_type == "pdf":
        chunks = page_based_chunking(file_path)
    else:
        if file_type == "pdf":
            text = extract_text_from_pdf(file_path)
        elif file_type == "docx":
            text = extract_text_from_docx(file_path)
        elif file_type == "pptx":
            text = extract_text_from_pptx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if strategy == "fixed":
            chunks = fixed_size_chunking(text, chunk_size, overlap)
        elif strategy == "semantic":
            chunks = semantic_chunking(text)
        elif strategy == "paragraph":
            chunks = paragraph_chunking(text)
        elif strategy == "heading":
            chunks = heading_chunking(text)
        elif strategy == "page" and file_type != "pdf":
            logger.warning(
                f"Page-based chunking not supported for {file_type}, falling back to paragraph chunking"
            )
            chunks = paragraph_chunking(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    total_chunks = len(chunks)
    avg_chunk_size = (
        sum(len(chunk["text"]) for chunk in chunks) / total_chunks
        if total_chunks > 0
        else 0
    )

    return {
        "file_type": file_type,
        "strategy": strategy,
        "total_chunks": total_chunks,
        "avg_chunk_size": avg_chunk_size,
        "chunks": chunks,
    }


def process_sync(payload: dict):
    """
    Synchronous processor for documents using strategy-based chunking.
    Supports remote file downloads and infers file type.
    """
    file_path = payload.get("filePath")
    strategy = payload.get("strategy", "semantic")
    chunk_size = payload.get("chunkSize", 1000)
    overlap = payload.get("overlap", 100)

    # Handle remote URLs
    if file_path.startswith("http"):
        logger.info(f"Downloading remote file from {file_path}")
        response = requests.get(file_path)
        response.raise_for_status()

        content_disposition = response.headers.get("content-disposition", "")
        _, ext = os.path.splitext(file_path or content_disposition or ".pdf")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        file_path = tmp_path
    else:
        _, ext = os.path.splitext(file_path)

    ext = ext.lower().lstrip(".")
    file_type = {"pdf": "pdf", "docx": "docx", "pptx": "pptx"}.get(ext)

    if not file_type:
        raise ValueError(f"Unsupported file extension: .{ext}")

    return process_document_chunking(
        file_path=file_path,
        file_type=file_type,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )
