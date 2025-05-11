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
    extract_text_from_doc,
    extract_text_from_xlsx,
    extract_text_from_xls,
    extract_text_from_odt,
    extract_text_from_ods,
    extract_text_from_odp,
    extract_text_from_txt,
    extract_text_from_rtf,
    extract_text_from_epub,
)

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
    Process a document file with the specified chunking strategy.

    Args:
        file_path: Path to the document file
        file_type: Type of document (pdf, docx, pptx, doc, xlsx, xls, odt, ods, odp, txt, rtf, epub)
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
    else:
        # Extract text based on file type
        if file_type == "pdf":
            text = extract_text_from_pdf(file_path)
        elif file_type == "docx":
            text = extract_text_from_docx(file_path)
        elif file_type == "pptx":
            text = extract_text_from_pptx(file_path)
        elif file_type == "doc":
            text = extract_text_from_doc(file_path)
        elif file_type == "xlsx":
            text = extract_text_from_xlsx(file_path)
        elif file_type == "xls":
            text = extract_text_from_xls(file_path)
        elif file_type == "odt":
            text = extract_text_from_odt(file_path)
        elif file_type == "ods":
            text = extract_text_from_ods(file_path)
        elif file_type == "odp":
            text = extract_text_from_odp(file_path)
        elif file_type == "txt":
            text = extract_text_from_txt(file_path)
        elif file_type == "rtf":
            text = extract_text_from_rtf(file_path)
        elif file_type == "epub":
            text = extract_text_from_epub(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Apply the selected chunking strategy
        if strategy == "fixed":
            chunks = fixed_size_chunking(text, chunk_size, overlap)
        elif strategy == "semantic":
            chunks = semantic_chunking(text)
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
