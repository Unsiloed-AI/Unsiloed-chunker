from Unsiloed.utils.chunking import (
    fixed_size_chunking,
    page_based_chunking,
    paragraph_chunking,
    heading_chunking,
    semantic_chunking,
)
from Unsiloed.utils.openai import (
    extract_text_from_pdf_cached,
    extract_text_from_docx_cached,
    extract_text_from_pptx_cached,
    get_file_hash,
)
import concurrent.futures
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
    Process a document file (PDF, DOCX, PPTX) with the specified chunking strategy.

    Args:
        file_path: Path to the document file
        file_type: Type of document (pdf, docx, pptx)
        strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy
        overlap: Overlap size for fixed strategy

    Returns:
        Dictionary with chunking results
    """
    logger.info(
        f"Processing {file_type.upper()} document with {strategy} chunking strategy"
    )

    # Get file hash for caching
    file_hash = get_file_hash(file_path)

    # Handle page-based chunking for PDFs only
    if strategy == "page" and file_type == "pdf":
        chunks = page_based_chunking(file_path)
    else:
        # Extract text based on file type using cached functions
        if file_type == "pdf":
            text = extract_text_from_pdf_cached(file_path, file_hash)
        elif file_type == "docx":
            text = extract_text_from_docx_cached(file_path, file_hash)
        elif file_type == "pptx":
            text = extract_text_from_pptx_cached(file_path, file_hash)
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

def process_documents_batch(
    documents,
    strategy="semantic",
    chunk_size=1000,
    overlap=100,
    max_workers=4,
):
    """
    Process multiple documents in parallel with optimized resource management.

    Args:
        documents: List of dictionaries containing file_path and file_type
        strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy
        overlap: Overlap size for fixed strategy
        max_workers: Maximum number of parallel workers

    Returns:
        List of processing results
    """
    results = []
    errors = []
    
    # Group documents by type for optimized processing
    doc_groups = {}
    for doc in documents:
        doc_type = doc["file_type"]
        if doc_type not in doc_groups:
            doc_groups[doc_type] = []
        doc_groups[doc_type].append(doc)
    
    # Process each document type group with appropriate number of workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for doc_type, docs in doc_groups.items():
            # Adjust workers based on document type and count
            type_workers = min(len(docs), max_workers)
            if doc_type == "pdf":
                # PDFs benefit more from parallel processing
                type_workers = min(type_workers * 2, max_workers)
            
            # Submit documents for processing
            for doc in docs:
                futures.append(
                    executor.submit(
                        process_document_chunking,
                        doc["file_path"],
                        doc_type,
                        strategy,
                        chunk_size,
                        overlap,
                    )
                )
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                errors.append(str(e))
    
    # Log processing summary
    logger.info(f"Batch processing completed: {len(results)} successful, {len(errors)} failed")
    if errors:
        logger.warning(f"Errors encountered: {errors}")
    
    return results
