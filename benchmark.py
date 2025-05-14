import os
import logging
from Unsiloed.services.chunking import process_document_chunking

logger = logging.getLogger(__name__)

def process_documents_batch(documents):
    """Process a batch of documents
    
    Args:
        documents: List of tuples (path, type)
    """
    results = []
    for path, doc_type in documents:
        try:
            result = process_document(path, doc_type)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing document {path}: {str(e)}")
            continue
    return results

def process_document(path, doc_type):
    """Process a single document
    
    Args:
        path: Path to the document
        doc_type: Type of document (pdf, docx, pptx)
    """
    strategy = "semantic"  # Using semantic chunking strategy
    return process_document_chunking(
        file_path=path,
        file_type=doc_type,
        strategy=strategy
    )