import logging
from fastapi import APIRouter

router = APIRouter(tags=["info"])

logger = logging.getLogger(__name__)


@router.get("/")
def read_root():
    """
    Root endpoint providing API information.

    Returns:
        API information
    """
    logger.info("Root endpoint accessed")
    return {
        "message": "Document Data Extractor API",
        "version": "2.1.0",
        "docs": "/docs",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API information"},
            {
                "path": "/document-chunking",
                "method": "POST",
                "description": "Chunk a document file (PDF, DOCX, PPTX) according to the specified strategy.",
            },
        ],
    }
