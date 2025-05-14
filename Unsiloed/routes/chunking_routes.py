from fastapi import APIRouter, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse
from Unsiloed.services.chunking import process_document_chunking
from Unsiloed.utils.chunking import ChunkingStrategy
import os
import tempfile
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chunking"])


@router.post("/chunking")
async def chunk_document(
    document_file: UploadFile,
    strategy: ChunkingStrategy = Form("semantic"),
    chunk_size: int = Form(1000),
    overlap: int = Form(100),
):
    """
    Chunk a document file (PDF, DOCX, PPTX) according to the specified strategy.

    Args:
        document_file: The document file to process
        strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy (in characters)
        overlap: Overlap size for fixed strategy (in characters)

    Returns:
        JSON response with chunks and metadata
    """
    logger.info(f"Received request for document chunking using {strategy} strategy")
    logger.debug(f"Document file name: {document_file.filename}")

    # Check file type from filename
    file_name = document_file.filename.lower()
    if file_name.endswith(".pdf"):
        file_type = "pdf"
        file_suffix = ".pdf"
    elif file_name.endswith(".docx"):
        file_type = "docx"
        file_suffix = ".docx"
    elif file_name.endswith(".pptx"):
        file_type = "pptx"
        file_suffix = ".pptx"
    elif file_name.endswith(".doc"):
        file_type = "doc"
        file_suffix = ".doc"
    elif file_name.endswith(".xlsx"):
        file_type = "xlsx"
        file_suffix = ".xlsx"
    elif file_name.endswith(".xls"):
        file_type = "xls"
        file_suffix = ".xls"
    elif file_name.endswith(".odt"):
        file_type = "odt"
        file_suffix = ".odt"
    elif file_name.endswith(".ods"):
        file_type = "ods"
        file_suffix = ".ods"
    elif file_name.endswith(".odp"):
        file_type = "odp"
        file_suffix = ".odp"
    elif file_name.endswith(".txt"):
        file_type = "txt"
        file_suffix = ".txt"
    elif file_name.endswith(".rtf"):
        file_type = "rtf"
        file_suffix = ".rtf"
    elif file_name.endswith(".epub"):
        file_type = "epub"
        file_suffix = ".epub"
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported formats: PDF, DOCX, PPTX, DOC, XLSX, XLS, ODT, ODS, ODP, TXT, RTF, EPUB",
        )

    file_path = None  # Initialize file_path
    try:
        # Read file content
        file_content = await document_file.read()

        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file.write(file_content)
            file_path = temp_file.name
            logger.debug(f"Document saved to temporary file: {file_path}")

        # Process the document with the requested chunking strategy
        result = process_document_chunking(
            file_path, file_type, strategy, chunk_size, overlap
        )

        logger.info(f"Document chunking completed with {result['total_chunks']} chunks")
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )

    finally:
        # Clean up the temporary file
        if file_path and os.path.exists(file_path):
            logger.debug(f"Cleaning up temporary file: {file_path}")
            os.unlink(file_path)
