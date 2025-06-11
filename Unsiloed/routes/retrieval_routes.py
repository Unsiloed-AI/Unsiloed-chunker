from fastapi import APIRouter, HTTPException, UploadFile, Form, File, Depends, Body
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
import uuid
import tempfile
import os
import logging
from Unsiloed.services.retrieval import AgenticRetrieval
from Unsiloed.utils.chunking import ChunkingStrategy

logger = logging.getLogger(__name__)

router = APIRouter(tags=["retrieval"])

# Global retrieval service instance
retrieval_service = AgenticRetrieval()


@router.post("/retrieval/index")
async def index_document(
    document_file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    strategy: ChunkingStrategy = Form("semantic"),
    chunk_size: int = Form(1000),
    overlap: int = Form(100),
):
    """
    Index a document for retrieval.
    
    Args:
        document_file: The document file to index
        document_id: Optional ID for the document (generated if not provided)
        strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy
        overlap: Overlap size for fixed strategy
        
    Returns:
        JSON response with indexing results
    """
    logger.info(f"Received request to index document using {strategy} strategy")
    
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
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF, DOCX, and PPTX are supported.",
        )
    
    file_path = None
    try:
        # Read file content
        file_content = await document_file.read()
        
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file.write(file_content)
            file_path = temp_file.name
            
        # Generate document ID if not provided
        if not document_id:
            document_id = str(uuid.uuid4())
            
        # Prepare options for indexing
        options = {
            "filePath": file_path,
            "fileType": file_type,
            "documentId": document_id,
            "strategy": strategy,
            "chunkSize": chunk_size,
            "overlap": overlap
        }
        
        # Index the document
        result = await retrieval_service.index_document(options)
        
        if result["status"] == "error":
            logger.error(f"Error indexing document: {result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Error indexing document: {result.get('error')}"
            )
            
        logger.info(f"Document indexed successfully with ID {document_id}")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error indexing document: {str(e)}"
        )
        
    finally:
        # Clean up the temporary file
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)


@router.post("/retrieval/query")
async def query_retrieval(
    query: str = Body(..., embed=True),
    top_k: int = Body(5, embed=True)
):
    """
    Perform agentic retrieval for a query.
    
    Args:
        query: The query to search for
        top_k: Number of top results to return per sub-query
        
    Returns:
        JSON response with retrieval results
    """
    logger.info(f"Received query: {query}")
    
    try:
        # Perform retrieval
        result = await retrieval_service.retrieve(query, top_k)
        
        if result.get("type") == "error":
            logger.error(f"Error in retrieval: {result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Error in retrieval: {result.get('error')}"
            )
            
        logger.info(f"Query processed successfully, found {len(result.get('chunks', []))} chunks")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.delete("/retrieval/document/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the retrieval index.
    
    Args:
        document_id: ID of the document to delete
        
    Returns:
        JSON response with deletion results
    """
    logger.info(f"Received request to delete document {document_id}")
    
    try:
        # Delete the document
        result = await retrieval_service.delete_document(document_id)
        
        if result["status"] == "error":
            logger.error(f"Error deleting document: {result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Error deleting document: {result.get('error')}"
            )
            
        logger.info(f"Document {document_id} deleted successfully")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        ) 