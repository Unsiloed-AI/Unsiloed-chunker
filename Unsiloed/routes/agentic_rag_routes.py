from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from Unsiloed.services.agentic_rag.service import process_query, reset_agentic_rag

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agentic_rag"])

class QueryRequest(BaseModel):
    """Request model for agentic RAG query endpoint."""
    query: str = Field(..., description="The query to process")
    chunks: List[Dict[str, Any]] = Field(..., description="Document chunks to use for retrieval")
    model_name: Optional[str] = Field("gpt-4o", description="The name of the OpenAI model to use")
    temperature: Optional[float] = Field(0, description="The temperature parameter for the LLM")
    cleanup_after: Optional[bool] = Field(True, description="Whether to clean up (delete) the vector store after processing")

@router.post("/agentic_rag/query")
async def agentic_rag_query(request: QueryRequest):
    """
    Process a query using the agentic RAG system.
    
    Args:
        request: QueryRequest object containing the query and document chunks
        
    Returns:
        JSON response with the answer and additional information
    """
    logger.info(f"Received agentic RAG query: {request.query}")
    
    try:
        # Process the query
        result = process_query(
            query=request.query,
            chunks=request.chunks,
            model_name=request.model_name,
            temperature=request.temperature,
            cleanup_after=request.cleanup_after
        )
        
        logger.info(f"Agentic RAG query processed successfully. Query type: {result.get('query_type', 'unknown')}")
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error processing agentic RAG query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {str(e)}"
        )

@router.post("/agentic_rag/reset")
async def reset_rag_system():
    """
    Reset the agentic RAG system.
    
    Returns:
        JSON response indicating success
    """
    try:
        reset_agentic_rag()
        return JSONResponse(content={"status": "success", "message": "Agentic RAG system reset successfully"})
    
    except Exception as e:
        logger.error(f"Error resetting agentic RAG system: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error resetting agentic RAG system: {str(e)}"
        )

@router.post("/agentic_rag/process_chunks")
async def process_chunks_and_query(
    query: str = Body(..., embed=True),
    chunks_result: Dict[str, Any] = Body(..., embed=True),
    model_name: str = Body("gpt-4o", embed=True),
    temperature: float = Body(0, embed=True),
    cleanup_after: bool = Body(True, embed=True)
):
    """
    Process chunks from the chunking service and query using the agentic RAG system.
    
    Args:
        query: The query to process
        chunks_result: Result from the chunking service
        model_name: The name of the OpenAI model to use
        temperature: The temperature parameter for the LLM
        cleanup_after: Whether to clean up (delete) the vector store after processing
        
    Returns:
        JSON response with the answer and additional information
    """
    logger.info(f"Received request to process chunks and query: {query}")
    
    try:
        # Extract chunks from the chunking service result
        chunks = chunks_result.get("chunks", [])
        
        if not chunks:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "No chunks found in the provided result"
                },
                status_code=400
            )
        
        # Process the query
        result = process_query(
            query=query,
            chunks=chunks,
            model_name=model_name,
            temperature=temperature,
            cleanup_after=cleanup_after
        )
        
        # Add metadata from the chunking result
        result["file_type"] = chunks_result.get("file_type", "unknown")
        result["strategy"] = chunks_result.get("strategy", "unknown")
        result["total_chunks"] = chunks_result.get("total_chunks", 0)
        
        logger.info(f"Chunks processed and query answered successfully. Query type: {result.get('query_type', 'unknown')}")
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error processing chunks and query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing chunks and query: {str(e)}"
        ) 