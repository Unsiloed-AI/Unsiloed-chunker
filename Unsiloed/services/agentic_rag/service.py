"""
Service layer for the agentic RAG system.
"""
import os
import tempfile
from typing import List, Dict, Any, Optional

from Unsiloed.services.agentic_rag.core import AgenticRAG
from Unsiloed.services.agentic_rag.utils import convert_chunks_to_rag_documents, format_rag_response

# Global instance of the AgenticRAG system
_agentic_rag_instance = None

def get_agentic_rag(persist_directory: Optional[str] = None) -> AgenticRAG:
    """
    Get or create the global AgenticRAG instance.
    
    Args:
        persist_directory: Directory to persist vector store (if None, a temp dir will be used)
        
    Returns:
        AgenticRAG instance
    """
    global _agentic_rag_instance
    
    if _agentic_rag_instance is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        _agentic_rag_instance = AgenticRAG(api_key=api_key, persist_directory=persist_directory)
    
    return _agentic_rag_instance

def process_query(
    query: str,
    chunks: List[Dict[str, Any]],
    model_name: str = "gpt-4o",
    temperature: float = 0,
    cleanup_after: bool = True  # Default to clean up after query
) -> Dict[str, Any]:
    """
    Process a query using the agentic RAG system.
    
    Args:
        query: The user query
        chunks: List of document chunks to use for retrieval
        model_name: The name of the OpenAI model to use
        temperature: The temperature parameter for the LLM
        cleanup_after: Whether to clean up (delete) the vector store after the query
        
    Returns:
        Dictionary containing the response and any additional information
    """
    # Create a temporary directory for this specific query
    temp_dir = tempfile.mkdtemp(prefix="agentic_rag_query_")
    
    try:
        # Create a new instance for this query rather than using the global one
        api_key = os.environ.get("OPENAI_API_KEY")
        instance = AgenticRAG(api_key=api_key, persist_directory=temp_dir)
        
        # Convert chunks to the format expected by the AgenticRAG system
        documents = convert_chunks_to_rag_documents(chunks)
        
        # Add documents to the RAG system
        instance.add_documents(documents)
        
        # Process the query
        response = instance.query(query)
        
        # Format the response for API output
        formatted_response = format_rag_response(response)
        
        return formatted_response
    
    finally:
        # Clean up if requested - cleanup the instance we just created
        if cleanup_after and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary vector store at {temp_dir}")
            except Exception as e:
                print(f"Error cleaning up vector store: {str(e)}")

def reset_agentic_rag():
    """
    Reset the global AgenticRAG instance.
    """
    global _agentic_rag_instance
    
    # Clean up if instance exists
    if _agentic_rag_instance:
        _agentic_rag_instance.cleanup()
    
    # Completely reset the instance to None
    _agentic_rag_instance = None
    
    # Re-initialize with fresh state
    api_key = os.environ.get("OPENAI_API_KEY")
    _agentic_rag_instance = AgenticRAG(api_key=api_key)
    
    return _agentic_rag_instance 