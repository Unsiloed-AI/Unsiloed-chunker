"""
Utility functions for the agentic RAG system.
"""
from typing import List, Dict, Any

def convert_chunks_to_rag_documents(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert document chunks from the chunking service format to the format expected by the AgenticRAG system.
    
    Args:
        chunks: List of document chunks from the chunking service
        
    Returns:
        List of documents in the format expected by the AgenticRAG system
    """
    rag_documents = []
    
    for i, chunk in enumerate(chunks):
        # Extract text and metadata
        text = chunk.get("text", "")
        
        # Create a document with text and metadata
        document = {
            "text": text,
            "id": f"chunk-{i}",
            "chunk_index": i
        }
        
        # Copy any additional metadata
        for key, value in chunk.items():
            if key != "text":
                document[key] = value
        
        rag_documents.append(document)
    
    return rag_documents

def format_rag_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the response from the AgenticRAG system for API output.
    
    Args:
        response: Response from the AgenticRAG system
        
    Returns:
        Formatted response for API output
    """
    return {
        "answer": response.get("answer", ""),
        "reasoning": response.get("reasoning", ""),
        "source_chunks": response.get("sources", []),
        "query_type": _determine_query_type(response)
    }

def _determine_query_type(response: Dict[str, Any]) -> str:
    """
    Determine the type of query based on the response.
    
    Args:
        response: Response from the AgenticRAG system
        
    Returns:
        Query type: "simple", "multi-hop", or "negation"
    """
    # Extract the reasoning to determine query type
    reasoning = response.get("reasoning", "")
    raw_result = response.get("raw_result", {})
    
    # Look for indicators in the reasoning
    if "multi-hop" in reasoning.lower() or "sub-question" in reasoning.lower():
        return "multi-hop"
    elif "negation" in reasoning.lower() or "exclude" in reasoning.lower() or "not" in reasoning.lower():
        return "negation"
    else:
        return "simple" 