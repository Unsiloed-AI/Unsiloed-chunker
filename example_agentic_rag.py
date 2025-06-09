"""
Example script to demonstrate the usage of the agentic RAG system.
"""
import os
import json
import requests
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up API endpoint
API_URL = "http://localhost:8000"

def chunk_document(file_path, strategy="semantic"):
    """
    Chunk a document using the chunking API.
    
    Args:
        file_path: Path to the document file
        strategy: Chunking strategy to use
        
    Returns:
        Dictionary with chunking results
    """
    # Get the file name and extension
    file_name = os.path.basename(file_path)
    
    # Prepare the request
    url = f"{API_URL}/chunking"
    files = {"document_file": (file_name, open(file_path, "rb"))}
    data = {"strategy": strategy}
    
    # Send the request
    response = requests.post(url, files=files, data=data)
    
    # Check for errors
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    # Return the chunking results
    return response.json()

def query_agentic_rag(query, chunks_result, cleanup_after=True):
    """
    Query the agentic RAG system with the given query and chunks.
    
    Args:
        query: The query to process
        chunks_result: Result from the chunking service
        cleanup_after: Whether to clean up the vector store after processing
        
    Returns:
        Dictionary with the answer and additional information
    """
    # Prepare the request
    url = f"{API_URL}/agentic_rag/process_chunks"
    data = {
        "query": query,
        "chunks_result": chunks_result,
        "model_name": "gpt-4o",
        "temperature": 0,
        "cleanup_after": cleanup_after
    }
    
    # Send the request
    response = requests.post(url, json=data)
    
    # Check for errors
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    # Return the response
    return response.json()

def main():
    """Main function to demonstrate the agentic RAG system."""
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Example document path - replace with your own
    document_path = "example.pdf"
    
    # Check if the document exists
    if not os.path.exists(document_path):
        print(f"Error: Document not found at {document_path}")
        return
    
    print(f"Chunking document: {document_path}")
    chunks_result = chunk_document(document_path)
    
    if not chunks_result:
        return
    
    print(f"Document chunked successfully. Total chunks: {chunks_result.get('total_chunks', 0)}")
    
    # Example queries
    queries = [
        "What is the main topic of this document?",  # Simple query
        "What are the key findings and their implications?",  # Multi-hop query
        "What topics are discussed excluding financial data?"  # Negation query
    ]
    
    # Process each query separately (each with its own vector store)
    for i, query in enumerate(queries):
        print(f"\nProcessing query {i+1}: {query}")
        result = query_agentic_rag(query, chunks_result, cleanup_after=True)
        
        if result:
            print(f"Query type: {result.get('query_type', 'unknown')}")
            print(f"Answer: {result.get('answer', '')}")
            print("\nReasoning:")
            print(result.get('reasoning', ''))
            print("\nVector store has been cleaned up after processing.")

if __name__ == "__main__":
    main() 