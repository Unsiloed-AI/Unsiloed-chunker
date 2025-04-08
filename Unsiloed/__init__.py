# App package
import os
import tempfile
import requests
from Unsiloed.services.chunking import process_document_chunking
from Unsiloed.utils.chunking import ChunkingStrategy

async def process(options):
    """
    Process a document file with OCR and chunking capabilities.
    
    Args:
        options (dict): A dictionary with the following keys:
            - filePath (str): URL or local path to the document
            - credentials (dict): containing apiKey for OpenAI
            - strategy (str, optional): Chunking strategy to use (default: "semantic")
            - chunkSize (int, optional): Size of chunks (default: 1000)
            - overlap (int, optional): Overlap size (default: 100)
    
    Returns:
        dict: A dictionary containing the processed chunks and metadata
    """
    # Set the OpenAI API key from credentials
    if "credentials" in options and "apiKey" in options["credentials"]:
        os.environ["OPENAI_API_KEY"] = options["credentials"]["apiKey"]
    
    # Get file path
    file_path = options.get("filePath")
    if not file_path:
        raise ValueError("filePath is required")
    
    # Get chunking options
    strategy = options.get("strategy", "semantic")
    chunk_size = options.get("chunkSize", 1000)
    overlap = options.get("overlap", 100)
    
    # Handle URLs by downloading the file
    temp_file = None
    local_file_path = file_path
    
    try:
        if file_path.startswith(("http://", "https://")):
            # Download the file to a temporary location
            response = requests.get(file_path)
            response.raise_for_status()
            
            # Determine file type from URL
            if file_path.lower().endswith(".pdf"):
                file_type = "pdf"
                suffix = ".pdf"
            elif file_path.lower().endswith(".docx"):
                file_type = "docx"
                suffix = ".docx"
            elif file_path.lower().endswith(".pptx"):
                file_type = "pptx"
                suffix = ".pptx"
            else:
                raise ValueError("Unsupported file type. Only PDF, DOCX, and PPTX are supported.")
                
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(response.content)
            temp_file.close()
            local_file_path = temp_file.name
        else:
            # Local file
            if file_path.lower().endswith(".pdf"):
                file_type = "pdf"
            elif file_path.lower().endswith(".docx"):
                file_type = "docx"
            elif file_path.lower().endswith(".pptx"):
                file_type = "pptx"
            else:
                raise ValueError("Unsupported file type. Only PDF, DOCX, and PPTX are supported.")
        
        # Process the document
        result = process_document_chunking(
            local_file_path, 
            file_type,
            strategy,
            chunk_size,
            overlap
        )
        
        return result
        
    finally:
        # Clean up temporary file if created
        if temp_file and os.path.exists(local_file_path):
            os.unlink(local_file_path)

# Also provide a synchronous version for simpler usage
def process_sync(options):
    """Synchronous version of the process function"""
    import asyncio
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(process(options))
    loop.close()
    return result
