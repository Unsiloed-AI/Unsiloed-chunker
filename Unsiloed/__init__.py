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
            - credentials (dict): containing apiKey for the model provider
            - strategy (str, optional): Chunking strategy to use (default: "semantic")
            - chunkSize (int, optional): Size of chunks (default: 1000)
            - overlap (int, optional): Overlap size (default: 100)
            - modelProvider (str, optional): Type of model provider to use (default: "openai")
            - modelConfig (dict, optional): Additional configuration for the model provider
    
    Returns:
        dict: A dictionary containing the processed chunks and metadata
    """
    # Set the API key from credentials
    if "credentials" in options and "apiKey" in options["credentials"]:
        if options.get("modelProvider") == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = options["credentials"]["apiKey"]
        else:
            os.environ["OPENAI_API_KEY"] = options["credentials"]["apiKey"]
    
    # Get file path
    file_path = options.get("filePath")
    if not file_path:
        raise ValueError("filePath is required")
    
    # Get chunking options
    strategy = options.get("strategy", "semantic")
    chunk_size = options.get("chunkSize", 1000)
    overlap = options.get("overlap", 100)
    
    # Get model provider configuration
    model_provider = options.get("modelProvider", "openai")
    model_config = options.get("modelConfig", {})
    
    # Handle URLs by downloading the file
    temp_file = None
    local_file_path = file_path
    
    if file_path.startswith(("http://", "https://")):
        try:
            response = requests.get(file_path)
            response.raise_for_status()
            
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(response.content)
            temp_file.close()
            
            local_file_path = temp_file.name
        except Exception as e:
            if temp_file:
                os.unlink(temp_file.name)
            raise ValueError(f"Failed to download file from URL: {str(e)}")
    
    try:
        # Process the document
        result = process_document_chunking(
            file_path=local_file_path,
            file_type=os.path.splitext(local_file_path)[1][1:].lower(),
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            model_provider=model_provider,
            model_config=model_config
        )
        
        return result
    finally:
        # Clean up temporary file if created
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

# Also provide a synchronous version for simpler usage
def process_sync(options):
    """Synchronous version of the process function"""
    import asyncio
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(process(options))
    loop.close()
    return result
