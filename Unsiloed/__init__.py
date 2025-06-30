# App package
import os
import tempfile
import requests
from Unsiloed.services.chunking import process_document_chunking_with_agentic_rag, process_document_chunking, determine_file_type_from_url
from Unsiloed.utils.chunking import ChunkingStrategy
from Unsiloed.utils.web_utils import validate_url, get_content_type_from_url

async def process(options):
    """
    Process a document file or URL with the specified chunking strategy.
    
    Args:
        options: Dictionary containing:
            - filePath: Path to file or URL
            - credentials: Dictionary with API credentials (optional)
                - apiKey: OpenAI API key
            - strategy: Chunking strategy ("semantic", "fixed", "paragraph", "heading", "page")
            - chunkSize: Size of chunks for fixed strategy (default: 1000)
            - overlap: Overlap size for fixed strategy (default: 100)
    
    Returns:
        Dictionary with chunking results
    """
    file_path = options.get("filePath")
    if not file_path:
        raise ValueError("filePath is required")
    
    # Handle credentials
    credentials = options.get("credentials", {})
    api_key = credentials.get("apiKey")
    
    # Set OpenAI API key in environment if provided
    original_api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif not original_api_key:
        # Check if semantic strategy is being used without API key
        strategy = options.get("strategy", "semantic")
        if strategy == "semantic":
            raise ValueError("OpenAI API key is required for semantic chunking. Please provide it in credentials.apiKey or set OPENAI_API_KEY environment variable.")
    
    # Get chunking options
    strategy = options.get("strategy", "semantic")
    chunk_size = options.get("chunkSize", 1000)
    overlap = options.get("overlap", 100)
    
    # Handle URLs and local files
    temp_file = None
    local_file_path = file_path
    
    try:
        if file_path.startswith(("http://", "https://")):
            # Handle URLs
            validated_url = validate_url(file_path)
            
            # Check if it's a direct file download or a web page
            content_type = get_content_type_from_url(validated_url)
            
            if validated_url.lower().endswith((".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".markdown")):
                # Direct file download
                response = requests.get(validated_url)
                response.raise_for_status()
                
                # Determine file type from URL
                if validated_url.lower().endswith(".pdf"):
                    file_type = "pdf"
                    suffix = ".pdf"
                elif validated_url.lower().endswith(".docx"):
                    file_type = "docx"
                    suffix = ".docx"
                elif validated_url.lower().endswith(".pptx"):
                    file_type = "pptx"
                    suffix = ".pptx"
                elif validated_url.lower().endswith((".html", ".htm")):
                    file_type = "html"
                    suffix = ".html"
                elif validated_url.lower().endswith((".md", ".markdown")):
                    file_type = "markdown"
                    suffix = ".md"
                else:
                    raise ValueError("Unsupported file type. Supported formats: PDF, DOCX, PPTX, HTML, Markdown.")
                    
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(response.content)
                temp_file.close()
                local_file_path = temp_file.name
            else:
                # Web page - process directly as URL
                file_type = determine_file_type_from_url(validated_url)
                local_file_path = validated_url
        else:
            # Local file
            if file_path.lower().endswith(".pdf"):
                file_type = "pdf"
            elif file_path.lower().endswith(".docx"):
                file_type = "docx"
            elif file_path.lower().endswith(".pptx"):
                file_type = "pptx"
            elif file_path.lower().endswith((".html", ".htm")):
                file_type = "html"
            elif file_path.lower().endswith((".md", ".markdown")):
                file_type = "markdown"
            else:
                raise ValueError("Unsupported file type. Supported formats: PDF, DOCX, PPTX, HTML, Markdown.")
        
        # Process the document
        query = options.get("query")
        if query:
            # Use the new Agentic RAG integration if a query is provided
            result = process_document_chunking_with_agentic_rag(
                local_file_path,
                file_type,
                strategy,
                query=query,
                chunk_size=chunk_size,
                overlap=overlap
            )
        else:
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
        
        # Restore original API key if we modified it
        if api_key:
            if original_api_key:
                os.environ["OPENAI_API_KEY"] = original_api_key
            else:
                # Remove the key we set if there wasn't one originally
                os.environ.pop("OPENAI_API_KEY", None)

# Also provide a synchronous version for simpler usage
def process_sync(options):
    """
    Synchronous version of the process function
    
    Args:
        options: Dictionary containing:
            - filePath: Path to file or URL
            - credentials: Dictionary with API credentials (optional)
                - apiKey: OpenAI API key
            - strategy: Chunking strategy ("semantic", "fixed", "paragraph", "heading", "page")
            - chunkSize: Size of chunks for fixed strategy (default: 1000)
            - overlap: Overlap size for fixed strategy (default: 100)
    
    Returns:
        Dictionary with chunking results
    """
    import asyncio
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(process(options))
    loop.close()
    return result
