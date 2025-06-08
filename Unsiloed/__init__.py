# App package
import os
import tempfile
import requests
from Unsiloed.services.chunking import process_document_chunking, OutputFormat, MarkdownTemplate, AnnotationModel
from Unsiloed.utils.chunking import ChunkingStrategy
from typing import Literal, Dict
import logging

logger = logging.getLogger(__name__)

async def process(options: Dict):
    """
    Process a document file with OCR and chunking capabilities.
    
    Args:
        options (dict): A dictionary with the following keys:
            - filePath (str): URL or local path to the document
            - credentials (dict): containing apiKey for OpenAI
            - strategy (str, optional): Chunking strategy to use (default: "semantic")
            - chunkSize (int, optional): Size of chunks (default: 1000)
            - overlap (int, optional): Overlap size (default: 100)
            - output_format (str, optional): Output format (json or markdown, default: "json")
            - use_collapsible (bool, optional): Use collapsible sections in Markdown (default: False)
            - markdown_template (str, optional): Markdown template style (table, list, compact, nested, default: "table")
            - metadata_fields (list, optional): Metadata fields to include in Markdown
            - annotate (bool, optional): Include content-based annotations in Markdown (default: False)
            - annotation_model (str, optional): OpenAI model for annotations (gpt-4o-mini or gpt-4o, default: "gpt-4o-mini")
            - download (bool, optional): Return Markdown output as a downloadable file (default: False)

    Returns:
        dict or str: Processed chunks and metadata (JSON) or Markdown string
    """

    if not isinstance(options, dict):
        logger.error(f"Invalid options type: expected dict, got {type(options).__name__}")
        raise ValueError(f"options must be a dictionary, got {type(options).__name__}")
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
    # CHANGED: Added output_format and use_collapsible options
    output_format: OutputFormat = options.get("output_format", os.environ.get("DEFAULT_OUTPUT_FORMAT", "json"))
    use_collapsible = options.get("use_collapsible", False),
    # CHANGED: Added new options
    markdown_template: MarkdownTemplate = options.get("markdown_template", os.environ.get("DEFAULT_MARKDOWN_TEMPLATE", "table"))
    metadata_fields = options.get("metadata_fields", None)
    annotate = options.get("annotate", False)
    # CHANGED: Added annotation_model option
    annotation_model: AnnotationModel = options.get("annotation_model", "gpt-4o-mini")
    download = options.get("download", False)
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
            overlap,
            output_format,
            use_collapsible,
            markdown_template,
            metadata_fields,
            annotate, 
            annotation_model,
            download
        )
        
        return result
        
    finally:
        # Clean up temporary file if created
        if temp_file and os.path.exists(local_file_path):
            os.unlink(local_file_path)

# Also provide a synchronous version for simpler usage
def process_sync(options: Dict):
    """Synchronous version of the process function"""
    import asyncio
    loop = asyncio.new_event_loop()
    # CHANGED: Added try-except for better error handling
    try:
        result = loop.run_until_complete(process(options))
        return result
    finally:
        loop.close()


"""

Nested Template with OpenAI Annotations:

curl -X POST "http://localhost:8000/chunking" \
  -F "document_file=@test.pdf" \
  -F "strategy=semantic" \
  -F "output_format=markdown" \
  -F "markdown_template=nested" \
  -F "metadata_fields=title,position" \
  -F "annotate=true" \
  -F "annotation_model=gpt-4o"

  


  Compact Template

  curl -X POST "http://localhost:8000/chunking" \
  -F "document_file=@test.pdf" \
  -F "strategy=semantic" \
  -F "output_format=markdown" \
  -F "markdown_template=compact"


"""