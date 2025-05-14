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
            - credentials (dict): containing API keys for model providers
            - modelProvider (str, optional): Model provider to use (default: from config)
            - strategy (str, optional): Chunking strategy to use (default: "semantic")
            - chunkSize (int, optional): Size of chunks (default: 1000)
            - overlap (int, optional): Overlap size (default: 100)

    Returns:
        dict: A dictionary containing the processed chunks and metadata
    """
    # Set API keys from credentials
    if "credentials" in options:
        credentials = options["credentials"]

        # Set OpenAI API key if provided
        if "apiKey" in credentials:
            os.environ["OPENAI_API_KEY"] = credentials["apiKey"]

        # Set Anthropic API key if provided
        if "anthropicApiKey" in credentials:
            os.environ["ANTHROPIC_API_KEY"] = credentials["anthropicApiKey"]

        # Set Hugging Face API key if provided
        if "huggingfaceApiKey" in credentials:
            os.environ["HUGGINGFACE_API_KEY"] = credentials["huggingfaceApiKey"]

        # Set local model path if provided
        if "localModelPath" in credentials:
            os.environ["LOCAL_MODEL_PATH"] = credentials["localModelPath"]

    # Get file path
    file_path = options.get("filePath")
    if not file_path:
        raise ValueError("filePath is required")

    # Get chunking options
    strategy = options.get("strategy", "semantic")
    chunk_size = options.get("chunkSize", 1000)
    overlap = options.get("overlap", 100)
    model_provider = options.get("modelProvider")

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
            elif file_path.lower().endswith(".doc"):
                file_type = "doc"
                suffix = ".doc"
            elif file_path.lower().endswith(".xlsx"):
                file_type = "xlsx"
                suffix = ".xlsx"
            elif file_path.lower().endswith(".xls"):
                file_type = "xls"
                suffix = ".xls"
            elif file_path.lower().endswith(".odt"):
                file_type = "odt"
                suffix = ".odt"
            elif file_path.lower().endswith(".ods"):
                file_type = "ods"
                suffix = ".ods"
            elif file_path.lower().endswith(".odp"):
                file_type = "odp"
                suffix = ".odp"
            elif file_path.lower().endswith(".txt"):
                file_type = "txt"
                suffix = ".txt"
            elif file_path.lower().endswith(".rtf"):
                file_type = "rtf"
                suffix = ".rtf"
            elif file_path.lower().endswith(".epub"):
                file_type = "epub"
                suffix = ".epub"
            else:
                raise ValueError("Unsupported file type. Supported formats: PDF, DOCX, PPTX, DOC, XLSX, XLS, ODT, ODS, ODP, TXT, RTF, EPUB.")

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
            elif file_path.lower().endswith(".doc"):
                file_type = "doc"
            elif file_path.lower().endswith(".xlsx"):
                file_type = "xlsx"
            elif file_path.lower().endswith(".xls"):
                file_type = "xls"
            elif file_path.lower().endswith(".odt"):
                file_type = "odt"
            elif file_path.lower().endswith(".ods"):
                file_type = "ods"
            elif file_path.lower().endswith(".odp"):
                file_type = "odp"
            elif file_path.lower().endswith(".txt"):
                file_type = "txt"
            elif file_path.lower().endswith(".rtf"):
                file_type = "rtf"
            elif file_path.lower().endswith(".epub"):
                file_type = "epub"
            else:
                raise ValueError("Unsupported file type. Supported formats: PDF, DOCX, PPTX, DOC, XLSX, XLS, ODT, ODS, ODP, TXT, RTF, EPUB.")

        # Process the document
        result = process_document_chunking(
            local_file_path,
            file_type,
            strategy,
            chunk_size,
            overlap,
            model_provider
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
