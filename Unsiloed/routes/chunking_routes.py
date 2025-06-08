from fastapi import APIRouter, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from Unsiloed.services.chunking import process_document_chunking, OutputFormat, MarkdownTemplate, AnnotationModel
from Unsiloed.utils.chunking import ChunkingStrategy
import os
import tempfile
import logging
import aiofiles
from starlette.background import BackgroundTask

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chunking"])


@router.post("/chunking")
async def chunk_document(
    document_file: UploadFile,
    strategy: ChunkingStrategy = Form("semantic"),
    chunk_size: int = Form(1000),
    overlap: int = Form(100),
    # CHANGED: Added output_format and use_collapsible parameters
    output_format: OutputFormat = Form("json"),
    use_collapsible: bool = Form(False),
    # CHANGED: Added new parameters
    markdown_template: MarkdownTemplate = Form("table"),
    metadata_fields: str = Form(None),  # Comma-separated string
    annotate: bool = Form(False),
    # CHANGED: Replaced save_to_file with download
    download: bool = Form(False),
    # CHANGED: Added annotation_model parameter
    annotation_model: AnnotationModel = Form("gpt-4o-mini")
):
    """
    Chunk a document file (PDF, DOCX, PPTX) according to the specified strategy.

    Args:
        document_file: The document file to process
        strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy (in characters)
        overlap: Overlap size for fixed strategy (in characters)
        output_format: Output format (json or markdown)
        use_collapsible: Use collapsible sections in Markdown output
        - markdown_template (str, optional): Markdown template style (table, list, compact, nested, default: "table")
        metadata_fields: Comma-separated list of metadata fields to include
        annotate: Include content-based annotations in Markdown
        download: Return Markdown output as a downloadable file
        annotation_model: OpenAI model for annotations (gpt-4o-mini or gpt-4o)

    Returns:
        JSON or Markdown response based on output_format and save_to_file
    """
    logger.info(f"Received request for document chunking using {strategy} strategy")
    logger.debug(f"Document file name: {document_file.filename}")

    # CHANGED: Parse metadata_fields
    metadata_fields_list = metadata_fields.split(",") if metadata_fields else None

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

    file_path = None  # Initialize file_path
    markdown_file_path = None
    try:
        # Read file content
        file_content = await document_file.read()

        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file.write(file_content)
            file_path = temp_file.name
            logger.debug(f"Document saved to temporary file: {file_path}")

        # Process the document with the requested chunking strategy
        result = process_document_chunking(
            file_path, file_type, strategy, chunk_size, overlap, output_format, use_collapsible, markdown_template, metadata_fields_list, annotate, annotation_model
        )

        # logger.info(f"Document chunking completed with {result['total_chunks']} chunks")
        logger.info(
            f"Document chunking completed with "
            f"{result['total_chunks'] if output_format == 'json' else 'Markdown output'}"
        )
        # CHANGED: Handle file export for Markdown
        if output_format == "markdown" and download:
            try:
                # Create a temporary file with a unique name
                markdown_file_path = os.path.join(tempfile.gettempdir(), f"{document_file.filename.rsplit('.', 1)[0]}_chunks_{os.urandom(4).hex()}.md")
                # Use aiofiles for asynchronous file writing
                async with aiofiles.open(markdown_file_path, "w", encoding="utf-8") as f:
                    await f.write(result)
                # Ensure file exists before serving
                if not os.path.exists(markdown_file_path):
                    raise FileNotFoundError(f"Markdown file not found: {markdown_file_path}")
                return FileResponse(
                    markdown_file_path,
                    media_type="text/markdown",
                    filename=f"{document_file.filename.rsplit('.', 1)[0]}_chunks.md",
                    background=BackgroundTask(os.unlink, markdown_file_path)
                )
            except Exception as e:
                logger.error(f"Failed to create Markdown file: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to create Markdown file: {str(e)}")
        
        # CHANGED: Return PlainTextResponse for Markdown
        if output_format == "markdown":
            return PlainTextResponse(content=result, media_type="text/markdown")
        return JSONResponse(content=result)

    # except Exception as e:
    #     logger.error(f"Error processing document: {str(e)}", exc_info=True)
    #     raise HTTPException(
    #         status_code=500, detail=f"Error processing document: {str(e)}"
    #     )

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    finally:
        # Clean up the temporary file
        if file_path and os.path.exists(file_path):
            logger.debug(f"Cleaning up temporary file: {file_path}")
            os.unlink(file_path)
        # if markdown_file_path and os.path.exists(markdown_file_path):
        #     logger.debug(f"Cleaning up temporary Markdown file: {markdown_file_path}")
        #     os.unlink(markdown_file_path)