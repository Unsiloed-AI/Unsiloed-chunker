from Unsiloed.utils.chunking import (
    fixed_size_chunking,
    page_based_chunking,
    paragraph_chunking,
    heading_chunking,
    semantic_chunking,
    ChunkingStrategy
)
from Unsiloed.utils.openai import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_pptx,
    get_openai_client
)

# CHANGED: Import template functions
from Unsiloed.config.templates import table_template, list_template, compact_template, nested_template
import logging
from typing import Literal, List, Dict, Any
import json

logger = logging.getLogger(__name__)

# CHANGED: Added OutputFormat type for JSON or Markdown output
OutputFormat = Literal["json", "markdown"]
# CHANGED: Added MarkdownTemplate type
MarkdownTemplate = Literal["table", "list", "compact", "nested"]
# CHANGED: Added annotation model type
AnnotationModel = Literal["gpt-4o-mini", "gpt-4o"]

# CHANGED: Added function for content-based annotations
# CHANGED: Replaced with OpenAI-powered annotation
def annotate_chunk(chunk: Dict[str, Any], model: AnnotationModel = "gpt-4o-mini") -> str:
    """
    Generate an annotation tag for a chunk based on its content.

    Args:
        chunk: Chunk dictionary with text and metadata
        model: OpenAI model to use for annotation
    Returns:
        # Annotation tag (e.g., Summary, Code, Quote)
        Annotation tag (e.g., Introduction, Technical Detail)
    """
    # text = chunk["text"].lower()
    # if "```" in text:
    #     return "Code"
    # elif text.startswith(">") or "quote" in text:
    #     return "Quote"
    # elif len(text.split()) < 50:
    #     return "Summary"
    # return "Content"

        # Simple cache to avoid redundant API calls
    cache: Dict[str, str] = {}
    text = chunk["text"][:1000]  # Limit to 1000 chars for efficiency
    text_hash = hash(text)
    if text_hash in cache:
        return cache[text_hash]

    try:
        openai_client = get_openai_client()
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing text. Provide a single, concise label (1-2 words) that describes the primary purpose or type of the following text chunk (e.g., Introduction, Technical Detail, Conclusion, Code, Quote).",
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            max_tokens=10,
            temperature=0.3,
        )
        annotation = response.choices[0].message.content.strip()
        cache[text_hash] = annotation
        return annotation
    except Exception as e:
        logger.warning(f"OpenAI annotation failed: {str(e)}. Using fallback.")
        # Fallback to basic heuristic
        if "```" in text:
            return "Code"
        elif text.startswith(">"):
            return "Quote"
        elif len(text.split()) < 50:
            return "Summary"
        return "Content"

# CHANGED: Updated to support templates, metadata filtering, and annotations
def format_chunks_as_markdown(chunks: List[dict],
    use_collapsible: bool = False,
    template: MarkdownTemplate = "table",
    metadata_fields: List[str] = None,
    annotate: bool = False,
    # CHANGED: Added annotation_model parameter
    annotation_model: AnnotationModel = "gpt-4o-mini"
) -> str:
    """
    Convert chunks to Markdown format with headings, tables, and optional collapsible sections.

    Args:
        chunks: List of chunk dictionaries with text and metadata
        use_collapsible: Whether to wrap chunks in collapsible sections
        template: Markdown template style (table or list)
        metadata_fields: List of metadata fields to include (None for all)
        annotate: Whether to include content-based annotations
        annotation_model: OpenAI model for annotations
    Returns:
        Markdown-formatted string
    """

    if metadata_fields is None:
        metadata_fields = ["title", "heading", "position", "start_char", "end_char", "strategy", "page"]

    # Select template
    # Select template
    template_funcs = {
        "table": table_template,
        "list": list_template,
        "compact": compact_template,
        "nested": nested_template
    }
    template_func = template_funcs.get(template, table_template)
    markdown = template_func(chunks, use_collapsible, metadata_fields)

    # Add annotations if requested
    if annotate:
        annotated_markdown = "# Document Chunks with Annotations\n\n"
        for i, chunk in enumerate(chunks, 1):
            tag = annotate_chunk(chunk, annotation_model)
            metadata = chunk.get("metadata", {})
            heading = metadata.get("title") or metadata.get("heading") or f"Chunk {i}"
            annotated_markdown += f"### {heading} [{tag}]\n\n"
            if use_collapsible:
                annotated_markdown += f"<details><summary>{heading} [{tag}]</summary>\n\n"
                annotated_markdown += f"{chunk['text']}\n\n"
                annotated_markdown += "</details>\n\n"
            else:
                annotated_markdown += f"{chunk['text']}\n\n"
        return annotated_markdown

    return markdown
    # if not chunks:
    #     return "# Document Chunks\n\nNo chunks generated."
    # markdown = "# Document Chunks\n\n"
    # markdown += "## Chunk Summary\n\n"
    # markdown += "| Index | Text Preview | Metadata |\n"
    # markdown += "|-------|--------------|----------|\n"
    # for i, chunk in enumerate(chunks, 1):
    #     text = chunk["text"].replace("\n", " ").strip()[:50] + "..."  # Preview
    #     metadata = chunk.get("metadata", {})
    #     metadata_str = ", ".join(f"{k}: {v}" for k, v in metadata.items())

        # Use title or heading if available, else generic
    #     heading = metadata.get("title") or metadata.get("heading") or f"Chunk {i}"
        
    #     # Add chunk details
    #     markdown += f"| {i} | {text} | {metadata_str} |\n"

    # markdown += "\n## Chunk Details\n\n"

    # for i, chunk in enumerate(chunks, 1):
    #     metadata = chunk.get("metadata", {})
    #     heading = metadata.get("title") or metadata.get("heading") or f"Chunk {i}"
    #     markdown += f"### {heading}\n\n"

    #     if use_collapsible:
    #         markdown += f"<details><summary>{heading}</summary>\n\n"
    #         markdown += f"{chunk['text']}\n\n"
    #         # CHANGED: Add blockquote for semantic chunk metadata if available
    #         if metadata.get("strategy") == "semantic" and metadata.get("position"):
    #             markdown += f"> Position: {metadata['position']}\n\n"
    #         markdown += "</details>\n\n"
    #     else:
    #         markdown += f"{chunk['text']}\n\n"
    #         if metadata.get("strategy") == "semantic" and metadata.get("position"):
    #             markdown += f"> Position: {metadata['position']}\n\n"

    # return markdown
def process_document_chunking(
    file_path,
    file_type,
    strategy: ChunkingStrategy,
    chunk_size=1000,
    overlap=100,
    # CHANGED: Added output_format and use_collapsible parameters
    output_format: OutputFormat = "json",
    use_collapsible: bool = False,
    # CHANGED: Added template, metadata_fields, and annotate parameters
    markdown_template: MarkdownTemplate = "table",
    metadata_fields: List[str] = None,
    annotate: bool = False,
    # CHANGED: Added annotation_model parameter
    annotation_model: AnnotationModel = "gpt-4o-mini",
    download: bool = False
):
    """
    Process a document file (PDF, DOCX, PPTX) with the specified chunking strategy.

    Args:
        file_path: Path to the document file
        file_type: Type of document (pdf, docx, pptx)
        strategy: Chunking strategy to use
        chunk_size: Size of chunks for fixed strategy
        overlap: Overlap size for fixed strategy
        output_format: Output format (json or markdown)
        use_collapsible: Use collapsible sections in Markdown output
        markdown_template (str, optional): Markdown template style (table, list, compact, nested, default: "table")
        metadata_fields: List of metadata fields to include in Markdown
        annotate: Include content-based annotations in Markdown
        annotation_model: OpenAI model for annotations
        download: Return Markdown output as a downloadable file
    Returns:
        Dictionary with chunking results (JSON) or Markdown string
    """
    logger.info(
        f"Processing {file_type.upper()} document with {strategy} chunking strategy"
    )

    # # CHANGED: Validate parameters
    # if output_format not in ["json", "markdown"]:
    #     raise ValueError(f"Invalid output_format: {output_format}. Must be 'json' or 'markdown'.")
    # if markdown_template not in ["table", "list"]:
    #     raise ValueError(f"Invalid markdown_template: {markdown_template}. Must be 'table' or 'list'.")
    # if metadata_fields:
    #     valid_fields = ["title", "heading", "position", "start_char", "end_char", "strategy", "page"]
    #     invalid_fields = [f for f in metadata_fields if f not in valid_fields]
    #     if invalid_fields:
    #         raise ValueError(f"Invalid metadata_fields: {invalid_fields}. Valid fields: {valid_fields}")
        

    # CHANGED: Updated template validation
    if output_format not in ["json", "markdown"]:
        raise ValueError(f"Invalid output_format: {output_format}. Must be 'json' or 'markdown'.")
    if markdown_template not in ["table", "list", "compact", "nested"]:
        raise ValueError(f"Invalid markdown_template: {markdown_template}. Must be 'table', 'list', 'compact', or 'nested'.")
    if metadata_fields:
        valid_fields = ["title", "heading", "position", "start_char", "end_char", "strategy", "page"]
        invalid_fields = [f for f in metadata_fields if f not in valid_fields]
        if invalid_fields:
            raise ValueError(f"Invalid metadata_fields: {invalid_fields}. Valid fields: {valid_fields}")
    # CHANGED: Validate annotation_model
    if annotate and annotation_model not in ["gpt-4o-mini", "gpt-4o"]:
        raise ValueError(f"Invalid annotation_model: {annotation_model}. Must be 'gpt-4o-mini' or 'gpt-4o'.")

    # Handle page-based chunking for PDFs only
    if strategy == "page" and file_type == "pdf":
        chunks = page_based_chunking(file_path)
    else:
        # Extract text based on file type
        if file_type == "pdf":
            text = extract_text_from_pdf(file_path)
        elif file_type == "docx":
            text = extract_text_from_docx(file_path)
        elif file_type == "pptx":
            text = extract_text_from_pptx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Apply the selected chunking strategy
        if strategy == "fixed":
            chunks = fixed_size_chunking(text, chunk_size, overlap)
        elif strategy == "semantic":
            chunks = semantic_chunking(text)
        elif strategy == "paragraph":
            chunks = paragraph_chunking(text)
        elif strategy == "heading":
            chunks = heading_chunking(text)
        elif strategy == "page" and file_type != "pdf":
            # For non-PDF files, fall back to paragraph chunking for page strategy
            logger.warning(
                f"Page-based chunking not supported for {file_type}, falling back to paragraph chunking"
            )
            chunks = paragraph_chunking(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
    # Calculate statistics
    total_chunks = len(chunks)
    avg_chunk_size = (
        sum(len(chunk["text"]) for chunk in chunks) / total_chunks
        if total_chunks > 0
        else 0
    )

    result = {
        "file_type": file_type,
        "strategy": strategy,
        "total_chunks": total_chunks,
        "avg_chunk_size": avg_chunk_size,
        "chunks": chunks,
    }

    # CHANGED: Pass new parameters to format_chunks_as_markdown
    if output_format == "markdown":
        return format_chunks_as_markdown(chunks, use_collapsible, markdown_template, metadata_fields, annotate, annotation_model)
    return result
