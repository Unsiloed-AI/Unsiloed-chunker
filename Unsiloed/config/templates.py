from typing import List, Dict, Any

def table_template(chunks: List[Dict[str, Any]], use_collapsible: bool, metadata_fields: List[str]) -> str:
    """
    Generate Markdown output using a table-based template.

    Args:
        chunks: List of chunk dictionaries with text and metadata
        use_collapsible: Whether to use collapsible sections
        metadata_fields: List of metadata fields to include

    Returns:
        Markdown-formatted string
    """
    if not chunks:
        return "# Document Chunks\n\nNo chunks generated."

    markdown = "# Document Chunks\n\n"
    markdown += "## Chunk Summary\n\n"
    markdown += "| Index | Text Preview | Metadata |\n"
    markdown += "|-------|--------------|----------|\n"

    for i, chunk in enumerate(chunks, 1):
        text = chunk["text"].replace("\n", " ").strip()[:50] + "..."  # Preview
        metadata = chunk.get("metadata", {})
        metadata_str = ", ".join(f"{k}: {v}" for k, v in metadata.items() if k in metadata_fields)

        markdown += f"| {i} | {text} | {metadata_str} |\n"

    markdown += "\n## Chunk Details\n\n"

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        heading = metadata.get("title") or metadata.get("heading") or f"Chunk {i}"
        markdown += f"### {heading}\n\n"

        if use_collapsible:
            markdown += f"<details><summary>{heading}</summary>\n\n"
            markdown += f"{chunk['text']}\n\n"
            if metadata.get("strategy") == "semantic" and "position" in metadata_fields and metadata.get("position"):
                markdown += f"> Position: {metadata['position']}\n\n"
            markdown += "</details>\n\n"
        else:
            markdown += f"{chunk['text']}\n\n"
            if metadata.get("strategy") == "semantic" and "position" in metadata_fields and metadata.get("position"):
                markdown += f"> Position: {metadata['position']}\n\n"

    return markdown

def list_template(chunks: List[Dict[str, Any]], use_collapsible: bool, metadata_fields: List[str]) -> str:
    """
    Generate Markdown output using a list-based template.

    Args:
        chunks: List of chunk dictionaries with text and metadata
        use_collapsible: Whether to use collapsible sections
        metadata_fields: List of metadata fields to include

    Returns:
        Markdown-formatted string
    """
    if not chunks:
        return "# Document Chunks\n\nNo chunks generated."

    markdown = "# Document Chunks\n\n"
    markdown += "## Chunk Summary\n\n"

    for i, chunk in enumerate(chunks, 1):
        text = chunk["text"].replace("\n", " ").strip()[:50] + "..."  # Preview
        metadata = chunk.get("metadata", {})
        metadata_str = ", ".join(f"{k}: {v}" for k, v in metadata.items() if k in metadata_fields)
        markdown += f"- **Chunk {i}**: {text} ({metadata_str})\n"

    markdown += "\n## Chunk Details\n\n"

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        heading = metadata.get("title") or metadata.get("heading") or f"Chunk {i}"
        markdown += f"### {heading}\n\n"

        if use_collapsible:
            markdown += f"<details><summary>{heading}</summary>\n\n"
            markdown += f"{chunk['text']}\n\n"
            if metadata.get("strategy") == "semantic" and "position" in metadata_fields and metadata.get("position"):
                markdown += f"> Position: {metadata['position']}\n\n"
            markdown += "</details>\n\n"
        else:
            markdown += f"{chunk['text']}\n\n"
            if metadata.get("strategy") == "semantic" and "position" in metadata_fields and metadata.get("position"):
                markdown += f"> Position: {metadata['position']}\n\n"

    return markdown

# CHANGED: Added compact template
def compact_template(chunks: List[Dict[str, Any]], use_collapsible: bool, metadata_fields: List[str]) -> str:
    """
    Generate minimal Markdown output with only chunk text and headings.

    Args:
        chunks: List of chunk dictionaries with text and metadata
        use_collapsible: Whether to use collapsible sections
        metadata_fields: Ignored in this template

    Returns:
        Markdown-formatted string
    """
    if not chunks:
        return "# Document Chunks\n\nNo chunks generated."

    markdown = "# Document Chunks\n\n"

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        heading = metadata.get("title") or metadata.get("heading") or f"Chunk {i}"
        markdown += f"### {heading}\n\n"

        if use_collapsible:
            markdown += f"<details><summary>{heading}</summary>\n\n"
            markdown += f"{chunk['text']}\n\n"
            markdown += "</details>\n\n"
        else:
            markdown += f"{chunk['text']}\n\n"

    return markdown

# CHANGED: Added nested template
def nested_template(chunks: List[Dict[str, Any]], use_collapsible: bool, metadata_fields: List[str]) -> str:
    """
    Generate Markdown output with nested lists based on chunk position.

    Args:
        chunks: List of chunk dictionaries with text and metadata
        use_collapsible: Whether to use collapsible sections
        metadata_fields: List of metadata fields to include

    Returns:
        Markdown-formatted string
    """
    if not chunks:
        return "# Document Chunks\n\nNo chunks generated."

    markdown = "# Document Chunks\n\n"
    markdown += "## Chunk Hierarchy\n\n"

    # Group chunks by position (beginning, middle, end)
    positions = {"beginning": [], "middle": [], "end": []}
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        pos = metadata.get("position", "middle").lower()
        positions[pos].append((i, chunk, metadata))

    for pos in ["beginning", "middle", "end"]:
        if positions[pos]:
            markdown += f"- **{pos.capitalize()}**\n"
            for i, chunk, metadata in positions[pos]:
                text = chunk["text"].replace("\n", " ").strip()[:50] + "..."
                metadata_str = ", ".join(f"{k}: {v}" for k, v in metadata.items() if k in metadata_fields)
                markdown += f"  - Chunk {i}: {text} ({metadata_str})\n"

    markdown += "\n## Chunk Details\n\n"

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        heading = metadata.get("title") or metadata.get("heading") or f"Chunk {i}"
        markdown += f"### {heading}\n\n"

        if use_collapsible:
            markdown += f"<details><summary>{heading}</summary>\n\n"
            markdown += f"{chunk['text']}\n\n"
            if metadata.get("strategy") == "semantic" and "position" in metadata_fields and metadata.get("position"):
                markdown += f"> Position: {metadata['position']}\n\n"
            markdown += "</details>\n\n"
        else:
            markdown += f"{chunk['text']}\n\n"
            if metadata.get("strategy") == "semantic" and "position" in metadata_fields and metadata.get("position"):
                markdown += f"> Position: {metadata['position']}\n\n"

    return markdown