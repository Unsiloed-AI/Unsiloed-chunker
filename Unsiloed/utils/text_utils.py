"""
Common text processing utilities to avoid circular imports.
"""

import logging

logger = logging.getLogger(__name__)

def paragraph_chunking(text):
    """
    Split text by paragraphs.

    Args:
        text: The text to chunk

    Returns:
        List of chunks with metadata
    """
    # Split text by double newlines to identify paragraphs
    paragraphs = text.split("\n\n")

    # Remove empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_position = 0

    for paragraph in paragraphs:
        start_position = text.find(paragraph, current_position)
        end_position = start_position + len(paragraph)

        chunks.append(
            {
                "text": paragraph,
                "metadata": {
                    "start_char": start_position,
                    "end_char": end_position,
                    "strategy": "paragraph",
                },
            }
        )

        current_position = end_position

    return chunks 