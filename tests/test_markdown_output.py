import pytest
from Unsiloed.utils.chunking import chunks_to_markdown

def test_chunks_to_markdown_basic():
    """Test basic Markdown conversion with simple chunks."""
    chunks = [
        {
            "text": "This is a simple chunk",
            "metadata": {
                "title": "Test Chunk 1",
                "strategy": "paragraph",
                "start_char": 0,
                "end_char": 20
            }
        }
    ]
    
    result = chunks_to_markdown(chunks)
    
    # Check if result contains expected Markdown elements
    assert "## Test Chunk 1" in result
    assert "<details>" in result
    assert "<summary>Metadata</summary>" in result
    assert "strategy" in result
    assert "paragraph" in result
    assert "---" in result

def test_chunks_to_markdown_with_table():
    """Test Markdown conversion with table content."""
    chunks = [
        {
            "text": """Type    Description    Wrapper
byte    8-bit signed    Byte
short   16-bit signed   Short""",
            "metadata": {
                "title": "Table Chunk",
                "strategy": "paragraph",
                "start_char": 0,
                "end_char": 100
            }
        }
    ]
    
    result = chunks_to_markdown(chunks)
    
    # Check if table is properly formatted
    assert "| Type" in result
    assert "| byte" in result
    assert "| short" in result
    assert "| Description" in result
    assert "| Wrapper" in result
    assert "| Byte" in result
    assert "| Short" in result

def test_chunks_to_markdown_with_headings():
    """Test Markdown conversion with heading content."""
    chunks = [
        {
            "text": "# Main Heading\n\nSome content\n\n## Sub Heading\n\nMore content",
            "metadata": {
                "title": "Heading Chunk",
                "strategy": "heading",
                "start_char": 0,
                "end_char": 100
            }
        }
    ]
    
    result = chunks_to_markdown(chunks)
    
    # Check if headings are preserved
    assert "# Main Heading" in result
    assert "## Sub Heading" in result
    assert "Some content" in result
    assert "More content" in result

def test_chunks_to_markdown_multiple_chunks():
    """Test Markdown conversion with multiple chunks."""
    chunks = [
        {
            "text": "First chunk content",
            "metadata": {
                "title": "First Chunk",
                "strategy": "paragraph",
                "start_char": 0,
                "end_char": 20
            }
        },
        {
            "text": "Second chunk content",
            "metadata": {
                "title": "Second Chunk",
                "strategy": "paragraph",
                "start_char": 21,
                "end_char": 41
            }
        }
    ]
    
    result = chunks_to_markdown(chunks)
    
    # Check if both chunks are present
    assert "## First Chunk" in result
    assert "## Second Chunk" in result
    assert "First chunk content" in result
    assert "Second chunk content" in result
    # Check if chunks are properly separated
    assert result.count("---") == 2

def test_chunks_to_markdown_empty_chunk():
    """Test Markdown conversion with an empty chunk."""
    chunks = [
        {
            "text": "",
            "metadata": {
                "title": "Empty Chunk",
                "strategy": "paragraph",
                "start_char": 0,
                "end_char": 0
            }
        }
    ]
    
    result = chunks_to_markdown(chunks)
    
    # Check if empty chunk is handled properly
    assert "## Empty Chunk" in result
    assert "<details>" in result
    assert "Empty Chunk" in result 