"""
Utils module for Unsiloed.
"""

from .web_utils import (
    extract_text_from_html_file,
    extract_text_from_html_content,
    extract_text_from_markdown,
    extract_text_from_markdown_content,
    scrape_website_sync,
    scrape_website_async,
    validate_url,
    get_content_type_from_url,
    extract_links_from_html,
    WebProcessingError,
    URLValidationError,
    WebScrapingError,
    MarkdownProcessingError,
    HTMLProcessingError,
)

__all__ = [
    'extract_text_from_html_file',
    'extract_text_from_html_content',
    'extract_text_from_markdown',
    'extract_text_from_markdown_content',
    'scrape_website_sync',
    'scrape_website_async',
    'validate_url',
    'get_content_type_from_url',
    'extract_links_from_html',
    'WebProcessingError',
    'URLValidationError',
    'WebScrapingError',
    'MarkdownProcessingError',
    'HTMLProcessingError',
] 