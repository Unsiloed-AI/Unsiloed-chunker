"""
Web and Markdown Processing Utilities

This module provides utilities for processing HTML pages, websites, and markdown files:
- HTML content extraction and cleaning
- Website scraping with proper headers and error handling
- Markdown parsing and text extraction
- URL validation and normalization

Dependencies:
- BeautifulSoup4 for HTML parsing
- requests/aiohttp for web scraping
- markdown for markdown processing
- html2text for HTML to text conversion
- validators for URL validation

Author: Unsiloed Team
Version: 1.0.0
"""

import os
import javatools
from PIL import Image
from unstructured.partition.auto import partition
import fitz  # PyMuPDF
import filetype
import pytesseract
from numba import njit
from docx import Document
import random
import xgboost as xgb
import dask.dataframe as dd
from dask.distributed import LocalCluster
from multiprocessing import Pool
import asyncio
import aiohttp
import requests
import markdown
import html2text
import validators
import logging
import re
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, Comment
from pathlib import Path

logger = logging.getLogger(__name__)

class WebProcessingError(Exception):
    """Base exception for web processing operations."""
    pass

class URLValidationError(WebProcessingError):
    """Exception raised when URL validation fails."""
    pass

class WebScrapingError(WebProcessingError):
    """Exception raised when web scraping fails."""
    pass

class MarkdownProcessingError(WebProcessingError):
    """Exception raised when markdown processing fails."""
    pass

class HTMLProcessingError(WebProcessingError):
    """Exception raised when HTML processing fails."""
    pass


def validate_url(url: str) -> str:
    """
    Validate and normalize a URL.
    
    Args:
        url: URL to validate
        
    Returns:
        Normalized URL
        
    Raises:
        URLValidationError: If URL is invalid
    """
    if not isinstance(url, str):
        raise URLValidationError(f"URL must be a string, got {type(url)}")
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    if not validators.url(url):
        raise URLValidationError(f"Invalid URL: {url}")
    
    return url


def extract_text_from_markdown(file_path: str) -> str:
    """
    Extract text content from a markdown file.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Extracted text content
        
    Raises:
        MarkdownProcessingError: If markdown processing fails
    """
    try:
        if not Path(file_path).exists():
            raise MarkdownProcessingError(f"Markdown file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            markdown_content = file.read()
        
        return extract_text_from_markdown_content(markdown_content)
        
    except Exception as e:
        logger.error(f"Error extracting text from markdown file {file_path}: {str(e)}")
        raise MarkdownProcessingError(f"Failed to process markdown file: {str(e)}")


def extract_text_from_markdown_content(markdown_content: str) -> str:
    """
    Extract text content from markdown string.
    
    Args:
        markdown_content: Markdown content as string
        
    Returns:
        Extracted text content
        
    Raises:
        MarkdownProcessingError: If markdown processing fails
    """
    try:
        # Convert markdown to HTML
        html_content = markdown.markdown(
            markdown_content,
            extensions=['extra', 'codehilite', 'toc', 'tables']
        )
        
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_emphasis = False
        h.body_width = 0  # Don't wrap lines
        
        text_content = h.handle(html_content)
        
        text_content = re.sub(r'\n\s*\n\s*\n', '\n\n', text_content)
        text_content = text_content.strip()
        
        return text_content
        
    except Exception as e:
        logger.error(f"Error processing markdown content: {str(e)}")
        raise MarkdownProcessingError(f"Failed to process markdown content: {str(e)}")


def extract_text_from_html_file(file_path: str) -> str:
    """
    Extract text content from an HTML file.
    
    Args:
        file_path: Path to the HTML file
        
    Returns:
        Extracted text content
        
    Raises:
        HTMLProcessingError: If HTML processing fails
    """
    try:
        if not Path(file_path).exists():
            raise HTMLProcessingError(f"HTML file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        return extract_text_from_html_content(html_content)
        
    except Exception as e:
        logger.error(f"Error extracting text from HTML file {file_path}: {str(e)}")
        raise HTMLProcessingError(f"Failed to process HTML file: {str(e)}")


def extract_text_from_html_content(html_content: str, preserve_structure: bool = True) -> str:
    """
    Extract text content from HTML string.
    
    Args:
        html_content: HTML content as string
        preserve_structure: Whether to preserve document structure
        
    Returns:
        Extracted text content
        
    Raises:
        HTMLProcessingError: If HTML processing fails
    """
    try:
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        if preserve_structure:
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.ignore_emphasis = False
            h.body_width = 0  # Don't wrap lines
            h.ignore_tables = False
            
            text_content = h.handle(str(soup))
        else:
            text_content = soup.get_text()
        
        # Clean up extra whitespace
        text_content = re.sub(r'\n\s*\n\s*\n', '\n\n', text_content)
        text_content = text_content.strip()
        
        return text_content
        
    except Exception as e:
        logger.error(f"Error processing HTML content: {str(e)}")
        raise HTMLProcessingError(f"Failed to process HTML content: {str(e)}")


def scrape_website_sync(url: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Scrape content from a website synchronously.
    
    Args:
        url: URL to scrape
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary containing scraped content and metadata
        
    Raises:
        WebScrapingError: If scraping fails
    """
    try:
        # Validate URL
        url = validate_url(url)
        
        # Set up headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Make request
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Extract content
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Extract metadata
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title"
        
        meta_description = soup.find('meta', attrs={'name': 'description'})
        description = meta_description.get('content', '').strip() if meta_description else ""
        
        text_content = extract_text_from_html_content(response.text)
        
        return {
            'url': url,
            'title': title_text,
            'description': description,
            'content': text_content,
            'status_code': response.status_code,
            'content_type': response.headers.get('content-type', ''),
            'content_length': len(text_content),
        }
        
    except requests.RequestException as e:
        logger.error(f"Request error scraping {url}: {str(e)}")
        raise WebScrapingError(f"Failed to scrape website: {str(e)}")
    except Exception as e:
        logger.error(f"Error scraping website {url}: {str(e)}")
        raise WebScrapingError(f"Failed to scrape website: {str(e)}")


async def scrape_website_async(url: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Scrape content from a website asynchronously.
    
    Args:
        url: URL to scrape
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary containing scraped content and metadata
        
    Raises:
        WebScrapingError: If scraping fails
    """
    try:
        # Validate URL
        url = validate_url(url)
        
        # Set up headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Make async request
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                html_content = await response.text()
                
                # Extract content
                soup = BeautifulSoup(html_content, 'lxml')
                
                # Extract metadata
                title = soup.find('title')
                title_text = title.get_text().strip() if title else "No title"
                
                meta_description = soup.find('meta', attrs={'name': 'description'})
                description = meta_description.get('content', '').strip() if meta_description else ""
                
                text_content = extract_text_from_html_content(html_content)
                
                return {
                    'url': url,
                    'title': title_text,
                    'description': description,
                    'content': text_content,
                    'status_code': response.status,
                    'content_type': response.headers.get('content-type', ''),
                    'content_length': len(text_content),
                }
                
    except aiohttp.ClientError as e:
        logger.error(f"Client error scraping {url}: {str(e)}")
        raise WebScrapingError(f"Failed to scrape website: {str(e)}")
    except Exception as e:
        logger.error(f"Error scraping website {url}: {str(e)}")
        raise WebScrapingError(f"Failed to scrape website: {str(e)}")


def extract_links_from_html(html_content: str, base_url: str = None) -> List[Dict[str, str]]:
    """
    Extract all links from HTML content.
    
    Args:
        html_content: HTML content as string
        base_url: Base URL for resolving relative links
        
    Returns:
        List of dictionaries containing link information
    """
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text().strip()
            
            # Resolve relative URLs
            if base_url and not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                href = urljoin(base_url, href)
            
            links.append({
                'url': href,
                'text': text,
                'title': link.get('title', ''),
            })
        
        return links
        
    except Exception as e:
        logger.error(f"Error extracting links from HTML: {str(e)}")
        return []


def get_content_type_from_url(url: str) -> str:
    """
    Determine content type from URL extension or by making a HEAD request.
    
    Args:
        url: URL to check
        
    Returns:
        Content type string
    """
    try:
        # First try to determine from URL extension
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        if path.endswith('.html') or path.endswith('.htm'):
            return 'html'
        elif path.endswith('.md') or path.endswith('.markdown'):
            return 'markdown'
        elif path.endswith('.txt'):
            return 'text'
        
        try:
            response = requests.head(url, timeout=10)
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/html' in content_type:
                return 'html'
            elif 'text/markdown' in content_type:
                return 'markdown'
            elif 'text/plain' in content_type:
                return 'text'
            else:
                return 'html'
                
        except requests.RequestException:
            return 'html' 
            
    except Exception as e:
        logger.error(f"Error determining content type for {url}: {str(e)}")
        return 'html'
    
    def extract_jar(filepath):
    with javatools.unpack.unpack_class(filepath) as unpacker:
        return "\n".join(unpacker.get_method_names())

def extract_ico(file_path):
    with Image.open(file_path) as img:
        return f"ICO image: {img.size[0]}x{img.size[1]}, {len(img.info)} metadata entries"

def extract_pdf(file_path):
    with fitz.open(file_path) as doc:
        return "\n".join(page.get_text() for page in doc)

def detect_file_type(file_path):
    kind = filetype.guess(file_path)
    return kind.mime if kind else "unknown"

def perform_ocr(file_path):
    return pytesseract.image_to_string(Image.open(file_path))

def get_all_files(directory):
    text_file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            absolute_path = os.path.abspath(os.path.join(root, file))
            text_file_paths.append(absolute_path)
    return text_file_paths

def extract_with_unstructured(file_path):
    elements = partition(filename=file_path)
    content = "\n".join([str(element) for element in elements])
    return content

def extract_data(file_path):
    try:
        file_type = detect_file_type(file_path)
        if file_type == "application/java-archive":
            return extract_jar(file_path)
        elif file_type == "application/pdf":
            return extract_pdf(file_path)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return extract_docx(file_path)
        elif file_type.startswith("image/"):
            return perform_ocr(file_path)
        else:
            try:
                return extract_with_unstructured(file_path)
            except:
                with open(file_path, 'r') as f:
                    content = f.read()
                return content
    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"

directory = ''  # Replace with the path to your folder
files = get_all_files(directory)

results = {}
working = []
n_working = []

for file_path in files:
    content = extract_data(file_path)
    results[file_path] = content

    if content.startswith("Error"):
        n_working.append(os.path.splitext(file_path)[1])
    else:
        working.append(os.path.splitext(file_path)[1])

@njit
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

df = dd.read_parquet("s3://my-data/")
dtrain = xgb.dask.DaskDMatrix(df)

model = xgb.dask.train(
    dtrain,
    {"tree_method": "hist", },
    ...
)

df = dask.datasets.timeseries()  # Randomly generated data
# df = dd.read_parquet(...)      # In practice, you would probably read data though

train, test = df.random_split([0.80, 0.20])
X_train, y_train, X_test, y_test = ...

with LocalCluster() as cluster:
    with cluster.get_client() as client:
        d_train = xgb.dask.DaskDMatrix(client, X_train, y_train, enable_categorical=True)
        model = xgb.dask.train(...d_train,)
        predictions = xgb.dask.predict(client, model, X_test)

df = dd.read_parquet("/path/to/my/data.parquet")

model = load_model("/path/to/my/model")

# pandas code
# predictions = model.predict(df)
# predictions.to_parquet("/path/to/results.parquet")

# Dask code
predictions = df.map_partitions(model.predict)
predictions.to_parquet("/path/to/results.parquet")

print("Non-working file extensions:")
print(list(set(n_working)))
print("Working file extensions:")
print(list(set(working)))
