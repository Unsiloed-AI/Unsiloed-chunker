import os
import base64
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import logging
import concurrent.futures
import PyPDF2
from dotenv import load_dotenv
import numpy as np
import cv2
from Unsiloed.utils.text_utils import paragraph_chunking

load_dotenv()

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

# Instead of initializing at import, create a function to get the client
client = None


def get_openai_client():
    """Get an OpenAI client with proper configuration"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        logger.debug("Attempting to create OpenAI client...")

        # Create client with explicit parameters
        client = OpenAI(api_key=api_key, timeout=60.0, max_retries=3)
        logger.debug(
            "OpenAI client created, now testing..."
        )  # Log after client creation

        # Test the client by listing available models
        models = client.models.list()
        if models and hasattr(models, "data") and len(models.data) > 0:
            logger.debug(
                f"OpenAI client initialized successfully, available models: {len(models.data)}"
            )
            return client
        else:
            logger.error("OpenAI client initialized but returned no models.")
            return None

    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {str(e)}")
        return None


def encode_image_to_base64(image_path):
    """
    Encode an image to base64.

    Args:
        image_path: Path to the image file or numpy array

    Returns:
        Base64 encoded string of the image
    """
    logger.debug("Encoding image to base64")

    # Handle numpy array (from CV2)
    if isinstance(image_path, np.ndarray):
        success, buffer = cv2.imencode(".jpg", image_path)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer).decode("utf-8")

    # Handle file path
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_extraction_prompt(schema: Dict[str, Any], page_count: int) -> str:
    """
    Create a prompt instructing the model how to extract data according to the schema.

    Args:
        schema: JSON schema defining the structure
        page_count: Number of pages in the document

    Returns:
        Prompt string for the model
    """
    # Create a compact JSON representation of the schema
    schema_str = json.dumps(schema, indent=2)

    prompt = f"""
    You are an expert at extracting structured data from documents.
    
    I have a document with {page_count} pages that has been converted to images. I need you to extract specific information from these images according to the following JSON schema:
    
    {schema_str}
    
    Please follow these instructions carefully:
    
    1. Examine all {page_count} images thoroughly to find the requested information.
    2. Extract the exact text from the document that matches each field in the schema.
    3. If you cannot find information for a specific field in any of the pages, return an empty string or null value for that field.
    4. For array fields, include all instances found throughout the document.
    5. Maintain the structure defined in the schema exactly.
    6. Return only the extracted data as a valid JSON object, matching the structure of the schema.
    7. Do not add any explanatory text or notes outside the JSON structure.
    8. Be precise and accurate in your extraction.
    9. If text is unclear or ambiguous, make your best guess based on context.
    10. For dates, numbers, and other formatted data, maintain the format as shown in the document.
    11. IMPORTANT: Your response MUST be a valid JSON object that exactly matches the structure of the provided schema.
    12. IMPORTANT: Do not include any explanations, just return the JSON object.
    
    Your response should be a valid JSON object containing only the extracted data.
    """

    return prompt


def semantic_chunk_with_structured_output(text: str) -> List[Dict[str, Any]]:
    """
    Use OpenAI API to create semantic chunks from text using JSON mode.

    Args:
        text: The text to chunk

    Returns:
        List of chunks with metadata
    """

    # If text is too long, split it first using a simpler method
    # and then process each part in parallel
    if len(text) > 25000:
        logger.info(
            "Text too long for direct semantic chunking, applying parallel processing"
        )
        return process_long_text_semantically(text)

    try:
        # Get the OpenAI client
        openai_client = get_openai_client()

        # Create a prompt for the OpenAI model with JSON mode
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing and dividing text into meaningful semantic chunks. Your output should be valid JSON.",
                },
                {
                    "role": "user",
                    "content": f"""Please analyze the following text and divide it into logical semantic chunks. 
                    Each chunk should represent a cohesive unit of information or a distinct section.
                    
                    Return your results as a JSON object with this structure:
                    {{
                        "chunks": [
                            {{
                                "text": "the text of the chunk",
                                "title": "a descriptive title for this chunk",
                                "position": "beginning/middle/end"
                            }},
                            ...
                        ]
                    }}
                    
                    Text to chunk:
                    
                    {text}""",
                },
            ],
            max_tokens=4000,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        # Parse the response
        result = json.loads(response.choices[0].message.content)

        # Convert the response to our standard chunk format
        chunks = []
        current_position = 0

        for i, chunk_data in enumerate(result.get("chunks", [])):
            chunk_text = chunk_data.get("text", "")
            # Find the chunk in the original text to get accurate character positions
            start_position = text.find(chunk_text, current_position)
            if start_position == -1:
                # If exact match not found, use approximate position
                start_position = current_position

            end_position = start_position + len(chunk_text)

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "title": chunk_data.get("title", f"Chunk {i + 1}"),
                        "position": chunk_data.get("position", "unknown"),
                        "start_char": start_position,
                        "end_char": end_position,
                        "strategy": "semantic",
                    },
                }
            )

            current_position = end_position

        return chunks

    except Exception as e:
        logger.error(f"Error in semantic chunking with JSON mode: {str(e)}")
        # Fall back to paragraph chunking if semantic chunking fails
        logger.info("Falling back to paragraph chunking")
        # We'll just do basic paragraph chunking here
        paragraphs = text.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_position = 0

        for i, paragraph in enumerate(paragraphs):
            start_position = text.find(paragraph, current_position)
            if start_position == -1:
                start_position = current_position

            end_position = start_position + len(paragraph)

            chunks.append(
                {
                    "text": paragraph,
                    "metadata": {
                        "title": f"Paragraph {i + 1}",
                        "position": "unknown",
                        "start_char": start_position,
                        "end_char": end_position,
                        "strategy": "paragraph",  # Fall back strategy
                    },
                }
            )

            current_position = end_position

        return chunks


def process_long_text_semantically(text: str) -> List[Dict[str, Any]]:
    """
    Process a long text by breaking it into smaller pieces and chunking each piece semantically.
    Uses parallel processing and JSON mode for better performance.

    Args:
        text: The long text to process

    Returns:
        List of semantic chunks
    """
    # Check if this might be Excel data by looking for patterns
    is_likely_excel = "Sheet:" in text and (", " in text or " | " in text)
    
    # For Excel data, use a special approach
    if is_likely_excel:
        logger.info("Detected Excel data, using Excel-specific chunking approach")
        return process_excel_text(text)
    
    # For regular text, use the standard approach
    # Create chunks of 20000 characters with 500 character overlap
    # Reduced from 25000 to improve reliability
    text_chunks = []
    chunk_size = 20000
    overlap = 500
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # Try to find a natural break point (newline) near the end to avoid cutting mid-sentence
        if end < text_length and end - start > chunk_size // 2:
            natural_break = text.rfind("\n", start + chunk_size // 2, end)
            if natural_break != -1:
                end = natural_break + 1

        text_chunks.append(text[start:end])
        start = end - overlap if end < text_length else text_length

    # Process each chunk in parallel
    all_semantic_chunks = []
    failed_chunks = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Define a worker function
        def process_chunk(chunk_idx, chunk_text):
            try:
                # Get the OpenAI client
                openai_client = get_openai_client()

                # Process this chunk with JSON mode
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at analyzing and dividing text into meaningful semantic chunks. Your output should be valid JSON.",
                        },
                        {
                            "role": "user",
                            "content": f"""Please analyze the following text and divide it into logical semantic chunks. 
                            Each chunk should represent a cohesive unit of information or a distinct section.
                            
                            Return your results as a JSON object with this structure:
                            {{
                                "chunks": [
                                    {{
                                        "text": "the text of the chunk",
                                        "title": "a descriptive title for this chunk",
                                        "position": "beginning/middle/end"
                                    }},
                                    ...
                                ]
                            }}
                            
                            IMPORTANT: Ensure your JSON is properly formatted with no unterminated strings.
                            IMPORTANT: Escape any special characters in the text that might break JSON formatting.
                            
                            Text to chunk:
                            
                            {chunk_text}""",
                        },
                    ],
                    max_tokens=4000,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )

                # Parse the response
                try:
                    result = json.loads(response.choices[0].message.content)
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON parsing error in chunk {chunk_idx}: {str(json_err)}")
                    # Return empty list and mark as failed
                    return [], chunk_idx

                # Convert the response to our standard chunk format
                sub_chunks = []
                current_position = 0

                for i, chunk_data in enumerate(result.get("chunks", [])):
                    chunk_text = chunk_data.get("text", "")
                    # Fix the position calculation
                    start_position = current_position
                    end_position = start_position + len(chunk_text)

                    sub_chunks.append(
                        {
                            "text": chunk_text,
                            "metadata": {
                                "title": chunk_data.get("title", f"Subchunk {i + 1}"),
                                "position": chunk_data.get("position", "unknown"),
                                "start_char": start_position,
                                "end_char": end_position,
                                "strategy": "semantic",
                            },
                        }
                    )

                    current_position = end_position

                return sub_chunks, None
            except Exception as e:
                logger.error(
                    f"Error processing semantic subchunk {chunk_idx} with JSON mode: {str(e)}"
                )
                return [], chunk_idx

        # Submit all tasks and gather results
        futures = [executor.submit(process_chunk, idx, chunk) for idx, chunk in enumerate(text_chunks)]
        for future in concurrent.futures.as_completed(futures):
            result, failed_idx = future.result()
            all_semantic_chunks.extend(result)
            if failed_idx is not None:
                failed_chunks.append(failed_idx)
    
    # If all chunks failed, fall back to paragraph chunking
    if len(failed_chunks) == len(text_chunks):
        logger.warning("All semantic chunks failed to process, falling back to paragraph chunking")
        return paragraph_chunking(text)
    
    # If some chunks failed but not all, process them with paragraph chunking
    if failed_chunks:
        logger.warning(f"{len(failed_chunks)} chunks failed, processing them with paragraph chunking")
        for idx in failed_chunks:
            paragraph_chunks = paragraph_chunking(text_chunks[idx])
            # Adjust the strategy to indicate fallback
            for chunk in paragraph_chunks:
                chunk["metadata"]["strategy"] = "semantic_fallback_paragraph"
            all_semantic_chunks.extend(paragraph_chunks)

    return all_semantic_chunks


def process_excel_text(text: str) -> List[Dict[str, Any]]:
    """
    Special processing for Excel data that avoids JSON parsing issues.
    Splits by sheets and then by rows to create more reliable chunks.
    
    Args:
        text: The Excel text to process
        
    Returns:
        List of chunks with metadata
    """
    logger.info("Processing Excel data with sheet-based chunking")
    
    # Split by double newlines to separate sheets
    sheets = text.split("\n\n")
    
    chunks = []
    current_position = 0
    
    for sheet in sheets:
        if not sheet.strip():
            continue
            
        # Check if this is a sheet header
        lines = sheet.split("\n")
        if lines and lines[0].startswith("Sheet:"):
            sheet_name = lines[0].replace("Sheet:", "").strip()
            
            # Create a chunk for the sheet
            chunks.append({
                "text": sheet,
                "metadata": {
                    "title": f"Sheet: {sheet_name}",
                    "position": "unknown",
                    "start_char": current_position,
                    "end_char": current_position + len(sheet),
                    "strategy": "excel_sheet",
                }
            })
            
            current_position += len(sheet) + 2  # +2 for the newlines
    
    # If no chunks were created, fall back to paragraph chunking
    if not chunks:
        logger.warning("Excel sheet-based chunking failed, falling back to paragraph chunking")
        return paragraph_chunking(text)
        
    return chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file with optimized performance.
    Uses parallel processing for multi-page PDFs.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text from the PDF
    """

    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            # Function to extract text from a page
            def extract_page_text(page_idx):
                try:
                    page = reader.pages[page_idx]
                    text = page.extract_text() or ""
                    return text
                except Exception as e:
                    logger.warning(
                        f"Error extracting text from page {page_idx}: {str(e)}"
                    )
                    return ""

            # For small PDFs, sequential processing is faster
            if len(reader.pages) <= 5:
                all_text = ""
                for i in range(len(reader.pages)):
                    all_text += extract_page_text(i) + "\n\n"
            else:
                # Process pages in parallel for larger PDFs
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = list(
                        executor.map(extract_page_text, range(len(reader.pages)))
                    )
                all_text = "\n\n".join(results)

        return all_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise


def extract_text_from_docx(docx_path: str) -> str:
    """
    Extract text from a DOCX file.

    Args:
        docx_path: Path to the DOCX file

    Returns:
        Extracted text from the DOCX
    """
    try:
        import docx  # python-docx package

        doc = docx.Document(docx_path)
        full_text = []

        # Extract text from paragraphs
        for para in doc.paragraphs:
            full_text.append(para.text)

        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    full_text.append(cell.text)

        return "\n\n".join(full_text)
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise


def extract_text_from_pptx(pptx_path: str) -> str:
    """
    Extract text from a PPTX file.

    Args:
        pptx_path: Path to the PPTX file

    Returns:
        Extracted text from the PPTX
    """
    try:
        from pptx import Presentation  # python-pptx package

        presentation = Presentation(pptx_path)
        full_text = []

        # Loop through slides
        for slide_number, slide in enumerate(presentation.slides, 1):
            slide_text = []
            slide_text.append(f"Slide {slide_number}")

            # Extract text from shapes (including text boxes)
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_text.append(shape.text)

                # Extract text from tables
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = []
                        for cell in row.cells:
                            if cell.text:
                                row_text.append(cell.text)
                        if row_text:
                            slide_text.append(" | ".join(row_text))

            full_text.append("\n".join(slide_text))

        return "\n\n".join(full_text)
    except Exception as e:
        logger.error(f"Error extracting text from PPTX: {str(e)}")
        raise


def extract_text_from_doc(doc_path: str) -> str:
    """
    Extract text from a DOC file using python-docx.

    Args:
        doc_path: Path to the DOC file

    Returns:
        Extracted text from the DOC
    """
    try:
        import docx
        doc = docx.Document(doc_path)
        full_text = []
        
        # Extract text from paragraphs
        for para in doc.paragraphs:
            full_text.append(para.text)
            
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    full_text.append(cell.text)
                    
        return "\n\n".join(full_text)
    except Exception as e:
        logger.error(f"Error extracting text from DOC: {str(e)}")
        raise


def extract_text_from_excel(excel_path: str) -> str:
    """
    Extract text from Excel files (XLSX, XLS) with improved formatting for semantic chunking.
    Applies special cleaning for better JSON compatibility.

    Args:
        excel_path: Path to the Excel file

    Returns:
        Extracted text from the Excel file
    """
    try:
        if excel_path.lower().endswith('.xlsx'):
            import openpyxl
            workbook = openpyxl.load_workbook(excel_path, data_only=True)
            full_text = []
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                sheet_text = [f"Sheet: {sheet_name}"]
                
                # Get max row and column with data to avoid processing empty cells
                max_row = sheet.max_row
                max_col = sheet.max_column
                
                # First, collect headers if they exist (usually first row)
                headers = []
                if max_row > 0:
                    for col_idx in range(1, max_col + 1):
                        cell = sheet.cell(row=1, column=col_idx)
                        if cell.value is not None:
                            # Clean header text
                            header_text = clean_cell_text(cell.value)
                            headers.append(header_text)
                
                # Process data rows
                for row_idx in range(2 if headers else 1, max_row + 1):
                    row_values = []
                    for col_idx in range(1, max_col + 1):
                        cell = sheet.cell(row=row_idx, column=col_idx)
                        if cell.value is not None:
                            # Clean cell text
                            cell_text = clean_cell_text(cell.value)
                            
                            # If we have headers, include them with the values
                            if headers and col_idx <= len(headers):
                                cell_text = f"{headers[col_idx-1]}: {cell_text}"
                                
                            row_values.append(cell_text)
                    
                    if row_values:
                        sheet_text.append("; ".join(row_values))
                
                full_text.append("\n".join(sheet_text))
        elif excel_path.lower().endswith('.xls'):
            import xlrd
            workbook = xlrd.open_workbook(excel_path)
            full_text = []
            
            for sheet in workbook.sheets():
                sheet_text = [f"Sheet: {sheet.name}"]
                
                # First, collect headers if they exist
                headers = []
                if sheet.nrows > 0:
                    for col_idx in range(sheet.ncols):
                        cell_value = sheet.cell_value(0, col_idx)
                        if cell_value:
                            # Clean header text
                            header_text = clean_cell_text(cell_value)
                            headers.append(header_text)
                
                # Process data rows
                for row_idx in range(1 if headers else 0, sheet.nrows):
                    row_values = []
                    for col_idx in range(sheet.ncols):
                        cell_value = sheet.cell_value(row_idx, col_idx)
                        if cell_value:
                            # Clean cell text
                            cell_text = clean_cell_text(cell_value)
                            
                            # If we have headers, include them with the values
                            if headers and col_idx < len(headers):
                                cell_text = f"{headers[col_idx]}: {cell_text}"
                                
                            row_values.append(cell_text)
                    
                    if row_values:
                        sheet_text.append("; ".join(row_values))
                
                full_text.append("\n".join(sheet_text))
        else:
            raise ValueError(f"Unsupported Excel format: {excel_path}")
        
        # Add double newlines between sheets for better paragraph separation
        return "\n\n".join(full_text)
    except Exception as e:
        logger.error(f"Error extracting text from Excel file: {str(e)}")
        raise


def clean_cell_text(value) -> str:
    """
    Clean cell text to avoid JSON parsing issues.
    
    Args:
        value: Cell value to clean
        
    Returns:
        Cleaned text
    """
    # Convert to string if not already
    text = str(value)
    
    # Replace problematic characters
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('"', "'")  # Replace double quotes with single quotes
    text = text.replace('\\', '/')  # Replace backslashes
    
    # Remove control characters
    text = ''.join(c for c in text if ord(c) >= 32 or c == '\n')
    
    # Trim extra whitespace
    text = ' '.join(text.split())
    
    return text


def extract_text_from_opendocument(od_path: str) -> str:
    """
    Extract text from OpenDocument files (ODT, ODS, ODP).

    Args:
        od_path: Path to the OpenDocument file

    Returns:
        Extracted text from the OpenDocument file
    """
    try:
        # Correct imports from the odf package 
        from odf.opendocument import load
        from odf.text import P
        from odf.table import Table, TableRow, TableCell
        from odf.teletype import extractText
        
        full_text = []
        
        # Load the document
        doc = load(od_path)
        
        if od_path.lower().endswith('.odt'):
            # Extract paragraphs from text document
            paragraphs = doc.getElementsByType(P)
            for para in paragraphs:
                # Use extractText instead of getText
                text_content = extractText(para)
                if text_content:
                    full_text.append(text_content)
                
        elif od_path.lower().endswith('.ods'):
            # Extract cells from spreadsheet
            tables = doc.getElementsByType(Table)
            for table in tables:
                sheet_text = [f"Sheet: {table.getAttribute('name') or 'Unnamed'}"]
                
                rows = table.getElementsByType(TableRow)
                for row in rows:
                    cells = row.getElementsByType(TableCell)
                    row_text = []
                    for cell in cells:
                        # Use extractText instead of getText
                        cell_text = extractText(cell)
                        if cell_text:
                            row_text.append(cell_text)
                    
                    if row_text:
                        sheet_text.append(" | ".join(row_text))
                        
                full_text.append("\n".join(sheet_text))
                
        elif od_path.lower().endswith('.odp'):
            # Just extract all paragraph text from presentation
            paragraphs = doc.getElementsByType(P)
            for para in paragraphs:
                # Use extractText instead of getText
                text_content = extractText(para)
                if text_content and text_content.strip():
                    full_text.append(text_content)
        else:
            raise ValueError(f"Unsupported OpenDocument format: {od_path}")
        
        return "\n\n".join(full_text)
    except ImportError as e:
        logger.error(f"Required module not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error extracting text from OpenDocument file: {str(e)}")
        raise


def extract_text_from_text_file(text_path: str) -> str:
    """
    Extract text from text files (TXT, RTF).

    Args:
        text_path: Path to the text file

    Returns:
        Extracted text from the text file
    """
    try:
        if text_path.lower().endswith('.rtf'):
            from striprtf.striprtf import rtf_to_text
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as file:
                rtf_content = file.read()
                return rtf_to_text(rtf_content)
        else:  # .txt
            with open(text_path, 'r', encoding='utf-8') as file:
                return file.read()
    except Exception as e:
        logger.error(f"Error extracting text from text file: {str(e)}")
        raise


def extract_text_from_epub(epub_path: str) -> str:
    """
    Extract text from EPUB files.

    Args:
        epub_path: Path to the EPUB file

    Returns:
        Extracted text from the EPUB
    """
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        
        book = epub.read_epub(epub_path)
        full_text = []
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text()
                if text.strip():
                    full_text.append(text)
        
        return "\n\n".join(full_text)
    except Exception as e:
        logger.error(f"Error extracting text from EPUB: {str(e)}")
        raise
