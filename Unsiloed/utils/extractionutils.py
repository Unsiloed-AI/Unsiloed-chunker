import os
import io
import base64
import logging
import pytesseract
from PIL import Image
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

def _extract_text_with_ocr(image: Image.Image) -> str:
    """Extract text using Tesseract OCR."""
    try:
        text = pytesseract.image_to_string(image, lang='eng').strip()
        return text if text else "[No text detected]"
    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        return "[OCR failed]"




async def _extract_table_with_openai_async(image: Image.Image) -> str:
    """Async version of table extraction using OpenAI Vision API."""
    try:
        # Create async OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI client not available, falling back to OCR for table")
            return _extract_text_with_ocr(image)
        
        async_client = AsyncOpenAI(api_key=api_key, timeout=60.0, max_retries=2)
        
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        response = await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing tables. Extract the table content and convert it to markdown format. Preserve the structure and all data."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this table image and convert it to markdown format. Include all rows, columns, and data. If it's not a clear table, describe what you see."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Async OpenAI table extraction failed: {str(e)}")
        return _extract_text_with_ocr(image)


async def _extract_image_with_openai_async(image: Image.Image, element_type: str) -> str:
    """Async version of image description using OpenAI Vision API."""
    try:
        # Create async OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI client not available, using placeholder for image")
            return f"[{element_type} - description not available]"
        
        async_client = AsyncOpenAI(api_key=api_key, timeout=60.0, max_retries=2)
        
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        prompt = "Describe this image in detail, focusing on its content and context within a document." if element_type == "Picture" else "Analyze this mathematical formula or equation and describe what it represents."
        
        response = await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert at analyzing {element_type.lower()}s in documents. Provide clear, concise descriptions."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Async OpenAI image extraction failed: {str(e)}")
        return f"[{element_type} - description failed]"
