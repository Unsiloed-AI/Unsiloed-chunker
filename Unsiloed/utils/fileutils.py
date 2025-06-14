from PIL import Image
from typing import List
import pdf2image


def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """
    Convert a PDF to a list of images.
    """
    return pdf2image.convert_from_path(pdf_path)
