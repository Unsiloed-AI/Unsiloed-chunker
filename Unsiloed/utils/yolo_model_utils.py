from ultralytics import YOLO
from Unsiloed.parse_config import MODEL_PATH, _model
import logging
from typing import List, Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)


def get_model() -> YOLO:
    """Returns a singleton YOLO model instance with error handling."""
    global _model
    if _model is None:
        try:
            _model = YOLO(MODEL_PATH)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading YOLO model from {MODEL_PATH}: {e}")
            raise RuntimeError("YOLO model could not be loaded.") from e
    return _model


def run_yolo_inference(image_list: List[Image.Image]) -> List[Dict[str, Any]]:
    """
    Run YOLO inference on a list of images.
    """
    model = get_model()
    yolo_results = model(image_list)
    return yolo_results
