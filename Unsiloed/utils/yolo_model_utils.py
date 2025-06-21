from ultralytics import YOLO
import os
from pathlib import Path
import logging
from typing import List, Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)

# Get absolute path to model file
def get_model_path():
    """Get absolute path to YOLO model."""
    current_dir = Path(__file__).parent.parent  # Go up to Unsiloed/
    model_path = current_dir / "models" / "yolo11_5_x_best.pt"
    return str(model_path.resolve())

MODEL_PATH = get_model_path()
_model = None


def get_model() -> YOLO:
    """Returns a singleton YOLO model instance with error handling."""
    global _model
    if _model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"YOLO model not found at: {MODEL_PATH}")
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