"""Utility functions for YOLO detection."""
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np

def normalize_bbox(x1: float, y1: float, x2: float, y2: float, img_width: int, img_height: int) -> Dict:
    """
    Normalize bounding box coordinates to 0-1 range.
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        img_width, img_height: Image dimensions
        
    Returns:
        Dictionary with normalized coordinates
    """
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return {
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
        "bbox": [x_center, y_center, width, height]
    }

def get_image_dimensions(image_path: str) -> tuple:
    """Get image width and height."""
    img = cv2.imread(image_path)
    if img is not None:
        return img.shape[1], img.shape[0]  # width, height
    return None, None

def format_detection_result(class_name: str, confidence: float, bbox: Dict, 
                          sku: str = None, shelf_location: str = None) -> Dict:
    """
    Format detection result dictionary.
    
    Args:
        class_name: Detected class name
        confidence: Confidence score
        bbox: Bounding box dictionary
        sku: Optional SKU identifier
        shelf_location: Optional shelf location
        
    Returns:
        Formatted detection dictionary
    """
    return {
        "class_name": class_name,
        "confidence": confidence,
        "x_center": bbox["x_center"],
        "y_center": bbox["y_center"],
        "width": bbox["width"],
        "height": bbox["height"],
        "bbox": bbox["bbox"],
        "sku": sku,
        "shelf_location": shelf_location
    }

