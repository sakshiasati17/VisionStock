"""YOLOv8 object detection module for VisionStock."""
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)

# Lazy import of YOLO to avoid slow import at module level
def _get_yolo():
    """Lazy import of YOLO."""
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as e:
        logger.error(f"Failed to import YOLO: {e}")
        raise


class SceneDetector:
    """YOLOv8-based object detector for retail shelf monitoring."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the detector with YOLOv8 model.
        
        Args:
            model_path: Path to YOLOv8 model weights. If None, uses default from config.
        """
        # Lazy import config to avoid circular imports
        try:
            from backend.config import MODEL_PATH, MODEL_CONFIDENCE, MODEL_IOU_THRESHOLD
            self.model_path = model_path or MODEL_PATH
            self.confidence = MODEL_CONFIDENCE
            self.iou_threshold = MODEL_IOU_THRESHOLD
        except Exception as e:
            logger.warning(f"Failed to import config, using defaults: {e}")
            # Default to baseline model (always available) instead of Hub model
            self.model_path = model_path or "yolov8n.pt"
            self.confidence = 0.25
            self.iou_threshold = 0.45
        
        # Load model (YOLO imported lazily)
        YOLO = _get_yolo()
        
        # Try loading the specified model first
        try:
            logger.info(f"Attempting to load YOLOv8 model from {self.model_path}")
            # If it's a Hub URL, YOLO will download it automatically
            # Add timeout handling for Hub downloads
            import signal
            
            self.model = YOLO(self.model_path)
            logger.info(f"Successfully loaded YOLOv8 model from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model from {self.model_path}: {e}")
            logger.warning(f"Error type: {type(e).__name__}, Error details: {str(e)}")
            
            # Try to load default YOLOv8n model (this is always available)
            try:
                logger.info("Attempting to load default YOLOv8n model as fallback...")
                self.model = YOLO("yolov8n.pt")
                logger.info("Successfully loaded default YOLOv8n model as fallback")
                # Update model_path to reflect the fallback
                self.model_path = "yolov8n.pt"
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                logger.error(f"Fallback error type: {type(e2).__name__}, Error details: {str(e2)}")
                raise RuntimeError(f"Failed to load any model. Original error: {e}, Fallback error: {e2}")
    
    def detect(self, image_path: str, track_time: bool = True) -> Tuple[List[Dict], Optional[float]]:
        """
        Detect objects in an image.
        
        Args:
            image_path: Path to the image file
            track_time: Whether to measure inference time
            
        Returns:
            Tuple of (detections list, inference_time_ms):
            - detections: List of detection dictionaries with keys:
              - class_name: Detected class name
              - confidence: Confidence score
              - bbox: [x_center, y_center, width, height] (normalized 0-1)
              - x_center, y_center, width, height: Individual bbox components
            - inference_time_ms: Inference time in milliseconds (None if track_time=False)
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Measure inference time
        start_time = time.time() if track_time else None
        
        # Run inference
        results = self.model.predict(
            image_path,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False
        )
        
        inference_time_ms = None
        if track_time and start_time:
            inference_time_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        detections = []
        result = results[0]
        
        # Get image dimensions for normalization
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]
        
        # Process detections
        for box in result.boxes:
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Convert to center, width, height (normalized)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # Get class and confidence
            class_id = int(box.cls[0].cpu().numpy())
            class_name = result.names[class_id]
            confidence = float(box.conf[0].cpu().numpy())
            
            detection = {
                "class_name": class_name,
                "confidence": confidence,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "bbox": [x_center, y_center, width, height]
            }
            
            detections.append(detection)
        
        logger.info(f"Detected {len(detections)} objects in {image_path} (inference: {inference_time_ms:.2f}ms)" if inference_time_ms else f"Detected {len(detections)} objects in {image_path}")
        return detections, inference_time_ms
    
    def detect_batch(self, image_paths: List[str]) -> Dict[str, Tuple[List[Dict], Optional[float]]]:
        """
        Detect objects in multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping image_path to (detections, inference_time_ms) tuple
        """
        results = {}
        for image_path in image_paths:
            try:
                detections, inference_time = self.detect(image_path)
                results[image_path] = (detections, inference_time)
            except Exception as e:
                logger.error(f"Error detecting in {image_path}: {e}")
                results[image_path] = ([], None)
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "confidence_threshold": self.confidence,
            "iou_threshold": self.iou_threshold,
            "model_type": "YOLOv8"
        }

