"""
Face detection service using OpenCV DNN for accurate face detection.
Uses a pre-trained ResNet SSD model for robust face detection.
"""

import logging
import cv2
import numpy as np
from typing import List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# DNN model configuration
_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# Local paths for cached models
_MODELS_DIR = Path(__file__).parent.parent / "models"
_PROTOTXT_PATH = _MODELS_DIR / "deploy.prototxt"
_MODEL_PATH = _MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# Global DNN network (lazy loaded)
_face_net = None


class FaceDetectionError(Exception):
    """Exception raised when face detection fails."""
    pass


def _download_model_if_needed() -> Tuple[str, str]:
    """Download DNN model files if not present."""
    import urllib.request
    
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not _PROTOTXT_PATH.exists():
        logger.info(f"Downloading face detector prototxt to {_PROTOTXT_PATH}")
        urllib.request.urlretrieve(_PROTOTXT_URL, _PROTOTXT_PATH)
    
    if not _MODEL_PATH.exists():
        logger.info(f"Downloading face detector model to {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    
    return str(_PROTOTXT_PATH), str(_MODEL_PATH)


def _get_face_net() -> cv2.dnn.Net:
    """Get or initialize the DNN face detector network."""
    global _face_net
    
    if _face_net is None:
        prototxt_path, model_path = _download_model_if_needed()
        logger.info("Loading DNN face detector model...")
        _face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        logger.info("DNN face detector model loaded successfully")
    
    return _face_net


def detect_faces(
    image: np.ndarray,
    confidence_threshold: float = 0.5
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in an image using OpenCV DNN.
    
    Args:
        image: Input image in BGR format (OpenCV)
        confidence_threshold: Minimum confidence for face detection (0.0-1.0)
        
    Returns:
        List of bounding boxes as (x, y, w, h) tuples
        
    Raises:
        FaceDetectionError: If detection fails
    """
    try:
        net = _get_face_net()
        
        (h, w) = image.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        # Pass the blob through the network
        net.setInput(blob)
        detections = net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Convert to (x, y, w, h) format
                face_w = x2 - x1
                face_h = y2 - y1
                
                if face_w > 0 and face_h > 0:
                    faces.append((x1, y1, face_w, face_h))
        
        return faces
        
    except Exception as e:
        logger.error(f"Face detection failed: {str(e)}")
        raise FaceDetectionError(f"Face detection failed: {str(e)}")


def validate_single_face(
    image: np.ndarray,
    confidence_threshold: float = 0.5
) -> Tuple[bool, int, str]:
    """
    Validate that exactly one face exists in the image.
    
    Args:
        image: Input image in BGR format (OpenCV)
        confidence_threshold: Minimum confidence for face detection
        
    Returns:
        Tuple of (is_valid, face_count, message)
    """
    faces = detect_faces(image, confidence_threshold)
    face_count = len(faces)
    
    if face_count == 0:
        return False, 0, "No face detected in the image. Please use a clear photo with a visible face."
    elif face_count == 1:
        return True, 1, "Face detected successfully."
    else:
        return False, face_count, f"Multiple faces detected ({face_count}). Please use a solo photo."
