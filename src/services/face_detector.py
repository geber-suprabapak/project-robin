"""
Face detection service using OpenCV DNN for accurate face detection.
Uses a pre-trained ResNet SSD model for robust face detection.
"""

import logging
import cv2
import numpy as np
from typing import List, Tuple

from src.config import settings
from src.services.model_downloader import ModelDownloadError, ensure_model_asset

logger = logging.getLogger(__name__)

# Global DNN network (lazy loaded)
_face_net = None


class FaceDetectionError(Exception):
    """Exception raised when face detection fails."""
    pass


def _resolve_model_paths() -> Tuple[str, str]:
    """Resolve face detector model files, downloading defaults when missing."""
    try:
        prototxt_path = ensure_model_asset(
            settings.face_detector_prototxt_path,
            url=settings.face_detector_prototxt_url,
            checksum=settings.face_detector_prototxt_checksum,
            auto_download=settings.auto_download_models,
            description="face detector prototxt",
        )
        model_path = ensure_model_asset(
            settings.face_detector_model_path,
            url=settings.face_detector_model_url,
            checksum=settings.face_detector_model_checksum,
            auto_download=settings.auto_download_models,
            description="face detector model",
        )
    except ModelDownloadError as e:
        raise FaceDetectionError(str(e)) from e

    return str(prototxt_path), str(model_path)


def _get_face_net() -> cv2.dnn.Net:
    """Get or initialize the DNN face detector network."""
    global _face_net
    
    if _face_net is None:
        prototxt_path, model_path = _resolve_model_paths()
        logger.info("Loading DNN face detector model...")
        _face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        logger.info("DNN face detector model loaded successfully")
    
    return _face_net


def is_face_detector_ready() -> bool:
    """Return whether the face detector can be loaded from local assets."""
    try:
        _get_face_net()
        return True
    except Exception as e:
        logger.warning(f"Face detector readiness check failed: {str(e)}")
        return False


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


def crop_face_from_image(
    image: np.ndarray,
    margin: float = 0.2,
    confidence_threshold: float = 0.5
) -> np.ndarray:
    """
    Detect and crop face from image with margin.
    
    Args:
        image: Input image in BGR format (OpenCV)
        margin: Margin to add around face (0.2 = 20% padding)
        confidence_threshold: Minimum confidence for face detection
        
    Returns:
        Cropped face image
        
    Raises:
        FaceDetectionError: If no face or multiple faces detected
    """
    faces = detect_faces(image, confidence_threshold)
    
    if len(faces) == 0:
        raise FaceDetectionError("No face detected for cropping")
    elif len(faces) > 1:
        raise FaceDetectionError(f"Multiple faces detected ({len(faces)}). Cannot crop.")
    
    # Get the single face bounding box
    x, y, w, h = faces[0]
    
    # Add margin
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    
    # Calculate new coordinates with margin
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(image.shape[1], x + w + margin_w)
    y2 = min(image.shape[0], y + h + margin_h)
    
    # Crop the face region
    cropped_face = image[y1:y2, x1:x2]
    
    logger.debug(f"Face cropped: original {image.shape}, cropped {cropped_face.shape}")
    
    return cropped_face

