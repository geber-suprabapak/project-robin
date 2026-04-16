"""
Image decoding service for converting Base64 strings to numpy arrays.
"""

import base64
import numpy as np
import cv2


class ImageDecodeError(Exception):
    """Custom exception for image decoding errors."""
    pass


def decode_base64_image(image_b64: str) -> np.ndarray:
    """
    Decode a base64 string to a numpy array (OpenCV format).
    
    Args:
        image_b64: Base64-encoded image string
        
    Returns:
        numpy array in BGR format (OpenCV default)
        
    Raises:
        ImageDecodeError: If decoding fails or image is corrupt
    """
    try:
        # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_b64, validate=True)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image using OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ImageDecodeError("Failed to decode image - corrupt or unsupported format")
        
        return image
        
    except base64.binascii.Error as e:
        raise ImageDecodeError(f"Invalid base64 encoding: {str(e)}")
    except cv2.error as e:
        raise ImageDecodeError(f"OpenCV decoding failed: {str(e)}")
    except Exception as e:
        raise ImageDecodeError(f"Unexpected error during image decoding: {str(e)}")


def preprocess_face_image(
    image: np.ndarray,
    target_size: int = 224,
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess face image for model inference.
    
    Args:
        image: Input image in BGR format (OpenCV)
        target_size: Target size for model input (default 224 for Hi-Res ArcFace)
        normalize: Whether to normalize pixel values to [-1, 1]
        
    Returns:
        Preprocessed image array in shape (1, 3, H, W) for ONNX
    """
    # Resize to target size
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 and normalize
    img_array = rgb.astype(np.float32)
    
    if normalize:
        # Normalize to [-1, 1] range (common for face recognition models)
        img_array = (img_array - 127.5) / 128.0
    else:
        # Normalize to [0, 1] range
        img_array /= 255.0
    
    # Transpose to (C, H, W) and add batch dimension -> (1, C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    
    return img_array


def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embedding vectors.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score (higher is more similar)
    """
    # Normalize vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    return float(similarity)


def calculate_cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine distance between two embedding vectors.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine distance (lower is more similar, range 0-2)
    """
    similarity = calculate_cosine_similarity(embedding1, embedding2)
    distance = 1.0 - similarity
    
    return float(distance)
