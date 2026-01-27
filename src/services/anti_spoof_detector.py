"""
Anti-Spoofing Detection Service using MiniFASNetV2.
Implements liveness detection to prevent spoof attacks (photos, screens, masks).
Uses multi-scale fusion for improved accuracy.
"""

import logging
import threading
import os
from typing import Optional, Tuple
import numpy as np
import cv2
import onnxruntime as ort

from src.config import settings

logger = logging.getLogger(__name__)

# MiniFASNet expects 80x80 input
MINI_FAS_INPUT_SIZE = 80


class AntiSpoofDetector:
    """
    Singleton Anti-Spoofing Detector using MiniFASNetV2.
    Uses multi-scale fusion (2.7x and 4.0x) for improved accuracy.
    """
    
    _instance: Optional["AntiSpoofDetector"] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> "AntiSpoofDetector":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the anti-spoof detector (only once)."""
        if hasattr(self, "_initialized"):
            return
        
        self._initialized = False
        self._session_scale27: Optional[ort.InferenceSession] = None
        self._session_scale40: Optional[ort.InferenceSession] = None
        self._inference_lock = threading.Lock()
        
        logger.info("AntiSpoofDetector instance created")
    
    def load_models(self) -> None:
        """
        Load MiniFASNetV2 ONNX models for multi-scale inference.
        """
        if self._initialized:
            logger.warning("Anti-spoof models already loaded. Skipping re-initialization.")
            return
        
        if not settings.anti_spoof_enabled:
            logger.info("Anti-spoofing is disabled. Skipping model loading.")
            return
        
        model_path_27 = settings.anti_spoof_model_path_v2_scale27
        model_path_40 = settings.anti_spoof_model_path_v2_scale40
        
        # Check if both model files exist
        if not os.path.exists(model_path_27):
            raise FileNotFoundError(
                f"Anti-spoof model (scale 2.7) not found: {model_path_27}\n"
                "Please download MiniFASNetV2 ONNX models."
            )
        
        if not os.path.exists(model_path_40):
            raise FileNotFoundError(
                f"Anti-spoof model (scale 4.0) not found: {model_path_40}\n"
                "Please download MiniFASNetV2 ONNX models."
            )
        
        try:
            # Configure execution providers (same as face inference engine)
            providers = self._configure_providers()
            
            # Session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            sess_options.log_severity_level = 3
            
            logger.info(f"Loading anti-spoof model (scale 2.7): {model_path_27}")
            self._session_scale27 = ort.InferenceSession(
                model_path_27,
                sess_options=sess_options,
                providers=providers
            )
            
            logger.info(f"Loading anti-spoof model (scale 4.0): {model_path_40}")
            self._session_scale40 = ort.InferenceSession(
                model_path_40,
                sess_options=sess_options,
                providers=providers
            )
            
            logger.info("✓ Anti-spoof models loaded successfully")
            logger.info(f"✓ Active provider: {self._session_scale27.get_providers()[0]}")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to load anti-spoof models: {str(e)}")
            raise RuntimeError(f"Anti-spoof model loading failed: {str(e)}")
    
    def _configure_providers(self) -> list:
        """Configure ONNX Runtime execution providers."""
        providers = []
        
        if settings.gpu_device_id >= 0:
            available_providers = ort.get_available_providers()
            
            if "CUDAExecutionProvider" in available_providers:
                cuda_options = {
                    "device_id": settings.gpu_device_id,
                    "gpu_mem_limit": settings.gpu_mem_limit,
                    "arena_extend_strategy": "kSameAsRequested",
                }
                providers.append(("CUDAExecutionProvider", cuda_options))
        
        providers.append("CPUExecutionProvider")
        return providers
    
    def _crop_face_with_scale(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int],
        scale: float
    ) -> np.ndarray:
        """
        Crop face region with a specific scale factor.
        
        Args:
            image: Input image in BGR format
            bbox: Face bounding box (x, y, w, h)
            scale: Scale factor for expanding the crop region
            
        Returns:
            Cropped and resized face image (80x80)
        """
        h, w = image.shape[:2]
        x, y, bw, bh = bbox
        
        # Calculate center
        cx = x + bw // 2
        cy = y + bh // 2
        
        # Calculate new size based on scale
        face_size = max(bw, bh)
        new_size = int(face_size * scale)
        half_size = new_size // 2
        
        # Calculate crop coordinates
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)
        
        # Crop
        cropped = image[y1:y2, x1:x2]
        
        # Resize to model input size
        if cropped.size == 0:
            # Fallback if crop failed
            cropped = image
        
        resized = cv2.resize(cropped, (MINI_FAS_INPUT_SIZE, MINI_FAS_INPUT_SIZE))
        
        return resized
    
    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for MiniFASNet.
        
        Args:
            face_image: BGR image (80x80)
            
        Returns:
            Preprocessed tensor (1, 3, 80, 80)
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W) then add batch dimension
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def _run_inference(
        self, 
        session: ort.InferenceSession, 
        preprocessed: np.ndarray
    ) -> float:
        """
        Run inference on a single scale model.
        
        Args:
            session: ONNX Runtime session
            preprocessed: Preprocessed input tensor
            
        Returns:
            Real face probability (0.0 to 1.0)
        """
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        outputs = session.run([output_name], {input_name: preprocessed})
        
        # Output is typically [batch, 2] where [0] is fake prob, [1] is real prob
        # Apply softmax if needed
        logits = outputs[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Return real face probability (index 1)
        real_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        
        return real_prob
    
    def check_liveness(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[bool, float]:
        """
        Check if the face is a real (live) face or a spoof.
        
        Uses multi-scale fusion with MiniFASNetV2 models for improved accuracy.
        
        Args:
            image: Input image in BGR format (OpenCV)
            bbox: Face bounding box as (x, y, w, h)
            
        Returns:
            Tuple of (is_live: bool, liveness_score: float)
            - is_live: True if the face passes liveness check
            - liveness_score: Confidence score (0.0 = fake, 1.0 = real)
        """
        if not self._initialized:
            if not settings.anti_spoof_enabled:
                # Anti-spoofing disabled, assume all faces are live
                logger.debug("Anti-spoofing disabled, skipping liveness check")
                return True, 1.0
            raise RuntimeError("Anti-spoof models not loaded. Call load_models() first.")
        
        with self._inference_lock:
            try:
                # Crop face at scale 2.7
                face_27 = self._crop_face_with_scale(image, bbox, scale=2.7)
                preprocessed_27 = self._preprocess(face_27)
                
                # Crop face at scale 4.0
                face_40 = self._crop_face_with_scale(image, bbox, scale=4.0)
                preprocessed_40 = self._preprocess(face_40)
                
                # Run inference on both scales
                prob_27 = self._run_inference(self._session_scale27, preprocessed_27)
                prob_40 = self._run_inference(self._session_scale40, preprocessed_40)
                
                # Fuse predictions (simple average)
                fused_score = (prob_27 + prob_40) / 2.0
                
                # Determine if live based on threshold
                is_live = fused_score >= settings.anti_spoof_threshold
                
                logger.debug(
                    f"Liveness check: scale2.7={prob_27:.3f}, scale4.0={prob_40:.3f}, "
                    f"fused={fused_score:.3f}, threshold={settings.anti_spoof_threshold}, "
                    f"is_live={is_live}"
                )
                
                return is_live, fused_score
                
            except Exception as e:
                logger.error(f"Liveness check failed: {str(e)}")
                # On error, default to allowing (fail open) or you can choose to deny
                # For security, it's better to deny on error
                return False, 0.0
    
    def warmup(self) -> None:
        """Run dummy inference to initialize CUDA context."""
        if not self._initialized:
            logger.warning("Cannot warmup: models not loaded")
            return
        
        logger.info("Warming up anti-spoof detector...")
        
        try:
            dummy_input = np.zeros((1, 3, MINI_FAS_INPUT_SIZE, MINI_FAS_INPUT_SIZE), dtype=np.float32)
            
            input_name_27 = self._session_scale27.get_inputs()[0].name
            output_name_27 = self._session_scale27.get_outputs()[0].name
            self._session_scale27.run([output_name_27], {input_name_27: dummy_input})
            
            input_name_40 = self._session_scale40.get_inputs()[0].name
            output_name_40 = self._session_scale40.get_outputs()[0].name
            self._session_scale40.run([output_name_40], {input_name_40: dummy_input})
            
            logger.info("✓ Anti-spoof detector warmup complete")
            
        except Exception as e:
            logger.warning(f"Anti-spoof warmup failed (non-critical): {str(e)}")
    
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._initialized


# Global anti-spoof detector instance (singleton)
anti_spoof_detector = AntiSpoofDetector()
