"""
Face Inference Engine using ONNX Runtime with GPU acceleration.
Implements singleton pattern for efficient model loading.
"""

import os
import threading
import logging
from typing import Optional, List
import numpy as np
import onnxruntime as ort

from config import settings

logger = logging.getLogger(__name__)


class FaceInferenceEngine:
    """
    Singleton Face Recognition Inference Engine.
    Loads ONNX model once and provides thread-safe inference.
    """
    
    _instance: Optional["FaceInferenceEngine"] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> "FaceInferenceEngine":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the inference engine (only once)."""
        # Prevent re-initialization
        if hasattr(self, "_initialized"):
            return
        
        self._initialized = False
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._inference_lock = threading.Lock()  # For thread-safe inference
        self._provider: Optional[str] = None
        
        logger.info("FaceInferenceEngine instance created")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load ONNX model with GPU support and CPU fallback.
        
        Args:
            model_path: Path to ONNX model file (uses config if not provided)
        """
        if self._initialized:
            logger.warning("Model already loaded. Skipping re-initialization.")
            return
        
        model_path = model_path or settings.model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please download an ArcFace ONNX model and place it at this location."
            )
        
        try:
            # Configure execution providers with GPU priority
            providers = self._configure_providers()
            
            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            sess_options.log_severity_level = 3  # Error only
            
            # Load model
            logger.info(f"Loading ONNX model from: {model_path}")
            logger.info(f"Execution providers: {providers}")
            
            self._session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output names
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            
            # Log model info
            active_provider = self._session.get_providers()[0]
            self._provider = active_provider
            
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"✓ Active execution provider: {active_provider}")
            logger.info(f"✓ Input name: {self._input_name}")
            logger.info(f"✓ Output name: {self._output_name}")
            logger.info(f"✓ Input shape: {self._session.get_inputs()[0].shape}")
            logger.info(f"✓ Output shape: {self._session.get_outputs()[0].shape}")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def _configure_providers(self) -> List[str]:
        """
        Configure ONNX Runtime execution providers with GPU priority and CPU fallback.
        
        Returns:
            List of providers in priority order
        """
        providers = []
        
        # Check if GPU is requested (gpu_device_id >= 0)
        if settings.gpu_device_id >= 0:
            # Check if CUDA is available
            available_providers = ort.get_available_providers()
            
            if "CUDAExecutionProvider" in available_providers:
                cuda_options = {
                    "device_id": settings.gpu_device_id,
                    "gpu_mem_limit": settings.gpu_mem_limit,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": True,
                }
                providers.append(("CUDAExecutionProvider", cuda_options))
                logger.info(f"CUDA provider configured (GPU {settings.gpu_device_id})")
            else:
                logger.warning(
                    "CUDA provider not available. Falling back to CPU. "
                    "Install onnxruntime-gpu for GPU support."
                )
        else:
            logger.info("CPU-only mode selected (gpu_device_id < 0)")
        
        # Always add CPU as fallback
        providers.append("CPUExecutionProvider")
        
        return providers
    
    def warmup(self) -> None:
        """
        Run a dummy inference to initialize CUDA context and reduce first-request latency.
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info("Warming up inference engine...")
        
        try:
            # Create dummy zero-array input with correct shape
            # ArcFace Hi-Res expects (1, 3, 224, 224)
            dummy_input = np.zeros(
                (1, 3, settings.model_input_size, settings.model_input_size),
                dtype=np.float32
            )
            
            # Run inference
            _ = self._session.run(
                [self._output_name],
                {self._input_name: dummy_input}
            )
            
            logger.info("✓ Warmup complete - CUDA context initialized")
            
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {str(e)}")
    
    def predict(self, image_array: np.ndarray) -> np.ndarray:
        """
        Generate face embedding from preprocessed image.
        Thread-safe inference using lock.
        
        Args:
            image_array: Preprocessed image array in shape (1, 3, H, W)
            
        Returns:
            Face embedding vector (typically 512-dimensional)
            
        Raises:
            RuntimeError: If model not loaded or inference fails
        """
        if not self._initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Ensure correct shape
        expected_shape = (1, 3, settings.model_input_size, settings.model_input_size)
        if image_array.shape != expected_shape:
            raise ValueError(
                f"Invalid input shape. Expected {expected_shape}, got {image_array.shape}"
            )
        
        # Thread-safe inference
        with self._inference_lock:
            try:
                outputs = self._session.run(
                    [self._output_name],
                    {self._input_name: image_array}
                )
                
                # Extract embedding vector and flatten
                embedding = outputs[0].flatten()
                
                # Normalize embedding (L2 normalization)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                return embedding
                
            except Exception as e:
                logger.error(f"Inference failed: {str(e)}")
                raise RuntimeError(f"Inference error: {str(e)}")
    
    def get_provider(self) -> Optional[str]:
        """Get the active execution provider (CUDA or CPU)."""
        return self._provider
    
    def is_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is active."""
        return self._provider == "CUDAExecutionProvider"
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._initialized


# Global inference engine instance (singleton)
inference_engine = FaceInferenceEngine()
