
import sys
import os
import time
import logging
import asyncio
import argparse
from pathlib import Path

import cv2
import numpy as np

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("benchmark")

# Force CPU mode before imports - CRITICAL
os.environ["GPU_DEVICE_ID"] = "-1"
# Optional: suppress other logs
os.environ["LOG_LEVEL"] = "ERROR" 

# Add project root to path to ensure we can import 'src'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
print(f"Running in: {PROJECT_ROOT}")
print("Force CPU Mode: Enabled (GPU_DEVICE_ID=-1)")

try:
    from src.core.inference_engine import inference_engine
    from src.services.face_detector import validate_single_face, crop_face_from_image, detect_faces
    from src.services.image_decoder import preprocess_face_image
    from src.services.supabase_client import supabase_service
    from src.config import settings
except ImportError as e:
    print(f"\nError importing project modules: {e}")
    print("Make sure you are running this script from the project root or 'scripts' directory.")
    sys.exit(1)



async def run_benchmark(image_path: str, iterations: int = 10):
    print(f"\n{'='*50}")
    print(f"BENCHMARK STARTING: {iterations} iterations")
    print(f"Target Image: {image_path}")
    print(f"{'='*50}\n")

    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # 2. Warmup
    print("Step 0: Warning up models...")
    try:
        # Warmup Detector
        detect_faces(img) 
        # Warmup Inference Engine (loads model if needed)
        inference_engine.load_model() 
        inference_engine.warmup()
        print("Warmup complete.\n")
    except Exception as e:
        print(f"Warmup failed: {e}")
        return

    # Storage for timings
    timings = {
        "detection": [],
        "cropping": [],
        "preprocessing": [],
        "inference": [],
        "search": [],
        "total": []
    }

    print(f"{'Iter':<5} | {'Detect':<8} | {'Crop':<8} | {'Preproc':<8} | {'Infer':<8} | {'Search':<8} | {'Total':<8}")
    print("-" * 75)

    for i in range(iterations):
        iter_start = time.perf_counter()
        
        try:
            # A. Detection (Validation)
            t0 = time.perf_counter()
            is_valid, _, _ = validate_single_face(img)
            t1 = time.perf_counter()
            if not is_valid:
                print(f"Iter {i+1}: Invalid face, skipping.")
                continue
                
            # B. Cropping (re-detects internally currently, let's measure the explicit crop call)
            # Note: crop_face_from_image in original code re-calls detect_faces.
            # This is inefficient but reflects the current codebase 'identification.py'.
            t2 = time.perf_counter()
            cropped_face = crop_face_from_image(img, margin=0.2)
            t3 = time.perf_counter()
            
            # C. Preprocessing
            t4 = time.perf_counter()
            preprocessed = preprocess_face_image(cropped_face, target_size=settings.model_input_size, normalize=True)
            t5 = time.perf_counter()
            
            # D. Inference
            t6 = time.perf_counter()
            embedding = inference_engine.predict(preprocessed)
            t7 = time.perf_counter()
            
            # E. Search (Database)
            t8 = time.perf_counter()
            # We await the async call
            match_result = await supabase_service.find_face_match(embedding, threshold=settings.face_match_threshold)
            t9 = time.perf_counter()

            # Record times (ms)
            detect_ms = (t1 - t0) * 1000
            crop_ms = (t3 - t2) * 1000
            prep_ms = (t5 - t4) * 1000
            infer_ms = (t7 - t6) * 1000
            search_ms = (t9 - t8) * 1000
            total_ms = (t9 - iter_start) * 1000

            timings["detection"].append(detect_ms)
            timings["cropping"].append(crop_ms)
            timings["preprocessing"].append(prep_ms)
            timings["inference"].append(infer_ms)
            timings["search"].append(search_ms)
            timings["total"].append(total_ms)

            print(f"{i+1:<5} | {detect_ms:<8.2f} | {crop_ms:<8.2f} | {prep_ms:<8.2f} | {infer_ms:<8.2f} | {search_ms:<8.2f} | {total_ms:<8.2f}")

        except Exception as e:
            print(f"Iter {i+1} failed: {e}")

    # Summary
    print(f"\n{'='*50}")
    print("BENCHMARK SUMMARY (CPU ONLY)")
    print(f"{'='*50}")
    
    if len(timings["total"]) == 0:
        print("No successful iterations.")
        return

    def stats(data):
        return f"Avg: {np.mean(data):.2f}ms | Min: {np.min(data):.2f}ms | Max: {np.max(data):.2f}ms"

    print(f"Detection   : {stats(timings['detection'])}")
    print(f"Cropping    : {stats(timings['cropping'])} (Note: Includes 2nd detection call)")
    print(f"Preprocess  : {stats(timings['preprocessing'])}")
    print(f"Inference   : {stats(timings['inference'])}")
    print(f"Search (DB) : {stats(timings['search'])}")
    print("-" * 50)
    print(f"TOTAL PER REQ: {stats(timings['total'])}")
    
    # Provider check
    print(f"\nInference Provider Used: {inference_engine.get_provider()}")
    print(f"DB Connected: {supabase_service.is_connected()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Face Recognition CPU Performance")
    parser.add_argument("--image", type=str, help="Path to test image", required=True)
    parser.add_argument("--iter", type=int, help="Number of iterations", default=5)
    
    args = parser.parse_args()
    
    if os.path.exists(args.image):
        asyncio.run(run_benchmark(args.image, args.iter))
    else:
        print(f"Error: Image verification failed. File not found at: {args.image}")
