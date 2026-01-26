
import sys
import os
import time
import logging
import asyncio
import argparse
import statistics
from pathlib import Path
import cv2
import numpy as np

# Configure basic logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger("stress_test")

# Force CPU mode
os.environ["GPU_DEVICE_ID"] = "-1"
os.environ["LOG_LEVEL"] = "ERROR"

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
print(f"Running in: {PROJECT_ROOT}")

try:
    from src.core.inference_engine import inference_engine
    from src.services.face_detector import validate_single_face, crop_face_from_image
    from src.services.image_decoder import preprocess_face_image
    from src.services.supabase_client import supabase_service
    from src.config import settings
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Disable detailed logs for clean output
logging.getLogger("src.core.inference_engine").setLevel(logging.ERROR)
logging.getLogger("src.services.supabase_client").setLevel(logging.ERROR)

async def single_user_request(user_id: int, img: np.ndarray, semaphore: asyncio.Semaphore):
    """
    Simulates one complete user presence flow: Detect -> Crop -> Embed -> Search
    """
    async with semaphore:  # Limit concurrency
        start_time = time.perf_counter()
        try:
            # 1. Detection
            is_valid, _, _ = validate_single_face(img)
            if not is_valid:
                return {"status": "failed", "time": 0}

            # 2. Crop
            cropped_face = crop_face_from_image(img, margin=0.2)

            # 3. Preprocess
            preprocessed = preprocess_face_image(cropped_face, target_size=settings.model_input_size, normalize=True)

            # 4. Inference (In separate thread to not block event loop excessively)
            # Note: inference_engine has its own lock, but we wrap it to be safe in async context
            embedding = await asyncio.to_thread(inference_engine.predict, preprocessed)

            # 5. Search
            match_result = await supabase_service.find_face_match(embedding, threshold=settings.face_match_threshold)
            
            total_time = (time.perf_counter() - start_time) * 1000
            return {"status": "success", "time": total_time}

        except Exception as e:
            return {"status": "error", "error": str(e), "time": 0}

async def run_stress_test(image_path: str, total_users: int, concurrency: int):
    print(f"\n{'='*60}")
    print(f"STRESS TEST: {total_users} Users")
    print(f"Concurrency: {concurrency} (Simulated parallel requests)")
    print(f"Target Image: {image_path}")
    print(f"{'='*60}\n")

    # Load image once
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Warmup
    print("Initializing models (Warmup)...")
    inference_engine.load_model()
    inference_engine.warmup()
    print("Warmup complete. Starting storm...\n")

    start_global = time.perf_counter()
    
    # Semaphore to control how many tasks run exactly at the same time
    semaphore = asyncio.Semaphore(concurrency)
    
    tasks = []
    print(f"🚀 Processing {total_users} users with concurrency {concurrency}...")
    
    # Create a simple progress reporter
    completed_count = 0
    print_lock = asyncio.Lock()

    async def tracked_request(uid, img, sem):
        nonlocal completed_count
        res = await single_user_request(uid, img, sem)
        async with print_lock:
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == total_users:
                print(f"\rProgress: {completed_count}/{total_users} ({(completed_count/total_users)*100:.1f}%)", end="", flush=True)
        return res

    tasks = [tracked_request(i, img, semaphore) for i in range(total_users)]
    
    # Run all tasks
    results = await asyncio.gather(*tasks)
    print("\nDone!")

    end_global = time.perf_counter()
    duration_sec = end_global - start_global

    # Analysis
    success_times = [r["time"] for r in results if r["status"] == "success"]
    failed = len([r for r in results if r["status"] != "success"])
    
    print(f"\n{'='*60}")
    print("STRESS TEST RESULTS")
    print(f"{'='*60}")
    
    if not success_times:
        print("All requests failed.")
        return

    avg_latency = statistics.mean(success_times)
    p95_latency = statistics.quantiles(success_times, n=20)[18] # 95th percentile
    p99_latency = statistics.quantiles(success_times, n=100)[98] # 99th percentile
    
    rps = total_users / duration_sec

    print(f"Total Time      : {duration_sec:.2f} seconds")
    print(f"Completed       : {len(success_times)} / {total_users}")
    print(f"Failed          : {failed}")
    print(f"{'-'*30}")
    print(f"Throughput (RPS): {rps:.2f} req/sec")
    print(f"Avg Latency     : {avg_latency:.2f} ms")
    print(f"P95 Latency     : {p95_latency:.2f} ms (95% user under this time)")
    print(f"P99 Latency     : {p99_latency:.2f} ms (Max load Lag)")
    print(f"{'='*60}")
    print("Interpretation:")
    print(f"Your server can handle ~{int(rps*60)} users per minute.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress Test Face Recognition System")
    parser.add_argument("--image", type=str, help="Path to test image", required=True)
    parser.add_argument("--total", type=int, help="Total users to simulate", default=100)
    parser.add_argument("--concurrency", type=int, help="Max concurrent requests", default=10)
    
    args = parser.parse_args()
    
    if os.path.exists(args.image):
        asyncio.run(run_stress_test(args.image, args.total, args.concurrency))
    else:
        print(f"Error: Image not found at {args.image}")
