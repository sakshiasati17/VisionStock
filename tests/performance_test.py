"""Performance testing for latency requirement (<=2s/image)."""
import time
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.inference import SceneDetector
from backend.config import MODEL_PATH

NUM_ITERATIONS = 50


def find_test_image():
    """Find the first available test image, or generate a synthetic one."""
    test_image_paths = [
        Path("data/custom/test/images"),
        Path("data/custom/val/images"),
        Path("data/custom/train/images"),
        Path("data/uploads"),
        Path("data/sample_uploads"),
    ]
    for path in test_image_paths:
        if path.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                images = list(path.glob(ext))[:1]
                if images:
                    return images[0]
    # Fallback: generate a synthetic shelf-like image so latency can still be measured
    import cv2
    import numpy as np
    path = Path("data/uploads")
    path.mkdir(parents=True, exist_ok=True)
    synthetic = path / "benchmark_synthetic.jpg"
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(synthetic), img)
    print(f"No dataset images found - generated synthetic image: {synthetic}")
    return synthetic


def percentile(sorted_values, p):
    """Linear-interpolated p-th percentile of an ascending-sorted list."""
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(k)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (k - lower)


def test_inference_latency(num_iterations=NUM_ITERATIONS):
    """Measure cold-start time, warm latency and percentiles over N iterations."""
    print("=" * 80)
    print(f"PERFORMANCE TESTING - {num_iterations} ITERATIONS")
    print("=" * 80)

    test_image = find_test_image()
    if not test_image or not test_image.exists():
        print("No test image found in any location")
        return

    print(f"Testing with image: {test_image}")
    print(f"Model: {MODEL_PATH}")

    # Cold start: model load + first inference
    print(f"\nMeasuring cold start (model load + first inference)...")
    cold_start_begin = time.time()
    detector = SceneDetector()
    first_detections, _ = detector.detect(str(test_image))
    cold_start_ms = (time.time() - cold_start_begin) * 1000
    print(f"  Cold start: {cold_start_ms:.1f} ms")

    # Warm latency: N iterations
    print(f"\nRunning {num_iterations} warm iterations...")
    times_ms = []
    for i in range(num_iterations):
        start = time.time()
        try:
            detector.detect(str(test_image))
            elapsed_ms = (time.time() - start) * 1000
            times_ms.append(elapsed_ms)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{num_iterations} iterations done")
        except Exception as e:
            print(f"  Iteration {i + 1} failed: {e}")

    if not times_ms:
        print("All test iterations failed")
        return

    times_sorted = sorted(times_ms)
    stats = {
        "cold_start_ms": round(cold_start_ms, 1),
        "iterations": num_iterations,
        "warm_avg_ms": round(sum(times_ms) / len(times_ms), 1),
        "warm_min_ms": round(times_sorted[0], 1),
        "warm_max_ms": round(times_sorted[-1], 1),
        "p50_ms": round(percentile(times_sorted, 50), 1),
        "p75_ms": round(percentile(times_sorted, 75), 1),
        "p95_ms": round(percentile(times_sorted, 95), 1),
        "p99_ms": round(percentile(times_sorted, 99), 1),
        "requirement_met": (sum(times_ms) / len(times_ms)) <= 2000,
        "model": str(MODEL_PATH),
        "test_image": str(test_image),
    }

    print("\n" + "=" * 80)
    print("PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Cold start (model load + 1st inference): {stats['cold_start_ms']} ms")
    print(f"Warm average latency: {stats['warm_avg_ms']} ms")
    print(f"p50: {stats['p50_ms']} ms | p75: {stats['p75_ms']} ms")
    print(f"p95: {stats['p95_ms']} ms | p99: {stats['p99_ms']} ms")
    print(f"Requirement: <=2000 ms -> {'MET' if stats['requirement_met'] else 'NOT MET'}")
    print("=" * 80)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "performance_test_results.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to results/performance_test_results.json")


if __name__ == "__main__":
    test_inference_latency()
