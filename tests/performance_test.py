"""Performance testing for latency requirement (≤2s/image)."""
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.inference import SceneDetector
from backend.config import MODEL_PATH

def test_inference_latency():
    """Test that inference meets ≤2s/image requirement."""
    print("="*80)
    print("PERFORMANCE TESTING - LATENCY VERIFICATION")
    print("="*80)
    
    print(f"Loading model from: {MODEL_PATH}")
    detector = SceneDetector()
    
    # Try multiple test image locations
    test_image_paths = [
        Path("data/custom/test/images"),
        Path("data/custom/val/images"),
        Path("data/custom/train/images"),
        Path("data/sample_uploads"),
    ]
    
    test_image = None
    for path in test_image_paths:
        if path.exists():
            images = list(path.glob("*.jpg"))[:1]
            if images:
                test_image = images[0]
                break
    
    if not test_image or not test_image.exists():
        print("⚠️  No test image found in any location")
        return
    
    print(f"Testing with image: {test_image}")
    
    # Run multiple iterations for average
    times = []
    num_iterations = 3
    
    for i in range(num_iterations):
        start = time.time()
        try:
            results = detector.detect(str(test_image))
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.3f}s")
        except Exception as e:
            print(f"  ⚠️  Iteration {i+1} failed: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print("\n" + "="*80)
        print("PERFORMANCE RESULTS")
        print("="*80)
        print(f"Average latency: {avg_time:.3f}s")
        print(f"Min latency: {min_time:.3f}s")
        print(f"Max latency: {max_time:.3f}s")
        print(f"Requirement: ≤2.0s")
        
        if avg_time <= 2.0:
            print("✅ Latency requirement MET")
        else:
            print("⚠️  Latency requirement NOT MET (exceeds 2s)")
        print("="*80)
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        import json
        with open(results_dir / "performance_test_results.json", "w") as f:
            json.dump({
                "average_latency_s": avg_time,
                "min_latency_s": min_time,
                "max_latency_s": max_time,
                "requirement_met": avg_time <= 2.0,
                "iterations": num_iterations,
                "test_image": str(test_image)
            }, f, indent=2)
        print(f"\n✅ Results saved to results/performance_test_results.json")
    else:
        print("❌ All test iterations failed")

if __name__ == "__main__":
    test_inference_latency()
