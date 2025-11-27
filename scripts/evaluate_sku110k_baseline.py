"""Evaluate baseline YOLOv8 model on SKU-110K dataset for Study 1."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_sku110k_baseline(dataset_url=None):
    """
    Evaluate COCO pre-trained YOLOv8n on SKU-110K dataset.
    This is for Study 1: Different datasets approach.
    
    Args:
        dataset_url: Ultralytics Hub dataset URL (optional)
    """
    logger.info("="*80)
    logger.info("STUDY 1: BASELINE EVALUATION ON SKU-110K DATASET")
    logger.info("="*80)
    
    # Load COCO pre-trained model
    model_path = "yolov8n.pt"
    logger.info(f"Loading baseline model: {model_path}")
    model = YOLO(model_path)
    
    # Try Ultralytics Hub dataset first (if URL provided)
    if dataset_url:
        logger.info(f"Using Ultralytics Hub dataset: {dataset_url}")
        try:
            results = model.val(data=dataset_url, imgsz=640, plots=True, save_json=True)
            return _extract_and_save_metrics(results, dataset_url, "Ultralytics Hub Dataset")
        except Exception as e:
            logger.warning(f"Could not use Hub dataset: {e}")
            logger.info("Trying local dataset...")
    
    # Check for SKU-110K dataset in multiple locations
    sku110k_paths = [
        project_root / "data" / "sku110k_subset",
        project_root / "data" / "sku110k",
        Path.home() / "datasets" / "sku110k",
    ]
    
    sku110k_yaml = None
    for path in sku110k_paths:
        yaml_file = path / "data.yaml" if path.is_dir() else path
        if yaml_file.exists():
            sku110k_yaml = str(yaml_file)
            logger.info(f"Found SKU-110K dataset at: {yaml_file}")
            break
    
    if not sku110k_yaml:
        logger.warning("⚠️  SKU-110K dataset not found locally")
        logger.info("\n" + "="*80)
        logger.info("SKU-110K DATASET SETUP INSTRUCTIONS")
        logger.info("="*80)
        logger.info("""
To complete Study 1, you need to download SKU-110K dataset:

Option 1: Download from official source
  - Visit: https://github.com/eg4000/SKU110K_CVPR19
  - Download dataset
  - Extract to: data/sku110k/
  - Create data.yaml with proper structure

Option 2: Use Roboflow
  - Search for "SKU-110K" on Roboflow Universe
  - Download in YOLO format
  - Place in: data/sku110k/

Option 3: Use representative subset
  - Use a smaller retail dataset for faster evaluation
  - Still demonstrates the methodology

For now, we'll create a placeholder result showing the methodology.
        """)
        
        # Create placeholder results showing methodology
        placeholder_results = {
            "study": "Study 1: Different Datasets (Methodology)",
            "baseline": {
                "model": "YOLOv8n (COCO pre-trained)",
                "dataset": "SKU-110K",
                "status": "Dataset needs to be downloaded",
                "expected_metrics": {
                    "note": "COCO pre-trained model on SKU-110K (retail dataset)",
                    "expected_range": "Higher than 0% (some retail object overlap)"
                }
            },
            "finetuned": {
                "model": "YOLOv8n (Fine-tuned)",
                "dataset": "Custom Retail",
                "metrics": {
                    "mAP50": 0.0404,
                    "mAP50_95": 0.0286,
                    "precision": 0.0423,
                    "recall": 0.1179,
                    "f1_score": 0.0622
                },
                "note": "From previous evaluation"
            },
            "methodology": "This study compares baseline on large retail dataset (SKU-110K) vs fine-tuned on custom dataset"
        }
        
        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "study1_methodology.json", "w") as f:
            json.dump(placeholder_results, f, indent=2)
        
        logger.info(f"\n✅ Methodology saved to: {results_dir / 'study1_methodology.json'}")
        return None
    
    try:
        logger.info(f"Evaluating on dataset: {sku110k_yaml}")
        logger.info("This may take a while...")
        
        # Evaluate model
        results = model.val(
            data=sku110k_yaml,
            imgsz=640,
            plots=True,
            save_json=True
        )
        
        # Extract metrics
        metrics = {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
            "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0.0)),
        }
        
        # Calculate F1
        if metrics["precision"] + metrics["recall"] > 0:
            metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
        else:
            metrics["f1_score"] = 0.0
        
        logger.info("\n" + "="*80)
        logger.info("STUDY 1: SKU-110K BASELINE RESULTS")
        logger.info("="*80)
        logger.info(f"mAP50:     {metrics['mAP50']:.4f}")
        logger.info(f"mAP50-95:  {metrics['mAP50_95']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
        logger.info("="*80)
        
        # Save results
        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)
        
        study1_results = {
            "study": "Study 1: Different Datasets",
            "baseline": {
                "model": "YOLOv8n (COCO pre-trained)",
                "dataset": "SKU-110K",
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            },
            "finetuned": {
                "model": "YOLOv8n (Fine-tuned)",
                "dataset": "Custom Retail",
                "metrics": {
                    "mAP50": 0.0404,
                    "mAP50_95": 0.0286,
                    "precision": 0.0423,
                    "recall": 0.1179,
                    "f1_score": 0.0622
                },
                "note": "From previous evaluation"
            }
        }
        
        with open(results_dir / "study1_comparison.json", "w") as f:
            json.dump(study1_results, f, indent=2)
        
        logger.info(f"\n✅ Results saved to: {results_dir / 'study1_comparison.json'}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Error evaluating SKU-110K: {e}")
        import traceback
        traceback.print_exc()
        return None


def _extract_and_save_metrics(results, dataset_source, dataset_name):
    """Extract metrics from results and save Study 1 comparison."""
    metrics = {
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0.0)),
    }
    
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    else:
        metrics["f1_score"] = 0.0
    
    logger.info("\n" + "="*80)
    logger.info("STUDY 1: SKU-110K BASELINE RESULTS")
    logger.info("="*80)
    logger.info(f"mAP50:     {metrics['mAP50']:.4f}")
    logger.info(f"mAP50-95:  {metrics['mAP50_95']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    logger.info("="*80)
    
    # Save results
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    study1_results = {
        "study": "Study 1: Different Datasets",
        "baseline": {
            "model": "YOLOv8n (COCO pre-trained)",
            "dataset": dataset_name,
            "dataset_source": dataset_source,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        },
        "finetuned": {
            "model": "YOLOv8n (Fine-tuned)",
            "dataset": "Custom Retail",
            "metrics": {
                "mAP50": 0.0404,
                "mAP50_95": 0.0286,
                "precision": 0.0423,
                "recall": 0.1179,
                "f1_score": 0.0622
            },
            "note": "From previous evaluation"
        },
        "improvement": {
            "mAP50": 0.0404 - metrics["mAP50"],
            "mAP50_95": 0.0286 - metrics["mAP50_95"],
            "precision": 0.0423 - metrics["precision"],
            "recall": 0.1179 - metrics["recall"],
            "f1_score": 0.0622 - metrics["f1_score"]
        }
    }
    
    with open(results_dir / "study1_comparison.json", "w") as f:
        json.dump(study1_results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {results_dir / 'study1_comparison.json'}")
    return metrics


def main():
    """Main function."""
    # Try with Ultralytics Hub dataset URL
    hub_dataset_url = "https://hub.ultralytics.com/datasets/6rEf4lEzkytF1wAEruxb"
    
    logger.info("Attempting to use Ultralytics Hub dataset...")
    metrics = evaluate_sku110k_baseline(dataset_url=hub_dataset_url)
    
    if metrics:
        logger.info("\n✅ Study 1 baseline evaluation complete!")
        logger.info("📊 Next: Compare with Study 2 results")
    else:
        logger.info("\n💡 Study 1 methodology documented")
        logger.info("📊 You can run this script again after downloading SKU-110K dataset")


if __name__ == "__main__":
    main()
