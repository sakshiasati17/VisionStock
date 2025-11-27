"""Script to run baseline YOLOv8 evaluation and register model in database."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from sqlalchemy.orm import Session
from backend.db_config import get_db, ModelVersion, ModelMetrics, init_db
from datetime import datetime, timezone
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_baseline_model(model_path: str = "yolov8n.pt", data_yaml: str = None):
    """
    Evaluate baseline YOLOv8 model.
    
    Args:
        model_path: Path to pre-trained YOLOv8 model
        data_yaml: Path to dataset YAML (optional, uses COCO if not provided)
    """
    logger.info("="*60)
    logger.info("BASELINE MODEL EVALUATION")
    logger.info("="*60)
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # If no dataset provided, evaluate on COCO validation set
    if data_yaml is None:
        logger.info("No dataset provided. Evaluating on COCO validation set...")
        logger.warning("⚠️  For SKU-110K evaluation, provide --data argument with dataset YAML")
        
        # Quick validation on COCO (this is just for testing)
        results = model.val(data="coco.yaml", split="val", imgsz=640)
    else:
        logger.info(f"Evaluating on dataset: {data_yaml}")
        results = model.val(data=data_yaml, imgsz=640)
    
    # Extract metrics
    metrics = {
        "map50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0.0)),
    }
    
    # Calculate F1
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    else:
        metrics["f1_score"] = 0.0
    
    logger.info("\n" + "="*60)
    logger.info("BASELINE EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"mAP50:     {metrics['map50']:.4f}")
    logger.info(f"mAP50-95:  {metrics['map50_95']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    logger.info("="*60)
    
    return metrics


def register_baseline_model(db: Session, model_path: str, metrics: dict):
    """Register baseline model in database."""
    logger.info("\n📝 Registering baseline model in database...")
    
    # Check if baseline already exists
    existing = db.query(ModelVersion).filter(ModelVersion.model_type == "baseline").first()
    
    if existing:
        logger.info("Baseline model already exists. Updating...")
        model_version = existing
    else:
        model_version = ModelVersion(
            version_name="yolov8n_baseline",
            model_type="baseline",
            model_path=model_path,
            epochs=0,
            created_at=datetime.now(timezone.utc)
        )
        db.add(model_version)
        db.flush()
    
    # Add metrics
    model_metrics = ModelMetrics(
        model_version_id=model_version.id,
        map50=metrics["map50"],
        map50_95=metrics["map50_95"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"],
        inference_time_ms=45.0  # Approximate baseline inference time
    )
    db.add(model_metrics)
    db.commit()
    
    logger.info(f"✅ Baseline model registered (ID: {model_version.id})")
    return model_version


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline YOLOv8 evaluation")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to baseline model")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset YAML (optional)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, just register model")
    
    args = parser.parse_args()
    
    # Initialize database
    init_db()
    
    # Get database session
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        if args.skip_eval:
            logger.info("Skipping evaluation (using sample metrics)...")
            metrics = {
                "map50": 0.55,
                "map50_95": 0.34,
                "precision": 0.82,
                "recall": 0.70,
                "f1_score": 0.75
            }
        else:
            # Evaluate model
            metrics = evaluate_baseline_model(args.model, args.data)
        
        # Register in database
        register_baseline_model(db, args.model, metrics)
        
        logger.info("\n✅ Baseline evaluation complete!")
        logger.info("💡 Next step: Fine-tune model and run comparison")
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    main()

