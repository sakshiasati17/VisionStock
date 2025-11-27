"""Fine-tuning script for YOLOv8 on custom retail shelf dataset."""
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dataset_yaml(data_dir: Path, output_path: Path):
    """
    Create YOLO dataset YAML file.
    
    Args:
        data_dir: Path to dataset directory with train/val/test splits
        output_path: Path to save the YAML file
    """
    dataset_config = {
        'path': str(data_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,  # Number of classes (adjust based on your dataset)
        'names': ['product']  # Class names (adjust based on your dataset)
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    logger.info(f"Created dataset YAML at {output_path}")
    return output_path


def train_model(
    model_path: str = "yolov8n.pt",
    data_yaml: str = None,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    lr0: float = 0.01,
    device: str = "0",
    project: str = "runs/detect",
    name: str = None,
    patience: int = 50,
    save_period: int = 10
):
    """
    Fine-tune YOLOv8 model on custom dataset.
    
    Args:
        model_path: Path to pre-trained YOLOv8 weights
        data_yaml: Path to dataset YAML file
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        lr0: Initial learning rate
        device: Device to use (0 for GPU, cpu for CPU)
        project: Project directory for saving results
        name: Experiment name
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
    """
    if name is None:
        name = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting fine-tuning: {name}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Epochs: {epochs}, Batch: {batch}, LR: {lr0}")
    
    # Load pre-trained model
    model = YOLO(model_path)
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save_period=save_period,
        # Data augmentation
        hsv_h=0.015,  # Image HSV-Hue augmentation
        hsv_s=0.7,    # Image HSV-Saturation augmentation
        hsv_v=0.4,    # Image HSV-Value augmentation
        degrees=10,   # Image rotation (+/- deg)
        translate=0.1,  # Image translation (+/- fraction)
        scale=0.5,    # Image scale (+/- gain)
        flipud=0.0,   # Image vertical flip (probability)
        fliplr=0.5,   # Image horizontal flip (probability)
        mosaic=1.0,   # Image mosaic (probability)
        mixup=0.1,    # Image mixup (probability)
        copy_paste=0.0,  # Segment copy-paste (probability)
        # Optimization
        optimizer='AdamW',  # Optimizer
        weight_decay=0.0005,  # Weight decay
        warmup_epochs=3,  # Warmup epochs
        warmup_momentum=0.8,  # Warmup initial momentum
        warmup_bias_lr=0.1,  # Warmup initial bias lr
        box=7.5,  # Box loss gain
        cls=0.5,  # Class loss gain
        dfl=1.5,  # DFL loss gain
        # Validation
        val=True,  # Validate/test during training
        plots=True,  # Save plots during training
        verbose=True
    )
    
    # Save training summary
    summary_path = Path(project) / name / "training_summary.json"
    summary = {
        "model": model_path,
        "dataset": data_yaml,
        "epochs": epochs,
        "batch_size": batch,
        "learning_rate": lr0,
        "image_size": imgsz,
        "device": device,
        "results": {
            "best_fitness": float(results.results_dict.get("metrics/fitness", 0)),
            "best_mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "best_mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0))
        } if hasattr(results, 'results_dict') else {},
        "timestamp": datetime.now().isoformat()
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Training completed. Results saved to {project}/{name}")
    logger.info(f"Best model: {Path(project) / name / 'weights' / 'best.pt'}")
    
    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for retail shelf detection")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pre-trained model path")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="0", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project directory")
    
    args = parser.parse_args()
    
    # Train model
    results, summary = train_model(
        model_path=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        device=args.device,
        project=args.project,
        name=args.name
    )
    
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(json.dumps(summary, indent=2))
    print("="*50)


if __name__ == "__main__":
    main()

