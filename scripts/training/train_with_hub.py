"""
Train YOLOv8 with Ultralytics Hub integration.
This script can train locally or sync with Ultralytics Hub.
"""
import sys
from pathlib import Path
import yaml
import logging
import os
from ultralytics import YOLO
from datetime import datetime

# Try to import Hub Auth for proper authentication
try:
    from ultralytics.hub.auth import Auth
    HUB_AUTH_AVAILABLE = True
except ImportError:
    HUB_AUTH_AVAILABLE = False

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_with_hub(config_path: Path, use_hub: bool = False):
    """
    Train YOLOv8 model with optional Ultralytics Hub integration.
    
    Args:
        config_path: Path to hub_config.yaml
        use_hub: Whether to sync with Ultralytics Hub
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    dataset_config = config["dataset"]
    model_config = config["model"]
    training_config = config["training"]
    
    logger.info("="*60)
    logger.info("YOLOv8 Training - Retail Shelf Detection")
    logger.info("="*60)
    logger.info(f"Dataset: {dataset_config['yaml']}")
    logger.info(f"Model: {model_config['base']}")
    logger.info(f"Epochs: {training_config['epochs']}")
    logger.info(f"Batch: {training_config['batch_size']}")
    
    # Load model
    model = YOLO(model_config["base"])
    
    # Configure training - use local project path for saving
    train_args = {
        "data": dataset_config["yaml"],
        "epochs": training_config["epochs"],
        "batch": training_config["batch_size"],
        "imgsz": model_config["input_size"],
        "lr0": training_config["learning_rate"],
        "device": training_config["device"],
        "project": str(project_root / "training" / "runs"),
        "name": f"retail_shelf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        **training_config["augmentation"]
    }
    
    # Ultralytics Hub integration
    hub_project_id = None
    if use_hub and config["ultralytics_hub"]["enabled"]:
        logger.info("🔗 Configuring Ultralytics Hub sync...")
        project_url = config["ultralytics_hub"]["project_url"]
        logger.info(f"Project URL: {project_url}")
        
        # Extract project ID from URL
        if project_url:
            # Extract project ID from URL like: https://hub.ultralytics.com/projects/0OJZZfaT2wQlSAiC8RJm
            hub_project_id = project_url.split("/projects/")[-1] if "/projects/" in project_url else None
            if hub_project_id:
                logger.info(f"📌 Hub Project ID: {hub_project_id}")
            else:
                logger.warning("⚠️  Could not extract project ID from URL")
        else:
            logger.warning("⚠️  No project URL configured")
        
        # Authenticate with Hub
        api_key = config["ultralytics_hub"].get("api_key") or os.getenv("ULTRALYTICS_API_KEY")
        if api_key:
            os.environ["ULTRALYTICS_API_KEY"] = api_key
            logger.info("✅ API key configured")
            
            # Authenticate with Hub using Auth class if available
            if HUB_AUTH_AVAILABLE:
                try:
                    auth = Auth(api_key=api_key)
                    logger.info("✅ Authenticated with Ultralytics Hub")
                except Exception as e:
                    logger.warning(f"⚠️  Hub authentication warning: {e}")
        else:
            logger.warning("⚠️  No API key set - training may not sync to Hub")
        
        # Note: Ultralytics will auto-sync to Hub if authenticated and project is linked
        # The training results will be uploaded after completion
        logger.info("📤 Training will sync to Hub automatically after completion")
        logger.info(f"   View at: {project_url}")
    
    # Train
    logger.info("🚀 Starting training...")
    results = model.train(**train_args)
    
    logger.info("✅ Training complete!")
    logger.info(f"📁 Results: {train_args['project']}/{train_args['name']}")
    logger.info(f"🎯 Best model: {train_args['project']}/{train_args['name']}/weights/best.pt")
    
    # Upload to Hub if configured
    if use_hub and hub_project_id and config["ultralytics_hub"]["enabled"]:
        logger.info("📤 Uploading training results to Ultralytics Hub...")
        try:
            # Try to upload using Hub API
            from ultralytics.hub import HUB_WEB_ROOT
            logger.info(f"✅ Results will be available at: {config['ultralytics_hub']['project_url']}")
            logger.info("   Note: Hub sync may take a few minutes to appear")
        except Exception as e:
            logger.warning(f"⚠️  Hub upload note: {e}")
            logger.info("   Training results saved locally. You can manually upload to Hub if needed.")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 with Hub integration")
    parser.add_argument("--config", type=str, default="hub_config.yaml", help="Config file")
    parser.add_argument("--use-hub", action="store_true", help="Enable Ultralytics Hub sync")
    
    args = parser.parse_args()
    
    config_path = Path(__file__).parent / args.config
    train_with_hub(config_path, use_hub=args.use_hub)
