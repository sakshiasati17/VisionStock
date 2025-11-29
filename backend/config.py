"""Configuration settings for VisionStock application."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent  # Project root
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
runs_dir = BASE_DIR / "runs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/scene2sql_db")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Model Configuration
# Use trained Hub model by default, fallback to local trained model or baseline
HUB_MODEL_URL = "https://hub.ultralytics.com/models/jfHGXJxP5esp8iuhi8Yl"
fine_tuned_model = MODELS_DIR / "yolov8-finetuned.pt"
baseline_model = MODELS_DIR / "yolov8n.pt"

# Prioritize Hub model (trained on Google Colab, synced to Ultralytics Hub)
# This is the production-ready model from training run train5
env_model_path = os.getenv("MODEL_PATH")

if env_model_path:
    # Use env var if explicitly set (allows override)
    MODEL_PATH = env_model_path
elif os.getenv("USE_HUB_MODEL", "true").lower() == "true":
    # Use Hub model by default (trained model with 50 epochs, mAP50: 0.0413)
    # Check for local trained model as fallback if Hub is not accessible
    fine_tuned_path = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
    if fine_tuned_path.exists():
        # Prefer local trained model if available
        MODEL_PATH = str(fine_tuned_path.resolve())
    elif fine_tuned_model.exists():
        # Fallback to saved fine-tuned model
        MODEL_PATH = str(fine_tuned_model.resolve())
    else:
        # Use Hub model URL (will be downloaded on first use)
        MODEL_PATH = HUB_MODEL_URL
else:
    # Use local trained model if available
    fine_tuned_path = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
    if fine_tuned_path.exists():
        MODEL_PATH = str(fine_tuned_path.resolve())
    elif fine_tuned_model.exists():
        MODEL_PATH = str(fine_tuned_model.resolve())
    else:
        # Fallback to baseline
        MODEL_PATH = str(baseline_model.resolve())
MODEL_CONFIDENCE = float(os.getenv("MODEL_CONFIDENCE", 0.25))
MODEL_IOU_THRESHOLD = float(os.getenv("MODEL_IOU_THRESHOLD", 0.45))

# Application Settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

