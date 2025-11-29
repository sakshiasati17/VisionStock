# VisionStock - Retail Inventory Detection System

**VisionStock** is an end-to-end computer vision system for automated retail shelf inventory detection and analysis. The system uses fine-tuned YOLOv8 object detection to identify products on shelves, compares detections against planogram expectations, and provides real-time inventory analytics through a Streamlit dashboard.

## 🎯 Project Overview

This project demonstrates the research question: **"Does fine-tuning YOLOv8 on a small, category-specific dataset significantly improve product detection performance on retail shelf images?"**

### Key Features

- ✅ **Two-Study Evaluation**: Comprehensive baseline vs fine-tuned comparison
  - Study 1: Different datasets (SKU-110K baseline, Custom fine-tuned)
  - Study 2: Same dataset (Custom baseline, Custom fine-tuned)
- ✅ **Baseline Evaluation**: Pre-trained YOLOv8n on SKU-110K samples
- ✅ **Fine-Tuning**: Custom dataset training (34 classes, 111 images) - **Trained on Google Colab**
- ✅ **Production Model**: Trained model hosted on [Ultralytics Hub](https://hub.ultralytics.com/models/jfHGXJxP5esp8iuhi8Yl) (50 epochs, mAP50: 4.13%)
- ✅ **REST API**: FastAPI backend for image upload and detection
- ✅ **Database Integration**: PostgreSQL for storing detections and planograms
- ✅ **SQL Analytics**: Automated discrepancy detection (missing, low stock, misplaced)
- ✅ **Interactive Dashboard**: Streamlit UI with 8 sections including two-study comparison
- ✅ **Docker Deployment**: Ready for local and cloud deployment
- ✅ **GCP Cloud Run Ready**: Pre-configured for Google Cloud Platform deployment

## 📊 Success Metrics

- **≥10% mAP improvement** after fine-tuning vs. baseline
- **85-90% precision/recall** on evaluation images
- **≤5% discrepancy error** for stock gap identification
- **≤2 seconds** end-to-end latency per image

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- pip or conda

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd VisionStock

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up database
createdb shelf_sense_db

# 4. Configure environment
cp env_template.txt .env
# Edit .env with your database credentials

# 5. Initialize database
python backend/init_database.py
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.

### Running the Application

**Option 1: Docker (Recommended)**
```bash
docker-compose up -d
```

**Option 2: Manual Start**

**Start FastAPI Backend:**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Start Streamlit Dashboard** (in new terminal):
```bash
streamlit run dashboard/app.py
```

**Access Services:**
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

### Deployment

**Local Docker:**
```bash
./scripts/deploy.sh
```

**GCP Cloud Run (Recommended for Large Images):**
```bash
# Quick deploy
./scripts/deploy_gcp.sh

# Or see detailed guide
# See docs/GCP_DEPLOYMENT.md for step-by-step instructions
```

**Production Model:**
The system uses the trained model from [Ultralytics Hub](https://hub.ultralytics.com/models/jfHGXJxP5esp8iuhi8Yl) by default. No local model files needed!

## 📁 Project Structure

```
VisionStock/
├── backend/                  # FastAPI application
│   ├── main.py              # API routes
│   ├── config.py            # Configuration
│   ├── db_config.py         # Database models
│   └── sql/                 # SQL scripts
├── dashboard/                # Streamlit UI
│   └── app.py               # Dashboard interface
├── scripts/                  # All scripts organized
│   ├── notebooks/           # Evaluation scripts
│   └── training/            # Training scripts
├── utils/                    # Utility functions
├── results/                  # Evaluation results
│   ├── study1_comparison.json
│   ├── study2_comparison.json
│   └── FINAL_TWO_STUDY_REPORT.md
├── data/                     # Dataset configs (YAML only)
├── models/                   # Model files
├── docs/                     # Documentation
└── tests/                    # Test scripts
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for complete structure.

## 📚 Documentation

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Complete directory structure
- [docs/GCP_DEPLOYMENT.md](docs/GCP_DEPLOYMENT.md) - GCP Cloud Run deployment guide

## 🔌 API Endpoints

### Detection
- `POST /api/detect` - Upload image and detect objects
- `GET /api/detections` - Get detection records

### Planograms
- `POST /api/planograms` - Create planogram entry
- `GET /api/planograms` - Get planogram records

### Analytics
- `POST /api/analyze` - Compare detections with planogram
- `GET /api/discrepancies` - Get discrepancy records
- `GET /api/summary` - Get summary statistics

See [USAGE.md](USAGE.md) for detailed API examples.

## 🧪 Training

### Baseline Evaluation
```bash
python scripts/notebooks/baseline_evaluation.py
```

### Fine-Tuning
```bash
python scripts/notebooks/fine_tuning.py
```

### Hub Integration (Training on Google Colab)
The model was trained on Google Colab and synced to Ultralytics Hub. For local training with Hub:
```bash
python scripts/training/train_with_hub.py
```

**Note**: The production model is already trained and available on [Ultralytics Hub](https://hub.ultralytics.com/models/jfHGXJxP5esp8iuhi8Yl).

## 📊 Results

### Two-Study Evaluation Approach

**Study 1: Different Datasets (As Per Original Proposal)**
- Baseline: COCO pre-trained on SKU-110K dataset
- Fine-Tuned: Custom Retail Dataset
- Results: See `results/study1_comparison.json`

**Study 2: Same Dataset (Before/After Fine-Tuning)**
- Baseline: COCO pre-trained on Custom Retail Dataset
- Fine-Tuned: Custom Retail Dataset
- Results: See `results/study2_comparison.json`

Training results and metrics are stored in `results/`:
- `study1_comparison.json` - Study 1 metrics
- `study2_comparison.json` - Study 2 metrics
- `FINAL_TWO_STUDY_REPORT.md` - Comprehensive comparison report

## 🛠️ Technology Stack

- **Computer Vision**: Ultralytics YOLOv8
- **Backend**: FastAPI, SQLAlchemy
- **Database**: PostgreSQL
- **Frontend**: Streamlit
- **Training**: Ultralytics Hub

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributors here]

## 🙏 Acknowledgments

- Ultralytics for YOLOv8
- Roboflow for dataset annotation tools
- Hugging Face for KanOps dataset

