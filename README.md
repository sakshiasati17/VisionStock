# VisionStock - Retail Inventory Detection System

**VisionStock** is an end-to-end computer vision system for automated retail shelf inventory detection and analysis. The system uses fine-tuned YOLOv8 object detection to identify products on shelves, compares detections against planogram expectations, and provides real-time inventory analytics through a Streamlit dashboard.

## 🎯 Project Overview

This project demonstrates the research question: **"Does fine-tuning YOLOv8 on a small, category-specific dataset significantly improve product detection performance on retail shelf images?"**

### Key Features

- ✅ **Baseline Evaluation**: Pre-trained YOLOv8n on SKU-110K samples
- ✅ **Fine-Tuning**: Custom dataset training (34 classes, 111 images) - **Trained on Google Colab**
- ✅ **Production Model**: Trained model hosted on [Ultralytics Hub](https://hub.ultralytics.com/models/jfHGXJxP5esp8iuhi8Yl) (50 epochs, mAP50: 4.13%)
- ✅ **REST API**: FastAPI backend for image upload and detection
- ✅ **Database Integration**: PostgreSQL for storing detections and planograms
- ✅ **SQL Analytics**: Automated discrepancy detection (missing, low stock, misplaced)
- ✅ **Interactive Dashboard**: Streamlit UI for model comparison and inventory analytics
- ✅ **Ultralytics Hub Integration**: Cloud-based training tracking and model versioning

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

## 📁 Project Structure

```
VisionStock/
├── data/
│   ├── baseline_images/      # SKU-110K samples
│   ├── fine_tune_dataset/    # Custom labeled dataset
│   └── sample_uploads/        # Demo images
├── models/
│   ├── yolov8-baseline.pt    # Pre-trained model
│   └── yolov8-finetuned.pt   # Fine-tuned model
├── backend/                  # FastAPI application
├── dashboard/                # Streamlit UI
├── notebooks/                # Jupyter notebooks for analysis
├── sql/                      # Database schemas and queries
├── utils/                    # Utility functions
├── tests/                    # Test scripts
└── results/                  # Training outputs and metrics
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for complete structure.

## 📚 Documentation

- [INSTALLATION.md](INSTALLATION.md) - Detailed setup guide
- [USAGE.md](USAGE.md) - Usage examples and API documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Directory structure

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
python notebooks/baseline_evaluation.py
```

### Fine-Tuning
```bash
python notebooks/fine_tuning.py
```

### Hub Integration
```bash
cd training/projects/retail_shelf_detection
python train_with_hub.py --use-hub
```

## 📊 Results

Training results and metrics are stored in `results/`:
- Baseline metrics comparison
- Fine-tuned model performance
- Detection examples
- Training curves

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

