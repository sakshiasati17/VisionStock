# VisionStock - Retail Inventory Detection System

**VisionStock** is an end-to-end computer vision system for automated retail shelf inventory detection and analysis. The system uses fine-tuned YOLOv8 object detection to identify products on shelves, compares detections against planogram expectations, and provides real-time inventory analytics through a Streamlit dashboard.

## 🎯 Project Overview

This project demonstrates the research question: **"Does fine-tuning YOLOv8 on a small, category-specific dataset significantly improve product detection performance on retail shelf images?"**

### Key Features

- ✅ **Two-Study Evaluation**: Comprehensive baseline vs fine-tuned comparison
  - Study 1: Different datasets (SKU-110K baseline, Custom fine-tuned)
  - Study 2: Same dataset (Custom baseline, Custom fine-tuned)
- ✅ **Baseline Evaluation**: Pre-trained YOLOv8n on SKU-110K samples
- ✅ **Fine-Tuning**: Custom dataset of 111 images across 34 classes (78 train / 22 val / 11 test) - **Trained on Google Colab**
- ✅ **Trained Model Artifact**: Fine-tuned model hosted on [Ultralytics Hub](https://hub.ultralytics.com/models/jfHGXJxP5esp8iuhi8Yl) (50 epochs, mAP50: 4.04%)
- ✅ **REST API**: FastAPI backend for image upload and detection
- ✅ **Database Integration**: PostgreSQL for storing detections and planograms
- ✅ **SQL Analytics**: Automated discrepancy detection (missing, low stock, misplaced)
- ✅ **Interactive Dashboard**: Streamlit UI with 5 pages including a two-study comparison view
- ✅ **Docker Deployment**: Ready for local and cloud deployment
- ✅ **GCP Cloud Run Ready**: Pre-configured for Google Cloud Platform deployment

## 📊 Original Project Targets

These were initial design targets, not achieved evaluation results. Actual performance is reported in the Results section below.

- ≥10% mAP improvement after fine-tuning vs. baseline *(achieved: 4.04% mAP50 on the same dataset, Study 2)*
- 85-90% precision/recall on evaluation images *(achieved: precision 4.23%, recall 11.79% — limited by the small dataset)*
- ≤5% discrepancy error for stock gap identification
- ≤2 seconds end-to-end latency per image *(achieved: 64.4 ms average — see Performance Benchmarks)*

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
createdb visionstock

# 4. Configure environment
cp the template file from the repo root as .env
# Edit .env with your database credentials

# 5. Initialize database
python backend/init_database.py
```

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
./scripts/deployment/deploy.sh
```

**GCP Cloud Run (Recommended for Large Images):**
```bash
./scripts/deployment/deploy_gcp.sh
```

**Trained Model Artifact:**
The system can use the trained model from [Ultralytics Hub](https://hub.ultralytics.com/models/jfHGXJxP5esp8iuhi8Yl) by default. No local model files needed!

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
│   ├── deployment/          # Deploy scripts + cloudbuild configs
│   ├── notebooks/           # Evaluation scripts
│   └── training/            # Training scripts
├── utils/                    # Utility functions
├── results/                  # Evaluation + benchmark results
├── data/                     # Dataset configs (YAML only)
├── models/                   # Model weights (yolov8n.pt)
└── tests/                    # Test scripts
```
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

**Study 1: Different Datasets (exploratory — baseline and fine-tuned models evaluated on different datasets, so results are not a direct improvement caused by fine-tuning)**
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
- `performance_test_results.json` - Latest latency benchmark (cold start, percentiles)

## ⚙️ Model Configuration

Both inference parameters live in `backend/config.py` and can be overridden via environment variables.

| Parameter | Default | Env override | Why this value |
|---|---|---|---|
| Confidence threshold | **0.25** (`MODEL_CONFIDENCE`) | `MODEL_CONFIDENCE` | A 0.25 confidence threshold prioritises recall in dense shelf scenes. This increases the number of candidate detections but may also increase false positives, so the trade-off is reported through precision and recall. |
| NMS IoU threshold | **0.45** (`MODEL_IOU_THRESHOLD`) | `MODEL_IOU_THRESHOLD` | Predictions with IoU above 0.45 are compared, and lower-confidence overlapping boxes are suppressed to reduce duplicate detections in dense shelf scenes. NMS reduces duplicates but cannot guarantee that a product is never counted twice. |

## ⚡ Performance Benchmarks

Measured locally (macOS, CPU-only, `yolov8n.pt`, 50 warm iterations) — reproduce with:

```bash
python3 tests/performance_test.py
```

| Metric | Value |
|---|---|
| Cold start (model load + first inference) | 995.6 ms |
| Warm average latency | 64.4 ms |
| p50 / p75 | 63.5 / 67.0 ms |
| p95 / p99 | 77.4 / 80.0 ms |
| Requirement (≤2 s/image) | ✅ MET |

Results are saved to `results/performance_test_results.json`. The model is included in the container image to avoid downloading weights during application startup; Cloud Run cold-start latency has not been measured.

## 💰 Illustrative Cloud Run Compute Estimate

Estimated cost per 1,000 images, calculated from [Cloud Run pricing](https://cloud.google.com/run/pricing) (request-based billing, us-central1, checked September 2026):

- CPU: $0.000018 per vCPU-second; Memory: $0.000002 per GiB-second
- Cloud Run bills per request rounded up to the nearest 100 ms; at ~65 ms warm latency each image is billed as 0.1 s on 1 vCPU / 512 MiB

**Per image**: 0.1 vCPU-s × $0.000018 + 0.05 GiB-s × $0.000002 ≈ $0.000002
**→ estimated ≈ $0.002 per 1,000 images** (illustrative estimate — excludes container startup, API overhead, image transfer, database operations, logging, failed requests, concurrency effects, and Mac-vs-Cloud Run CPU differences)

Free tier (request-based billing): 180,000 vCPU-seconds + 360,000 GiB-seconds + 2 million requests per month
**→ estimated ≈ 1.8 million images/month free tier capacity** (CPU-bound at 0.1 vCPU-s per image). These are estimates from public pricing, not measured production costs.

## ⚠️ Dataset Limitations

- **Limited training data**: the custom retail dataset contains 111 images (78 used for training, 22 validation, 11 test), which constrains achievable accuracy
- **Small evaluation set**: 11 test images limits statistical confidence in the reported metrics
- **Class imbalance**: some product classes have far fewer examples than others
- **Precision trade-off**: the fine-tuned model has lower precision than the COCO baseline on the large SKU-110K dataset (Study 1), accepted in exchange for much higher recall
- **Model size**: performance could improve with more data and larger backbones (YOLOv8s/m)

## 🛠️ Technology Stack

- **Computer Vision**: Ultralytics YOLOv8
- **Backend**: FastAPI, SQLAlchemy
- **Database**: PostgreSQL
- **Frontend**: Streamlit
- **Training**: Ultralytics Hub

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.

## 👥 Contributors

- [Sakshi Asati](https://github.com/sakshiasati17)

## 🙏 Acknowledgments

- Ultralytics for YOLOv8
- Roboflow for dataset annotation tools
- Hugging Face for KanOps dataset

