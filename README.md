# VisionStock

Automated retail shelf inventory detection using fine-tuned YOLOv8. Detects products on shelf images, compares against planograms, and surfaces inventory gaps through a REST API and dashboard.

## Results

| Metric | Baseline (COCO) | Fine-Tuned |
|---|---|---|
| mAP50 (custom dataset) | 0% | 4.04% |
| Recall | 0.28% | 11.79% |
| Precision | 0% | 4.23% |

- 42× recall improvement after fine-tuning on 78 training images across 34 product classes
- Baseline scored 0% on all metrics (COCO classes have zero overlap with retail products)
- Training: 50 epochs, Google Colab T4 GPU, SGD momentum 0.937

## Performance

| Environment | Inference latency |
|---|---|
| MacBook Air (CPU) | avg 59.7ms, cold start 130.5ms |
| Linux server (CPU) | p50 68.4ms, p95 78.2ms |
| GCP Cloud Run full API | < 2s |

Cost on GCP Cloud Run: ~$0.003 per 1,000 images (~25,000 images/month free tier).

## Stack

- **Model**: YOLOv8n (3.2M params), fine-tuned on [Ultralytics Hub](https://hub.ultralytics.com/models/jfHGXJxP5esp8iuhi8Yl)
- **Backend**: FastAPI + SQLAlchemy + PostgreSQL
- **Dashboard**: Streamlit (5 pages: overview, model performance, detection viewer, inventory, reports)
- **Infra**: Docker Compose, GCP Cloud Run

## Quick Start

**Docker (recommended):**
```bash
docker-compose up -d
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

**Manual:**
```bash
pip install -r requirements.txt
cp env_template.txt .env          # fill in DB credentials
python backend/init_database.py
uvicorn backend.main:app --reload --port 8000
streamlit run dashboard/app.py
```

## API Endpoints

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/detect` | Upload shelf image, get bounding boxes |
| GET | `/api/detections` | Detection history |
| POST/GET | `/api/planograms` | Planogram management |
| POST | `/api/analyze` | Compare detections vs planogram |
| GET | `/api/discrepancies` | Missing / misplaced products |
| GET | `/api/summary` | Inventory summary stats |

## Model Config

- Confidence threshold: 0.25 (high recall priority)
- NMS IoU threshold: 0.45
- Input size: 640×640, max upload: 10MB
- Fallback chain: Hub → `best.pt` → `yolov8-finetuned.pt` → `yolov8n.pt`

## Dataset

111 images (78 train / 22 val / 11 test), 34 retail product classes, annotated with Roboflow.
See `results/PERFORMANCE_ANALYSIS.md` for full benchmark details and dataset limitations.
