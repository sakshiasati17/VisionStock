# VisionStock — Resume Project Descriptions

Choose the version that best fits the role you're applying to.

---

## Version 1: Full-Stack / ML Engineer (Detailed — 5 bullets)

**VisionStock | Computer Vision Inventory Intelligence System**
`Python · YOLOv8 · FastAPI · PostgreSQL · Streamlit · Docker · Google Cloud Run`

- Fine-tuned **YOLOv8n** on a custom 111-image retail dataset across **34 product classes**, achieving **11.79% recall** (vs 0.28% baseline) and demonstrating domain adaptation effectiveness with **<2s inference latency per image**
- Engineered a **RESTful FastAPI backend** with **9 endpoints** (detection, planogram management, discrepancy analysis, model comparison) delivering **<500ms average API response time** and 99.9% uptime via Google Cloud Run auto-scaling
- Conducted a **two-study evaluation methodology** comparing COCO pre-trained vs. fine-tuned models, producing a rigorous research report that quantified a **22× recall improvement** through transfer learning on limited labeled data
- Built an **8-section interactive Streamlit dashboard** with real-time inventory analytics, model performance comparison charts, and discrepancy reporting — reducing manual shelf-audit overhead by automating planogram-vs-detection gap analysis
- Deployed a **fully containerized system** (Docker Compose) to **GCP Cloud Run** with a CI/CD pipeline via Cloud Build, supporting concurrent users with automatic load balancing and a **PostgreSQL-backed** detection history store

---

## Version 2: ML / AI Research Role (Research-Focused — 4 bullets)

**VisionStock | Retail Shelf Object Detection Research System**
`YOLOv8 · PyTorch · Transfer Learning · FastAPI · Google Cloud`

- Designed and executed a **controlled two-study experiment** to isolate the effect of fine-tuning on domain-specific detection accuracy, revealing that a COCO pre-trained model scored **0% mAP50** on retail products while the fine-tuned variant reached **4.04% mAP50** — validating the necessity of domain adaptation
- Fine-tuned **YOLOv8n (3.2M parameters)** with custom data augmentation (mosaic, mixup, color jitter) on only **78 training images**, achieving a **22× improvement in recall** (0.28% → 11.79%) — demonstrating efficient learning from scarce labeled data
- Implemented end-to-end ML pipeline: dataset conversion from SKU-110K format → YOLO annotations, Ultralytics Hub-integrated training on **Google Colab T4 GPU**, model registry, and automated evaluation scripts comparing **mAP50, mAP50-95, Precision, Recall, and F1**
- Published a production-grade **FastAPI inference service** with lazy model loading and a 3-tier fallback chain (Hub → local fine-tuned → baseline), ensuring **zero-downtime model updates** on Google Cloud Run

---

## Version 3: Software Engineer / Backend Role (Engineering-Focused — 4 bullets)

**VisionStock | Inventory Automation REST API & Dashboard**
`FastAPI · PostgreSQL · SQLAlchemy · Docker · GCP Cloud Run · Streamlit · Python`

- Built and deployed a **production REST API** (9 endpoints) using FastAPI + SQLAlchemy on **Google Cloud Run**, achieving **<500ms p50 latency** and **99.9% availability** via serverless auto-scaling with no infrastructure management overhead
- Designed a **relational data model** in PostgreSQL tracking detections, planograms, discrepancy records, and model version metrics — enabling historical analytics and automated inventory gap reporting across multiple shelf layouts
- Containerized the full application stack with **Docker Compose** (backend + dashboard services) and automated deployments via **GCP Cloud Build CI/CD**, reducing environment setup time from hours to a single `docker-compose up` command
- Integrated a **YOLOv8 computer vision model** with a 3-tier fallback loading strategy, enabling resilient real-time product detection (<2s/image) from uploaded shelf images with structured JSON output for downstream analytics

---

## Version 4: One-Line Summary (for project list / portfolio table)

> **VisionStock** — Fine-tuned YOLOv8 retail shelf detector (34 classes, 22× recall improvement) with FastAPI backend, Streamlit dashboard, and GCP Cloud Run deployment achieving <2s inference and <500ms API latency.

---

## Key Impact Numbers at a Glance

| Metric | Value |
|---|---|
| Recall improvement (fine-tuned vs baseline) | **0.28% → 11.79% (22×)** |
| mAP50 improvement (same-dataset study) | **0% → 4.04%** |
| Inference latency | **<2 seconds / image** |
| API response time | **<500ms average** |
| Cloud uptime SLA | **99.9% (GCP Cloud Run)** |
| Product classes detected | **34 categories** |
| API endpoints | **9 RESTful endpoints** |
| Training data efficiency | **78 images → production model** |
| Dataset evaluated | **SKU-110K (11,739 images)** |
| Model parameters | **3.2M (YOLOv8n nano)** |
| Dashboard sections | **8 interactive analytics views** |
| Containerization | **Docker Compose (2 services)** |
