# System Architecture

## Overview
VisionStock is an AI-powered retail inventory management system using YOLOv8 for product detection.

## Components

### 1. Backend (FastAPI)
- RESTful API for detection and analysis
- PostgreSQL database integration
- Model inference service

### 2. Dashboard (Streamlit)
- Interactive web interface
- Real-time visualization
- Two-study comparison display

### 3. Model Training
- YOLOv8 fine-tuning pipeline
- Two-study evaluation approach
- Ultralytics Hub integration

### 4. Data Pipeline
- Custom retail dataset (111 images)
- SKU-110K dataset (11,739 images)
- YOLO format conversion

## Data Flow
1. Image upload → FastAPI
2. YOLOv8 inference → Detection results
3. PostgreSQL storage → Detection records
4. Streamlit dashboard → Visualization
