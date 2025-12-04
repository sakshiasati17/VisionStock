# VisionStock: AI-Powered Retail Inventory Detection System
## Project Proposal

### Research Question
**Does fine-tuning YOLOv8 on a small, category-specific retail dataset significantly improve product detection performance on shelf images compared to baseline pre-trained models?**

### Dataset
- **Primary Dataset**: Custom retail dataset (111 images, 34 product classes)
- **Baseline Dataset**: SKU-110K (11,739 images, 1 class) for comparison
- **Format**: YOLO format with bounding box annotations
- **Source**: Custom annotated retail shelf images + public SKU-110K dataset

### Proposed Methodology
1. **Baseline Evaluation**: Evaluate COCO pre-trained YOLOv8n on SKU-110K dataset
2. **Fine-Tuning**: Fine-tune YOLOv8n on custom retail dataset (50 epochs)
3. **Two-Study Comparison**:
   - Study 1: Baseline (SKU-110K) vs Fine-tuned (Custom) - different datasets
   - Study 2: Baseline (Custom) vs Fine-tuned (Custom) - same dataset
4. **Performance Metrics**: mAP50, mAP50-95, Precision, Recall, F1-Score
5. **Deployment**: FastAPI backend + Streamlit dashboard for real-time inference

### Success Metrics
- **Primary**: ≥10% mAP50 improvement after fine-tuning vs baseline
- **Secondary**: 
  - 85-90% precision/recall on evaluation images
  - ≤5% discrepancy error for stock gap identification
  - ≤2 seconds end-to-end latency per image
- **Deployment**: Successfully deploy on GCP Cloud Run with working dashboard

### Expected Outcomes
- Demonstrate transfer learning effectiveness for domain-specific CV tasks
- Provide production-ready inventory detection system
- Showcase complete ML pipeline from data to deployment
- Achieve competitive performance on retail product detection task

### Timeline
- **Weeks 1-2**: Data collection and annotation
- **Weeks 3-4**: Baseline evaluation and model training
- **Weeks 5-6**: Fine-tuning and optimization
- **Weeks 7-8**: Two-study evaluation and analysis
- **Weeks 9-10**: Dashboard development and deployment
- **Weeks 11-12**: Documentation and final report

### Technology Stack
- **Model**: YOLOv8 (Ultralytics)
- **Framework**: PyTorch (via Ultralytics)
- **Backend**: FastAPI, PostgreSQL
- **Frontend**: Streamlit
- **Deployment**: Docker, GCP Cloud Run



