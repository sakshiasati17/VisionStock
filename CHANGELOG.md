# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-11-26

### Added
- Two-study evaluation approach (Study 1: Different datasets, Study 2: Same dataset)
- FastAPI backend with comprehensive endpoints
- Streamlit dashboard with 8 sections
- SKU-110K dataset integration (11,739 images)
- Custom retail dataset (111 images, 34 classes)
- Ultralytics Hub model integration
- PostgreSQL database integration
- Model performance comparison system
- Planogram management system
- Inventory analysis and discrepancy detection

### Study Results
- Study 1: SKU-110K baseline (8.12% mAP50) vs Custom fine-tuned (4.04% mAP50)
- Study 2: Custom baseline (0%) vs Custom fine-tuned (4.04% mAP50) ✅

### Technical
- YOLOv8 fine-tuning pipeline
- Data augmentation (mosaic, mixup, HSV)
- Professional project structure
- Comprehensive documentation
