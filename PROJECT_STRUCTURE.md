# Project Structure

```
VisionStock/
├── README.md                 # Main project documentation
├── CONTRIBUTING.md           # Contribution guidelines
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── .env.example             # Environment variables template
│
├── backend/                  # FastAPI backend
│   ├── __init__.py
│   ├── main.py              # API routes
│   ├── config.py            # Configuration
│   ├── db_config.py         # Database models
│   ├── schemas.py           # Pydantic schemas
│   └── init_database.py     # Database initialization
│
├── dashboard/                # Streamlit dashboard
│   ├── __init__.py
│   └── app.py               # Dashboard UI
│
├── data/                     # Datasets
│   ├── custom/              # Custom retail dataset
│   │   ├── data.yaml
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── sku110k/             # SKU-110K dataset
│       └── SKU-110K/
│
├── models/                   # Trained models
│   ├── yolov8n.pt          # Baseline model
│   └── yolov8-finetuned.pt # Fine-tuned model
│
├── notebooks/                # Evaluation notebooks
│   ├── baseline_evaluation.py
│   └── fine_tuning.py
│
├── scripts/                  # Utility scripts
│   ├── complete_workflow.py
│   ├── convert_sku110k_to_yolo.py
│   ├── evaluate_sku110k_baseline.py
│   └── run_complete_pipeline.py
│
├── utils/                    # Utility modules
│   ├── inference.py        # YOLO inference wrapper
│   ├── planogram_utils.py  # Planogram utilities
│   └── timer.py            # Performance timing
│
├── sql/                      # SQL scripts
│   ├── create_tables.sql
│   ├── discrepancy_queries.sql
│   └── planogram_table.sql
│
├── tests/                    # Test files
│   ├── api_tests.http
│   └── performance_test.py
│
├── results/                  # Study results
│   ├── study1_comparison.json
│   ├── study2_comparison.json
│   └── FINAL_TWO_STUDY_REPORT.md
│
├── training/                 # Training scripts
│   └── projects/
│       └── retail_shelf_detection/
│
└── docs/                     # Documentation
    ├── API.md
    ├── ARCHITECTURE.md
    └── DEPLOYMENT.md
```
