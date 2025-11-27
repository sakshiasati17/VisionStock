# Project Structure

```
VisionStock/
├── README.md                    # Main project documentation
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
├── .env.example                 # Environment variables template
├── docker-compose.yml           # Docker Compose configuration
├── railway.toml                 # Railway deployment config
├── .railwayignore               # Railway build exclusions
├── Makefile                     # Common commands
├── LICENSE                      # MIT License
│
├── backend/                     # FastAPI backend
│   ├── __init__.py
│   ├── main.py                  # API routes
│   ├── config.py                # Configuration
│   ├── db_config.py             # Database models
│   ├── schemas.py               # Pydantic schemas
│   ├── health.py                 # Health check endpoint
│   ├── init_database.py          # Database initialization
│   ├── Dockerfile                # Backend Docker image
│   └── sql/                      # SQL scripts
│       ├── create_tables.sql
│       ├── planogram_table.sql
│       └── discrepancy_queries.sql
│
├── dashboard/                   # Streamlit dashboard
│   ├── __init__.py
│   ├── app.py                   # Dashboard UI
│   └── Dockerfile                # Dashboard Docker image
│
├── data/                        # Datasets (YAML configs only)
│   ├── custom/
│   │   └── data.yaml            # Custom dataset config
│   └── sku110k/
│       └── SKU-110K/
│           └── data.yaml        # SKU-110K dataset config
│
├── models/                      # Model files
│   └── yolov8n.pt              # Baseline YOLOv8n model
│
├── scripts/                     # All scripts organized
│   ├── deploy.sh                # Local deployment script
│   ├── convert_sku110k_to_yolo.py
│   ├── evaluate_sku110k_baseline.py
│   ├── run_complete_pipeline.py
│   ├── notebooks/               # Evaluation scripts
│   │   ├── baseline_evaluation.py
│   │   └── fine_tuning.py
│   └── training/                # Training scripts
│       ├── train_with_hub.py
│       └── hub_config.yaml
│
├── utils/                       # Utility modules
│   ├── inference.py             # YOLO inference wrapper
│   ├── planogram_utils.py       # Planogram utilities
│   └── timer.py                 # Performance timing
│
├── results/                     # Evaluation results
│   ├── study1_comparison.json   # Study 1 metrics
│   ├── study2_comparison.json   # Study 2 metrics
│   └── FINAL_TWO_STUDY_REPORT.md
│
├── tests/                       # Tests
│   ├── api_tests.http
│   └── performance_test.py
│
└── docs/                        # Documentation
    ├── API.md
    ├── ARCHITECTURE.md
    ├── DEPLOYMENT.md
    └── RAILWAY_DEPLOYMENT.md
```

## Key Directories

- **backend/**: FastAPI application with all API endpoints
- **dashboard/**: Streamlit web interface
- **scripts/**: All utility, training, and evaluation scripts
- **utils/**: Reusable utility functions
- **results/**: Model evaluation results and reports
- **docs/**: Project documentation

## Excluded from Git

The following are excluded via `.gitignore`:
- Large model files (`*.pt` except `yolov8n.pt`)
- Dataset images and labels
- Training runs (`runs/`)
- Cache files (`*.cache`)
- Logs (`logs/`)
- Environment files (`.env`)
