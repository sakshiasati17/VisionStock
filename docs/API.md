# API Documentation

## FastAPI Backend Endpoints

### Detection Endpoints
- `POST /api/detect` - Detect objects in uploaded image
- `GET /api/detections` - Get all detection records
- `GET /api/detections/{id}` - Get specific detection

### Planogram Endpoints
- `GET /api/planograms` - Get all planograms
- `POST /api/planograms` - Create new planogram
- `GET /api/planograms/{id}` - Get specific planogram

### Analysis Endpoints
- `GET /api/analyze` - Analyze shelf inventory
- `GET /api/summary` - Get summary statistics

### Model Endpoints
- `GET /api/models` - Get all registered models
- `GET /api/models/comparison` - Compare baseline vs fine-tuned

See `backend/main.py` for full implementation.
