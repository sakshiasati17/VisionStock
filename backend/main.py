"""FastAPI backend for ShelfSense."""
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timezone
import shutil
from pathlib import Path
import logging

from backend.db_config import get_db, Detection, Planogram, Discrepancy, ModelVersion, ModelMetrics, init_db
from utils.inference import SceneDetector
from backend.config import UPLOAD_DIR, MAX_UPLOAD_SIZE
from backend.schemas import (
    DetectionResponse, DetectionCreate, PlanogramCreate, 
    PlanogramResponse, DiscrepancyResponse, DetectionSummary
)
from backend.health import router as health_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ShelfSense API",
    description="Camera-Aware Inventory Management System API",
    version="1.0.0"
)

# Initialize detector
detector = SceneDetector()

# Include health check router
app.include_router(health_router)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("Database initialized")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "ShelfSense API",
        "version": "1.0.0",
        "endpoints": {
            "detect": "/api/detect",
            "detections": "/api/detections",
            "planograms": "/api/planograms",
            "discrepancies": "/api/discrepancies",
            "summary": "/api/summary"
        }
    }


@app.post("/api/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    shelf_location: Optional[str] = Query(None, description="Shelf location identifier"),
    db: Session = Depends(get_db)
):
    """
    Upload an image and detect objects using YOLOv8.
    Stores detection results in the database.
    """
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_path = UPLOAD_DIR / f"{timestamp}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Run detection
        detections, inference_time_ms = detector.detect(str(file_path))
        
        # Store detections in database
        db_detections = []
        for det in detections:
            db_detection = Detection(
                image_path=str(file_path),
                timestamp=datetime.now(timezone.utc),
                class_name=det["class_name"],
                confidence=det["confidence"],
                x_center=det["x_center"],
                y_center=det["y_center"],
                width=det["width"],
                height=det["height"],
                shelf_location=shelf_location,
                meta_data={"bbox": det["bbox"]}
            )
            db.add(db_detection)
            db_detections.append(db_detection)
        
        db.commit()
        
        # Refresh to get IDs
        for db_det in db_detections:
            db.refresh(db_det)
        
        return DetectionResponse(
            image_path=str(file_path),
            timestamp=datetime.utcnow(),
            detections_count=len(detections),
            detections=[det for det in detections]
        )
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/api/detections", response_model=List[DetectionResponse])
async def get_detections(
    sku: Optional[str] = Query(None),
    shelf_location: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get detection records with optional filters."""
    query = db.query(Detection)
    
    if sku:
        query = query.filter(Detection.sku == sku)
    if shelf_location:
        query = query.filter(Detection.shelf_location == shelf_location)
    
    detections = query.order_by(Detection.timestamp.desc()).limit(limit).all()
    
    # Group by image_path
    results = {}
    for det in detections:
        if det.image_path not in results:
            results[det.image_path] = {
                "image_path": det.image_path,
                "timestamp": det.timestamp,
                "detections": []
            }
        results[det.image_path]["detections"].append({
            "class_name": det.class_name,
            "confidence": det.confidence,
            "x_center": det.x_center,
            "y_center": det.y_center,
            "width": det.width,
            "height": det.height,
            "sku": det.sku,
            "shelf_location": det.shelf_location
        })
    
    return [
        DetectionResponse(
            image_path=r["image_path"],
            timestamp=r["timestamp"],
            detections_count=len(r["detections"]),
            detections=r["detections"]
        )
        for r in results.values()
    ]


@app.post("/api/planograms", response_model=PlanogramResponse)
async def create_planogram(
    planogram: PlanogramCreate,
    db: Session = Depends(get_db)
):
    """Create a new planogram entry."""
    db_planogram = Planogram(
        planogram_name=planogram.planogram_name,
        sku=planogram.sku,
        product_name=planogram.product_name,
        shelf_location=planogram.shelf_location,
        expected_count=planogram.expected_count,
        x_position=planogram.x_position,
        y_position=planogram.y_position,
        meta_data=planogram.meta_data
    )
    db.add(db_planogram)
    db.commit()
    db.refresh(db_planogram)
    
    return PlanogramResponse.model_validate(db_planogram)


@app.get("/api/planograms", response_model=List[PlanogramResponse])
async def get_planograms(
    planogram_name: Optional[str] = Query(None),
    sku: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get planogram records."""
    query = db.query(Planogram)
    
    if planogram_name:
        query = query.filter(Planogram.planogram_name == planogram_name)
    if sku:
        query = query.filter(Planogram.sku == sku)
    
    planograms = query.all()
    return [PlanogramResponse.model_validate(p) for p in planograms]


@app.get("/api/discrepancies", response_model=List[DiscrepancyResponse])
async def get_discrepancies(
    planogram_name: Optional[str] = Query(None),
    sku: Optional[str] = Query(None),
    discrepancy_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get discrepancy records."""
    query = db.query(Discrepancy)
    
    if planogram_name:
        query = query.join(Planogram).filter(Planogram.planogram_name == planogram_name)
    if sku:
        query = query.filter(Discrepancy.sku == sku)
    if discrepancy_type:
        query = query.filter(Discrepancy.discrepancy_type == discrepancy_type)
    
    discrepancies = query.order_by(Discrepancy.timestamp.desc()).limit(limit).all()
    return [DiscrepancyResponse.model_validate(d) for d in discrepancies]


@app.post("/api/analyze")
async def analyze_discrepancies(
    planogram_name: str,
    image_path: Optional[str] = None,
    shelf_location: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Compare detections with planogram and identify discrepancies.
    """
    # Get planogram data
    planogram_entries = db.query(Planogram).filter(
        Planogram.planogram_name == planogram_name
    ).all()
    
    if not planogram_entries:
        raise HTTPException(status_code=404, detail="Planogram not found")
    
    # Get recent detections
    query = db.query(Detection)
    if image_path:
        query = query.filter(Detection.image_path == image_path)
    if shelf_location:
        query = query.filter(Detection.shelf_location == shelf_location)
    
    detections = query.all()
    
    # Group detections by SKU and shelf location
    detection_counts = {}
    for det in detections:
        key = (det.sku or det.class_name, det.shelf_location or "unknown")
        detection_counts[key] = detection_counts.get(key, 0) + 1
    
    # Compare with planogram
    discrepancies = []
    for planogram in planogram_entries:
        key = (planogram.sku, planogram.shelf_location)
        detected_count = detection_counts.get(key, 0)
        expected_count = planogram.expected_count
        
        if detected_count != expected_count:
            discrepancy_type = "missing" if detected_count < expected_count else "extra"
            
            discrepancy = Discrepancy(
                planogram_id=planogram.id,
                sku=planogram.sku,
                shelf_location=planogram.shelf_location,
                discrepancy_type=discrepancy_type,
                expected_count=expected_count,
                detected_count=detected_count,
                timestamp=datetime.now(timezone.utc)
            )
            db.add(discrepancy)
            discrepancies.append(discrepancy)
    
    db.commit()
    
    return {
        "planogram_name": planogram_name,
        "discrepancies_found": len(discrepancies),
        "discrepancies": [
            {
                "sku": d.sku,
                "shelf_location": d.shelf_location,
                "type": d.discrepancy_type,
                "expected": d.expected_count,
                "detected": d.detected_count
            }
            for d in discrepancies
        ]
    }


@app.get("/api/summary", response_model=DetectionSummary)
async def get_summary(
    shelf_location: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get summary statistics of detections."""
    query = db.query(Detection)
    if shelf_location:
        query = query.filter(Detection.shelf_location == shelf_location)
    
    total_detections = query.count()
    unique_skus = query.distinct(Detection.sku).count()
    unique_classes = query.distinct(Detection.class_name).count()
    
    # Average confidence
    from sqlalchemy import func
    avg_confidence = db.query(func.avg(Detection.confidence)).scalar() or 0.0
    
    return DetectionSummary(
        total_detections=total_detections,
        unique_skus=unique_skus,
        unique_classes=unique_classes,
        average_confidence=float(avg_confidence)
    )


@app.post("/api/models")
async def register_model(
    version_name: str,
    model_type: str,
    model_path: str,
    base_model: Optional[str] = None,
    epochs: Optional[int] = None,
    dataset_path: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Register a new model version in the database."""
    if model_type not in ["baseline", "finetuned"]:
        raise HTTPException(status_code=400, detail="model_type must be 'baseline' or 'finetuned'")
    
    # Check if version name already exists
    existing = db.query(ModelVersion).filter(ModelVersion.version_name == version_name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Model version '{version_name}' already exists")
    
    model_version = ModelVersion(
        version_name=version_name,
        model_type=model_type,
        model_path=model_path,
        base_model=base_model,
        epochs=epochs,
        dataset_path=dataset_path,
        is_active=0
    )
    db.add(model_version)
    db.commit()
    db.refresh(model_version)
    
    return {
        "id": model_version.id,
        "version_name": model_version.version_name,
        "model_type": model_version.model_type,
        "model_path": model_version.model_path,
        "created_at": model_version.created_at.isoformat()
    }


@app.get("/api/models")
async def get_models(
    model_type: Optional[str] = None,
    active_only: bool = False,
    db: Session = Depends(get_db)
):
    """Get list of registered model versions."""
    query = db.query(ModelVersion)
    
    if model_type:
        query = query.filter(ModelVersion.model_type == model_type)
    if active_only:
        query = query.filter(ModelVersion.is_active == 1)
    
    models = query.order_by(ModelVersion.created_at.desc()).all()
    
    result = []
    for model in models:
        # Get latest metrics if available
        latest_metrics = db.query(ModelMetrics).filter(
            ModelMetrics.model_version_id == model.id
        ).order_by(ModelMetrics.evaluation_date.desc()).first()
        
        model_data = {
            "id": model.id,
            "version_name": model.version_name,
            "model_type": model.model_type,
            "model_path": model.model_path,
            "base_model": model.base_model,
            "epochs": model.epochs,
            "is_active": bool(model.is_active),
            "created_at": model.created_at.isoformat(),
            "latest_metrics": None
        }
        
        if latest_metrics:
            model_data["latest_metrics"] = {
                "map50": latest_metrics.map50,
                "map50_95": latest_metrics.map50_95,
                "precision": latest_metrics.precision,
                "recall": latest_metrics.recall,
                "f1_score": latest_metrics.f1_score,
                "inference_time_ms": latest_metrics.inference_time_ms,
                "evaluation_date": latest_metrics.evaluation_date.isoformat()
            }
        
        result.append(model_data)
    
    return result


@app.post("/api/models/{model_id}/metrics")
async def add_model_metrics(
    model_id: int,
    map50: float,
    map50_95: float,
    precision: float,
    recall: float,
    f1_score: float,
    inference_time_ms: Optional[float] = None,
    test_dataset_path: Optional[str] = None,
    num_test_images: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Add evaluation metrics for a model version."""
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")
    
    metrics = ModelMetrics(
        model_version_id=model_id,
        map50=map50,
        map50_95=map50_95,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        inference_time_ms=inference_time_ms,
        test_dataset_path=test_dataset_path,
        num_test_images=num_test_images
    )
    db.add(metrics)
    db.commit()
    db.refresh(metrics)
    
    return {
        "id": metrics.id,
        "model_version_id": metrics.model_version_id,
        "map50": metrics.map50,
        "map50_95": metrics.map50_95,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "inference_time_ms": metrics.inference_time_ms,
        "evaluation_date": metrics.evaluation_date.isoformat()
    }


@app.get("/api/models/comparison")
async def compare_models(
    baseline_id: int,
    finetuned_id: int,
    db: Session = Depends(get_db)
):
    """Compare baseline and fine-tuned model metrics."""
    baseline = db.query(ModelVersion).filter(ModelVersion.id == baseline_id).first()
    finetuned = db.query(ModelVersion).filter(ModelVersion.id == finetuned_id).first()
    
    if not baseline or not finetuned:
        raise HTTPException(status_code=404, detail="One or both models not found")
    
    # Get latest metrics
    baseline_metrics = db.query(ModelMetrics).filter(
        ModelMetrics.model_version_id == baseline_id
    ).order_by(ModelMetrics.evaluation_date.desc()).first()
    
    finetuned_metrics = db.query(ModelMetrics).filter(
        ModelMetrics.model_version_id == finetuned_id
    ).order_by(ModelMetrics.evaluation_date.desc()).first()
    
    if not baseline_metrics or not finetuned_metrics:
        raise HTTPException(status_code=404, detail="Metrics not found for one or both models")
    
    # Calculate improvements
    improvements = {
        "map50_improvement": finetuned_metrics.map50 - baseline_metrics.map50,
        "map50_improvement_pct": ((finetuned_metrics.map50 - baseline_metrics.map50) / baseline_metrics.map50 * 100) if baseline_metrics.map50 > 0 else 0,
        "map50_95_improvement": finetuned_metrics.map50_95 - baseline_metrics.map50_95,
        "map50_95_improvement_pct": ((finetuned_metrics.map50_95 - baseline_metrics.map50_95) / baseline_metrics.map50_95 * 100) if baseline_metrics.map50_95 > 0 else 0,
        "precision_improvement": finetuned_metrics.precision - baseline_metrics.precision,
        "recall_improvement": finetuned_metrics.recall - baseline_metrics.recall,
        "f1_improvement": finetuned_metrics.f1_score - baseline_metrics.f1_score
    }
    
    return {
        "baseline": {
            "version_name": baseline.version_name,
            "map50": baseline_metrics.map50,
            "map50_95": baseline_metrics.map50_95,
            "precision": baseline_metrics.precision,
            "recall": baseline_metrics.recall,
            "f1_score": baseline_metrics.f1_score,
            "inference_time_ms": baseline_metrics.inference_time_ms
        },
        "finetuned": {
            "version_name": finetuned.version_name,
            "map50": finetuned_metrics.map50,
            "map50_95": finetuned_metrics.map50_95,
            "precision": finetuned_metrics.precision,
            "recall": finetuned_metrics.recall,
            "f1_score": finetuned_metrics.f1_score,
            "inference_time_ms": finetuned_metrics.inference_time_ms
        },
        "improvements": improvements
    }

