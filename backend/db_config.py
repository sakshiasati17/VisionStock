"""Database models and connection for VisionStock."""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timezone
from backend.config import DATABASE_URL

Base = declarative_base()

# Database engine - use pool_pre_ping to handle connection issues gracefully
try:
    engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True, connect_args={"connect_timeout": 10})
except Exception:
    # Fallback to in-memory SQLite if DATABASE_URL is invalid
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Invalid DATABASE_URL, using in-memory SQLite")
    engine = create_engine("sqlite:///:memory:", echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Detection(Base):
    """Stores object detection results from YOLOv8."""
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Detection details
    sku = Column(String, nullable=True, index=True)  # Product SKU if identified
    class_name = Column(String, nullable=False)  # YOLO class name
    confidence = Column(Float, nullable=False)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)  # Which model was used
    
    # Bounding box coordinates (normalized 0-1)
    x_center = Column(Float, nullable=False)
    y_center = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    
    # Additional metadata
    shelf_location = Column(String, nullable=True)  # e.g., "A1", "B2"
    meta_data = Column(JSON, nullable=True)  # Renamed from 'metadata' (reserved in SQLAlchemy)


class Planogram(Base):
    """Stores expected product layout (planogram) data."""
    __tablename__ = "planograms"
    
    id = Column(Integer, primary_key=True, index=True)
    planogram_name = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Product information
    sku = Column(String, nullable=False, index=True)
    product_name = Column(String, nullable=True)
    
    # Expected location
    shelf_location = Column(String, nullable=False)  # e.g., "A1", "B2"
    expected_count = Column(Integer, default=1, nullable=False)
    
    # Position in planogram
    x_position = Column(Float, nullable=True)  # Normalized position
    y_position = Column(Float, nullable=True)
    
    meta_data = Column(JSON, nullable=True)  # Renamed from 'metadata' (reserved in SQLAlchemy)


class Discrepancy(Base):
    """Stores detected discrepancies between planogram and detections."""
    __tablename__ = "discrepancies"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    planogram_id = Column(Integer, ForeignKey("planograms.id"), nullable=False)
    
    # Discrepancy details
    sku = Column(String, nullable=False, index=True)
    shelf_location = Column(String, nullable=False)
    discrepancy_type = Column(String, nullable=False)  # "missing", "extra", "misplaced"
    
    # Counts
    expected_count = Column(Integer, nullable=False)
    detected_count = Column(Integer, nullable=False)
    
    # Additional info
    confidence_score = Column(Float, nullable=True)
    meta_data = Column(JSON, nullable=True)  # Renamed from 'metadata' (reserved in SQLAlchemy)
    
    # Relationship
    planogram = relationship("Planogram", backref="discrepancies")


class ModelVersion(Base):
    """Stores model version information and paths."""
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    version_name = Column(String, nullable=False, unique=True, index=True)
    model_type = Column(String, nullable=False)  # "baseline" or "finetuned"
    model_path = Column(String, nullable=False)
    base_model = Column(String, nullable=True)  # Original pre-trained model
    
    # Training info
    training_date = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    epochs = Column(Integer, nullable=True)
    dataset_path = Column(String, nullable=True)
    
    # Status
    is_active = Column(Integer, default=0, nullable=False)  # 0=inactive, 1=active
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    
    meta_data = Column(JSON, nullable=True)  # Renamed from 'metadata' (reserved in SQLAlchemy)


class ModelMetrics(Base):
    """Stores evaluation metrics for model versions."""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False, index=True)
    evaluation_date = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Performance metrics
    map50 = Column(Float, nullable=False)
    map50_95 = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    
    # Speed metrics
    inference_time_ms = Column(Float, nullable=True)  # Average inference time in milliseconds
    
    # Dataset info
    test_dataset_path = Column(String, nullable=True)
    num_test_images = Column(Integer, nullable=True)
    
    meta_data = Column(JSON, nullable=True)  # Renamed from 'metadata' (reserved in SQLAlchemy)
    
    # Relationship
    model_version = relationship("ModelVersion", backref="metrics")


def init_db():
    """Initialize database tables."""
    try:
    Base.metadata.create_all(bind=engine)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to initialize database tables: {e}")


def get_db():
    """Dependency for FastAPI to get database session."""
    try:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to get database session: {e}")
        # Return None if database is not available
        yield None

