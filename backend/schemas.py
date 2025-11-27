"""Pydantic schemas for API request/response models."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class DetectionBase(BaseModel):
    """Base detection schema."""
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    x_center: float = Field(..., ge=0.0, le=1.0)
    y_center: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., ge=0.0, le=1.0)
    height: float = Field(..., ge=0.0, le=1.0)
    sku: Optional[str] = None
    shelf_location: Optional[str] = None


class DetectionCreate(DetectionBase):
    """Schema for creating a detection."""
    pass


class DetectionResponse(BaseModel):
    """Schema for detection response."""
    image_path: str
    timestamp: datetime
    detections_count: int
    detections: List[Dict[str, Any]]

    class Config:
        from_attributes = True
        populate_by_name = True


class PlanogramCreate(BaseModel):
    """Schema for creating a planogram."""
    planogram_name: str
    sku: str
    product_name: Optional[str] = None
    shelf_location: str
    expected_count: int = Field(default=1, ge=0)
    x_position: Optional[float] = None
    y_position: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class PlanogramResponse(BaseModel):
    """Schema for planogram response."""
    id: int
    planogram_name: str
    sku: str
    product_name: Optional[str]
    shelf_location: str
    expected_count: int
    x_position: Optional[float]
    y_position: Optional[float]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True


class DiscrepancyResponse(BaseModel):
    """Schema for discrepancy response."""
    id: int
    timestamp: datetime
    sku: str
    shelf_location: str
    discrepancy_type: str
    expected_count: int
    detected_count: int
    confidence_score: Optional[float]

    class Config:
        from_attributes = True
        populate_by_name = True


class DetectionSummary(BaseModel):
    """Schema for detection summary statistics."""
    total_detections: int
    unique_skus: int
    unique_classes: int
    average_confidence: float

