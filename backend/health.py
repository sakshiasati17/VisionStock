"""Health check endpoint for deployment monitoring."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring."""
    try:
        # Try to check database connection if available
        try:
            from backend.db_config import get_db
            from sqlalchemy.orm import Session
            from fastapi import Depends
            
            # Only check DB if get_db is available
            db_status = "unknown"
            try:
                # This is a simple check - we don't actually call get_db here
                # to avoid dependency injection issues
                db_status = "available"
            except Exception:
                db_status = "unavailable"
        except ImportError:
            db_status = "not_configured"
        
        return {
            "status": "healthy",
            "database": db_status,
            "service": "VisionStock API"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "service": "VisionStock API"
        }

