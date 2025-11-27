"""Health check endpoint for deployment monitoring."""
from fastapi import APIRouter
from backend.db_config import get_db
from sqlalchemy.orm import Session
from fastapi import Depends

router = APIRouter()


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint for deployment monitoring."""
    try:
        # Check database connection
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "service": "VisionStock API"
    }

