"""Script to initialize the database."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.db_config import init_db, engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Initializing database...")
    try:
        init_db()
        logger.info("✅ Database initialized successfully!")
        logger.info("✅ All tables created!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        logger.error("Please check your DATABASE_URL in .env file")

