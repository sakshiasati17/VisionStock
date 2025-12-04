FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n model (fallback model that's always available)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "Model download attempted"

# Copy backend code
COPY backend/ ./backend/
COPY utils/ ./utils/

# Create upload directory
RUN mkdir -p /app/data/uploads

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV USE_HUB_MODEL=false
ENV MODEL_PATH=yolov8n.pt

# Expose port (Cloud Run uses PORT env var, defaults to 8080)
EXPOSE 8080

# Run API (Cloud Run sets PORT automatically)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]

