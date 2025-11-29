FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/
COPY utils/ ./utils/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (Cloud Run uses PORT env var, defaults to 8080)
EXPOSE 8080

# Run API (Cloud Run sets PORT automatically)
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}

