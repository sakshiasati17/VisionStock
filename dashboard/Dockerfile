FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dashboard code
COPY dashboard/ ./dashboard/

# Expose port (Railway uses dynamic PORT)
EXPOSE $PORT

# Run Streamlit (Railway sets PORT automatically)
CMD streamlit run dashboard/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0

