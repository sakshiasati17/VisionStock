#!/bin/bash
# Deploy VisionStock Backend to Cloud Run with all fixes
# This script builds and deploys with USE_HUB_MODEL=false and MODEL_PATH=yolov8n.pt

set -e

# Add gcloud to PATH
if [ -d "$HOME/google-cloud-sdk/bin" ]; then
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi

# Configuration
PROJECT_ID="visionstock-146728282882"
SERVICE_NAME="visionstock-backend"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying VisionStock Backend with all fixes..."
echo ""

# Check gcloud
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud not found. Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "⚠️  Please authenticate: gcloud auth login"
    exit 1
fi

# Set project
echo "📋 Setting project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID} 2>&1 || {
    echo "⚠️  Could not set project. You may need to use a different account."
    echo "   Current account: $(gcloud auth list --filter=status:ACTIVE --format='value(account)')"
    echo ""
    echo "   Please use Cloud Console to deploy instead (see DEPLOY_INSTRUCTIONS.md)"
    exit 1
}

# Enable APIs
echo "📦 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com --project=${PROJECT_ID} 2>/dev/null || true

# Build using Cloud Build
echo "🔨 Building Docker image with Cloud Build..."
gcloud builds submit \
    --config scripts/deployment/cloudbuild.yaml \
    --timeout=20m \
    --project=${PROJECT_ID} 2>&1

# Deploy to Cloud Run with environment variables
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --region ${REGION} \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --set-env-vars "PYTHONPATH=/app,PYTHONUNBUFFERED=1,USE_HUB_MODEL=false,MODEL_PATH=yolov8n.pt" \
    --project ${PROJECT_ID}

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region ${REGION} \
    --format 'value(status.url)' \
    --project ${PROJECT_ID} 2>/dev/null || echo "N/A")

echo ""
echo "✅ Deployment complete!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo ""
echo "🧪 Test the API:"
echo "   curl ${SERVICE_URL}/health"
echo ""
echo "📊 View logs:"
echo "   gcloud run services logs read ${SERVICE_NAME} --region ${REGION} --project ${PROJECT_ID}"


