#!/bin/bash
# GCP Cloud Run Deployment Script for VisionStock
# This script builds and deploys the backend to Google Cloud Run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 VisionStock GCP Cloud Run Deployment${NC}"
echo ""

# Add gcloud to PATH if installed in home directory
if [ -d "$HOME/google-cloud-sdk/bin" ]; then
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Error: gcloud CLI is not installed${NC}"
    echo "Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}⚠️  Not authenticated with gcloud. Please run: gcloud auth login${NC}"
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}⚠️  No GCP project set. Please set it with: gcloud config set project YOUR_PROJECT_ID${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Using GCP Project: ${PROJECT_ID}${NC}"

# Set region (default: us-central1)
REGION=${REGION:-us-central1}
SERVICE_NAME="visionstock-backend"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo -e "${GREEN}✓ Region: ${REGION}${NC}"
echo -e "${GREEN}✓ Service Name: ${SERVICE_NAME}${NC}"
echo ""

# Enable required APIs
echo -e "${YELLOW}📦 Enabling required GCP APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    --project=${PROJECT_ID} 2>/dev/null || true

# Build the Docker image
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:latest -f backend/Dockerfile .

# Tag with commit SHA if available
COMMIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
if [ "$COMMIT_SHA" != "latest" ]; then
    docker tag ${IMAGE_NAME}:latest ${IMAGE_NAME}:${COMMIT_SHA}
fi

# Push to Container Registry
echo -e "${YELLOW}📤 Pushing image to Container Registry...${NC}"
docker push ${IMAGE_NAME}:latest
if [ "$COMMIT_SHA" != "latest" ]; then
    docker push ${IMAGE_NAME}:${COMMIT_SHA}
fi

# Deploy to Cloud Run
echo -e "${YELLOW}🚀 Deploying to Cloud Run...${NC}"
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
    --set-env-vars "PYTHONPATH=/app,PYTHONUNBUFFERED=1" \
    --project ${PROJECT_ID}

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region ${REGION} \
    --format 'value(status.url)' \
    --project ${PROJECT_ID})

echo ""
echo -e "${GREEN}✅ Deployment successful!${NC}"
echo -e "${GREEN}🌐 Service URL: ${SERVICE_URL}${NC}"
echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo "1. Set environment variables (database, etc.):"
echo "   gcloud run services update ${SERVICE_NAME} --region ${REGION} --update-env-vars KEY=VALUE"
echo ""
echo "2. View logs:"
echo "   gcloud run services logs read ${SERVICE_NAME} --region ${REGION}"
echo ""
echo "3. Test the API:"
echo "   curl ${SERVICE_URL}/health"

