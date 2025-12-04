#!/bin/bash
# Quick fix script to set USE_HUB_MODEL=false in Cloud Run
# This makes the backend use yolov8n.pt (baseline model) instead of Hub model

set -e

# Configuration
PROJECT_ID="visionstock-146728282882"
SERVICE_NAME="visionstock-backend"
REGION="us-central1"

echo "🔧 Setting USE_HUB_MODEL=false for $SERVICE_NAME..."

# Update the Cloud Run service with the environment variable
gcloud run services update $SERVICE_NAME \
  --region=$REGION \
  --project=$PROJECT_ID \
  --update-env-vars USE_HUB_MODEL=false \
  --quiet

echo "✅ Successfully updated $SERVICE_NAME"
echo "📋 The service will now use yolov8n.pt (baseline model) instead of Hub model"
echo ""
echo "🔄 The service is being updated. This may take a minute..."
echo "📊 Check status: gcloud run services describe $SERVICE_NAME --region=$REGION --project=$PROJECT_ID"


