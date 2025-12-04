#!/bin/bash
# Quick fix to set USE_HUB_MODEL=false in Cloud Run
# Run this script to fix the 500 error immediately

set -e

# Add gcloud to PATH if installed in home directory
if [ -d "$HOME/google-cloud-sdk/bin" ]; then
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi

# Configuration
PROJECT_ID="visionstock-146728282882"
SERVICE_NAME="visionstock-backend"
REGION="us-central1"

echo "🔧 Fixing backend model configuration..."
echo "📋 Setting USE_HUB_MODEL=false to use baseline model (yolov8n.pt)"
echo ""

# Check if gcloud is available
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI is not installed or not in PATH"
    echo ""
    echo "Please run this command manually:"
    echo ""
    echo "gcloud run services update $SERVICE_NAME \\"
    echo "  --region=$REGION \\"
    echo "  --project=$PROJECT_ID \\"
    echo "  --update-env-vars USE_HUB_MODEL=false"
    echo ""
    echo "Or install gcloud: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "⚠️  Not authenticated. Please run: gcloud auth login"
    exit 1
fi

# Update the service
echo "🔄 Updating Cloud Run service..."
gcloud run services update $SERVICE_NAME \
  --region=$REGION \
  --project=$PROJECT_ID \
  --update-env-vars USE_HUB_MODEL=false \
  --quiet

echo ""
echo "✅ Successfully updated $SERVICE_NAME!"
echo "📋 The service will now use yolov8n.pt (baseline model) instead of Hub model"
echo ""
echo "⏳ The service is being updated. This may take 1-2 minutes..."
echo ""
echo "🧪 Test the API after a minute:"
echo "   curl https://visionstock-backend-146728282882.us-central1.run.app/health"
echo ""
echo "📊 Check service status:"
echo "   gcloud run services describe $SERVICE_NAME --region=$REGION --project=$PROJECT_ID"


