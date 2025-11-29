# GCP Cloud Run Deployment Guide

This guide walks you through deploying VisionStock backend to Google Cloud Run.

## Prerequisites

1. **Google Cloud Account**: Sign up at [cloud.google.com](https://cloud.google.com)
2. **GCP Project**: Create a new project or use an existing one
3. **gcloud CLI**: Install from [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install)
4. **Docker**: Install Docker Desktop or Docker Engine

## Initial Setup

### 1. Install and Authenticate gcloud CLI

```bash
# Install gcloud (if not already installed)
# macOS:
brew install --cask google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Verify
gcloud config list
```

### 2. Enable Required APIs

```bash
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com
```

### 3. Set Up Database

You have two options:

**Option A: Cloud SQL (PostgreSQL) - Recommended for Production**
```bash
# Create Cloud SQL instance
gcloud sql instances create visionstock-db \
    --database-version=POSTGRES_14 \
    --tier=db-f1-micro \
    --region=us-central1

# Create database
gcloud sql databases create shelf_sense_db --instance=visionstock-db

# Create user
gcloud sql users create visionstock_user \
    --instance=visionstock-db \
    --password=YOUR_SECURE_PASSWORD
```

**Option B: External PostgreSQL**
- Use any PostgreSQL database (e.g., Supabase, Neon, Cloud SQL)
- Just provide the connection string in environment variables

## Deployment Methods

### Method 1: Automated Script (Recommended)

```bash
# Make script executable
chmod +x scripts/deploy_gcp.sh

# Deploy (uses default region: us-central1)
./scripts/deploy_gcp.sh

# Or specify a region
REGION=us-west1 ./scripts/deploy_gcp.sh
```

The script will:
1. Build the Docker image
2. Push to Google Container Registry
3. Deploy to Cloud Run
4. Provide the service URL

### Method 2: Manual Deployment

```bash
# Set variables
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1
SERVICE_NAME=visionstock-backend
IMAGE_NAME=gcr.io/${PROJECT_ID}/${SERVICE_NAME}

# Build image
docker build -t ${IMAGE_NAME}:latest -f backend/Dockerfile .

# Push to Container Registry
docker push ${IMAGE_NAME}:latest

# Deploy to Cloud Run
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
    --set-env-vars "PYTHONPATH=/app,PYTHONUNBUFFERED=1"
```

### Method 3: Cloud Build (CI/CD)

For automated deployments from GitHub:

1. **Connect Repository**:
   - Go to Cloud Build > Triggers
   - Connect your GitHub repository
   - Create a trigger for `main` branch

2. **Use cloudbuild.yaml**:
   - The `cloudbuild.yaml` file in the root will be used automatically
   - Each push to `main` will trigger a build and deployment

## Configuration

### Set Environment Variables

After deployment, configure your database and other settings:

```bash
gcloud run services update visionstock-backend \
    --region us-central1 \
    --update-env-vars \
    DATABASE_URL=postgresql://user:pass@host:5432/dbname,\
    MODEL_HUB_URL=https://hub.ultralytics.com/models/jfHGXJxP5esp8iuhi8Yl
```

### For Cloud SQL Connection

If using Cloud SQL, you'll need to:

1. **Add Cloud SQL connection**:
```bash
gcloud run services update visionstock-backend \
    --region us-central1 \
    --add-cloudsql-instances PROJECT_ID:REGION:visionstock-db \
    --update-env-vars DATABASE_URL=postgresql://user:pass@/shelf_sense_db?host=/cloudsql/PROJECT_ID:REGION:visionstock-db
```

2. **Update DATABASE_URL** in your environment variables to use Unix socket path.

## Resource Limits

Cloud Run automatically scales based on traffic. Default settings:

- **Memory**: 2Gi (adjustable: 128Mi - 8Gi)
- **CPU**: 2 vCPU (adjustable: 1-8)
- **Timeout**: 300 seconds (max: 3600)
- **Max Instances**: 10 (adjustable: 1-1000)
- **Concurrency**: 80 requests per instance (default)

To adjust:
```bash
gcloud run services update visionstock-backend \
    --region us-central1 \
    --memory 4Gi \
    --cpu 4 \
    --max-instances 20
```

## Monitoring and Logs

### View Logs

```bash
# Real-time logs
gcloud run services logs tail visionstock-backend --region us-central1

# Recent logs
gcloud run services logs read visionstock-backend --region us-central1 --limit 50
```

### View Metrics

- Go to Cloud Run console: https://console.cloud.google.com/run
- Click on `visionstock-backend` service
- View metrics, logs, and revisions

## Cost Estimation

**Free Tier** (Always Free):
- 2 million requests/month
- 360,000 GB-seconds memory
- 180,000 vCPU-seconds

**Beyond Free Tier**:
- $0.40 per million requests
- $0.0000025 per GB-second
- $0.0000100 per vCPU-second

**Example**: 100K requests/month, 2GB memory, 2 vCPU, 1s avg response
- Requests: Free (within 2M limit)
- Memory: ~200K GB-seconds = $0.50
- vCPU: ~200K vCPU-seconds = $2.00
- **Total: ~$2.50/month**

## Troubleshooting

### Build Fails

```bash
# Check build logs
gcloud builds list --limit 5
gcloud builds log BUILD_ID
```

### Service Won't Start

```bash
# Check service logs
gcloud run services logs read visionstock-backend --region us-central1

# Check service status
gcloud run services describe visionstock-backend --region us-central1
```

### Database Connection Issues

- Verify DATABASE_URL is correct
- For Cloud SQL: Ensure Cloud SQL Admin API is enabled
- Check firewall rules if using external database

### Image Too Large

If you hit size limits (unlikely on GCP):
- Current image: ~8.4GB (acceptable on Cloud Run)
- Cloud Run supports up to 10GB images
- Consider multi-stage builds if needed

## Updating the Service

### Update Code

```bash
# Rebuild and redeploy
./scripts/deploy_gcp.sh
```

### Rollback

```bash
# List revisions
gcloud run revisions list --service visionstock-backend --region us-central1

# Rollback to previous revision
gcloud run services update-traffic visionstock-backend \
    --region us-central1 \
    --to-revisions REVISION_NAME=100
```

## Next Steps

1. **Set up custom domain** (optional):
   ```bash
   gcloud run domain-mappings create \
       --service visionstock-backend \
       --domain api.yourdomain.com \
       --region us-central1
   ```

2. **Set up monitoring alerts** in Cloud Console

3. **Configure CI/CD** with Cloud Build triggers

4. **Deploy Streamlit dashboard** separately (or use Streamlit Cloud)

## Support

For issues:
- Check [Cloud Run documentation](https://cloud.google.com/run/docs)
- View service logs: `gcloud run services logs read`
- Check [VisionStock GitHub Issues](https://github.com/sakshiasati17/VisionStock/issues)

