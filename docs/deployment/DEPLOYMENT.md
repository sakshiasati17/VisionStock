# Deployment Guide

## 🚀 Quick Start

### Local Development (Docker)
```bash
# Using Makefile
make setup
make run-api      # Terminal 1
make run-dashboard # Terminal 2

# Or using Docker Compose
./scripts/deploy.sh
```

### Local Development (Manual)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up database
python backend/init_database.py

# 3. Configure environment
cp .env.example .env
# Edit .env with your settings

# 4. Start services
cd backend && uvicorn main:app --reload  # Terminal 1
streamlit run dashboard/app.py            # Terminal 2
```

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up -d --build
```

### Services
- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Database**: localhost:5432

---

## ☁️ Cloud Deployment

### Option 1: GCP Cloud Run (Recommended)

See [docs/GCP_DEPLOYMENT.md](docs/GCP_DEPLOYMENT.md) for detailed instructions.

Quick deploy:
```bash
./scripts/deploy_gcp.sh
```

### Option 2: Render.com

1. **Create Account**: Sign up at https://render.com
2. **New Web Service**: Connect GitHub repo
3. **Add PostgreSQL**: Create managed PostgreSQL database
4. **Configure**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Environment: Set all required variables
5. **Deploy**: Render auto-deploys on git push

### Option 3: Heroku

```bash
# Install Heroku CLI
heroku login
heroku create visionstock

# Add PostgreSQL
heroku addons:create heroku-postgresql:hobby-dev

# Set environment variables
heroku config:set HUB_MODEL_URL=https://hub.ultralytics.com/models/jfHGXJxP5esp8iuhi8Yl
heroku config:set USE_HUB_MODEL=true

# Deploy
git push heroku main
```

### Option 4: Docker Hub + Any Platform

```bash
# Build images
docker build -f backend/Dockerfile -t visionstock/backend:latest .
docker build -f dashboard/Dockerfile -t visionstock/dashboard:latest .

# Push to Docker Hub
docker push visionstock/backend:latest
docker push visionstock/dashboard:latest

# Deploy to any container platform (AWS ECS, Google Cloud Run, etc.)
```

---

## 📋 Environment Variables

Required variables (see `.env.example`):

```bash
# Database
DATABASE_URL=postgresql://user:password@host:5432/dbname

# Model
HUB_MODEL_URL=https://hub.ultralytics.com/models/jfHGXJxP5esp8iuhi8Yl
USE_HUB_MODEL=true
# OR
MODEL_PATH=models/yolov8-finetuned.pt

# API
API_HOST=0.0.0.0
API_PORT=8000

# Ultralytics Hub
ULTRALYTICS_API_KEY=your_api_key
```

---

## 🔧 Production Checklist

- [ ] Set up PostgreSQL database
- [ ] Configure environment variables
- [ ] Initialize database schema
- [ ] Set up model files or Hub connection
- [ ] Configure CORS (if needed)
- [ ] Set up SSL/TLS certificates
- [ ] Configure logging
- [ ] Set up monitoring
- [ ] Configure backups
- [ ] Test all endpoints
- [ ] Test dashboard functionality

---

## 📊 Health Checks

### API Health
```bash
curl http://localhost:8000/health
```

### Database Connection
```bash
python backend/init_database.py
```

### Model Loading
```bash
curl -X POST http://localhost:8000/api/detect \
  -F "file=@data/custom/test/images/kanops_011.jpg"
```

---

## 🐛 Troubleshooting

### Database Connection Issues
- Check `DATABASE_URL` format
- Verify PostgreSQL is running
- Check firewall/network settings

### Model Loading Issues
- Verify `HUB_MODEL_URL` is correct
- Check `ULTRALYTICS_API_KEY` is set
- Ensure model files exist (if using local model)

### Port Conflicts
- Change ports in `docker-compose.yml`
- Update `API_PORT` and `DASHBOARD_PORT` in `.env`
