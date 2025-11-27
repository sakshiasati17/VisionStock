# Railway.app Deployment Guide

This guide will help you deploy VisionStock to Railway.app, a modern platform that makes deployment simple.

## 🚀 Quick Start

### Step 1: Create Railway Account
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub (recommended) or email
3. Verify your account

### Step 2: Create New Project
1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose your repository: `sakshiasati17/VisionStock`
4. Railway will automatically detect the project

### Step 3: Add PostgreSQL Database
1. In your Railway project, click **"+ New"**
2. Select **"Database"** → **"Add PostgreSQL"**
3. Railway will create a PostgreSQL instance
4. Copy the connection string (you'll need it later)

### Step 4: Deploy Backend Service
1. Click **"+ New"** → **"GitHub Repo"**
2. Select `sakshiasati17/VisionStock`
3. Railway will auto-detect the Dockerfile
4. Configure the service:
   - **Name**: `visionstock-backend`
   - **Root Directory**: `/` (default)
   - **Dockerfile Path**: `backend/Dockerfile`

### Step 5: Configure Backend Environment Variables
In the backend service settings, add these environment variables:

```env
DATABASE_URL=<PostgreSQL connection string from Step 3>
MODEL_PATH=https://hub.ultralytics.com/models/jfHGXJxP5esp8iuhi8Yl
USE_HUB_MODEL=true
API_HOST=0.0.0.0
API_PORT=$PORT
PYTHONPATH=/app
PYTHONUNBUFFERED=1
```

**Important**: Railway automatically sets `$PORT`, so use that instead of hardcoding `8000`.

### Step 6: Deploy Dashboard Service
1. Click **"+ New"** → **"GitHub Repo"**
2. Select `sakshiasati17/VisionStock` again
3. Configure:
   - **Name**: `visionstock-dashboard`
   - **Root Directory**: `/`
   - **Dockerfile Path**: `dashboard/Dockerfile`

### Step 7: Configure Dashboard Environment Variables
Add this environment variable:

```env
API_BASE_URL=<Your backend service URL>
```

To find your backend URL:
1. Go to backend service → **Settings** → **Networking**
2. Click **"Generate Domain"**
3. Copy the URL (e.g., `https://visionstock-backend-production.up.railway.app`)
4. Use this as `API_BASE_URL` (without `/api` suffix)

### Step 8: Generate Public URLs
1. For **Backend**: Settings → Networking → **"Generate Domain"**
2. For **Dashboard**: Settings → Networking → **"Generate Domain"**

### Step 9: Initialize Database
Once backend is deployed, initialize the database:

1. Get your backend service URL
2. Visit: `https://your-backend-url.railway.app/api/init-db`
3. Or use Railway's CLI:
   ```bash
   railway run python backend/init_database.py
   ```

## 📋 Environment Variables Summary

### Backend Service
| Variable | Value | Description |
|----------|-------|-------------|
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection string |
| `MODEL_PATH` | `https://hub.ultralytics.com/models/jfHGXJxP5esp8iuhi8Yl` | Ultralytics Hub model URL |
| `USE_HUB_MODEL` | `true` | Use Hub model instead of local |
| `API_HOST` | `0.0.0.0` | Bind to all interfaces |
| `API_PORT` | `$PORT` | Railway's dynamic port |
| `PYTHONPATH` | `/app` | Python path |
| `PYTHONUNBUFFERED` | `1` | Unbuffered output |

### Dashboard Service
| Variable | Value | Description |
|----------|-------|-------------|
| `API_BASE_URL` | `https://your-backend-url.railway.app` | Backend API URL |

## 🔧 Railway CLI (Optional)

For advanced management, install Railway CLI:

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Link to your project
railway link

# Deploy
railway up

# View logs
railway logs

# Run commands
railway run python backend/init_database.py
```

## 🎯 Post-Deployment Checklist

- [ ] Backend service is running
- [ ] Dashboard service is running
- [ ] PostgreSQL database is connected
- [ ] Database tables are initialized
- [ ] Backend health check: `https://your-backend-url.railway.app/health`
- [ ] Dashboard is accessible: `https://your-dashboard-url.railway.app`
- [ ] API endpoints are working

## 🐛 Troubleshooting

### Backend won't start
- Check logs: Railway Dashboard → Service → **Logs**
- Verify `DATABASE_URL` is correct
- Ensure `$PORT` is used (Railway sets this automatically)

### Database connection errors
- Verify PostgreSQL service is running
- Check `DATABASE_URL` format: `postgresql://user:pass@host:port/dbname`
- Ensure database is initialized: visit `/api/init-db`

### Dashboard can't connect to backend
- Verify `API_BASE_URL` is correct (no trailing slash)
- Check backend service is running
- Ensure backend URL is accessible

### Model loading issues
- Hub model downloads automatically on first request
- Check internet connectivity in Railway logs
- Verify `USE_HUB_MODEL=true` is set

## 💰 Railway Pricing

**Free Tier Includes:**
- $5 credit/month
- 500 hours of usage
- Perfect for development/testing

**Paid Plans:**
- Developer: $5/month + usage
- Team: $20/month + usage

## 📚 Additional Resources

- [Railway Documentation](https://docs.railway.app)
- [Railway Discord](https://discord.gg/railway)
- [Project Repository](https://github.com/sakshiasati17/VisionStock)

## ✅ Success!

Once deployed, you'll have:
- **Backend API**: `https://your-backend-url.railway.app`
- **Dashboard**: `https://your-dashboard-url.railway.app`
- **Database**: Managed PostgreSQL instance

Your VisionStock application is now live! 🎉

