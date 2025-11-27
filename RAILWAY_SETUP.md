# 🚂 Railway.app Deployment - Quick Setup

## Step-by-Step Guide

### 1️⃣ Sign Up & Create Project
1. Go to [railway.app](https://railway.app) and sign up with GitHub
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose: `sakshiasati17/VisionStock`
5. Railway will auto-detect your project

### 2️⃣ Add PostgreSQL Database
1. In your project, click **"+ New"**
2. Select **"Database"** → **"Add PostgreSQL"**
3. Wait for it to provision
4. Go to **Settings** → Copy the **Connection String** (you'll need this)

### 3️⃣ Deploy Backend Service
1. Click **"+ New"** → **"GitHub Repo"**
2. Select `sakshiasati17/VisionStock`
3. Railway will show a service - name it `backend`
4. Go to **Settings** → **Build & Deploy**:
   - **Root Directory**: `/` (default)
   - **Dockerfile Path**: `backend/Dockerfile`

### 4️⃣ Configure Backend Environment Variables
In backend service → **Variables** tab, add:

```
DATABASE_URL=<paste PostgreSQL connection string>
MODEL_PATH=https://hub.ultralytics.com/models/jfHGXJxP5esp8iuhi8Yl
USE_HUB_MODEL=true
API_HOST=0.0.0.0
PYTHONPATH=/app
PYTHONUNBUFFERED=1
```

**Note**: Railway automatically sets `PORT`, so don't set `API_PORT`.

### 5️⃣ Generate Backend Public URL
1. Backend service → **Settings** → **Networking**
2. Click **"Generate Domain"**
3. Copy the URL (e.g., `https://backend-production-xxxx.up.railway.app`)

### 6️⃣ Deploy Dashboard Service
1. Click **"+ New"** → **"GitHub Repo"**
2. Select `sakshiasati17/VisionStock` again
3. Name it `dashboard`
4. **Settings** → **Build & Deploy**:
   - **Root Directory**: `/`
   - **Dockerfile Path**: `dashboard/Dockerfile`

### 7️⃣ Configure Dashboard Environment Variables
In dashboard service → **Variables** tab, add:

```
API_BASE_URL=<your backend URL from step 5>
```

**Important**: Use the full URL without `/api` (e.g., `https://backend-production-xxxx.up.railway.app`)

### 8️⃣ Generate Dashboard Public URL
1. Dashboard service → **Settings** → **Networking**
2. Click **"Generate Domain"**
3. Copy the URL

### 9️⃣ Initialize Database
Once backend is deployed:

**Option A: Via Browser**
- Visit: `https://your-backend-url.railway.app/api/init-db`

**Option B: Via Railway CLI**
```bash
railway run python backend/init_database.py
```

### ✅ Verify Deployment
1. **Backend Health**: `https://your-backend-url.railway.app/health`
   - Should return: `{"status":"healthy","timestamp":"..."}`

2. **Dashboard**: `https://your-dashboard-url.railway.app`
   - Should show the Streamlit dashboard

3. **API Test**: 
   ```bash
   curl https://your-backend-url.railway.app/api/models
   ```

## 🎯 Your Live URLs
After deployment, you'll have:
- **Backend API**: `https://your-backend-url.railway.app`
- **Dashboard**: `https://your-dashboard-url.railway.app`
- **API Docs**: `https://your-backend-url.railway.app/docs`

## 💡 Pro Tips
- Railway auto-deploys on every GitHub push
- Check logs in Railway dashboard if something fails
- Free tier gives $5/month credit (plenty for testing)
- Database backups are automatic

## 🐛 Troubleshooting
- **Backend won't start**: Check logs, verify `DATABASE_URL` is correct
- **Dashboard can't connect**: Verify `API_BASE_URL` matches backend URL exactly
- **Database errors**: Make sure you initialized the DB (step 9)

## 📚 Full Documentation
See `docs/RAILWAY_DEPLOYMENT.md` for detailed instructions.

---

**Ready to deploy?** Follow the steps above! 🚀

