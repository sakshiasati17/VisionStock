#!/bin/bash

# Cloud Deployment Script (Heroku, Railway, Render, etc.)

set -e

echo "================================================================================
☁️  VisionStock Cloud Deployment Guide
================================================================================"

echo "
📋 DEPLOYMENT OPTIONS:

1. 🚂 Railway.app
   - Connect GitHub repo
   - Add PostgreSQL service
   - Set environment variables
   - Deploy automatically

2. 🎯 Render.com
   - Create Web Service
   - Connect GitHub repo
   - Add PostgreSQL database
   - Set environment variables
   - Deploy

3. 🟣 Heroku
   - heroku create visionstock
   - heroku addons:create heroku-postgresql
   - git push heroku main

4. 🐳 Docker Hub
   - docker build -t visionstock/backend ./backend
   - docker push visionstock/backend
   - Deploy to any container platform

📝 ENVIRONMENT VARIABLES NEEDED:
   - DATABASE_URL
   - MODEL_PATH or HUB_MODEL_URL
   - USE_HUB_MODEL
   - ULTRALYTICS_API_KEY

See docs/DEPLOYMENT.md for detailed instructions.
"

