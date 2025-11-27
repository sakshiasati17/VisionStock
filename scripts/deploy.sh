#!/bin/bash

# VisionStock Deployment Script

set -e

echo "================================================================================
🚀 VisionStock Deployment Script
================================================================================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo -e "${BLUE}📦 Building Docker images...${NC}"
docker-compose build

echo -e "${BLUE}🗄️  Starting PostgreSQL database...${NC}"
docker-compose up -d postgres

echo -e "${BLUE}⏳ Waiting for database to be ready...${NC}"
sleep 10

echo -e "${BLUE}🔧 Initializing database...${NC}"
docker-compose run --rm backend python backend/init_database.py

echo -e "${BLUE}🚀 Starting all services...${NC}"
docker-compose up -d

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""
echo "Services are running:"
echo "  📊 Dashboard: http://localhost:8501"
echo "  🔌 API: http://localhost:8000"
echo "  📚 API Docs: http://localhost:8000/docs"
echo "  🗄️  Database: localhost:5432"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"

