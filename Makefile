.PHONY: help install setup run-api run-dashboard test clean

help:
	@echo "VisionStock - AI Inventory Management"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup       - Setup database and environment"
	@echo "  make run-api     - Start FastAPI server"
	@echo "  make run-dashboard - Start Streamlit dashboard"
	@echo "  make test        - Run tests"
	@echo "  make clean       - Clean temporary files"

install:
	pip install -r requirements.txt

setup:
	python backend/init_database.py

run-api:
	cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard/app.py --server.port 8501

test:
	python -m pytest tests/

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
