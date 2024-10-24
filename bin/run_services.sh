#!/bin/bash

# Run Prefect server
echo "Starting Prefect server..."
prefect server start --host 0.0.0.0 --port 4201 &

# Give Prefect server some time to start (optional)
sleep 10

# Run FastAPI service
echo "Starting FastAPI service..."
uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
