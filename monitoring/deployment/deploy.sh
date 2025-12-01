#!/bin/bash
# Model Deployment Script

# Configuration
MODEL_URI="models:/HouseRentalPredictor/1"
PORT=5002
HOST="0.0.0.0"

echo "=========================================="
echo "Deploying House Rental Prediction Model"
echo "=========================================="
echo ""
echo "Model URI: $MODEL_URI"
echo "Host: $HOST:$PORT"
echo ""

# Start MLflow model server
echo "Starting MLflow model server..."
mlflow models serve \
    --model-uri "$MODEL_URI" \
    --host "$HOST" \
    --port "$PORT" \
    --no-conda \
    --env-manager local

echo ""
echo "Model server started successfully!"
echo "Test with: curl -X POST http://$HOST:$PORT/invocations -H 'Content-Type: application/json' -d @sample_input.json"
