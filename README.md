# Breast Cancer Prediction ML App

This project contains a machine learning model that predicts breast cancer diagnosis using scikit-learn's breast cancer dataset and a Flask API to serve predictions.

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model
```bash
cd src
python train.py
cd ..
```
This creates `model.pkl` in the src directory.

### 3. Build Docker Image
```bash
docker build -t mlops-app:v1 .
```

### 4. Run Docker Container
```bash
docker run -p 5000:5000 mlops-app:v1
```

The API will be available at `http://localhost:5000`

**Note:** Make sure to activate the virtual environment before training the model locally.

## API Endpoints

### GET /
Home endpoint with API information

### GET /health
Health check endpoint

### POST /predict
Make a prediction with 30 features

**Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
```

**Response:**
```json
{
  "prediction": 0,
  "diagnosis": "malignant",
  "probability": {
    "malignant": 0.92,
    "benign": 0.08
  }
}
```

## Testing with curl

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
```

## Dataset

The breast cancer dataset contains 30 features computed from digitized images of fine needle aspirate (FNA) of breast masses. The model predicts whether a tumor is malignant (0) or benign (1).

---

# MLOps Deployment Guide

## Prerequisites
- Docker installed
- Kubernetes cluster running (minikube, kind, or cloud provider)
- kubectl configured

## Step 1: Build the Docker Container

```bash
# Navigate to your application directory
cd /home/karthikeyan/training-temp/MLOps

# Build the Docker image
docker build -t mlops-app:v1 .

# Verify the image was created
docker images | grep mlops-app
```

## Step 2: Push to Container Registry (Optional)

If using a remote Kubernetes cluster:

```bash
# Tag the image for your registry
docker tag mlops-app:v1 <your-registry>/mlops-app:v1

# Login to your registry
docker login <your-registry>

# Push the image
docker push <your-registry>/mlops-app:v1
```