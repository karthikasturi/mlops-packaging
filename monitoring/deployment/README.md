# Model Deployment Guide

This directory contains all necessary files for deploying the House Rental Prediction model with monitoring.

## 📦 Deployment Options

### Option 1: Local Deployment with MLflow

Start the model server locally:

```bash
# Using the deployment script
cd monitoring/deployment
./deploy.sh

# Or directly with MLflow
mlflow models serve \
    --model-uri models:/HouseRentalPredictor/1 \
    --host 0.0.0.0 \
    --port 5002 \
    --no-conda \
    --env-manager local
```

The server will be available at: http://localhost:5002

### Option 2: Docker Deployment

Deploy using Docker Compose (includes Prometheus monitoring):

```bash
cd monitoring/deployment
docker-compose up -d
```

This will start:
- **Model Server** on port 5002
- **Prometheus** on port 9090

To stop:
```bash
docker-compose down
```

## 🧪 Testing the Deployment

### 1. Health Check

```bash
python health_check.py
```

### 2. Single Prediction

Using curl:
```bash
curl -X POST http://localhost:5002/invocations \
     -H 'Content-Type: application/json' \
     -d @sample_input.json
```

Using Python:
```python
import requests
import json

with open('sample_input.json', 'r') as f:
    data = json.load(f)

response = requests.post(
    'http://localhost:5002/invocations',
    json=data,
    headers={'Content-Type': 'application/json'}
)

print('Prediction:', response.json())
```

### 3. Load Testing

Run load test with 100 requests and 10 concurrent workers:
```bash
python load_test.py 100 10
```

Custom load test:
```bash
python load_test.py <num_requests> <concurrency>
```

## 📊 Monitoring

### Prometheus Metrics

Access Prometheus dashboard:
- **URL**: http://localhost:9090
- **Model metrics endpoint**: http://localhost:5002/metrics

### Available Metrics

The model server exposes the following Prometheus metrics:

1. **Request Count**
   - Metric: `http_requests_total`
   - Description: Total number of HTTP requests

2. **Latency**
   - Metric: `http_request_duration_seconds`
   - Description: Request duration in seconds
   - Percentiles: p50, p95, p99

3. **Prediction Count**
   - Metric: `prediction_requests_total`
   - Description: Total number of predictions

4. **Error Rate**
   - Metric: `http_request_errors_total`
   - Description: Total number of failed requests

5. **System Metrics**
   - CPU usage
   - Memory usage
   - Disk I/O

### Prometheus Queries

Example queries for monitoring:

```promql
# Request rate (requests per second)
rate(http_requests_total[5m])

# Average latency
rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])

# P95 latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_request_errors_total[5m]) / rate(http_requests_total[5m])

# Throughput
sum(rate(prediction_requests_total[5m]))
```

## 🏗️ Architecture

```
┌─────────────────┐
│   Client App    │
└────────┬────────┘
         │ HTTP POST
         ▼
┌─────────────────┐
│  Model Server   │◄── MLflow Models
│  (Port 5002)    │
└────────┬────────┘
         │ Metrics
         ▼
┌─────────────────┐
│   Prometheus    │
│  (Port 9090)    │
└─────────────────┘
```

## 📋 Configuration Files

### `serving_config.json`
Configuration for the MLflow model server:
- Port: 5002
- Host: 0.0.0.0
- Workers: 2

### `docker-compose.yml`
Docker Compose configuration for:
- Model server container
- Prometheus monitoring
- Network and volume configuration

### `Dockerfile`
Container image for the model server:
- Base: Python 3.12 slim
- Includes MLflow and dependencies
- Exposes port 5002
- Health check enabled

### `prometheus.yml`
Prometheus scrape configuration:
- Scrape interval: 15s
- Target: model-server:5002

## 🔧 Troubleshooting

### Model Server Issues

**Server won't start:**
```bash
# Check if port is already in use
lsof -i :5002

# Check logs
docker-compose logs model-server
```

**Prediction errors:**
```bash
# Test with health check
python health_check.py

# Check model logs
docker-compose logs -f model-server
```

### Prometheus Issues

**Metrics not showing:**
```bash
# Check if metrics endpoint is accessible
curl http://localhost:5002/metrics

# Verify Prometheus targets
# Open http://localhost:9090/targets
```

## 🚀 Production Deployment

For production deployment, consider:

1. **Resource Limits**
   - Add resource constraints in docker-compose.yml
   - Configure appropriate worker count

2. **Security**
   - Enable authentication/authorization
   - Use HTTPS with SSL certificates
   - Implement rate limiting

3. **Scaling**
   - Use Kubernetes for orchestration
   - Implement horizontal pod autoscaling
   - Add load balancer

4. **Monitoring**
   - Set up Grafana dashboards
   - Configure alerting rules
   - Implement log aggregation

5. **CI/CD**
   - Automate model deployment pipeline
   - Implement blue-green deployment
   - Add automated testing

## 📚 API Documentation

### Prediction Endpoint

**URL**: `POST /invocations`

**Request Format**:
```json
{
  "dataframe_split": {
    "columns": ["area_sqft", "bedrooms", "bathrooms", "parking", ...],
    "data": [[1500.0, 2.0, 1.5, 1.0, ...]]
  }
}
```

**Response Format**:
```json
{
  "predictions": [2150.50]
}
```

### Health Endpoint

**URL**: `GET /health`

**Response**:
```json
{
  "status": "healthy"
}
```

### Metrics Endpoint

**URL**: `GET /metrics`

**Response**: Prometheus format metrics

## 📞 Support

For issues or questions:
1. Check the logs: `docker-compose logs`
2. Review health check output
3. Test with sample input
4. Verify Prometheus metrics

---

Generated by MLflow Model Deployment System
