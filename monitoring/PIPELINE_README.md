# MLOps Complete Pipeline - House Rental Prediction

This repository demonstrates a complete MLOps pipeline for a house rental price prediction model, covering data ingestion, model training, experiment tracking, deployment, and monitoring.

## 🎯 Overview

The pipeline includes:

1. **Data Ingestion & Preparation** - MLflow Tracking for data versioning
2. **Model Training & Experiments** - Hyperparameter tracking and model comparison
3. **Model Registry** - Version control and stage management
4. **Deployment** - Containerization and serving endpoints
5. **Monitoring** - Prometheus metrics and health checks

## 📁 Project Structure

```
MLOps/monitoring/
├── data/                           # Versioned datasets
│   ├── house_rental_raw.csv
│   ├── house_rental_train_processed.csv
│   ├── house_rental_test_processed.csv
│   └── *_metadata.json
├── models/                         # Saved model artifacts
│   ├── house_rental_best_model.pkl
│   └── house_rental_best_model_metadata.json
├── deployment/                     # Deployment configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── prometheus.yml
│   ├── monitored_server.py
│   ├── health_check.py
│   ├── load_test.py
│   └── README.md
├── data_ingestion.py              # Data versioning pipeline
├── train_model.py                 # Model training with tracking
├── query_data_versions.py         # Data version management
├── detect_data_drift.py           # Drift detection
├── deploy_model.py                # Deployment setup
└── run_example.py                 # Complete pipeline demo
```

## 🚀 Quick Start

### 1. Data Ingestion & Versioning

```bash
python data_ingestion.py
```

**What it does:**
- Generates synthetic house rental data
- Extracts features and preprocesses data
- Logs all parameters and metadata to MLflow
- Creates versioned datasets with hashes
- Tracks preprocessing transformations

**MLflow Artifacts:**
- Raw data
- Processed train/test datasets
- Preprocessing parameters
- Data statistics and metadata

### 2. Model Training & Experiments

```bash
python train_model.py
```

**What it does:**
- Trains multiple models (Random Forest, Gradient Boosting, Ridge)
- Tests different hyperparameter configurations
- Logs all experiments to MLflow
- Registers best model to Model Registry
- Saves model as pickle file for deployment

**Models Trained:**
- Random Forest (100 trees, depth=10)
- Random Forest (200 trees, depth=15)
- Gradient Boosting (100 trees, lr=0.1)
- Gradient Boosting (150 trees, lr=0.05)
- Ridge Regression (alpha=1.0)

**MLflow Tracking:**
- Hyperparameters
- Training/test metrics (RMSE, MAE, R², MAPE)
- Model artifacts
- Feature importance
- Model signatures

### 3. Query Data Versions

```bash
python query_data_versions.py
```

**What it does:**
- Lists all data ingestion runs
- Compares different data versions
- Shows preprocessing strategies
- Loads specific data versions

### 4. Model Deployment Setup

```bash
python deploy_model.py
```

**What it does:**
- Creates deployment configurations
- Generates Dockerfile and docker-compose.yml
- Sets up Prometheus monitoring
- Creates health check and load test scripts
- Prepares sample inputs

### 5. Deploy Model

**Option A: Local Deployment**
```bash
cd deployment
./deploy.sh
```

**Option B: Docker Deployment**
```bash
cd deployment
docker-compose up -d
```

### 6. Test Deployment

```bash
# Health check
python deployment/health_check.py

# Load test
python deployment/load_test.py 100 10
```

## 📊 MLflow Tracking

### Experiments

1. **house-rental-data-ingestion**
   - Data versioning
   - Preprocessing parameters
   - Dataset statistics

2. **house-rental-model-training**
   - Model experiments
   - Hyperparameter tuning
   - Performance metrics

### Model Registry

**Registered Models:**
- `HouseRentalPredictor` - Best performing model in Production stage

### Accessing MLflow UI

```bash
# MLflow UI is running at:
http://localhost:5001
```

**Navigate to:**
- **Experiments** - View all training runs
- **Models** - Access Model Registry
- **Compare Runs** - Side-by-side comparison

## 📈 Monitoring

### Prometheus Metrics

After deploying with Docker Compose:

```bash
# Prometheus UI
http://localhost:9090

# Model metrics endpoint
http://localhost:5002/metrics
```

### Available Metrics

1. **Request Metrics**
   - `model_requests_total` - Total requests
   - `model_request_duration_seconds` - Latency histogram
   - `model_server_active_requests` - Active requests

2. **Prediction Metrics**
   - `model_predictions_total` - Total predictions

3. **Error Metrics**
   - `model_errors_total` - Error count by type

4. **System Metrics**
   - `model_server_cpu_usage_percent` - CPU usage
   - `model_server_memory_usage_mb` - Memory usage

### Prometheus Queries

```promql
# Request rate (requests/second)
rate(model_requests_total[5m])

# Average latency (milliseconds)
rate(model_request_duration_seconds_sum[5m]) / rate(model_request_duration_seconds_count[5m]) * 1000

# P95 latency
histogram_quantile(0.95, rate(model_request_duration_seconds_bucket[5m]))

# P99 latency
histogram_quantile(0.99, rate(model_request_duration_seconds_bucket[5m]))

# Error rate
rate(model_errors_total[5m]) / rate(model_requests_total[5m])

# Throughput (predictions/minute)
rate(model_predictions_total[1m]) * 60
```

## 🧪 API Usage

### Prediction Endpoint

```bash
curl -X POST http://localhost:5002/invocations \
     -H 'Content-Type: application/json' \
     -d '{
       "dataframe_split": {
         "columns": ["area_sqft", "bedrooms", "bathrooms", "parking", "age_years", 
                     "floor", "has_gym", "has_pool", "price_per_sqft", "room_bath_ratio",
                     "total_rooms", "amenities_score", "is_new", "is_spacious"],
         "data": [[1500.0, 2.0, 1.5, 1.0, 5.0, 3.0, 1.0, 0.0, 1.2, 1.33, 3.5, 2.0, 1.0, 1.0]]
       }
     }'
```

### Python Client

```python
import requests
import json

# Prepare input
data = {
    "dataframe_split": {
        "columns": ["area_sqft", "bedrooms", "bathrooms", "parking", "age_years",
                    "floor", "has_gym", "has_pool", "price_per_sqft", "room_bath_ratio",
                    "total_rooms", "amenities_score", "is_new", "is_spacious"],
        "data": [[1500.0, 2.0, 1.5, 1.0, 5.0, 3.0, 1.0, 0.0, 1.2, 1.33, 3.5, 2.0, 1.0, 1.0]]
    }
}

# Make prediction
response = requests.post(
    'http://localhost:5002/invocations',
    json=data,
    headers={'Content-Type': 'application/json'}
)

print('Predicted Rent:', response.json()['predictions'][0])
```

## 🏗️ Complete Pipeline Workflow

```
┌─────────────────────┐
│  Data Generation    │
│  & Preprocessing    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  MLflow Tracking    │◄─── Data Versioning
│  (Data Ingestion)   │     Preprocessing Params
└──────────┬──────────┘     Dataset Metadata
           │
           ▼
┌─────────────────────┐
│  Model Training     │
│  & Experiments      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  MLflow Tracking    │◄─── Hyperparameters
│  (Experiments)      │     Metrics (RMSE, R²)
└──────────┬──────────┘     Model Artifacts
           │
           ▼
┌─────────────────────┐
│  Model Registry     │◄─── Model Versions
│  (Best Model)       │     Stage Management
└──────────┬──────────┘     Model Metadata
           │
           ▼
┌─────────────────────┐
│  Model Deployment   │◄─── Docker Container
│  (Serving)          │     MLflow Models Serve
└──────────┬──────────┘     Load Balancing
           │
           ▼
┌─────────────────────┐
│  Monitoring         │◄─── Prometheus Metrics
│  (Prometheus)       │     Grafana Dashboards
└─────────────────────┘     Alerts & Logging
```

## 📋 Checklist

- [x] Data ingestion with versioning
- [x] Feature engineering and preprocessing
- [x] Multiple model training experiments
- [x] Hyperparameter tracking
- [x] Model comparison and selection
- [x] Model registry and versioning
- [x] Pickle file creation for deployment
- [x] Docker containerization
- [x] MLflow model serving
- [x] Prometheus monitoring
- [x] Health checks and load testing
- [x] API documentation
- [x] Deployment guides

## 🎓 Key Learnings

1. **Data Versioning**: Every dataset change is tracked with hashes and timestamps
2. **Experiment Tracking**: All hyperparameters and metrics logged automatically
3. **Model Registry**: Centralized model versioning with stage management
4. **Reproducibility**: Complete lineage from data to deployed model
5. **Monitoring**: Real-time insights into model performance and system health

## 📚 Next Steps

1. **Add Grafana**: Visual dashboards for metrics
2. **Implement Alerts**: Set up Prometheus alerting rules
3. **A/B Testing**: Deploy multiple model versions
4. **Model Retraining**: Automated retraining pipeline
5. **Data Drift Detection**: Continuous monitoring of input data
6. **CI/CD Integration**: Automate deployment pipeline

## 🔧 Troubleshooting

See [deployment/README.md](deployment/README.md) for detailed troubleshooting guide.

## 📞 Resources

- MLflow Documentation: https://mlflow.org/docs/latest/
- Prometheus Documentation: https://prometheus.io/docs/
- Docker Documentation: https://docs.docker.com/

---

**MLOps Pipeline** - Complete end-to-end machine learning operations
