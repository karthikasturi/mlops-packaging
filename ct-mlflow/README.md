# Continuous Training with MLflow

This example demonstrates continuous training with MLflow experiment tracking on a local environment.

## Prerequisites

- Python 3.9+
- MLflow server already running on port 5001
- Virtual environment with scikit-learn installed

## Setup

### Verify MLflow Server

Ensure MLflow tracking server is running on port 5001:

```bash
# Check if MLflow is accessible
curl http://localhost:5001

# Or visit in browser
# http://localhost:5001
```

The MLflow UI should be available at: `http://localhost:5001`

## Running Continuous Training

### Basic Usage

```bash
cd ct-mlflow
python mlflow_continuous_training.py
```

### What It Does

1. **Data Drift Detection**: Checks if input feature distributions have changed
2. **Performance Drift Detection**: Monitors model accuracy degradation
3. **Automated Retraining**: Triggers retraining if drift is detected
4. **Model Comparison**: Compares new model with current production model
5. **MLflow Tracking**: Logs all experiments, metrics, and models
6. **Model Registry**: Registers and versions models in MLflow

### Output Example

```
==================================================================
CONTINUOUS TRAINING WITH MLFLOW TRACKING
==================================================================

[STEP 1] Loading breast cancer dataset...
   Training samples: 455
   Test samples: 114

[STEP 2] Loading current production model...
   Current model loaded successfully
   Baseline accuracy: 0.9737

[STEP 3] Detecting data drift...
   Average drift score: 1.2345
   ⚠️  DATA DRIFT DETECTED!

[STEP 4] Detecting performance drift...
   Current accuracy: 0.9649
   ⚠️  PERFORMANCE DRIFT DETECTED!

[DECISION] 🔄 RETRAINING INITIATED

[STEP 5] Training new model...
   ✓ Model training completed

[STEP 6] Evaluating new model...
   Accuracy:  0.9825
   Precision: 0.9828
   Recall:    0.9825
   F1 Score:  0.9825

[STEP 7] Model comparison and deployment...
   ✓ New model performs better or equal
   ✓ Model saved to ../model/model.pkl
   ✓ Model logged to MLflow
   ✓ Model registered: breast_cancer_model v2

   Accuracy improvement: 0.88%

==================================================================
✓ CONTINUOUS TRAINING COMPLETED - NEW MODEL DEPLOYED
==================================================================
```

## Compare Models

View and compare all training runs:

```bash
python compare_models.py
```

This will show:
- All training runs with their metrics
- Best performing model
- Registered model versions

## MLflow UI Features

Access the MLflow UI at `http://localhost:5001`:

### 1. Experiments
- View all training runs
- Compare metrics across runs
- Filter and sort by performance

### 2. Model Registry
- Browse registered models
- View model versions
- Track model lineage

### 3. Metrics Visualization
- Accuracy trends over time
- Performance comparisons
- Drift scores

### 4. Parameters
- Training hyperparameters
- Data statistics
- Drift detection settings

### 5. Artifacts
- Trained models
- Training plots
- Model metadata

## Tracked Metrics

The following metrics are logged to MLflow:

**Performance Metrics:**
- `accuracy` - Model accuracy on test set
- `precision` - Weighted precision score
- `recall` - Weighted recall score
- `f1_score` - Weighted F1 score

**Drift Metrics:**
- `data_drift_score` - Average feature drift score
- `baseline_accuracy` - Previous model accuracy
- `current_accuracy` - Current model accuracy
- `accuracy_improvement` - Improvement over baseline

**Parameters:**
- `n_samples_train` - Training set size
- `n_samples_test` - Test set size
- `n_features` - Number of features
- `n_estimators` - Random Forest trees
- `max_depth` - Maximum tree depth
- `data_drift_detected` - Boolean flag
- `performance_drift_detected` - Boolean flag
- `deployment_status` - deployed/rejected

## Scheduling Continuous Training

### Using Cron

Add to crontab for daily runs at 2 AM:

```bash
0 2 * * * cd /path/to/MLOps/ct-mlflow && /path/to/venv/bin/python mlflow_continuous_training.py >> ct_mlflow.log 2>&1
```

### Using Python Scheduler

```python
import schedule
import time
import subprocess

def run_training():
    subprocess.run(['python', 'ct-mlflow/mlflow_continuous_training.py'])

# Run daily at 2 AM
schedule.every().day.at("02:00").do(run_training)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Model Versioning

Models are automatically versioned in MLflow:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5001")

# Get latest model version
latest_versions = client.get_latest_versions("breast_cancer_model")

# Load specific version
import mlflow.sklearn
model = mlflow.sklearn.load_model(f"models:/breast_cancer_model/1")
```

## Production Deployment

### Load Model from MLflow

```python
import mlflow.sklearn

# Load production model
model = mlflow.sklearn.load_model("models:/breast_cancer_model/Production")

# Make predictions
predictions = model.predict(X_new)
```

### Promote Model to Production

```bash
# Via MLflow CLI
mlflow models serve -m "models:/breast_cancer_model/2" -p 5002

# Or promote in UI
# Navigate to Models > breast_cancer_model > Version 2 > Transition to Production
```

## Directory Structure

```
ct-mlflow/
├── mlflow_continuous_training.py  # Main training pipeline
├── compare_models.py              # Model comparison tool
├── requirements-mlflow.txt        # MLflow dependencies
└── README.md                      # This file

mlruns/                            # MLflow tracking data
└── <experiment_id>/
    └── <run_id>/
        ├── artifacts/
        ├── metrics/
        ├── params/
        └── tags/
```

## Troubleshooting

### MLflow Server Not Running

Error: `Connection refused`

Solution: Ensure your MLflow server is running and accessible at `http://localhost:5001`

### Connection Issues

Error: `Cannot connect to MLflow tracking server`

Solution:
```bash
# Verify MLflow server is accessible
curl http://localhost:5001

# Check if port 5001 is open
netstat -an | grep 5001
```

### Model Registry Not Working

Error: `No registered model found`

Solution: Ensure your MLflow server was started with a backend store for model registry support

## Best Practices

1. **Regular Monitoring**: Run drift detection periodically
2. **Baseline Tracking**: Always maintain baseline metrics
3. **Model Comparison**: Compare new models before deployment
4. **Version Control**: Use MLflow's model versioning
5. **Experiment Naming**: Use descriptive experiment names
6. **Parameter Logging**: Log all hyperparameters
7. **Artifact Storage**: Store training artifacts for reproducibility
8. **Model Staging**: Use staging environments before production
9. **Rollback Strategy**: Keep previous model versions for rollback
10. **Performance Monitoring**: Track metrics in production

## Advanced Features

### A/B Testing

```python
# Deploy multiple model versions
mlflow.sklearn.load_model("models:/breast_cancer_model/1")  # Version A
mlflow.sklearn.load_model("models:/breast_cancer_model/2")  # Version B
```

### Custom Metrics

```python
with mlflow.start_run():
    mlflow.log_metric("custom_metric", value)
    mlflow.log_param("custom_param", value)
```

### Model Tagging

```python
client = MlflowClient()
client.set_model_version_tag("breast_cancer_model", "1", "validation", "passed")
```

## Next Steps

1. Integrate with CI/CD pipeline
2. Add automated testing
3. Implement blue-green deployment
4. Set up monitoring alerts
5. Add data quality checks
6. Implement feature store
7. Add model explainability
8. Set up A/B testing framework
