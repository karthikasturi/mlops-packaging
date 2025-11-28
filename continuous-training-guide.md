# Continuous Training Guide

This guide demonstrates how to implement continuous training based on model drift detection.

## Overview

The continuous training system consists of three main components:

1. **Drift Detection** (`detect_drift.py`) - Monitors for data and performance drift
2. **Continuous Training** (`continuous_training.py`) - Retrains models when needed
3. **Automated Monitor** (`monitor.py`) - Combines detection and retraining

## Components

### 1. Drift Detection (`detect_drift.py`)

Detects two types of drift:

- **Data Drift**: Changes in input feature distributions
- **Performance Drift**: Degradation in model accuracy

```bash
# Run drift detection
python src/detect_drift.py
```

**Output:**
- Checks for statistical changes in feature distributions
- Measures current model accuracy vs baseline
- Generates drift report
- Recommends retraining if drift detected

### 2. Continuous Training (`continuous_training.py`)

Handles automated model retraining:

```bash
# Run continuous training
python src/continuous_training.py

# Force retrain (skip model comparison)
python src/continuous_training.py --force
```

**Features:**
- Backs up current model before retraining
- Trains new model on latest data
- Compares new vs old model performance
- Deploys new model only if it performs better
- Maintains training history

### 3. Automated Monitor (`monitor.py`)

Combines drift detection and retraining in one workflow:

```bash
# Run full monitoring and retraining pipeline
python src/monitor.py
```

**Workflow:**
1. Detects drift in data and performance
2. If drift detected → triggers automatic retraining
3. Compares models and deploys if better
4. Generates reports and logs

## Usage Examples

### Example 1: Manual Drift Check

```bash
cd src

# Check for model drift
python detect_drift.py
```

If drift is detected, you'll see:
```
⚠️  DATA DRIFT DETECTED!
⚠️  PERFORMANCE DRIFT DETECTED!
🔄 RETRAINING RECOMMENDED
```

### Example 2: Manual Retraining

```bash
# Retrain the model manually
python continuous_training.py
```

The pipeline will:
- Backup your current model to `model_backups/`
- Train a new model
- Compare performance
- Deploy if better

### Example 3: Automated Monitoring

```bash
# Run complete monitoring pipeline
python monitor.py
```

This combines drift detection and retraining automatically.

## Scheduling Continuous Training

### Using Cron (Linux/Mac)

Edit crontab:
```bash
crontab -e
```

Add entry to run daily at 2 AM:
```bash
0 2 * * * cd /path/to/MLOps/src && /path/to/venv/bin/python monitor.py >> /var/log/mlops-monitor.log 2>&1
```

Run every 6 hours:
```bash
0 */6 * * * cd /path/to/MLOps/src && /path/to/venv/bin/python monitor.py >> /var/log/mlops-monitor.log 2>&1
```

### Using Python Script (Cross-platform)

Create `scheduler.py`:
```python
import schedule
import time
import subprocess

def run_monitoring():
    print("Running model monitoring...")
    subprocess.run(['python', 'src/monitor.py'])

# Schedule to run every day at 2 AM
schedule.every().day.at("02:00").do(run_monitoring)

# Or run every 6 hours
schedule.every(6).hours.do(run_monitoring)

while True:
    schedule.run_pending()
    time.sleep(60)
```

Run the scheduler:
```bash
pip install schedule
python scheduler.py
```

## Model Backup and Versioning

All model backups are stored in `model/model_backups/`:

```bash
# List all model backups
ls -lh model/model_backups/

# Restore a specific backup
cp model/model_backups/model_backup_20231127_143022.pkl model/model.pkl
```

## Drift Detection Thresholds

You can adjust drift detection sensitivity in `detect_drift.py`:

```python
# Performance drift threshold (default: 5% accuracy drop)
def check_model_performance_drift(model, X_test, y_test, baseline_accuracy, threshold=0.05):
    # Lower threshold = more sensitive to performance changes
    # threshold=0.03 means retrain if accuracy drops by 3%
```

```python
# Data drift threshold (default: 2 standard deviations)
if abs(new_mean - train_mean) > 2 * train_std:
    # Lower multiplier = more sensitive to data changes
    # Use 1.5 * train_std for higher sensitivity
```

## Integration with Production

### Step 1: Deploy with Docker

After retraining, rebuild and deploy:

```bash
# Rebuild Docker image with new model
docker build -t mlops-app:v2 .

# Run updated container
docker run -d -p 5000:5000 mlops-app:v2
```

### Step 2: Kubernetes Deployment

Update deployment with new model:

```bash
# Build and tag new version
docker build -t username/mlops-app:v2 .
docker push username/mlops-app:v2

# Update Kubernetes deployment
kubectl set image deployment/mlops-app mlops-container=username/mlops-app:v2

# Verify rollout
kubectl rollout status deployment/mlops-app
```

### Step 3: CI/CD Integration

Add to `.github/workflows/retrain.yml`:

```yaml
name: Continuous Training

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:  # Manual trigger

jobs:
  monitor-and-retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run monitoring
        run: |
          cd src
          python monitor.py
      
      - name: Build and push if retrained
        if: success()
        run: |
          docker build -t ${{ secrets.DOCKER_USERNAME }}/mlops-app:latest .
          docker push ${{ secrets.DOCKER_USERNAME }}/mlops-app:latest
```

## Monitoring Metrics

Key metrics to track:

1. **Drift Frequency**: How often drift is detected
2. **Retraining Success Rate**: Percentage of successful retrains
3. **Model Accuracy Trend**: Track accuracy over time
4. **Training Time**: Monitor retraining duration
5. **Data Volume**: Track incoming data volume

## Best Practices

1. **Always backup models** before retraining
2. **Set appropriate thresholds** based on your use case
3. **Monitor logs** for retraining activities
4. **Test new models** before production deployment
5. **Keep training history** for audit and debugging
6. **Version your models** with timestamps
7. **Alert on failures** in the retraining pipeline
8. **Gradually roll out** new models (canary/blue-green)

## Troubleshooting

### Issue: Drift detected too frequently

Solution: Increase drift thresholds in `detect_drift.py`

### Issue: Model performance not improving

Solution: Check if you need:
- More training data
- Feature engineering
- Different model architecture
- Hyperparameter tuning

### Issue: Retraining takes too long

Solution:
- Reduce dataset size (use sampling)
- Optimize model complexity
- Use incremental learning
- Schedule during off-peak hours

## Next Steps

1. Implement A/B testing for new models
2. Add monitoring dashboard (Grafana, Prometheus)
3. Integrate with MLflow for experiment tracking
4. Set up alerting (email, Slack) for drift detection
5. Implement feature store for consistent features
6. Add data validation checks
7. Create rollback mechanism for failed deployments
