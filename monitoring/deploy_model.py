"""
Model Deployment with MLflow Models
Containerization and Serving Endpoints

This module demonstrates:
1. Packaging models with Docker environments
2. Serving models using MLflow Models
3. Creating deployment configurations
4. Building Docker containers for models
5. Testing model serving endpoints
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import json
import subprocess
import requests
import time


class ModelDeployment:
    """
    Handles model deployment with MLflow
    """
    
    def __init__(self, tracking_uri="http://localhost:5001"):
        """
        Initialize deployment manager
        
        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        
    def get_best_model_uri(self, experiment_name="house-rental-model-training"):
        """
        Get the URI of the best performing model
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Model URI and run info
        """
        print("\n" + "=" * 70)
        print("FINDING BEST MODEL")
        print("=" * 70)
        
        experiment = self.client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            print(f"❌ Experiment '{experiment_name}' not found!")
            return None, None
        
        # Get best run based on test_r2
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_r2 DESC"],
            max_results=1
        )
        
        if not runs:
            print("❌ No runs found!")
            return None, None
        
        best_run = runs[0]
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        
        print(f"\n✓ Best Model Found:")
        print(f"  Run ID: {run_id[:8]}...")
        print(f"  Model Type: {best_run.data.params.get('model_type', 'N/A')}")
        print(f"  Test R²: {best_run.data.metrics.get('test_r2', 0):.4f}")
        print(f"  Test RMSE: ${best_run.data.metrics.get('test_rmse', 0):.2f}")
        print(f"  Model URI: {model_uri}")
        
        return model_uri, best_run
    
    def get_registered_model_uri(self, model_name="HouseRentalPredictor", stage="Production"):
        """
        Get URI for a registered model at a specific stage
        
        Args:
            model_name: Name of registered model
            stage: Model stage (Production, Staging, etc.)
            
        Returns:
            Model URI
        """
        print("\n" + "=" * 70)
        print(f"GETTING REGISTERED MODEL: {model_name}")
        print("=" * 70)
        
        try:
            # Get latest version in the specified stage
            model_versions = self.client.search_model_versions(
                filter_string=f"name='{model_name}'"
            )
            
            if not model_versions:
                print(f"❌ Model '{model_name}' not found in registry!")
                return None
            
            # Find version in the specified stage
            production_version = None
            for version in model_versions:
                # MLflow 2.9+ doesn't use stages, check current_stage or tags
                if hasattr(version, 'current_stage') and version.current_stage == stage:
                    production_version = version
                    break
            
            if not production_version:
                # If no stage found, use latest version
                production_version = model_versions[0]
                print(f"⚠️  No {stage} version found, using latest version")
            
            model_uri = f"models:/{model_name}/{production_version.version}"
            
            print(f"\n✓ Model Details:")
            print(f"  Name: {model_name}")
            print(f"  Version: {production_version.version}")
            print(f"  Run ID: {production_version.run_id[:8]}...")
            print(f"  URI: {model_uri}")
            
            return model_uri
            
        except Exception as e:
            print(f"❌ Error getting registered model: {str(e)}")
            return None
    
    def create_deployment_config(self, model_uri, output_path="monitoring/deployment"):
        """
        Create deployment configuration files
        
        Args:
            model_uri: MLflow model URI
            output_path: Directory to save configs
            
        Returns:
            Path to config directory
        """
        print("\n" + "=" * 70)
        print("CREATING DEPLOYMENT CONFIGURATION")
        print("=" * 70)
        
        os.makedirs(output_path, exist_ok=True)
        
        # 1. Create serving config
        serving_config = {
            "model_uri": model_uri,
            "port": 5002,
            "host": "0.0.0.0",
            "workers": 2,
            "enable_mlserver": False
        }
        
        serving_config_path = os.path.join(output_path, "serving_config.json")
        with open(serving_config_path, 'w') as f:
            json.dump(serving_config, f, indent=2)
        
        print(f"\n✓ Serving config: {serving_config_path}")
        
        # 2. Create Docker environment config
        docker_env = {
            "python": "3.12",
            "build_command": "pip install -r requirements.txt"
        }
        
        docker_env_path = os.path.join(output_path, "docker_env.json")
        with open(docker_env_path, 'w') as f:
            json.dump(docker_env, f, indent=2)
        
        print(f"✓ Docker env config: {docker_env_path}")
        
        # 3. Create deployment script
        deployment_script = f"""#!/bin/bash
# Model Deployment Script

# Configuration
MODEL_URI="{model_uri}"
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
mlflow models serve \\
    --model-uri "$MODEL_URI" \\
    --host "$HOST" \\
    --port "$PORT" \\
    --no-conda \\
    --env-manager local

echo ""
echo "Model server started successfully!"
echo "Test with: curl -X POST http://$HOST:$PORT/invocations -H 'Content-Type: application/json' -d @sample_input.json"
"""
        
        deployment_script_path = os.path.join(output_path, "deploy.sh")
        with open(deployment_script_path, 'w') as f:
            f.write(deployment_script)
        
        os.chmod(deployment_script_path, 0o755)
        print(f"✓ Deployment script: {deployment_script_path}")
        
        # 4. Create Dockerfile
        dockerfile_content = f"""# MLflow Model Deployment Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install MLflow and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts (will be mounted or copied)
ENV MODEL_URI="{model_uri}"
ENV MLFLOW_TRACKING_URI="{self.tracking_uri}"

# Expose port
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
    CMD curl -f http://localhost:5002/health || exit 1

# Run MLflow model server
CMD mlflow models serve \\
    --model-uri $MODEL_URI \\
    --host 0.0.0.0 \\
    --port 5002 \\
    --no-conda \\
    --env-manager local
"""
        
        dockerfile_path = os.path.join(output_path, "Dockerfile")
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        print(f"✓ Dockerfile: {dockerfile_path}")
        
        # 5. Create docker-compose.yml
        docker_compose = f"""version: '3.8'

services:
  model-server:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: house-rental-predictor
    ports:
      - "5002:5002"
    environment:
      - MODEL_URI={model_uri}
      - MLFLOW_TRACKING_URI={self.tracking_uri}
    networks:
      - ml-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - ml-network
    restart: unless-stopped

networks:
  ml-network:
    driver: bridge

volumes:
  prometheus-data:
"""
        
        docker_compose_path = os.path.join(output_path, "docker-compose.yml")
        with open(docker_compose_path, 'w') as f:
            f.write(docker_compose)
        
        print(f"✓ Docker Compose: {docker_compose_path}")
        
        # 6. Create Prometheus configuration
        prometheus_config = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mlflow-model'
    static_configs:
      - targets: ['model-server:5002']
    metrics_path: '/metrics'
"""
        
        prometheus_config_path = os.path.join(output_path, "prometheus.yml")
        with open(prometheus_config_path, 'w') as f:
            f.write(prometheus_config)
        
        print(f"✓ Prometheus config: {prometheus_config_path}")
        
        # 7. Create requirements.txt for deployment
        requirements = """mlflow==3.6.0
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.3
gunicorn==23.0.0
prometheus-client==0.21.0
"""
        
        requirements_path = os.path.join(output_path, "requirements.txt")
        with open(requirements_path, 'w') as f:
            f.write(requirements)
        
        print(f"✓ Requirements: {requirements_path}")
        
        # 8. Create sample input for testing
        sample_input = {
            "dataframe_split": {
                "columns": [
                    "area_sqft", "bedrooms", "bathrooms", "parking", "age_years",
                    "floor", "has_gym", "has_pool", "price_per_sqft", "room_bath_ratio",
                    "total_rooms", "amenities_score", "is_new", "is_spacious"
                ],
                "data": [
                    [1500.0, 2.0, 1.5, 1.0, 5.0, 3.0, 1.0, 0.0, 1.2, 1.33, 3.5, 2.0, 1.0, 1.0]
                ]
            }
        }
        
        sample_input_path = os.path.join(output_path, "sample_input.json")
        with open(sample_input_path, 'w') as f:
            json.dump(sample_input, f, indent=2)
        
        print(f"✓ Sample input: {sample_input_path}")
        
        print("\n" + "=" * 70)
        print("✅ DEPLOYMENT CONFIGURATION CREATED")
        print("=" * 70)
        
        return output_path
    
    def test_local_serving(self, model_uri, port=5002):
        """
        Test model serving locally
        
        Args:
            model_uri: MLflow model URI
            port: Port to serve on
        """
        print("\n" + "=" * 70)
        print("TESTING LOCAL MODEL SERVING")
        print("=" * 70)
        
        print(f"\nℹ️  To start the model server, run:")
        print(f"   mlflow models serve --model-uri {model_uri} --port {port} --no-conda --env-manager local")
        print(f"\n   Server will be available at: http://localhost:{port}")
        
        print(f"\nℹ️  To test the endpoint, use:")
        print(f"   curl -X POST http://localhost:{port}/invocations \\")
        print(f"        -H 'Content-Type: application/json' \\")
        print(f"        -d @monitoring/deployment/sample_input.json")
        
        print("\nℹ️  Or use Python:")
        print(f"""
import requests
import json

# Load sample input
with open('monitoring/deployment/sample_input.json', 'r') as f:
    data = json.load(f)

# Make prediction
response = requests.post(
    'http://localhost:{port}/invocations',
    json=data,
    headers={{'Content-Type': 'application/json'}}
)

print('Prediction:', response.json())
""")
    
    def create_monitoring_script(self, output_path="monitoring/deployment"):
        """
        Create monitoring and health check scripts
        
        Args:
            output_path: Directory to save scripts
        """
        print("\n" + "=" * 70)
        print("CREATING MONITORING SCRIPTS")
        print("=" * 70)
        
        # Health check script
        health_check_script = """#!/usr/bin/env python3
\"\"\"
Health Check Script for Model Server
\"\"\"

import requests
import time
import sys

def check_health(url="http://localhost:5002/health", timeout=5):
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ Model server is healthy")
            return True
        else:
            print(f"⚠️  Model server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to model server at {url}")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Health check timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def check_prediction(url="http://localhost:5002/invocations"):
    \"\"\"Test prediction endpoint\"\"\"
    sample_data = {
        "dataframe_split": {
            "columns": [
                "area_sqft", "bedrooms", "bathrooms", "parking", "age_years",
                "floor", "has_gym", "has_pool", "price_per_sqft", "room_bath_ratio",
                "total_rooms", "amenities_score", "is_new", "is_spacious"
            ],
            "data": [[1500.0, 2.0, 1.5, 1.0, 5.0, 3.0, 1.0, 0.0, 1.2, 1.33, 3.5, 2.0, 1.0, 1.0]]
        }
    }
    
    try:
        response = requests.post(
            url,
            json=sample_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            prediction = response.json()
            print(f"✅ Prediction successful: {prediction}")
            return True
        else:
            print(f"⚠️  Prediction failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Model Server Health Check\\n")
    
    # Check health endpoint
    health_ok = check_health()
    
    if health_ok:
        # Check prediction endpoint
        print("\\nTesting prediction endpoint...")
        pred_ok = check_prediction()
        
        if pred_ok:
            print("\\n✅ All checks passed!")
            sys.exit(0)
        else:
            print("\\n⚠️  Prediction check failed")
            sys.exit(1)
    else:
        print("\\n❌ Health check failed")
        sys.exit(1)
"""
        
        health_check_path = os.path.join(output_path, "health_check.py")
        with open(health_check_path, 'w') as f:
            f.write(health_check_script)
        
        os.chmod(health_check_path, 0o755)
        print(f"✓ Health check script: {health_check_path}")
        
        # Load testing script
        load_test_script = """#!/usr/bin/env python3
\"\"\"
Load Testing Script for Model Server
\"\"\"

import requests
import time
import concurrent.futures
import statistics
from datetime import datetime

def make_prediction(url, data, request_num):
    \"\"\"Make a single prediction request\"\"\"
    start_time = time.time()
    
    try:
        response = requests.post(
            url,
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            'request_num': request_num,
            'status_code': response.status_code,
            'latency_ms': latency,
            'success': response.status_code == 200,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'request_num': request_num,
            'status_code': 0,
            'latency_ms': (time.time() - start_time) * 1000,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def run_load_test(url="http://localhost:5002/invocations", 
                  num_requests=100, 
                  concurrency=10):
    \"\"\"
    Run load test on model server
    
    Args:
        url: Prediction endpoint URL
        num_requests: Total number of requests
        concurrency: Number of concurrent requests
    \"\"\"
    print(f"\\n{'=' * 70}")
    print("LOAD TESTING MODEL SERVER")
    print(f"{'=' * 70}")
    print(f"\\nURL: {url}")
    print(f"Total Requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print()
    
    # Sample data
    sample_data = {
        "dataframe_split": {
            "columns": [
                "area_sqft", "bedrooms", "bathrooms", "parking", "age_years",
                "floor", "has_gym", "has_pool", "price_per_sqft", "room_bath_ratio",
                "total_rooms", "amenities_score", "is_new", "is_spacious"
            ],
            "data": [[1500.0, 2.0, 1.5, 1.0, 5.0, 3.0, 1.0, 0.0, 1.2, 1.33, 3.5, 2.0, 1.0, 1.0]]
        }
    }
    
    results = []
    start_time = time.time()
    
    # Execute requests with thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(make_prediction, url, sample_data, i)
            for i in range(num_requests)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            
            if len(results) % 10 == 0:
                print(f"Progress: {len(results)}/{num_requests} requests completed")
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    latencies = [r['latency_ms'] for r in successful]
    
    print(f"\\n{'=' * 70}")
    print("LOAD TEST RESULTS")
    print(f"{'=' * 70}")
    
    print(f"\\n📊 Summary:")
    print(f"   Total Requests: {num_requests}")
    print(f"   Successful: {len(successful)} ({len(successful)/num_requests*100:.1f}%)")
    print(f"   Failed: {len(failed)} ({len(failed)/num_requests*100:.1f}%)")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Throughput: {num_requests/total_time:.2f} req/s")
    
    if latencies:
        print(f"\\n⏱️  Latency Statistics (ms):")
        print(f"   Mean: {statistics.mean(latencies):.2f}")
        print(f"   Median: {statistics.median(latencies):.2f}")
        print(f"   Min: {min(latencies):.2f}")
        print(f"   Max: {max(latencies):.2f}")
        print(f"   Std Dev: {statistics.stdev(latencies) if len(latencies) > 1 else 0:.2f}")
        
        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        print(f"\\n📈 Percentiles:")
        print(f"   P50: {p50:.2f} ms")
        print(f"   P95: {p95:.2f} ms")
        print(f"   P99: {p99:.2f} ms")
    
    if failed:
        print(f"\\n❌ Error Summary:")
        error_types = {}
        for f in failed:
            error = f.get('error', f'HTTP {f.get("status_code", "unknown")}')
            error_types[error] = error_types.get(error, 0) + 1
        
        for error, count in error_types.items():
            print(f"   {error}: {count}")
    
    print(f"\\n{'=' * 70}")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    run_load_test(num_requests=num_requests, concurrency=concurrency)
"""
        
        load_test_path = os.path.join(output_path, "load_test.py")
        with open(load_test_path, 'w') as f:
            f.write(load_test_script)
        
        os.chmod(load_test_path, 0o755)
        print(f"✓ Load test script: {load_test_path}")
        
        print("\n" + "=" * 70)
        print("✅ MONITORING SCRIPTS CREATED")
        print("=" * 70)


def main():
    """
    Main deployment workflow
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "MODEL DEPLOYMENT WORKFLOW")
    print("=" * 80)
    
    deployment = ModelDeployment()
    
    # Step 1: Get best model
    print("\n\n" + "█" * 80)
    print("STEP 1: SELECTING MODEL FOR DEPLOYMENT")
    print("█" * 80)
    
    # Try to get registered model first
    model_uri = deployment.get_registered_model_uri("HouseRentalPredictor")
    
    if not model_uri:
        # Fallback to best model from experiment
        model_uri, best_run = deployment.get_best_model_uri()
    
    if not model_uri:
        print("\n❌ No model found for deployment!")
        return
    
    # Step 2: Create deployment configuration
    print("\n\n" + "█" * 80)
    print("STEP 2: CREATING DEPLOYMENT CONFIGURATION")
    print("█" * 80)
    
    config_path = deployment.create_deployment_config(model_uri)
    
    # Step 3: Create monitoring scripts
    print("\n\n" + "█" * 80)
    print("STEP 3: CREATING MONITORING SCRIPTS")
    print("█" * 80)
    
    deployment.create_monitoring_script(config_path)
    
    # Step 4: Instructions for deployment
    print("\n\n" + "█" * 80)
    print("STEP 4: DEPLOYMENT INSTRUCTIONS")
    print("█" * 80)
    
    print("\n📦 Option 1: Local Deployment")
    print("=" * 70)
    print(f"cd {config_path}")
    print("./deploy.sh")
    print("\nOR:")
    print(f"mlflow models serve --model-uri {model_uri} --port 5002 --no-conda")
    
    print("\n\n🐳 Option 2: Docker Deployment")
    print("=" * 70)
    print(f"cd {config_path}")
    print("docker-compose up -d")
    
    print("\n\n🧪 Testing the Deployment")
    print("=" * 70)
    print("1. Health check:")
    print(f"   python {config_path}/health_check.py")
    print("\n2. Single prediction:")
    print(f"   curl -X POST http://localhost:5002/invocations \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d @{config_path}/sample_input.json")
    print("\n3. Load testing:")
    print(f"   python {config_path}/load_test.py 100 10")
    
    print("\n\n📊 Monitoring")
    print("=" * 70)
    print("After starting with Docker Compose:")
    print("- Model Server: http://localhost:5002")
    print("- Prometheus: http://localhost:9090")
    print("- Model metrics: http://localhost:5002/metrics")
    
    print("\n\n" + "=" * 80)
    print("✅ DEPLOYMENT SETUP COMPLETE")
    print("=" * 80)
    
    print("\n📁 Generated Files:")
    print(f"   {config_path}/deploy.sh")
    print(f"   {config_path}/Dockerfile")
    print(f"   {config_path}/docker-compose.yml")
    print(f"   {config_path}/prometheus.yml")
    print(f"   {config_path}/requirements.txt")
    print(f"   {config_path}/sample_input.json")
    print(f"   {config_path}/health_check.py")
    print(f"   {config_path}/load_test.py")
    
    print("\n🚀 Ready for deployment!")


if __name__ == "__main__":
    main()
