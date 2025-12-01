"""
Enhanced Model Server with Prometheus Metrics

This wrapper adds comprehensive monitoring capabilities to the MLflow model server:
- Request counting
- Latency tracking (p50, p95, p99)
- Error rate monitoring
- CPU and memory usage
- Prediction throughput
"""

from flask import Flask, request, jsonify
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import psutil
import os
from functools import wraps
from datetime import datetime


# Prometheus Metrics
REQUEST_COUNT = Counter(
    'model_requests_total',
    'Total number of requests to the model',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'model_request_duration_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total number of predictions made'
)

ERROR_COUNT = Counter(
    'model_errors_total',
    'Total number of errors',
    ['error_type']
)

CPU_USAGE = Gauge(
    'model_server_cpu_usage_percent',
    'CPU usage percentage'
)

MEMORY_USAGE = Gauge(
    'model_server_memory_usage_mb',
    'Memory usage in MB'
)

ACTIVE_REQUESTS = Gauge(
    'model_server_active_requests',
    'Number of requests currently being processed'
)

MODEL_INFO = Gauge(
    'model_info',
    'Model information',
    ['model_uri', 'version', 'framework']
)


class MonitoredModelServer:
    """
    Enhanced model server with monitoring
    """
    
    def __init__(self, model_uri, port=5002):
        """
        Initialize the monitored model server
        
        Args:
            model_uri: MLflow model URI
            port: Port to serve on
        """
        self.model_uri = model_uri
        self.port = port
        self.app = Flask(__name__)
        
        print(f"\n{'=' * 70}")
        print("INITIALIZING MONITORED MODEL SERVER")
        print(f"{'=' * 70}")
        
        # Load model
        print(f"\n📥 Loading model from: {model_uri}")
        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            print("✓ Model loaded successfully")
            
            # Set model info metric
            MODEL_INFO.labels(
                model_uri=model_uri,
                version='1.0',
                framework='mlflow'
            ).set(1)
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
        
        # Setup routes
        self._setup_routes()
        
        # Start system metrics collector
        self._start_system_metrics_collector()
    
    def _start_system_metrics_collector(self):
        """Start background thread for system metrics"""
        import threading
        
        def collect_metrics():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    CPU_USAGE.set(cpu_percent)
                    
                    # Memory usage
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    MEMORY_USAGE.set(memory_mb)
                    
                except Exception as e:
                    print(f"Error collecting system metrics: {str(e)}")
                
                time.sleep(5)  # Collect every 5 seconds
        
        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()
        print("✓ System metrics collector started")
    
    def monitor_request(self, endpoint):
        """
        Decorator to monitor requests
        
        Args:
            endpoint: Name of the endpoint
        """
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                ACTIVE_REQUESTS.inc()
                start_time = time.time()
                status = 200
                
                try:
                    response = f(*args, **kwargs)
                    return response
                    
                except Exception as e:
                    status = 500
                    ERROR_COUNT.labels(error_type=type(e).__name__).inc()
                    return jsonify({'error': str(e)}), 500
                    
                finally:
                    # Record metrics
                    latency = time.time() - start_time
                    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
                    REQUEST_COUNT.labels(
                        method=request.method,
                        endpoint=endpoint,
                        status=status
                    ).inc()
                    ACTIVE_REQUESTS.dec()
            
            return wrapper
        return decorator
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        @self.monitor_request('health')
        def health():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'model_uri': self.model_uri,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/invocations', methods=['POST'])
        @self.monitor_request('invocations')
        def predict():
            """Prediction endpoint"""
            try:
                # Get input data
                data = request.get_json()
                
                if not data:
                    ERROR_COUNT.labels(error_type='InvalidInput').inc()
                    return jsonify({'error': 'No input data provided'}), 400
                
                # Make prediction
                predictions = self.model.predict(data)
                PREDICTION_COUNT.inc()
                
                # Return predictions
                return jsonify({
                    'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
                })
                
            except Exception as e:
                ERROR_COUNT.labels(error_type=type(e).__name__).inc()
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            """Prometheus metrics endpoint"""
            return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
        
        @self.app.route('/info', methods=['GET'])
        @self.monitor_request('info')
        def info():
            """Model information endpoint"""
            return jsonify({
                'model_uri': self.model_uri,
                'framework': 'mlflow',
                'version': '1.0',
                'endpoints': {
                    'health': '/health',
                    'predict': '/invocations',
                    'metrics': '/metrics',
                    'info': '/info'
                }
            })
    
    def run(self, host='0.0.0.0', debug=False):
        """
        Run the server
        
        Args:
            host: Host to bind to
            debug: Enable debug mode
        """
        print(f"\n{'=' * 70}")
        print("STARTING MODEL SERVER")
        print(f"{'=' * 70}")
        print(f"\n🚀 Server starting on http://{host}:{self.port}")
        print(f"\n📍 Endpoints:")
        print(f"   Health:      http://{host}:{self.port}/health")
        print(f"   Prediction:  http://{host}:{self.port}/invocations")
        print(f"   Metrics:     http://{host}:{self.port}/metrics")
        print(f"   Info:        http://{host}:{self.port}/info")
        print(f"\n{'=' * 70}\n")
        
        self.app.run(host=host, port=self.port, debug=debug)


def main():
    """
    Main entry point
    """
    import sys
    
    # Get model URI from command line or environment
    if len(sys.argv) > 1:
        model_uri = sys.argv[1]
    else:
        model_uri = os.environ.get('MODEL_URI', 'models:/HouseRentalPredictor/1')
    
    port = int(os.environ.get('PORT', 5002))
    
    # Create and run server
    server = MonitoredModelServer(model_uri, port=port)
    server.run(host='0.0.0.0', debug=False)


if __name__ == '__main__':
    main()
