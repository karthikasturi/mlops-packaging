"""
FastAPI application for MLflow model serving with Prometheus metrics
"""
import time
import os
import pickle
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import psutil

# Prometheus metrics
REQUEST_COUNT = Counter(
    'model_request_count',
    'Total number of prediction requests',
    ['endpoint', 'http_status']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Time spent processing prediction requests',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total number of predictions made'
)

ERROR_COUNT = Counter(
    'model_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'Current CPU usage percentage'
)

MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'Current memory usage percentage'
)

ACTIVE_REQUESTS = Gauge(
    'model_active_requests',
    'Number of requests currently being processed'
)

# Model configuration
MODEL_PATH = "/app/house_rental_best_model.pkl"
model = None
model_version = "1.0"
feature_names = None

def load_model():
    """Load trained model from pickle file"""
    global model, model_version, feature_names
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_package = pickle.load(f)
        
        # Handle both dict package and direct model
        if isinstance(model_package, dict):
            model = model_package['model']
            feature_names = model_package.get('feature_names', None)
            model_version = model_package.get('version', '1.0')
            print(f"Model loaded successfully from package: {MODEL_PATH}")
            print(f"Model type: {model_package.get('model_type', 'unknown')}")
            print(f"Features: {len(feature_names) if feature_names else 'unknown'}")
        else:
            model = model_package
            print(f"Model loaded successfully (direct): {MODEL_PATH}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Request/Response models
class PredictionRequest(BaseModel):
    area_sqft: float
    bedrooms: int
    bathrooms: int
    parking: int
    age_years: int
    floor: int
    has_gym: int
    has_pool: int
    price_per_sqft: float
    room_bath_ratio: float
    total_rooms: int
    amenities_score: float
    is_new: int
    is_spacious: int

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup and cleanup on shutdown"""
    load_model()
    yield
    # Cleanup if needed

# Initialize FastAPI app with lifespan
app = FastAPI(title="House Rental Predictor", version="1.0.0", lifespan=lifespan)
    
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(endpoint='/health', http_status='200').inc()
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_version or "unknown"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction"""
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        if model is None:
            ERROR_COUNT.labels(error_type='model_not_loaded').inc()
            REQUEST_COUNT.labels(endpoint='/predict', http_status='500').inc()
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Prepare input features in the correct order
        features = np.array([[
            request.area_sqft,
            request.bedrooms,
            request.bathrooms,
            request.parking,
            request.age_years,
            request.floor,
            request.has_gym,
            request.has_pool,
            request.price_per_sqft,
            request.room_bath_ratio,
            request.total_rooms,
            request.amenities_score,
            request.is_new,
            request.is_spacious
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Update metrics
        processing_time = time.time() - start_time
        PREDICTION_LATENCY.observe(processing_time)
        PREDICTION_COUNT.inc()
        REQUEST_COUNT.labels(endpoint='/predict', http_status='200').inc()
        
        return PredictionResponse(
            prediction=float(prediction),
            model_version=model_version or "unknown",
            processing_time_ms=processing_time * 1000
        )
        
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(error_type='prediction_error').inc()
        REQUEST_COUNT.labels(endpoint='/predict', http_status='500').inc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        ACTIVE_REQUESTS.dec()

@app.post("/invocations", response_model=PredictionResponse)
async def invocations(request: PredictionRequest):
    """MLflow compatible invocations endpoint"""
    return await predict(request)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    # Update system metrics
    CPU_USAGE.set(psutil.cpu_percent(interval=0.1))
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    
    # Generate metrics
    metrics_output = generate_latest()
    return Response(content=metrics_output, media_type=CONTENT_TYPE_LATEST)

@app.get("/info")
async def model_info():
    """Get model information"""
    REQUEST_COUNT.labels(endpoint='/info', http_status='200').inc()
    return {
        "model_path": MODEL_PATH,
        "model_version": model_version,
        "model_loaded": model is not None
    }

@app.get("/")
async def root():
    """Root endpoint"""
    REQUEST_COUNT.labels(endpoint='/', http_status='200').inc()
    return {
        "service": "House Rental Predictor",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "invocations": "/invocations",
            "metrics": "/metrics",
            "info": "/info"
        }
    }
