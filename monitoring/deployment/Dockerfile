# FastAPI Model Deployment Dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy trained model
COPY ../models/house_rental_best_model.pkl /app/house_rental_best_model.pkl

# Expose port
EXPOSE 5002

# Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    # CMD curl -f http://localhost:5002/health || exit 1

# Run FastAPI application with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5002", "--workers", "1"]
#CMD ["python", "app.py"]
