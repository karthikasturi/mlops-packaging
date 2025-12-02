# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy application code
COPY src/ /app/

# Copy model directory
COPY model/ /model/

# Expose Flask port
EXPOSE 5000

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Run the Flask app
CMD ["python", "app.py"]
