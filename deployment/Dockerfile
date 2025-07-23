# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p Models static uploads

# Set environment variables for Hugging Face Spaces
ENV PYTHONPATH=/app
ENV PORT=7860

# Set Hugging Face cache to a writable directory
ENV HF_HOME=/data/hf_cache
RUN mkdir -p /data/hf_cache
# Expose the port
EXPOSE 7860

# Health check - simplified and more tolerant
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Command to run the application - use app:app format for HF Spaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "30"]