# 🐳 Titanic ML API - Production Docker Image
# =============================================

FROM python:3.9-slim

# Metadata
LABEL maintainer="Titanic ML Team"
LABEL description="Production-ready Titanic Survival Prediction API"
LABEL version="2.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src:/app" \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=4

# Create non-root user for security
RUN groupadd -r titanic && useradd -r -g titanic titanic

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements/base.txt /app/requirements/base.txt
COPY api/requirements.txt /app/api/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements/base.txt \
    && pip install --no-cache-dir -r api/requirements.txt \
    && pip install --no-cache-dir gunicorn

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models \
    && chown -R titanic:titanic /app

# Copy models if they exist (optional)
# COPY models/ /app/models/

# Switch to non-root user
USER titanic

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["gunicorn", "api.app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-"]