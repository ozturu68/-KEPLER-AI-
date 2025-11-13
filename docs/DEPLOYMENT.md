# 🚀 Deployment Guide

**Last Updated:** 2025-11-13  
**Status:** 🚧 In Development

## Overview

Bu döküman Kepler Exoplanet ML projesinin farklı ortamlara deployment sürecini açıklar.

## Table of Contents

- [Deployment Options](#deployment-options)
- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [API Deployment](#api-deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Deployment Options

### Available Deployment Methods

```
1. Local Development
   ├─ Virtual environment
   ├─ Manual setup
   └─ For: Development & Testing

2. Docker Container
   ├─ Containerized application
   ├─ Reproducible environment
   └─ For: Production-ready deployment

3. Cloud Platforms
   ├─ AWS (EC2, Lambda, SageMaker)
   ├─ Google Cloud (Compute Engine, Cloud Run)
   ├─ Azure (VM, Container Instances)
   └─ For: Scalable production

4. Serverless
   ├─ AWS Lambda
   ├─ Google Cloud Functions
   ├─ Azure Functions
   └─ For: Event-driven, cost-effective

5. Kubernetes
   ├─ Multi-container orchestration
   ├─ Auto-scaling
   └─ For: Large-scale production
```

---

## Local Deployment

### Development Setup

```bash
# 1. Clone repository
git clone https://github.com/sulegogh/kepler-new.git
cd kepler-new

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import src; print('✅ Installation successful')"

# 5. Run tests
pytest tests/ -v

# 6. Start development server (when API is ready)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables

```bash
# Create .env file
cat > .env << EOF
# Application
APP_NAME="Kepler Exoplanet ML"
APP_VERSION="0.7.0"
ENVIRONMENT="development"

# Model
MODEL_PATH="models/catboost_exoplanet_v1.pkl"
MODEL_VERSION="1.0.0"

# Data
DATA_DIR="data"
RAW_DATA_DIR="data/raw"
PROCESSED_DATA_DIR="data/processed"

# Logging
LOG_LEVEL="INFO"
LOG_FILE="logs/app.log"

# API (when ready)
API_HOST="0.0.0.0"
API_PORT="8000"
API_WORKERS="4"

# Database (future)
# DATABASE_URL="postgresql://user:password@localhost:5432/kepler"

# Cache (future)
# REDIS_URL="redis://localhost:6379/0"
EOF

# Load environment variables
source .env
```

---

## Docker Deployment

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Copy configuration
COPY pytest.ini .
COPY .env.example .env

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: "3.8"

services:
  app:
    build: .
    container_name: kepler-ml
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - MODEL_PATH=/app/models/catboost_exoplanet_v1.pkl
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    networks:
      - kepler-network
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    container_name: kepler-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - kepler-network
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    container_name: kepler-postgres
    environment:
      - POSTGRES_USER=kepler
      - POSTGRES_PASSWORD=kepler_password
      - POSTGRES_DB=kepler
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - kepler-network
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: kepler-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    networks:
      - kepler-network
    restart: unless-stopped

networks:
  kepler-network:
    driver: bridge

volumes:
  redis-data:
  postgres-data:
```

### Build and Run

```bash
# Build Docker image
docker build -t kepler-ml:latest .

# Run container
docker run -d \
  --name kepler-ml \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e ENVIRONMENT=production \
  kepler-ml:latest

# Check logs
docker logs -f kepler-ml

# Stop container
docker stop kepler-ml

# Using Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Multi-stage Build (Optimized)

```dockerfile
# Dockerfile.optimized
# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application
COPY src/ ./src/
COPY models/ ./models/
COPY .env.example .env

# Update PATH
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Cloud Deployment

### AWS Deployment

#### EC2 Instance

```bash
# 1. Launch EC2 instance (Ubuntu 22.04)
# Instance type: t3.medium (2 vCPU, 4 GB RAM)

# 2. Connect to instance
ssh -i "key.pem" ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# 3. Update system
sudo apt update && sudo apt upgrade -y

# 4. Install Docker
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# 5. Clone repository
git clone https://github.com/sulegogh/kepler-new.git
cd kepler-new

# 6. Deploy with Docker Compose
docker-compose up -d

# 7. Configure security group
# Allow inbound: 80 (HTTP), 443 (HTTPS), 8000 (API)
```

#### AWS Lambda (Serverless)

```python
# lambda_function.py
import json
import pickle
import pandas as pd
from typing import Dict, Any


def load_model():
    """Load model from S3 or local."""
    with open('/tmp/model.pkl', 'rb') as f:
        return pickle.load(f)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for predictions.

    Args:
        event: Lambda event with prediction data
        context: Lambda context

    Returns:
        API Gateway response
    """
    try:
        # Parse input
        body = json.loads(event.get('body', '{}'))
        features = body.get('features', {})

        # Load model
        model = load_model()

        # Make prediction
        df = pd.DataFrame([features])
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0]

        # Response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'prediction': int(prediction),
                'probabilities': proba.tolist(),
                'success': True
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': str(e),
                'success': False
            })
        }
```

```bash
# Deploy to Lambda
# 1. Package dependencies
pip install -r requirements.txt -t package/
cp lambda_function.py package/
cd package && zip -r ../lambda.zip . && cd ..

# 2. Create Lambda function
aws lambda create-function \
  --function-name kepler-ml-predict \
  --runtime python3.10 \
  --role arn:aws:iam::ACCOUNT:role/lambda-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda.zip \
  --timeout 30 \
  --memory-size 512

# 3. Create API Gateway
aws apigateway create-rest-api \
  --name kepler-ml-api \
  --description "Kepler Exoplanet ML API"
```

---

### Google Cloud Deployment

#### Cloud Run

```bash
# 1. Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/kepler-ml

# 2. Deploy to Cloud Run
gcloud run deploy kepler-ml \
  --image gcr.io/PROJECT_ID/kepler-ml \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --set-env-vars ENVIRONMENT=production

# 3. Get service URL
gcloud run services describe kepler-ml \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)'
```

---

### Azure Deployment

#### Azure Container Instances

```bash
# 1. Create resource group
az group create --name kepler-ml-rg --location eastus

# 2. Create container registry
az acr create --resource-group kepler-ml-rg \
  --name keplermlregistry --sku Basic

# 3. Build and push image
az acr build --registry keplermlregistry \
  --image kepler-ml:latest .

# 4. Deploy container
az container create \
  --resource-group kepler-ml-rg \
  --name kepler-ml \
  --image keplermlregistry.azurecr.io/kepler-ml:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server keplermlregistry.azurecr.io \
  --ports 8000 \
  --dns-name-label kepler-ml
```

---

## API Deployment

### Nginx Configuration

```nginx
# nginx.conf
upstream kepler_api {
    server app:8000;
}

server {
    listen 80;
    server_name api.kepler-ml.example.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.kepler-ml.example.com;

    # SSL configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Logging
    access_log /var/log/nginx/kepler_access.log;
    error_log /var/log/nginx/kepler_error.log;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # API endpoints
    location / {
        proxy_pass http://kepler_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://kepler_api/health;
        access_log off;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;
}
```

### Systemd Service

```ini
# /etc/systemd/system/kepler-ml.service
[Unit]
Description=Kepler Exoplanet ML API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/kepler-new
Environment="PATH=/home/ubuntu/kepler-new/venv/bin"
ExecStart=/home/ubuntu/kepler-new/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable kepler-ml
sudo systemctl start kepler-ml
sudo systemctl status kepler-ml

# View logs
sudo journalctl -u kepler-ml -f
```

---

## Monitoring

### Health Checks

```python
# src/api/main.py (when implemented)
from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.7.0"
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return {
        "requests_total": 1000,
        "requests_success": 950,
        "requests_failed": 50,
        "avg_response_time_ms": 123.45
    }
```

### Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: "3.8"

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
    networks:
      - kepler-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - kepler-network

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
    networks:
      - kepler-network

volumes:
  prometheus-data:
  grafana-data:

networks:
  kepler-network:
    external: true
```

---

## Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000
# or
netstat -tulpn | grep 8000

# Kill process
kill -9 PID
```

#### Docker Build Fails

```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t kepler-ml:latest .
```

#### Permission Denied

```bash
# Fix file permissions
sudo chown -R $USER:$USER /path/to/kepler-new

# Docker permissions
sudo usermod -aG docker $USER
newgrp docker
```

#### Out of Memory

```bash
# Increase Docker memory limit
# Docker Desktop: Settings → Resources → Memory

# Linux: Edit /etc/docker/daemon.json
{
  "default-runtime": "runc",
  "default-shm-size": "2G"
}

sudo systemctl restart docker
```

---

## Performance Tuning

### Uvicorn Configuration

```bash
# Production settings
uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --loop uvloop \
  --http httptools \
  --limit-concurrency 100 \
  --timeout-keep-alive 5
```

### Gunicorn with Uvicorn Workers

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

---

## Security Checklist

```markdown
- [ ] Use HTTPS in production
- [ ] Set strong passwords
- [ ] Enable firewall rules
- [ ] Use environment variables for secrets
- [ ] Implement rate limiting
- [ ] Enable CORS properly
- [ ] Keep dependencies updated
- [ ] Use non-root user in containers
- [ ] Implement authentication
- [ ] Monitor logs for suspicious activity
- [ ] Regular security audits
- [ ] Backup data regularly
```

---

## Deployment Checklist

```markdown
Pre-Deployment:

- [ ] All tests passing
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] Environment variables configured
- [ ] SSL certificates ready
- [ ] Monitoring setup
- [ ] Backup plan ready

Deployment:

- [ ] Deploy to staging first
- [ ] Run smoke tests
- [ ] Check health endpoints
- [ ] Verify API responses
- [ ] Test authentication
- [ ] Check monitoring dashboards

Post-Deployment:

- [ ] Monitor error logs
- [ ] Check performance metrics
- [ ] Verify database connections
- [ ] Test all endpoints
- [ ] Notify team
- [ ] Update documentation
```

---

**Status:** 🚧 In Development (API not yet implemented)  
**Maintainer:** sulegogh  
**Last Updated:** 2025-11-13
