# PowerScope Production Deployment Guide

## Prerequisites
- Docker and Docker Compose installed
- Trained models in `artifacts/` directory
- Environment variables configured

## Quick Start

1. **Build and start all services:**
   ```bash
   cd docker
   docker-compose up --build
   ```

2. **Access the application:**
   - Web UI: http://localhost (nginx proxy)
   - API directly: http://localhost:8000
   - Streamlit directly: http://localhost:8501

## Services

### PowerScope API (powerscope-api)
- **Port:** 8000
- **Health check:** GET /health
- **Prediction endpoint:** POST /predict
- **Model:** LightGBM or LSTM (configurable via MODEL_KIND env var)

### PowerScope UI (powerscope-ui)
- **Port:** 8501
- **Framework:** Streamlit
- **Features:** Interactive forecasting, model comparison

### Nginx Reverse Proxy
- **Port:** 80, 443
- **Routes:**
  - `/api/` → powerscope-api:8000
  - `/health` → powerscope-api:8000/health
  - `/` → powerscope-ui:8501

## Configuration

### Environment Variables
- `MODEL_KIND`: "lightgbm" or "lstm" (default: lightgbm)
- `PYTHONPATH`: /app

### Volume Mounts
- `../artifacts:/app/artifacts:ro` - Trained models (read-only)
- `../data:/app/data:ro` - Data files (read-only)

## Health Monitoring

All services include health checks:
- API: HTTP health endpoint with 30s intervals
- Containers: Restart policy "unless-stopped"
- Nginx: Depends on API and UI services

## Production Notes

1. **SSL/TLS:** Add certificates to nginx configuration for HTTPS
2. **Scaling:** Use Docker Swarm or Kubernetes for multi-node deployment
3. **Monitoring:** Add logging and metrics collection
4. **Security:** Configure firewall rules and access controls
5. **Backup:** Regular backup of artifacts and configuration

## Troubleshooting

### Common Issues
1. **Models not found:** Ensure artifacts/ directory contains trained models
2. **Port conflicts:** Change port mappings in docker-compose.yml
3. **Memory issues:** Increase Docker memory limits for ML models
4. **GPU support:** Add nvidia-docker runtime for GPU acceleration

### Logs
```bash
# View all service logs
docker-compose logs

# View specific service logs
docker-compose logs powerscope-api
docker-compose logs powerscope-ui
docker-compose logs nginx
```

### Service Status
```bash
# Check running services
docker-compose ps

# Restart specific service
docker-compose restart powerscope-api
```