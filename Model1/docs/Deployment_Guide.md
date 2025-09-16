# Ocean Hazard Platform - Model A Deployment Guide

## Overview
This guide covers the deployment of Model A (Multilingual Hazard Detection & Extraction System) for the Ocean Hazard Platform.

## Prerequisites

### System Requirements
- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: Minimum 50GB free space
- **Network**: Stable internet connection for model downloads

### Software Requirements
- Docker 20.10+
- Docker Compose 2.0+
- Git
- (Optional) NVIDIA GPU with CUDA 11.8+ for faster processing

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Model1
```

### 2. Environment Setup
Create a `.env` file:
```bash
# Copy example environment file
cp .env.example .env

# Edit environment variables
nano .env
```

**Required Environment Variables:**
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Database Configuration
MONGODB_URL=mongodb://mongodb:27017
DATABASE_NAME=ocean_hazard_platform

# Model Configuration
MODEL_ENV=production
TORCH_DEVICE=cpu
# Set to 'cuda' if GPU available

# Security (set strong passwords in production)
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=your_secure_password

# Optional: API Keys for external services
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
NOMINATIM_USER_AGENT=ocean_hazard_platform
```

### 3. Deploy with Docker Compose
```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f model-a-ml
```

### 4. Verify Deployment
```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "2025-09-16T10:30:00Z",
#   "version": "1.0.0",
#   "models_loaded": {
#     "nlp_engine": true,
#     "geolocation_extractor": true
#   }
# }
```

## Production Deployment

### 1. Security Configuration

**Update default passwords:**
```bash
# Generate secure passwords
openssl rand -base64 32

# Update in .env file
MONGO_ROOT_PASSWORD=<generated_password>
```

**Configure SSL/TLS:**
```bash
# Create SSL certificates directory
mkdir -p nginx/ssl

# Generate self-signed certificates (or use Let's Encrypt)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout nginx/ssl/nginx.key \
    -out nginx/ssl/nginx.crt
```

**Nginx Configuration:**
Create `nginx/nginx.conf`:
```nginx
events {
    worker_connections 1024;
}

http {
    upstream model_a_backend {
        server model-a-ml:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/nginx.crt;
        ssl_certificate_key /etc/nginx/ssl/nginx.key;

        location / {
            proxy_pass http://model_a_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### 2. Resource Scaling

**For High Load Scenarios:**
Update `docker-compose.yml`:
```yaml
services:
  model-a-ml:
    # ... existing configuration
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    environment:
      - WORKERS=4  # Increase worker processes
```

### 3. Monitoring Setup

**Add monitoring services to docker-compose.yml:**
```yaml
  # Prometheus monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  # Grafana dashboard
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  prometheus_data:
  grafana_data:
```

## GPU Support

### 1. NVIDIA GPU Setup
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. Update Docker Compose for GPU
```yaml
services:
  model-a-ml:
    # ... existing configuration
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - TORCH_DEVICE=cuda
```

## Backup and Recovery

### 1. Database Backup
```bash
# Create backup script
cat > scripts/backup-db.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/app/backups"
mkdir -p $BACKUP_DIR

docker-compose exec -T mongodb mongodump \
    --host localhost:27017 \
    --db ocean_hazard_platform \
    --archive=$BACKUP_DIR/mongodb_backup_$DATE.archive
EOF

chmod +x scripts/backup-db.sh

# Schedule daily backups
crontab -e
# Add: 0 2 * * * /path/to/model1/scripts/backup-db.sh
```

### 2. Model and Config Backup
```bash
# Backup important directories
tar -czf model_backup_$(date +%Y%m%d).tar.gz \
    config/ \
    models/ \
    data/ \
    .env
```

## Performance Optimization

### 1. Model Optimization
```python
# Update config/config.yaml for production
MODELS:
  OPTIMIZATION:
    USE_QUANTIZATION: true
    BATCH_SIZE: 32
    MAX_SEQUENCE_LENGTH: 256
    ENABLE_CACHING: true
    
PROCESSING:
  ASYNC_WORKERS: 8
  QUEUE_SIZE: 1000
  TIMEOUT_SECONDS: 30
```

### 2. Database Optimization
```javascript
// MongoDB optimization script
// scripts/optimize-mongodb.js
db.hazard_reports.createIndex({ "timestamp": 1 });
db.hazard_reports.createIndex({ "location.coordinates": "2dsphere" });
db.hazard_reports.createIndex({ "hazard_type": 1, "severity": 1 });
db.anomalies.createIndex({ "detection_time": 1 });
db.operator_feedback.createIndex({ "report_id": 1 });
```

## Health Monitoring

### 1. Service Health Checks
```bash
# Create health check script
cat > scripts/health-check.sh << 'EOF'
#!/bin/bash

# Check API health
API_HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status')
if [ "$API_HEALTH" != "healthy" ]; then
    echo "API unhealthy, restarting service..."
    docker-compose restart model-a-ml
fi

# Check database connectivity
DB_STATUS=$(docker-compose exec -T mongodb mongo --eval "db.runCommand('ping')" --quiet)
if [[ $DB_STATUS != *"ok"* ]]; then
    echo "Database connection failed"
    exit 1
fi

echo "All services healthy"
EOF

chmod +x scripts/health-check.sh
```

### 2. Log Management
```yaml
# Add to docker-compose.yml
services:
  model-a-ml:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

## Troubleshooting

### Common Issues

**1. Models not loading:**
```bash
# Check available disk space
df -h

# Check model download logs
docker-compose logs model-a-ml | grep -i "model"

# Manually download models
docker-compose exec model-a-ml python -c "
from transformers import AutoModel
AutoModel.from_pretrained('xlm-roberta-large')
"
```

**2. High memory usage:**
```bash
# Monitor memory usage
docker stats

# Reduce batch size in config
MODELS:
  BATCH_SIZE: 8  # Reduce from 16
```

**3. Database connection issues:**
```bash
# Check MongoDB logs
docker-compose logs mongodb

# Test connection
docker-compose exec model-a-ml python -c "
import pymongo
client = pymongo.MongoClient('mongodb://mongodb:27017')
print(client.admin.command('ping'))
"
```

**4. API timeout errors:**
```bash
# Increase timeout in nginx
proxy_read_timeout 300s;
proxy_connect_timeout 300s;

# Check processing queue
curl http://localhost:8000/stats/pipeline
```

## Scaling Guidelines

### Horizontal Scaling
- Deploy multiple instances behind a load balancer
- Use Redis for caching and session management
- Implement database sharding for large datasets

### Vertical Scaling
- Increase CPU/memory allocation
- Use GPU instances for faster processing
- Optimize model parameters for memory usage

## Security Best Practices

1. **Network Security:**
   - Use private networks for inter-service communication
   - Implement firewall rules
   - Regular security updates

2. **Data Security:**
   - Encrypt sensitive data at rest
   - Use secure connections (TLS/SSL)
   - Regular backup testing

3. **Access Control:**
   - Implement API authentication
   - Use role-based access control
   - Monitor and log API usage

## Support and Maintenance

### Regular Maintenance Tasks
- Weekly: Check logs and performance metrics
- Monthly: Update dependencies and security patches
- Quarterly: Review and optimize model performance
- Annually: Security audit and disaster recovery testing

### Support Contacts
- **Technical Issues**: ML Development Team
- **Infrastructure**: DevOps Team
- **Emergency**: On-call rotation

## References
- [Docker Documentation](https://docs.docker.com/)
- [MongoDB Operations](https://docs.mongodb.com/manual/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Transformers Library](https://huggingface.co/docs/transformers/)