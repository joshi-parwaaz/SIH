# Docker Quick Start Guide

## Prerequisites
- Docker Engine 20.10+ and Docker Compose 2.0+
- 4GB RAM minimum, 8GB recommended
- 10GB disk space for images and data

## Quick Start Commands

### Development Environment
```bash
# Start basic services
docker-compose up --build

# Access points:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Health: http://localhost:8000/health
```

### Production Environment
```bash
# Start full production stack
docker-compose -f docker-compose.prod.yml up --build -d

# Start with nginx proxy
docker-compose -f docker-compose.prod.yml --profile production up -d

# Access points:
# - Nginx Proxy: http://localhost:80
# - Direct API: http://localhost:8000
```

### Management Commands
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f disaster-api

# Stop services
docker-compose down

# Cleanup everything
docker-compose down -v
docker system prune -f
```

### Testing Docker Build
```bash
# Build image only
docker build -t disaster-mgmt:test .

# Run single container
docker run -p 8000:8000 disaster-mgmt:test

# Test health endpoint
curl http://localhost:8000/health
```

## Service Architecture

### Development Stack
- **disaster-api**: Main FastAPI application (port 8000)
- **logs**: Persistent log storage
- **model-cache**: ML model cache storage

### Production Stack
- **disaster-api**: Scaled API service (2 replicas)
- **nginx**: Load balancer and reverse proxy (port 80/443)
- **redis**: Caching layer (port 6379)
- **postgres**: Database storage (optional, port 5432)
- **prometheus**: Monitoring (optional, port 9090)

## Environment Variables

Copy `.env.production` to `.env` and customize:

```bash
# Essential settings
APP_ENV=production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Database (if using postgres profile)
POSTGRES_PASSWORD=your_secure_password

# Security
SECRET_KEY=your_secret_key
JWT_SECRET_KEY=your_jwt_secret
```

## Docker Profiles

Use profiles to start specific service combinations:

```bash
# Basic API only
docker-compose up

# With nginx proxy
docker-compose --profile production up

# With database
docker-compose --profile database up

# With monitoring
docker-compose --profile monitoring up

# Full production stack
docker-compose --profile production --profile database --profile monitoring up
```

## Health Checks

All services include health checks:
- **API**: HTTP health endpoint at `/health`
- **Nginx**: Proxy pass to API health
- **Redis**: Built-in health check
- **Postgres**: Connection test

## Volume Management

### Persistent Data
- `logs`: Application logs
- `model-cache`: ML model cache
- `redis-data`: Redis cache persistence
- `postgres-data`: Database files
- `prometheus-data`: Monitoring metrics

### Backup Strategy
```bash
# Create backup
docker-compose exec disaster-api python -c "
import json, shutil
from datetime import datetime
backup_time = datetime.now().strftime('%Y%m%d_%H%M%S')
shutil.copy('hazard_history.json', f'hazard_history_backup_{backup_time}.json')
print(f'Backup created: hazard_history_backup_{backup_time}.json')
"

# Volume backup
docker run --rm -v model2_logs:/data -v $(pwd)/backups:/backup alpine tar czf /backup/logs_backup.tar.gz -C /data .
```

## Production Deployment

### Step 1: Prepare Environment
```bash
# Clone repository
git clone <repo-url>
cd Model2

# Create production environment
cp .env.production .env
# Edit .env with your settings

# Create SSL certificates (if using HTTPS)
mkdir -p nginx/ssl
# Copy your cert.pem and key.pem files
```

### Step 2: Deploy Services
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up --build -d

# Verify deployment
docker-compose -f docker-compose.prod.yml ps
curl -f http://localhost/health
```

### Step 3: Configure Load Balancer
For high availability, place behind external load balancer:
- AWS ALB, GCP Load Balancer, or Azure Load Balancer
- Configure health check: `GET /health`
- Enable auto-scaling based on CPU/memory

## Troubleshooting

### Common Issues

**Build Fails**
```bash
# Clear Docker cache
docker system prune -a
docker-compose build --no-cache
```

**Memory Issues**
```bash
# Reduce resource limits in docker-compose.prod.yml
resources:
  limits:
    memory: 1G  # Reduce from 2G
```

**Network Issues**
```bash
# Reset Docker network
docker network prune
docker-compose down
docker-compose up
```

**Permission Issues**
```bash
# Fix file permissions
sudo chown -R $USER:$USER logs/
sudo chmod -R 755 logs/
```

### Monitoring

**Check Resource Usage**
```bash
docker stats
```

**View Service Logs**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f disaster-api
docker-compose logs -f nginx
```

**Health Check Status**
```bash
# API health
curl -f http://localhost:8000/health

# Through nginx
curl -f http://localhost/health

# Service health in compose
docker-compose ps
```

## Security Considerations

### Production Security
1. **Change default passwords** in `.env`
2. **Use HTTPS** with valid SSL certificates
3. **Enable firewall** rules for ports 80, 443 only
4. **Regular updates** of base images
5. **Monitor logs** for suspicious activity

### Network Security
- All services communicate via internal Docker network
- Only nginx and API expose external ports
- Redis and Postgres are internal only
- Rate limiting configured in nginx

### Data Security
- Sensitive data in environment variables
- Logs stored in Docker volumes
- Database passwords encrypted
- JWT tokens for API authentication

This Docker setup provides a production-ready deployment of the Disaster Management System with scalability, monitoring, and security features.