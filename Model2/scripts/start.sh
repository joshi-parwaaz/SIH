#!/bin/bash

# Ocean Hazard Prediction System - Start Script
echo "Starting Ocean Hazard Prediction System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose."
    exit 1
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs data/models data/exports data/maps notebooks nginx/logs monitoring

# Set permissions
chmod 755 logs data notebooks

# Check for environment file
if [ ! -f .env ]; then
    echo "Creating default .env file..."
    cat > .env << EOL
# Ocean Hazard Prediction System Environment Variables
ENVIRONMENT=production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Database Configuration (CHANGE THESE PASSWORDS!)
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=your_secure_mongo_password_here
MONGO_DATABASE=ocean_hazard_db
REDIS_PASSWORD=your_secure_redis_password_here
MONGODB_URI=mongodb://admin:\${MONGO_ROOT_PASSWORD}@mongo:27017/ocean_hazard_db?authSource=admin
REDIS_URL=redis://:\${REDIS_PASSWORD}@redis:6379/0

# Service Passwords (CHANGE THESE!)
GRAFANA_PASSWORD=your_secure_grafana_password_here
JUPYTER_TOKEN=your_secure_jupyter_token_here

# API Keys (replace with actual keys)
WEATHER_API_KEY=your_weather_api_key_here
EARTHQUAKE_API_KEY=your_earthquake_api_key_here
SOCIAL_MEDIA_API_KEY=your_social_media_api_key_here

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@example.com
SMTP_PASSWORD=your_app_password

# Webhook Configuration
WEBHOOK_SECRET=your_webhook_secret_here

# Security
JWT_SECRET_KEY=your_jwt_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here
EOL
    echo "Please edit .env file with your actual API keys and configuration"
fi

# Pull latest images
echo "Pulling latest Docker images..."
docker-compose pull

# Build and start services
echo "Building and starting services..."
docker-compose up -d --build

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check service health
echo "Checking service health..."
services=("ocean-hazard-api" "ocean-hazard-mongo" "ocean-hazard-redis")

for service in "${services[@]}"; do
    if docker-compose ps $service | grep -q "Up"; then
        echo "✓ $service is running"
    else
        echo "✗ $service failed to start"
        docker-compose logs $service
    fi
done

# Show service URLs
echo ""
echo "=== Ocean Hazard Prediction System Started ==="
echo "API Documentation: http://localhost:8000/docs"
echo "API Health Check: http://localhost:8000/health"
echo "Grafana Dashboard: http://localhost:3000 (admin/[check .env file])"
echo "Jupyter Notebook: http://localhost:8888 (token: [check .env file])"
echo "Prometheus: http://localhost:9090"
echo ""
echo "IMPORTANT: Change default passwords in .env file before production use!"
echo "To view logs: docker-compose logs -f"
echo "To stop system: ./scripts/stop.sh"
echo "================================================"