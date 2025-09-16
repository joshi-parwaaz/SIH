#!/bin/bash

# Ocean Hazard Platform - Model A Startup Script

echo "Starting Ocean Hazard Platform - Model A..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your configuration before running again."
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs models config

# Build and start services
echo "Building and starting services..."
docker-compose up -d --build

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Health check
echo "Checking service health..."
HEALTH_STATUS=$(curl -s http://localhost:8000/health | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

if [ "$HEALTH_STATUS" = "healthy" ]; then
    echo "✅ Model A service is healthy and ready!"
    echo "🌐 API available at: http://localhost:8000"
    echo "📚 API documentation: http://localhost:8000/docs"
    echo "💾 MongoDB available at: mongodb://localhost:27017"
else
    echo "❌ Service health check failed. Checking logs..."
    docker-compose logs --tail=50 model-a-ml
    exit 1
fi

echo "🚀 Ocean Hazard Platform - Model A is now running!"
echo ""
echo "Quick commands:"
echo "  View logs:        docker-compose logs -f model-a-ml"
echo "  Stop services:    docker-compose down"
echo "  Restart:          docker-compose restart"
echo "  Health check:     curl http://localhost:8000/health"