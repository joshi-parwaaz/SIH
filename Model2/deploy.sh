#!/bin/bash

# Enhanced Disaster Management API - Quick Deployment Script
# This script allows easy deployment of the new enhanced ML pipeline
# while maintaining compatibility with existing backend integration.

echo "🚀 Enhanced Disaster Management API Deployment"
echo "============================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Stop any existing containers
echo "🛑 Stopping any existing containers..."
docker-compose down 2>/dev/null || true

# Build and start the new enhanced API
echo "🔨 Building and starting enhanced disaster management API..."
docker-compose up --build -d

# Wait for service to be ready
echo "⏳ Waiting for API to be ready..."
sleep 10

# Health check
echo "🔍 Checking API health..."
max_retries=30
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    if curl -f -s http://localhost:8000/health > /dev/null; then
        echo "✅ API is healthy and ready!"
        break
    fi
    
    retry_count=$((retry_count + 1))
    echo "⏳ Waiting... ($retry_count/$max_retries)"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    echo "❌ API failed to start properly. Check logs with: docker-compose logs"
    exit 1
fi

# Show API status
echo ""
echo "🎉 Enhanced Disaster Management API is now running!"
echo "============================================="
echo "📍 API Base URL: http://localhost:8000"
echo "🏥 Health Check: http://localhost:8000/health"
echo "📊 Main Analysis: http://localhost:8000/analyze"
echo "📈 System Status: http://localhost:8000/status"
echo "📖 API Documentation: http://localhost:8000/docs"
echo ""
echo "🔄 Backward Compatible Endpoints:"
echo "   - POST/GET /api/analyze"
echo "   - POST/GET /api/hazard-analysis"
echo ""
echo "📊 Quick Test:"
curl -s http://localhost:8000/health | head -n 5
echo ""
echo ""
echo "🛠️ Management Commands:"
echo "   - View logs: docker-compose logs -f"
echo "   - Stop API: docker-compose down"
echo "   - Restart: docker-compose restart"
echo ""
echo "✅ Your existing backend should work without any changes!"
echo "   Just point it to the same endpoints at http://localhost:8000"