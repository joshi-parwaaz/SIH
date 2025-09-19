#!/bin/bash
# Docker deployment script for Ocean Hazard Analysis System

set -e

echo "🌊 Ocean Hazard Analysis System - Docker Deployment"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start     Start the application (default)"
    echo "  stop      Stop the application"
    echo "  restart   Restart the application"
    echo "  logs      Show application logs"
    echo "  status    Show container status"
    echo "  clean     Clean up containers and volumes"
    echo "  build     Build the Docker image"
    echo ""
    echo "Options:"
    echo "  --prod    Use production profile (includes nginx, redis, postgres)"
    echo "  --dev     Use development profile (api only)"
    echo "  --help    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start --dev          # Start in development mode"
    echo "  $0 start --prod         # Start with all services"
    echo "  $0 logs hazard-api      # Show API logs"
    echo "  $0 clean                # Clean up everything"
}

# Parse command line arguments
COMMAND=${1:-start}
PROFILE="dev"

while [[ $# -gt 0 ]]; do
    case $1 in
        --prod)
            PROFILE="prod"
            shift
            ;;
        --dev)
            PROFILE="dev" 
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            if [[ -z "$COMMAND" ]]; then
                COMMAND=$1
            fi
            shift
            ;;
    esac
done

# Set Docker Compose profiles
if [[ "$PROFILE" == "prod" ]]; then
    COMPOSE_PROFILES="--profile production --profile cache --profile database"
    echo "🚀 Using production profile (nginx + redis + postgres)"
else
    COMPOSE_PROFILES=""
    echo "🔧 Using development profile (api only)"
fi

# Execute commands
case $COMMAND in
    start)
        echo "📦 Starting Ocean Hazard Analysis System..."
        docker-compose up -d $COMPOSE_PROFILES
        echo "✅ System started successfully!"
        echo ""
        echo "🌐 Available endpoints:"
        if [[ "$PROFILE" == "prod" ]]; then
            echo "  Main API: http://localhost"
            echo "  Direct API: http://localhost:8000"
        else
            echo "  API: http://localhost:8000"
        fi
        echo "  API Docs: http://localhost:8000/docs"
        echo "  Health Check: http://localhost:8000/health"
        echo ""
        echo "📊 Check status: $0 status"
        echo "📋 View logs: $0 logs"
        ;;
    
    stop)
        echo "🛑 Stopping Ocean Hazard Analysis System..."
        docker-compose down
        echo "✅ System stopped successfully!"
        ;;
    
    restart)
        echo "🔄 Restarting Ocean Hazard Analysis System..."
        docker-compose down
        docker-compose up -d $COMPOSE_PROFILES
        echo "✅ System restarted successfully!"
        ;;
    
    logs)
        SERVICE=${2:-hazard-api}
        echo "📋 Showing logs for $SERVICE..."
        docker-compose logs -f $SERVICE
        ;;
    
    status)
        echo "📊 Container Status:"
        docker-compose ps
        echo ""
        echo "🔍 Health Checks:"
        docker-compose exec hazard-api curl -s http://localhost:8000/health | jq . || echo "API not responding"
        ;;
    
    clean)
        echo "🧹 Cleaning up Docker containers and volumes..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        echo "✅ Cleanup completed!"
        ;;
    
    build)
        echo "🔨 Building Docker image..."
        docker-compose build --no-cache
        echo "✅ Build completed!"
        ;;
    
    *)
        echo "❌ Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac