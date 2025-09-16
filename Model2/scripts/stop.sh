#!/bin/bash

# Ocean Hazard Prediction System - Stop Script
echo "Stopping Ocean Hazard Prediction System..."

# Stop all services
echo "Stopping all services..."
docker-compose down

# Optional: Remove volumes (uncomment if you want to clear all data)
# echo "Removing volumes..."
# docker-compose down -v

# Optional: Remove images (uncomment if you want to remove built images)
# echo "Removing images..."
# docker-compose down --rmi all

echo "System stopped successfully."

# Show status
echo "Checking remaining containers..."
docker ps | grep ocean-hazard || echo "No ocean-hazard containers running."