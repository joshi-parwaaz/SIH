#!/bin/bash

# Ocean Hazard Platform - Model A Stop Script

echo "Stopping Ocean Hazard Platform - Model A..."

# Stop all services
docker-compose down

# Optional: Remove volumes (uncomment if you want to reset all data)
# echo "Removing volumes..."
# docker-compose down -v

echo "✅ All services stopped successfully."
echo ""
echo "To start again, run: ./scripts/start.sh"