# Integration Guide - Ocean Hazard Analysis System (1pro)

## Overview
This guide helps integrate the **1pro Ocean Hazard Analysis System** with your existing frontend and backend infrastructure.

## System Architecture

```
Frontend (Your App) → Backend (Your API) → 1pro Service → ML Pipeline + Data Sources
```

## Quick Integration Options

### Option 1: Microservice Integration (Recommended)

Deploy 1pro as a separate service and call it from your backend:

```python
# In your backend API
import requests

async def get_ocean_hazards():
    try:
        response = requests.get("http://localhost:8001/analyze", timeout=60)
        return response.json()
    except Exception as e:
        return {"error": f"Hazard service unavailable: {str(e)}"}
```

### Option 2: Direct Code Integration

Copy the ML components directly into your backend:

```python
# Copy these folders to your backend:
# - app/pipeline.py
# - scrapers/
# - models/
from app.pipeline import HazardAnalysisPipeline

pipeline = HazardAnalysisPipeline()
results = await pipeline.analyze()
```

## Docker Deployment Integration

### 1. Add to Your Docker Compose

```yaml
# Add to your existing docker-compose.yml
services:
  # Your existing services...
  
  ocean-hazard-api:
    build: ./1pro
    ports:
      - "8001:8000"  # Different port to avoid conflicts
    environment:
      - ENVIRONMENT=production
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - your-network

networks:
  your-network:
    driver: bridge
```

### 2. Network Configuration

```yaml
# If using internal networking
services:
  your-backend:
    environment:
      - HAZARD_API_URL=http://ocean-hazard-api:8000
  
  ocean-hazard-api:
    # No need to expose port externally
    expose:
      - "8000"
```

## API Integration Examples

### Backend Integration (Node.js/Express)

```javascript
// routes/hazards.js
const axios = require('axios');

router.get('/ocean-hazards', async (req, res) => {
  try {
    const response = await axios.get('http://localhost:8001/analyze', {
      timeout: 60000
    });
    
    res.json({
      success: true,
      data: response.data,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Hazard analysis service unavailable'
    });
  }
});
```

### Backend Integration (Python/FastAPI)

```python
# main.py - Add to your existing FastAPI app
from fastapi import HTTPException
import httpx

@app.get("/api/ocean-hazards")
async def get_ocean_hazards():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "http://localhost:8001/analyze",
                timeout=60.0
            )
            return response.json()
        except Exception as e:
            raise HTTPException(
                status_code=503, 
                detail="Hazard analysis service unavailable"
            )
```

### Frontend Integration (React)

```javascript
// components/OceanHazards.jsx
import { useState, useEffect } from 'react';

function OceanHazards() {
  const [hazards, setHazards] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchHazards = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/ocean-hazards');
      const data = await response.json();
      setHazards(data);
    } catch (error) {
      console.error('Failed to fetch hazards:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <button onClick={fetchHazards} disabled={loading}>
        {loading ? 'Analyzing...' : 'Get Ocean Hazards'}
      </button>
      
      {hazards && (
        <div>
          <h3>Current Threat Level: {hazards.overall_threat_level}</h3>
          <p>Active Alerts: {hazards.active_alerts}</p>
          {/* Render hazard data */}
        </div>
      )}
    </div>
  );
}
```

## Configuration Management

### Environment Variables

```bash
# Add to your .env file
HAZARD_API_URL=http://localhost:8001
HAZARD_API_TIMEOUT=60
HAZARD_CACHE_TTL=300  # Cache results for 5 minutes
```

### Production Configuration

```yaml
# production.docker-compose.yml
services:
  ocean-hazard-api:
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
    environment:
      - WORKERS=2
      - LOG_LEVEL=info
```

## Data Flow Integration

### 1. Real-time Dashboard Integration

```javascript
// Real-time updates
const EventSource = require('eventsource');

const hazardStream = new EventSource('http://localhost:8001/stream');
hazardStream.onmessage = (event) => {
  const hazardData = JSON.parse(event.data);
  updateDashboard(hazardData);
};
```

### 2. Database Integration

```python
# Store hazard data in your database
async def store_hazard_analysis(hazard_data):
    await db.hazard_reports.insert_one({
        "timestamp": datetime.utcnow(),
        "threat_level": hazard_data["overall_threat_level"],
        "alerts": hazard_data["active_alerts"],
        "sources": hazard_data["data_sources"],
        "raw_data": hazard_data
    })
```

## Performance Optimization

### 1. Caching Strategy

```python
# Add Redis caching
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

async def get_cached_hazards():
    cached = redis_client.get("ocean_hazards")
    if cached:
        return json.loads(cached)
    
    # Fetch fresh data
    hazards = await fetch_hazards()
    redis_client.setex("ocean_hazards", 300, json.dumps(hazards))  # 5 min cache
    return hazards
```

### 2. Background Processing

```python
# Celery task for background analysis
from celery import Celery

celery_app = Celery('hazard_analysis')

@celery_app.task
def analyze_hazards_background():
    # Run analysis in background
    result = requests.get("http://localhost:8001/analyze")
    # Store in database or cache
    return result.json()
```

## Monitoring Integration

### 1. Health Checks

```python
# Add to your health check endpoint
async def health_check():
    services = {}
    
    # Check hazard service
    try:
        response = await client.get("http://localhost:8001/health")
        services["ocean_hazards"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        services["ocean_hazards"] = "down"
    
    return services
```

### 2. Logging Integration

```python
# Structured logging
import logging

logger = logging.getLogger(__name__)

async def analyze_with_logging():
    logger.info("Starting ocean hazard analysis")
    try:
        result = await get_hazards()
        logger.info(f"Analysis completed: {result['overall_threat_level']}")
        return result
    except Exception as e:
        logger.error(f"Hazard analysis failed: {str(e)}")
        raise
```

## Testing Integration

### Unit Tests

```python
# test_hazard_integration.py
import pytest
from unittest.mock import patch

@pytest.mark.asyncio
async def test_hazard_api_integration():
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {
            "overall_threat_level": "medium",
            "active_alerts": 2
        }
        
        result = await get_ocean_hazards()
        assert result["overall_threat_level"] == "medium"
```

## Deployment Checklist

- [ ] Clone 1pro repository to production server
- [ ] Update docker-compose.yml with correct ports
- [ ] Configure environment variables
- [ ] Set up networking between services
- [ ] Test API connectivity from your backend
- [ ] Configure monitoring and logging
- [ ] Set up health checks
- [ ] Test failover scenarios
- [ ] Configure caching if needed
- [ ] Set up backup and recovery

## Troubleshooting

### Common Issues

1. **Port Conflicts**: Change 1pro port from 8000 to 8001
2. **Network Issues**: Ensure services are on same Docker network
3. **Timeout Issues**: Increase timeout to 60+ seconds for ML processing
4. **Memory Issues**: Allocate at least 1GB RAM for ML models

### Debug Commands

```bash
# Check service health
curl http://localhost:8001/health

# View logs
docker-compose logs ocean-hazard-api

# Test connectivity
docker exec -it your-backend-container curl http://ocean-hazard-api:8000/health
```

## Support

For integration issues, check:
1. API documentation at `http://localhost:8001/docs`
2. Service logs for error details
3. Network connectivity between services
4. Resource allocation and performance metrics

---

**Ready for Integration!** 🚀

The 1pro Ocean Hazard Analysis System is designed to integrate seamlessly with your existing infrastructure. Choose the integration approach that best fits your architecture and requirements.