# API Documentation

## Ocean Hazard Prediction API v1.0

### Base URL
```
http://localhost:8000
```

### Authentication
Currently, the API does not require authentication for development purposes. In production, implement API key authentication.

## API Endpoints

### System Management

#### GET /
Get system information and available endpoints.

**Response:**
```json
{
  "message": "Ocean Hazard Prediction API",
  "version": "1.0.0",
  "status": "operational",
  "timestamp": "2024-01-20T10:30:00",
  "endpoints": {
    "docs": "/docs",
    "health": "/health",
    "risk_assessment": "/risk/assess",
    "batch_assessment": "/risk/batch",
    "hotspots": "/hotspots/identify",
    "alerts": "/alerts"
  }
}
```

#### GET /health
System health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:30:00",
  "components": {
    "data_aggregation": "operational",
    "feature_engineering": "operational",
    "predictive_modeling": "operational",
    "risk_scoring": "operational",
    "hotspot_mapping": "operational",
    "alert_generation": "operational"
  }
}
```

### Risk Assessment

#### POST /risk/assess
Assess risk for a single location.

**Request Body:**
```json
{
  "location": {
    "latitude": 35.6762,
    "longitude": 139.6503
  },
  "hazard_type": "tsunami",
  "time_window": "short_term",
  "include_environmental": true,
  "include_historical": true
}
```

**Parameters:**
- `location.latitude` (float): Latitude in degrees (-90 to 90)
- `location.longitude` (float): Longitude in degrees (-180 to 180)
- `hazard_type` (string): Type of hazard ("tsunami", "storm_surge", "coastal_flooding")
- `time_window` (string): Prediction window ("short_term", "medium_term", "long_term")
- `include_environmental` (boolean): Include environmental data
- `include_historical` (boolean): Include historical analysis

**Response:**
```json
{
  "success": true,
  "risk_assessment": {
    "overall_risk_score": 0.75,
    "risk_level": "High",
    "component_scores": {
      "temporal_risk": 0.65,
      "spatial_risk": 0.80,
      "environmental_risk": 0.70,
      "historical_risk": 0.85
    },
    "confidence_score": 0.88,
    "prediction_details": {
      "hazard_type": "tsunami",
      "time_window": "short_term",
      "location": [35.6762, 139.6503]
    },
    "factors": [
      "High seismic activity in region",
      "Coastal vulnerability assessment indicates high risk",
      "Historical tsunami events recorded"
    ]
  },
  "assessment_timestamp": "2024-01-20T10:30:00"
}
```

#### POST /risk/batch
Perform risk assessment for multiple locations.

**Request Body:**
```json
{
  "locations": [
    {
      "latitude": 35.6762,
      "longitude": 139.6503
    },
    {
      "latitude": 34.0522,
      "longitude": -118.2437
    }
  ],
  "hazard_type": "tsunami",
  "time_window": "short_term"
}
```

**Response:**
```json
{
  "success": true,
  "batch_results": [
    {
      "location": [35.6762, 139.6503],
      "risk_assessment": {
        "overall_risk_score": 0.75,
        "risk_level": "High"
      }
    },
    {
      "location": [34.0522, -118.2437],
      "risk_assessment": {
        "overall_risk_score": 0.45,
        "risk_level": "Medium"
      }
    }
  ],
  "total_locations": 2,
  "assessment_timestamp": "2024-01-20T10:30:00"
}
```

### Hotspot Analysis

#### POST /hotspots/identify
Identify risk hotspots based on recent assessments.

**Query Parameters:**
- `min_risk_score` (float): Minimum risk score for hotspots (default: 0.6)
- `min_events` (int): Minimum events for hotspot (default: 3)
- `spatial_radius` (float): Spatial radius in km (default: 50)
- `hazard_type` (string): Hazard type to analyze (default: "tsunami")

**Response:**
```json
{
  "success": true,
  "hotspots": [
    {
      "hotspot_id": "hotspot_001",
      "center_location": [35.6762, 139.6503],
      "radius_km": 25.5,
      "risk_score": 0.85,
      "event_count": 12,
      "affected_locations": [
        [35.6762, 139.6503],
        [35.6895, 139.6917]
      ]
    }
  ],
  "total_hotspots": 1,
  "criteria": {
    "min_risk_score": 0.6,
    "min_events": 3,
    "spatial_radius_km": 50
  },
  "identified_at": "2024-01-20T10:30:00"
}
```

#### POST /hotspots/map
Create visual hotspot map.

**Response:**
```json
{
  "success": true,
  "message": "Hotspot map creation started",
  "hotspots_count": 3,
  "timestamp": "2024-01-20T10:30:00"
}
```

### Alert Management

#### POST /alerts/generate
Generate alert based on risk assessment.

**Request Body:**
```json
{
  "location": {
    "latitude": 35.6762,
    "longitude": 139.6503
  },
  "hazard_type": "tsunami",
  "time_window": "short_term"
}
```

**Response:**
```json
{
  "success": true,
  "alert_result": {
    "alert_generated": true,
    "alert_id": "alert_20240120_103000",
    "alert_level": "HIGH",
    "notification_channels": ["email", "webhook"],
    "delivery_status": {
      "email": "sent",
      "webhook": "sent",
      "sms": "not_configured"
    }
  },
  "risk_assessment": {
    "overall_risk_score": 0.85,
    "risk_level": "High"
  },
  "generated_at": "2024-01-20T10:30:00"
}
```

#### GET /alerts/active
Get currently active alerts.

**Query Parameters:**
- `risk_level` (string): Filter by risk level ("Low", "Medium", "High", "Critical")
- `hazard_type` (string): Filter by hazard type

**Response:**
```json
{
  "success": true,
  "active_alerts": [
    {
      "alert_id": "alert_20240120_103000",
      "location": [35.6762, 139.6503],
      "hazard_type": "tsunami",
      "alert_level": "HIGH",
      "created_at": "2024-01-20T10:30:00",
      "status": "active",
      "expires_at": "2024-01-20T16:30:00"
    }
  ],
  "total_count": 1,
  "filters": {
    "risk_level": null,
    "hazard_type": null
  },
  "retrieved_at": "2024-01-20T10:30:00"
}
```

#### GET /alerts/summary
Get alert summary for specified time period.

**Query Parameters:**
- `time_period_hours` (int): Time period for summary in hours (default: 24)

**Response:**
```json
{
  "success": true,
  "summary": {
    "total_alerts": 15,
    "by_level": {
      "LOW": 5,
      "MEDIUM": 7,
      "HIGH": 2,
      "CRITICAL": 1
    },
    "by_hazard_type": {
      "tsunami": 8,
      "storm_surge": 5,
      "coastal_flooding": 2
    },
    "delivery_stats": {
      "email_success_rate": 0.98,
      "webhook_success_rate": 0.95,
      "sms_success_rate": 0.92
    },
    "time_period_hours": 24
  }
}
```

#### POST /alerts/configure
Configure alert notification channels.

**Request Body:**
```json
{
  "channel_type": "email",
  "configuration": {
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "username": "alerts@example.com",
    "password": "app_password",
    "recipients": ["admin@example.com", "ops@example.com"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully configured email channel",
  "configured_at": "2024-01-20T10:30:00"
}
```

### Data Management

#### GET /data/collect
Trigger data collection from various sources.

**Query Parameters:**
- `data_type` (string): Type of data to collect ("all", "historical", "sensor", "geospatial", "social")
- `location_lat` (float): Location latitude (optional)
- `location_lon` (float): Location longitude (optional)
- `radius_km` (float): Collection radius in km (default: 100)

**Response:**
```json
{
  "success": true,
  "collection_results": {
    "historical": {
      "events_collected": 150,
      "status": "success"
    },
    "sensor": {
      "readings_collected": 2500,
      "status": "success"
    },
    "geospatial": {
      "data_points": 1,
      "status": "success"
    },
    "social": {
      "signals_collected": 75,
      "status": "success"
    }
  },
  "collected_at": "2024-01-20T10:30:00"
}
```

### Model Management

#### POST /models/train
Trigger model training pipeline.

**Response:**
```json
{
  "success": true,
  "message": "Model training started in background",
  "started_at": "2024-01-20T10:30:00"
}
```

#### GET /models/status
Get status of trained models.

**Response:**
```json
{
  "success": true,
  "model_status": {
    "time_series_models": 3,
    "classification_models": 5,
    "ensemble_models": 2,
    "clustering_models": 2,
    "last_training": "2024-01-19T15:45:00",
    "model_performance": "Available through training reports"
  },
  "checked_at": "2024-01-20T10:30:00"
}
```

### Data Export

#### GET /export/data
Export system data for analysis or backup.

**Query Parameters:**
- `data_type` (string): Type of data to export ("all", "alerts", "hotspots")
- `format` (string): Export format ("json", "csv", "xml")

**Response:**
```json
{
  "success": true,
  "exported_files": [
    {
      "type": "alerts",
      "file": "/exports/alerts_20240120.json"
    },
    {
      "type": "hotspots",
      "file": "/exports/hotspots_20240120.json"
    }
  ],
  "export_format": "json",
  "exported_at": "2024-01-20T10:30:00"
}
```

## Error Handling

### Error Response Format
```json
{
  "detail": "Error message describing what went wrong",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": "2024-01-20T10:30:00"
}
```

### Common HTTP Status Codes
- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Example Error Response
```json
{
  "detail": "Latitude must be between -90 and 90 degrees",
  "error_code": "INVALID_LATITUDE",
  "timestamp": "2024-01-20T10:30:00"
}
```

## Rate Limiting

### Current Limits
- General API endpoints: 10 requests per second per IP
- Documentation endpoints: 5 requests per second per IP
- Health check: No limit

### Rate Limit Headers
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 8
X-RateLimit-Reset: 1642680600
```

## SDKs and Libraries

### Python SDK Example
```python
import requests

class OceanHazardAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def assess_risk(self, latitude, longitude, hazard_type="tsunami"):
        response = requests.post(
            f"{self.base_url}/risk/assess",
            json={
                "location": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "hazard_type": hazard_type
            }
        )
        return response.json()

# Usage
api = OceanHazardAPI()
result = api.assess_risk(35.6762, 139.6503)
print(f"Risk Level: {result['risk_assessment']['risk_level']}")
```

### JavaScript/Node.js Example
```javascript
const axios = require('axios');

class OceanHazardAPI {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async assessRisk(latitude, longitude, hazardType = 'tsunami') {
        const response = await axios.post(`${this.baseUrl}/risk/assess`, {
            location: {
                latitude: latitude,
                longitude: longitude
            },
            hazard_type: hazardType
        });
        return response.data;
    }
}

// Usage
const api = new OceanHazardAPI();
api.assessRisk(35.6762, 139.6503)
   .then(result => {
       console.log(`Risk Level: ${result.risk_assessment.risk_level}`);
   });
```

## Webhook Integration

### Webhook Payload Format
When alerts are generated, webhooks are called with:

```json
{
  "event_type": "alert_generated",
  "alert": {
    "alert_id": "alert_20240120_103000",
    "location": [35.6762, 139.6503],
    "hazard_type": "tsunami",
    "alert_level": "HIGH",
    "risk_score": 0.85,
    "created_at": "2024-01-20T10:30:00"
  },
  "timestamp": "2024-01-20T10:30:00",
  "signature": "sha256=webhook_signature"
}
```

### Webhook Verification
Verify webhook authenticity using HMAC-SHA256:

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected_signature}", signature)
```

---

For more detailed information, visit the interactive API documentation at `/docs` when the server is running.