# API Documentation

## Ocean Hazard Platform - Model A API

### Overview
The Model A API provides real-time multilingual hazard detection and extraction capabilities for the Ocean Hazard Platform. It processes social media posts, government alerts, and user reports to identify ocean-related hazards and extract actionable intelligence.

### Base URL
```
http://localhost:8000
```

### Authentication
Currently, the API does not require authentication. In production, implement appropriate authentication mechanisms.

## Endpoints

### Health Check
**GET** `/health`

Check the health status of the ML service.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-16T10:30:00Z",
  "version": "1.0.0",
  "models_loaded": {
    "nlp_engine": true,
    "geolocation_extractor": true
  }
}
```

### Single Report Analysis
**POST** `/analyze/single`

Analyze a single hazard report.

**Request Body:**
```json
{
  "content": "Heavy waves hitting Mumbai coastline, flooding in Bandra area",
  "source": "twitter",
  "timestamp": "2025-09-16T10:30:00Z",
  "location": {
    "lat": 19.0596,
    "lon": 72.8295
  },
  "metadata": {
    "user_id": "user123",
    "tweet_id": "1234567890"
  },
  "media_urls": ["https://example.com/image.jpg"]
}
```

**Response:**
```json
{
  "report_id": "api_20250916_103000_1234",
  "processed_report": {
    "id": "api_20250916_103000_1234",
    "original_content": "Heavy waves hitting Mumbai coastline, flooding in Bandra area",
    "normalized_content": "heavy waves hitting mumbai coastline flooding in bandra area",
    "detected_language": "en",
    "confidence_score": 0.95
  },
  "hazard_analysis": {
    "report_id": "api_20250916_103000_1234",
    "is_hazard": true,
    "hazard_type": "high_waves",
    "confidence": 0.87,
    "severity": "high",
    "urgency": "immediate",
    "sentiment": "negative",
    "sentiment_score": 0.25,
    "misinformation_probability": 0.1,
    "entities": [
      {
        "text": "Mumbai",
        "label": "LOCATION",
        "coordinates": [19.0760, 72.8777],
        "confidence": 0.9
      }
    ],
    "processing_metadata": {
      "analyzed_at": "2025-09-16T10:30:01Z",
      "language": "en",
      "analysis_version": "1.0"
    }
  },
  "geolocation": {
    "report_id": "api_20250916_103000_1234",
    "extracted_locations": [
      {
        "text": "Mumbai",
        "label": "GPE",
        "coordinates": [19.0760, 72.8777],
        "confidence": 0.9,
        "address": "Mumbai, Maharashtra, India"
      }
    ],
    "primary_location": {
      "text": "Mumbai",
      "coordinates": [19.0760, 72.8777],
      "address": "Mumbai, Maharashtra, India"
    },
    "confidence_score": 0.9,
    "processing_metadata": {
      "processed_at": "2025-09-16T10:30:01Z"
    }
  },
  "processing_time_ms": 1250.5,
  "status": "success"
}
```

### Batch Report Analysis
**POST** `/analyze/batch`

Analyze multiple reports in a single request.

**Request Body:**
```json
{
  "reports": [
    {
      "content": "Tsunami warning issued for Tamil Nadu coast",
      "source": "government_alert",
      "timestamp": "2025-09-16T10:00:00Z"
    },
    {
      "content": "Strange waves at Marina Beach Chennai",
      "source": "user_report", 
      "timestamp": "2025-09-16T10:05:00Z"
    }
  ],
  "process_anomalies": true
}
```

**Response:**
```json
{
  "batch_id": "batch_20250916_103000",
  "total_reports": 2,
  "processed_reports": 2,
  "hazard_predictions": 2,
  "geolocation_results": 2,
  "anomaly_alerts": 1,
  "processing_time_ms": 3200.0,
  "results": {
    "hazard_analyses": [...],
    "geolocation_results": [...],
    "anomaly_alerts": [
      {
        "alert_id": "spatial_1_20250916_103001",
        "alert_type": "geographic_cluster",
        "severity": "high",
        "location": "Chennai",
        "coordinates": [13.0827, 80.2707],
        "detection_time": "2025-09-16T10:30:01Z",
        "message": "Spatial cluster detected: 5 reports within 2.5km radius near Chennai",
        "metrics": {
          "cluster_size": 5,
          "radius_km": 2.5,
          "density_per_km2": 0.25
        },
        "affected_reports": ["report1", "report2"],
        "confidence_score": 0.8
      }
    ]
  },
  "status": "success"
}
```

### Recent Anomalies
**GET** `/anomalies/recent?hours=24`

Get recent anomaly alerts.

**Parameters:**
- `hours` (optional): Time window in hours (default: 24)

**Response:**
```json
{
  "time_window_hours": 24,
  "total_alerts": 3,
  "alerts": [
    {
      "alert_id": "volume_spike_20250916_090000",
      "alert_type": "volume_spike",
      "severity": "medium",
      "location": null,
      "coordinates": null,
      "detection_time": "2025-09-16T09:00:00Z",
      "message": "Volume spike detected: 25 reports vs 8.5 baseline (Z-score: 2.8)",
      "metrics": {
        "current_count": 25,
        "baseline_mean": 8.5,
        "z_score": 2.8
      },
      "affected_reports": ["report1", "report2", "..."],
      "confidence_score": 0.7
    }
  ]
}
```

### Submit Feedback
**POST** `/feedback/submit`

Submit operator feedback for model improvement.

**Request Body:**
```json
{
  "report_id": "api_20250916_103000_1234",
  "operator_id": "operator123",
  "feedback_type": "hazard_classification",
  "action": "update_label",
  "original_prediction": {
    "hazard_type": "high_waves",
    "confidence": 0.87
  },
  "corrected_prediction": {
    "hazard_type": "storm_surge",
    "confidence": 0.95
  },
  "confidence_score": 0.9,
  "notes": "Misclassified storm surge as high waves"
}
```

**Response:**
```json
{
  "feedback_id": "fb_20250916_103001_uuid",
  "status": "submitted",
  "timestamp": "2025-09-16T10:30:01Z"
}
```

### Pipeline Statistics
**GET** `/stats/pipeline`

Get ML pipeline performance statistics.

**Response:**
```json
{
  "preprocessing": {
    "total_processed": 1000,
    "language_distribution": {
      "en": 450,
      "hi": 300,
      "ta": 150,
      "te": 100
    },
    "processing_errors": 5
  },
  "nlp_analysis": {
    "total_analyzed": 995,
    "hazards_detected": 387,
    "high_urgency_reports": 45,
    "misinformation_flagged": 23
  },
  "geolocation": {
    "total_processed": 995,
    "entities_extracted": 1250,
    "successfully_geocoded": 980,
    "geocoding_errors": 15
  },
  "anomaly_detection": {
    "total_alerts_generated": 12,
    "alerts_by_type": {
      "volume_spike": 5,
      "geographic_cluster": 4,
      "hazard_type_anomaly": 3
    }
  },
  "feedback": {
    "total_feedback_received": 45,
    "average_operator_confidence": 0.85
  },
  "timestamp": "2025-09-16T10:30:00Z"
}
```

### Performance Metrics
**GET** `/feedback/metrics?days=30`

Get model performance metrics from operator feedback.

**Parameters:**
- `days` (optional): Time window in days (default: 30)

**Response:**
```json
{
  "metrics_period_days": 30,
  "performance_metrics": {
    "hazard_classification": {
      "accuracy": 0.87,
      "precision": 0.89,
      "recall": 0.85,
      "f1_score": 0.87,
      "total_feedback": 150
    }
  },
  "retraining_recommendations": [
    {
      "model_component": "hazard_classification",
      "priority": "medium",
      "reason": "Sufficient feedback data available: 150 samples",
      "recommended_action": "schedule_retraining_next_cycle",
      "estimated_improvement": 0.05
    }
  ],
  "timestamp": "2025-09-16T10:30:00Z"
}
```

### Data Ingestion Status
**GET** `/ingestion/status`

Get status of data ingestion processes.

**Response:**
```json
{
  "ingestion_statistics": {
    "total_ingested": 5000,
    "twitter_reports": 3000,
    "government_alerts": 500,
    "user_reports": 1500,
    "ingestion_errors": 25
  },
  "active_sources": {
    "twitter": "enabled",
    "government_alerts": "enabled",
    "user_reports": "enabled"
  },
  "timestamp": "2025-09-16T10:30:00Z"
}
```

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common HTTP status codes:
- `400 Bad Request`: Invalid input data
- `422 Unprocessable Entity`: Validation errors
- `500 Internal Server Error`: Server processing errors

## Data Types

### Hazard Types
- `tsunami`
- `storm_surge`
- `high_waves`
- `coastal_flooding`
- `swell_surge`
- `rip_current`
- `erosion`
- `other`

### Severity Levels
- `low`
- `medium`
- `high`
- `critical`
- `immediate`

### Urgency Levels
- `low`
- `medium`
- `high`
- `immediate`

### Sentiment Categories
- `positive`
- `neutral`
- `negative`

### Feedback Types
- `hazard_classification`
- `sentiment_analysis`
- `misinformation_detection`
- `location_extraction`
- `anomaly_detection`
- `overall_accuracy`

### Feedback Actions
- `correct`
- `incorrect`
- `partially_correct`
- `missing`
- `false_positive`
- `update_label`

## Rate Limits

Current implementation has no rate limits. In production, implement appropriate rate limiting based on usage patterns.

## SDK Examples

### Python SDK Usage
```python
import requests
import json

# Analyze single report
report_data = {
    "content": "Heavy waves at Juhu Beach Mumbai",
    "source": "user_report",
    "metadata": {"user_id": "user123"}
}

response = requests.post(
    "http://localhost:8000/analyze/single",
    json=report_data
)

result = response.json()
print(f"Hazard detected: {result['hazard_analysis']['is_hazard']}")
print(f"Location: {result['geolocation']['primary_location']['text']}")
```

### JavaScript SDK Usage
```javascript
// Analyze single report
const reportData = {
  content: "Tsunami warning for Visakhapatnam coast",
  source: "government_alert",
  timestamp: new Date().toISOString()
};

fetch('http://localhost:8000/analyze/single', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(reportData)
})
.then(response => response.json())
.then(data => {
  console.log('Hazard Analysis:', data.hazard_analysis);
  console.log('Location:', data.geolocation.primary_location);
});
```

## Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Check service health
curl http://localhost:8000/health

# View logs
docker-compose logs -f model-a-ml
```

### Environment Variables
Set these environment variables for production:
- `MONGODB_URL`: MongoDB connection string
- `DATABASE_NAME`: Database name
- `API_HOST`: API host (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)
- `MODEL_ENV`: Environment (production/development)

## Support

For technical support or questions about the API, contact the AI/ML development team.