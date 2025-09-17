"""
API integration tests for Model1.
Testing the FastAPI endpoints and API functionality.
"""

import pytest
import asyncio
import json
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

# Import API server
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.api_server import app


class TestAPIEndpoints:
    """Test all API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_process_single_report_valid(self, client):
        """Test processing a single valid report."""
        # Mock all the ML components
        with patch('api.api_server.preprocessing_pipeline') as mock_preprocess, \
             patch('api.api_server.nlp_engine') as mock_nlp, \
             patch('api.api_server.geolocation_extractor') as mock_geo, \
             patch('api.api_server.anomaly_detector') as mock_anomaly:
            
            # Setup mocks
            mock_preprocess.process_text.return_value = Mock(
                original_text="Tsunami warning for Chennai",
                processed_text="tsunami warning chennai",
                language="en",
                confidence=0.95
            )
            
            mock_nlp.analyze_hazard.return_value = Mock(
                is_hazard=True,
                hazard_type="tsunami",
                confidence=0.92,
                urgency_score=0.88,
                sentiment="panic"
            )
            
            mock_geo.extract_locations.return_value = [
                Mock(
                    location="Chennai",
                    coordinates=(13.0827, 80.2707),
                    confidence=0.94,
                    location_type="city"
                )
            ]
            
            # Test request
            test_data = {
                "text": "Tsunami warning issued for Chennai coast",
                "source": "twitter",
                "timestamp": datetime.now().isoformat()
            }
            
            response = client.post("/api/v1/process/single", json=test_data)
            assert response.status_code == 200
            
            result = response.json()
            assert result["success"] == True
            assert result["is_hazard"] == True
            assert result["hazard_type"] == "tsunami"
            assert "locations" in result
            assert len(result["locations"]) >= 1
    
    def test_process_single_report_invalid_input(self, client):
        """Test processing with invalid input."""
        # Missing required fields
        test_data = {
            "text": "",  # Empty text
            "source": "twitter"
            # Missing timestamp
        }
        
        response = client.post("/api/v1/process/single", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_process_batch_reports(self, client):
        """Test batch processing of multiple reports."""
        with patch('api.api_server.preprocessing_pipeline') as mock_preprocess, \
             patch('api.api_server.nlp_engine') as mock_nlp, \
             patch('api.api_server.geolocation_extractor') as mock_geo:
            
            # Setup mocks for batch processing
            mock_preprocess.process_text.side_effect = [
                Mock(processed_text="tsunami warning chennai", language="en", confidence=0.95),
                Mock(processed_text="high waves mumbai", language="en", confidence=0.93)
            ]
            
            mock_nlp.analyze_hazard.side_effect = [
                Mock(is_hazard=True, hazard_type="tsunami", confidence=0.92),
                Mock(is_hazard=True, hazard_type="high_waves", confidence=0.85)
            ]
            
            mock_geo.extract_locations.side_effect = [
                [Mock(location="Chennai", coordinates=(13.0827, 80.2707))],
                [Mock(location="Mumbai", coordinates=(19.0760, 72.8777))]
            ]
            
            # Test batch request
            test_data = {
                "reports": [
                    {
                        "text": "Tsunami warning for Chennai",
                        "source": "twitter",
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "text": "High waves at Mumbai beach",
                        "source": "user_report",
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            }
            
            response = client.post("/api/v1/process/batch", json=test_data)
            assert response.status_code == 200
            
            result = response.json()
            assert result["success"] == True
            assert "results" in result
            assert len(result["results"]) == 2
            assert result["processed_count"] == 2
    
    def test_get_statistics(self, client):
        """Test statistics endpoint."""
        with patch('api.api_server.data_manager') as mock_data:
            mock_data.get_statistics.return_value = {
                "total_reports": 1250,
                "hazard_reports": 890,
                "non_hazard_reports": 360,
                "accuracy": 0.94,
                "avg_processing_time": 1.2,
                "most_common_hazards": ["tsunami", "high_waves", "storm_surge"],
                "top_locations": ["Chennai", "Mumbai", "Kolkata"]
            }
            
            response = client.get("/api/v1/statistics")
            assert response.status_code == 200
            
            data = response.json()
            assert "total_reports" in data
            assert "hazard_reports" in data
            assert "accuracy" in data
            assert data["accuracy"] > 0.9
    
    def test_anomaly_alerts_endpoint(self, client):
        """Test anomaly alerts endpoint."""
        with patch('api.api_server.anomaly_detector') as mock_anomaly:
            mock_anomaly.get_recent_alerts.return_value = [
                {
                    "id": "alert_001",
                    "type": "spatial_cluster",
                    "severity": "high",
                    "location": "Chennai",
                    "coordinates": [13.0827, 80.2707],
                    "confidence": 0.91,
                    "timestamp": datetime.now().isoformat(),
                    "description": "Unusual cluster of tsunami reports"
                }
            ]
            
            response = client.get("/api/v1/anomalies/recent")
            assert response.status_code == 200
            
            data = response.json()
            assert "alerts" in data
            assert len(data["alerts"]) >= 1
            assert data["alerts"][0]["severity"] == "high"
    
    def test_feedback_submission(self, client):
        """Test feedback submission endpoint."""
        with patch('api.api_server.feedback_system') as mock_feedback:
            mock_feedback.submit_feedback.return_value = {
                "feedback_id": "fb_001",
                "status": "accepted",
                "message": "Feedback recorded successfully"
            }
            
            feedback_data = {
                "report_id": "report_123",
                "feedback_type": "correction",
                "original_prediction": {
                    "is_hazard": True,
                    "hazard_type": "tsunami"
                },
                "correct_prediction": {
                    "is_hazard": False,
                    "hazard_type": "none"
                },
                "operator_id": "operator_001",
                "comments": "False alarm - just high tide"
            }
            
            response = client.post("/api/v1/feedback", json=feedback_data)
            assert response.status_code == 200
            
            result = response.json()
            assert result["status"] == "accepted"
            assert "feedback_id" in result
    
    def test_export_data_geojson(self, client):
        """Test GeoJSON data export."""
        with patch('api.api_server.data_manager') as mock_data:
            mock_data.export_geojson.return_value = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [80.2707, 13.0827]
                        },
                        "properties": {
                            "hazard_type": "tsunami",
                            "severity": "high",
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                ]
            }
            
            response = client.get("/api/v1/export/geojson?hours=24")
            assert response.status_code == 200
            
            data = response.json()
            assert data["type"] == "FeatureCollection"
            assert "features" in data
            assert len(data["features"]) >= 1
    
    def test_multilingual_processing(self, client):
        """Test multilingual text processing."""
        with patch('api.api_server.preprocessing_pipeline') as mock_preprocess, \
             patch('api.api_server.nlp_engine') as mock_nlp:
            
            mock_preprocess.process_text.return_value = Mock(
                original_text="चेन्नई में सुनामी की चेतावनी",
                processed_text="chennai tsunami warning",
                language="hi",
                confidence=0.89
            )
            
            mock_nlp.analyze_hazard.return_value = Mock(
                is_hazard=True,
                hazard_type="tsunami",
                confidence=0.87
            )
            
            # Hindi text
            test_data = {
                "text": "चेन्नई में सुनामी की चेतावनी जारी की गई है",
                "source": "government_alert",
                "timestamp": datetime.now().isoformat()
            }
            
            response = client.post("/api/v1/process/single", json=test_data)
            assert response.status_code == 200
            
            result = response.json()
            assert result["success"] == True
            assert result["language"] == "hi"
            assert result["is_hazard"] == True
    
    def test_rate_limiting(self, client):
        """Test API rate limiting."""
        # This would test rate limiting if implemented
        # For now, just test that multiple requests work
        for i in range(5):
            response = client.get("/health")
            assert response.status_code == 200
    
    def test_error_handling(self, client):
        """Test API error handling."""
        # Test malformed JSON
        response = client.post(
            "/api/v1/process/single", 
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        
        # Test internal server error handling
        with patch('api.api_server.preprocessing_pipeline') as mock_preprocess:
            mock_preprocess.process_text.side_effect = Exception("Internal error")
            
            test_data = {
                "text": "Test text",
                "source": "test",
                "timestamp": datetime.now().isoformat()
            }
            
            response = client.post("/api/v1/process/single", json=test_data)
            assert response.status_code == 500


class TestAPIPerformance:
    """Test API performance and load handling."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_response_times(self, client):
        """Test API response times."""
        import time
        
        # Mock fast responses
        with patch('api.api_server.preprocessing_pipeline'), \
             patch('api.api_server.nlp_engine'), \
             patch('api.api_server.geolocation_extractor'):
            
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            
            assert response.status_code == 200
            assert (end_time - start_time) < 1.0  # Should respond within 1 second
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get("/health")
            results.append(response.status_code)
        
        # Create 10 concurrent threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(results) == 10
        assert all(status == 200 for status in results)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])