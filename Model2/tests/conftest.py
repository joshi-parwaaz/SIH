"""Test configuration and fixtures for Ocean Hazard Prediction System."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os

# Test data fixtures
@pytest.fixture
def sample_location():
    """Sample location coordinates."""
    return (35.6762, 139.6503)  # Tokyo

@pytest.fixture
def sample_locations():
    """Multiple sample locations for batch testing."""
    return [
        (35.6762, 139.6503),  # Tokyo
        (34.0522, -118.2437), # Los Angeles
        (-6.2088, 106.8456),  # Jakarta
        (37.7749, -122.4194), # San Francisco
        (40.7128, -74.0060)   # New York
    ]

@pytest.fixture
def sample_historical_data():
    """Sample historical event data."""
    return [
        {
            'event_id': 'test_tsunami_001',
            'hazard_type': 'tsunami',
            'location': (35.6762, 139.6503),
            'event_date': datetime.now() - timedelta(days=365),
            'magnitude': 8.5,
            'description': 'Test tsunami event',
            'casualties': 1000,
            'economic_impact': 50000000
        },
        {
            'event_id': 'test_storm_surge_001',
            'hazard_type': 'storm_surge',
            'location': (34.0522, -118.2437),
            'event_date': datetime.now() - timedelta(days=180),
            'magnitude': 4,
            'description': 'Test storm surge event',
            'casualties': 50,
            'economic_impact': 5000000
        }
    ]

@pytest.fixture
def sample_sensor_data():
    """Sample sensor readings."""
    return {
        'sensor_id': 'test_buoy_001',
        'sensor_type': 'ocean_buoy',
        'location': (35.6762, 139.6503),
        'timestamp': datetime.now(),
        'readings': {
            'wave_height': 2.5,
            'sea_temperature': 25.0,
            'wind_speed': 15.0,
            'atmospheric_pressure': 1013.0
        },
        'status': 'active'
    }

@pytest.fixture
def sample_training_data():
    """Sample training data for ML models."""
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'latitude': np.random.uniform(-90, 90, n_samples),
        'longitude': np.random.uniform(-180, 180, n_samples),
        'sea_temperature': np.random.uniform(15, 30, n_samples),
        'wave_height': np.random.uniform(0.5, 8, n_samples),
        'wind_speed': np.random.uniform(5, 50, n_samples),
        'historical_events': np.random.randint(0, 10, n_samples),
        'coastal_distance': np.random.uniform(0, 100, n_samples),
        'elevation': np.random.uniform(-10, 100, n_samples),
        'risk_level': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_samples)
    })
    
    return data

@pytest.fixture
def test_config():
    """Test configuration dictionary."""
    return {
        'model': {
            'ensemble_size': 3,
            'validation_split': 0.2,
            'random_state': 42
        },
        'risk_assessment': {
            'weights': {
                'temporal': 0.25,
                'spatial': 0.25,
                'environmental': 0.25,
                'historical': 0.25
            },
            'thresholds': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8
            }
        },
        'hotspot_detection': {
            'min_risk_score': 0.6,
            'min_events': 3,
            'spatial_radius_km': 50,
            'temporal_window_hours': 24
        },
        'alerts': {
            'cooldown_minutes': 30,
            'max_alerts_per_hour': 10,
            'notification_channels': ['email', 'webhook']
        }
    }

@pytest.fixture
def temp_directory():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_api_responses():
    """Mock API responses for external services."""
    return {
        'weather_api': {
            'temperature': 25.0,
            'wind_speed': 15.0,
            'atmospheric_pressure': 1013.0,
            'humidity': 65.0
        },
        'earthquake_api': {
            'magnitude': 6.5,
            'depth': 10.0,
            'location': (35.6762, 139.6503),
            'timestamp': datetime.now().isoformat()
        },
        'social_media_api': {
            'signals': [
                {
                    'platform': 'twitter',
                    'content': 'Earthquake felt in Tokyo',
                    'location': (35.6762, 139.6503),
                    'timestamp': datetime.now().isoformat(),
                    'sentiment': 'concern'
                }
            ]
        }
    }

@pytest.fixture
def sample_risk_assessment():
    """Sample risk assessment result."""
    return {
        'overall_risk_score': 0.75,
        'risk_level': 'High',
        'component_scores': {
            'temporal_risk': 0.65,
            'spatial_risk': 0.80,
            'environmental_risk': 0.70,
            'historical_risk': 0.85
        },
        'confidence_score': 0.88,
        'prediction_details': {
            'hazard_type': 'tsunami',
            'time_window': 'short_term',
            'location': (35.6762, 139.6503)
        },
        'factors': [
            'High seismic activity in region',
            'Coastal vulnerability assessment indicates high risk',
            'Historical tsunami events recorded'
        ]
    }

# Test utilities
class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_time_series_data(length=100, features=5):
        """Generate synthetic time series data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
        data = np.random.randn(length, features)
        return pd.DataFrame(data, index=dates, columns=[f'feature_{i}' for i in range(features)])
    
    @staticmethod
    def generate_spatial_data(n_points=100):
        """Generate synthetic spatial data points."""
        np.random.seed(42)
        return [
            (np.random.uniform(-90, 90), np.random.uniform(-180, 180))
            for _ in range(n_points)
        ]
    
    @staticmethod
    def generate_alert_data(n_alerts=10):
        """Generate synthetic alert data."""
        alerts = []
        for i in range(n_alerts):
            alerts.append({
                'alert_id': f'test_alert_{i:03d}',
                'location': (np.random.uniform(-90, 90), np.random.uniform(-180, 180)),
                'hazard_type': np.random.choice(['tsunami', 'storm_surge', 'coastal_flooding']),
                'alert_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']),
                'created_at': datetime.now() - timedelta(hours=np.random.randint(1, 72)),
                'status': np.random.choice(['active', 'resolved', 'expired'])
            })
        return alerts

# Mock classes for testing
class MockDataAggregator:
    """Mock data aggregator for testing."""
    
    def __init__(self, sample_data=None):
        self.sample_data = sample_data or {}
    
    def collect_historical_events(self, location, radius_km=100):
        return self.sample_data.get('historical', [])
    
    def collect_sensor_data(self):
        return self.sample_data.get('sensor', {})
    
    def collect_geospatial_data(self, location):
        return self.sample_data.get('geospatial', {})
    
    def collect_social_signals(self, location):
        return self.sample_data.get('social', [])

class MockMLModel:
    """Mock ML model for testing."""
    
    def __init__(self, prediction_value=0.75):
        self.prediction_value = prediction_value
        self.is_trained = False
    
    def fit(self, X, y):
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        return np.full(len(X), self.prediction_value)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained")
        proba = np.random.random((len(X), 2))
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

# Test markers
pytestmark = [
    pytest.mark.asyncio,  # For async tests
]

# Test configuration
pytest_plugins = ["pytest_asyncio"]