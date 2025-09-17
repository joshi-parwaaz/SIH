"""
Unit tests for Model1 ML components.
Testing individual components of the hazard detection system.
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Import Model1 components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.text_processor import PreprocessingPipeline, ProcessedReport
from nlp_analysis.hazard_analyzer import NLPAnalysisEngine, HazardPrediction
from geolocation.location_extractor import GeolocationExtractor, GeolocationResult
from anomaly_detection.anomaly_detector import AnomalyDetectionEngine, AnomalyAlert


class TestPreprocessingPipeline:
    """Test the text preprocessing pipeline."""
    
    @pytest.fixture
    def preprocessor(self):
        return PreprocessingPipeline()
    
    def test_language_detection_english(self, preprocessor):
        """Test English language detection."""
        text = "Tsunami warning issued for Chennai coast. Please evacuate immediately."
        result = preprocessor.detect_language(text)
        assert result['language'] == 'en'
        assert result['confidence'] > 0.8
    
    def test_language_detection_hindi(self, preprocessor):
        """Test Hindi language detection."""
        text = "चेन्नई तट के लिए सुनामी चेतावनी जारी की गई है। कृपया तुरंत निकलें।"
        result = preprocessor.detect_language(text)
        assert result['language'] == 'hi'
        assert result['confidence'] > 0.7
    
    def test_text_normalization(self, preprocessor):
        """Test text normalization and cleaning."""
        text = "🌊 URGENT!!! Tsunami warning for Mumbai beach 😱😱 #tsunami #emergency"
        normalized = preprocessor.normalize_text(text)
        
        # Should contain key hazard information
        assert 'tsunami' in normalized.lower()
        assert 'mumbai' in normalized.lower()
        assert 'warning' in normalized.lower()
        # Should remove excessive punctuation and emojis
        assert '!!!' not in normalized
        assert '😱' not in normalized
    
    def test_code_mixed_processing(self, preprocessor):
        """Test Hindi-English code-mixed text processing."""
        text = "Mumbai me tsunami ka warning hai, please evacuate करें"
        result = preprocessor.process_code_mixed_text(text)
        
        assert result['has_code_mixing'] == True
        assert 'tsunami' in result['processed_text'].lower()
        assert 'mumbai' in result['processed_text'].lower()
    
    @pytest.mark.asyncio
    async def test_full_preprocessing_pipeline(self, preprocessor):
        """Test complete preprocessing pipeline."""
        raw_text = "🚨 ALERT: High waves reported at Marina Beach Chennai! Stay away from coast 🌊"
        
        result = await preprocessor.process_text(raw_text)
        
        assert isinstance(result, ProcessedReport)
        assert result.original_text == raw_text
        assert result.processed_text is not None
        assert result.language is not None
        assert result.confidence > 0.5


class TestNLPAnalysisEngine:
    """Test the NLP analysis engine."""
    
    @pytest.fixture
    async def nlp_engine(self):
        engine = NLPAnalysisEngine()
        # Mock the model loading for faster tests
        with patch.object(engine, 'load_models', new_callable=AsyncMock):
            await engine.load_models()
        return engine
    
    @pytest.mark.asyncio
    async def test_hazard_classification_positive(self, nlp_engine):
        """Test hazard classification for positive cases."""
        text = "Tsunami warning issued for Chennai coast. High waves expected."
        
        with patch.object(nlp_engine, '_classify_hazard_type') as mock_classify:
            mock_classify.return_value = {
                'hazard_type': 'tsunami',
                'confidence': 0.92,
                'is_hazard': True
            }
            
            result = await nlp_engine.analyze_hazard(text)
            
            assert isinstance(result, HazardPrediction)
            assert result.is_hazard == True
            assert result.hazard_type == 'tsunami'
            assert result.confidence > 0.9
    
    @pytest.mark.asyncio
    async def test_hazard_classification_negative(self, nlp_engine):
        """Test hazard classification for non-hazard text."""
        text = "Beautiful sunset at Marina Beach today. Perfect weather for photography."
        
        with patch.object(nlp_engine, '_classify_hazard_type') as mock_classify:
            mock_classify.return_value = {
                'hazard_type': 'none',
                'confidence': 0.95,
                'is_hazard': False
            }
            
            result = await nlp_engine.analyze_hazard(text)
            
            assert result.is_hazard == False
            assert result.hazard_type == 'none'
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, nlp_engine):
        """Test sentiment analysis for urgency detection."""
        urgent_text = "URGENT! Tsunami waves hitting the shore NOW! Everyone run!"
        
        with patch.object(nlp_engine, '_analyze_sentiment') as mock_sentiment:
            mock_sentiment.return_value = {
                'sentiment': 'negative',
                'urgency_score': 0.95,
                'panic_level': 0.85
            }
            
            result = await nlp_engine.analyze_sentiment(urgent_text)
            
            assert result['urgency_score'] > 0.8
            assert result['panic_level'] > 0.7
    
    @pytest.mark.asyncio
    async def test_misinformation_detection(self, nlp_engine):
        """Test misinformation detection."""
        suspicious_text = "Scientists confirm aliens caused the tsunami in Chennai! Share to warn others!"
        
        with patch.object(nlp_engine, '_detect_misinformation') as mock_misinfo:
            mock_misinfo.return_value = {
                'is_misinformation': True,
                'confidence': 0.87,
                'warning_flags': ['unverified_claim', 'conspiracy_theory']
            }
            
            result = await nlp_engine.detect_misinformation(suspicious_text)
            
            assert result['is_misinformation'] == True
            assert result['confidence'] > 0.8


class TestGeolocationExtractor:
    """Test the geolocation extraction system."""
    
    @pytest.fixture
    def geo_extractor(self):
        return GeolocationExtractor()
    
    @pytest.mark.asyncio
    async def test_location_extraction_exact_match(self, geo_extractor):
        """Test exact location name extraction."""
        text = "Tsunami warning for Chennai Marina Beach area. Evacuate immediately."
        
        with patch.object(geo_extractor, '_extract_locations') as mock_extract:
            mock_extract.return_value = [
                {
                    'location': 'Chennai',
                    'confidence': 0.95,
                    'coordinates': (13.0827, 80.2707),
                    'type': 'city'
                },
                {
                    'location': 'Marina Beach',
                    'confidence': 0.88,
                    'coordinates': (13.0499, 80.2824),
                    'type': 'landmark'
                }
            ]
            
            result = await geo_extractor.extract_locations(text)
            
            assert isinstance(result, list)
            assert len(result) >= 1
            assert result[0]['location'] == 'Chennai'
            assert result[0]['coordinates'] is not None
    
    @pytest.mark.asyncio
    async def test_fuzzy_location_matching(self, geo_extractor):
        """Test fuzzy matching for misspelled locations."""
        text = "High waves at Puri bech, Odisha. Very dangerous situation."
        
        with patch.object(geo_extractor, '_fuzzy_location_match') as mock_fuzzy:
            mock_fuzzy.return_value = [
                {
                    'original': 'Puri bech',
                    'matched': 'Puri Beach',
                    'confidence': 0.82,
                    'coordinates': (19.8135, 85.8312),
                    'type': 'beach'
                }
            ]
            
            result = await geo_extractor.extract_locations(text)
            
            assert len(result) >= 1
            assert 'Puri' in result[0]['matched']
            assert result[0]['confidence'] > 0.8
    
    def test_coordinate_validation(self, geo_extractor):
        """Test coordinate validation for Indian coastal regions."""
        # Valid Indian coastal coordinates
        valid_coords = (19.0760, 72.8777)  # Mumbai
        assert geo_extractor._validate_coordinates(valid_coords) == True
        
        # Invalid coordinates (outside India)
        invalid_coords = (40.7128, -74.0060)  # New York
        assert geo_extractor._validate_coordinates(invalid_coords) == False


class TestAnomalyDetectionEngine:
    """Test the anomaly detection system."""
    
    @pytest.fixture
    def anomaly_detector(self):
        return AnomalyDetectionEngine()
    
    def test_spatial_clustering(self, anomaly_detector):
        """Test spatial clustering for anomaly detection."""
        # Create sample data points clustered around Chennai
        sample_data = pd.DataFrame({
            'latitude': [13.08, 13.09, 13.07, 13.08, 13.10],
            'longitude': [80.27, 80.28, 80.26, 80.29, 80.25],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='H'),
            'hazard_type': ['tsunami', 'tsunami', 'high_waves', 'tsunami', 'tsunami']
        })
        
        with patch.object(anomaly_detector, '_cluster_spatial_data') as mock_cluster:
            mock_cluster.return_value = {
                'clusters': [{'size': 4, 'center': (13.084, 80.272)}],
                'anomaly_score': 0.85
            }
            
            result = anomaly_detector.detect_spatial_anomalies(sample_data)
            
            assert result['anomaly_score'] > 0.8
            assert len(result['clusters']) >= 1
    
    def test_temporal_spike_detection(self, anomaly_detector):
        """Test temporal spike detection."""
        # Create sample data with a spike in reports
        timestamps = pd.date_range('2024-01-01', periods=24, freq='H')
        counts = [1, 2, 1, 3, 2, 1, 15, 18, 12, 2, 1, 3] + [1]*12  # Spike in hours 6-8
        
        sample_data = pd.DataFrame({
            'timestamp': timestamps,
            'report_count': counts
        })
        
        with patch.object(anomaly_detector, '_detect_temporal_spikes') as mock_spike:
            mock_spike.return_value = {
                'spike_detected': True,
                'spike_time': '2024-01-01 06:00:00',
                'spike_magnitude': 5.2,
                'baseline': 2.1
            }
            
            result = anomaly_detector.detect_temporal_anomalies(sample_data)
            
            assert result['spike_detected'] == True
            assert result['spike_magnitude'] > 3.0
    
    @pytest.mark.asyncio
    async def test_anomaly_alert_generation(self, anomaly_detector):
        """Test anomaly alert generation."""
        anomaly_data = {
            'anomaly_type': 'spatial_cluster',
            'severity': 'high',
            'location': 'Chennai',
            'coordinates': (13.0827, 80.2707),
            'confidence': 0.91,
            'timestamp': datetime.now()
        }
        
        alert = await anomaly_detector.generate_alert(anomaly_data)
        
        assert isinstance(alert, AnomalyAlert)
        assert alert.anomaly_type == 'spatial_cluster'
        assert alert.severity == 'high'
        assert alert.confidence > 0.9


class TestIntegration:
    """Integration tests for the complete ML pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self):
        """Test the complete pipeline from raw input to final output."""
        # Sample multilingual input
        raw_input = "🌊 Chennai marina beach mein tsunami aa raha hai! Bahut dangerous waves 😰"
        
        # Initialize all components
        preprocessor = PreprocessingPipeline()
        nlp_engine = NLPAnalysisEngine()
        geo_extractor = GeolocationExtractor()
        
        # Mock the heavy model loading
        with patch.object(nlp_engine, 'load_models', new_callable=AsyncMock), \
             patch.object(geo_extractor, 'load_models', new_callable=AsyncMock):
            
            await nlp_engine.load_models()
            await geo_extractor.load_models()
            
            # Step 1: Preprocessing
            processed = await preprocessor.process_text(raw_input)
            assert processed.original_text == raw_input
            assert processed.processed_text is not None
            
            # Step 2: NLP Analysis (mocked)
            with patch.object(nlp_engine, 'analyze_hazard') as mock_nlp:
                mock_nlp.return_value = HazardPrediction(
                    is_hazard=True,
                    hazard_type='tsunami',
                    confidence=0.89,
                    urgency_score=0.92,
                    sentiment='panic'
                )
                
                hazard_result = await nlp_engine.analyze_hazard(processed.processed_text)
                
                assert hazard_result.is_hazard == True
                assert hazard_result.hazard_type == 'tsunami'
                assert hazard_result.urgency_score > 0.8
            
            # Step 3: Geolocation extraction (mocked)
            with patch.object(geo_extractor, 'extract_locations') as mock_geo:
                mock_geo.return_value = [
                    GeolocationResult(
                        location='Chennai',
                        coordinates=(13.0827, 80.2707),
                        confidence=0.94,
                        location_type='city'
                    ),
                    GeolocationResult(
                        location='Marina Beach',
                        coordinates=(13.0499, 80.2824),
                        confidence=0.87,
                        location_type='landmark'
                    )
                ]
                
                locations = await geo_extractor.extract_locations(processed.processed_text)
                
                assert len(locations) >= 1
                assert locations[0].location == 'Chennai'
                assert locations[0].coordinates is not None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])