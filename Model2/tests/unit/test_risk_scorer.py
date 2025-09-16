"""Unit tests for Risk Scorer module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.output_generation.risk_scorer import RiskScorer


class TestRiskScorer:
    """Test suite for RiskScorer class."""
    
    @pytest.fixture
    def risk_scorer(self, test_config):
        """Create RiskScorer instance for testing."""
        return RiskScorer(test_config)
    
    def test_risk_scorer_initialization(self, risk_scorer, test_config):
        """Test RiskScorer initialization."""
        assert risk_scorer.config == test_config
        assert 'temporal' in risk_scorer.config['risk_assessment']['weights']
        assert 'spatial' in risk_scorer.config['risk_assessment']['weights']
        assert 'environmental' in risk_scorer.config['risk_assessment']['weights']
        assert 'historical' in risk_scorer.config['risk_assessment']['weights']
    
    def test_calculate_temporal_risk(self, risk_scorer, sample_location):
        """Test temporal risk calculation."""
        # Test with current time (should be higher risk)
        current_risk = risk_scorer.calculate_temporal_risk(
            location=sample_location,
            current_time=datetime.now()
        )
        assert 0.0 <= current_risk <= 1.0
        
        # Test with specific time patterns
        # High risk time (peak tsunami season)
        high_risk_time = datetime(2024, 3, 15, 14, 30)  # Afternoon in tsunami season
        high_risk_score = risk_scorer.calculate_temporal_risk(
            location=sample_location,
            current_time=high_risk_time
        )
        assert 0.0 <= high_risk_score <= 1.0
    
    def test_calculate_spatial_risk(self, risk_scorer, sample_location):
        """Test spatial risk calculation."""
        # Mock historical events near location
        mock_events = [
            {
                'location': (35.7, 139.7),  # Close to Tokyo
                'magnitude': 8.0,
                'event_date': datetime.now() - timedelta(days=100)
            },
            {
                'location': (36.0, 140.0),  # Further from Tokyo
                'magnitude': 7.5,
                'event_date': datetime.now() - timedelta(days=200)
            }
        ]
        
        spatial_risk = risk_scorer.calculate_spatial_risk(
            location=sample_location,
            nearby_events=mock_events
        )
        assert 0.0 <= spatial_risk <= 1.0
    
    def test_calculate_environmental_risk(self, risk_scorer, sample_location):
        """Test environmental risk calculation."""
        # Mock environmental data
        env_data = {
            'sea_temperature': 28.0,  # High temperature
            'wave_height': 5.0,       # High waves
            'wind_speed': 25.0,       # Strong winds
            'atmospheric_pressure': 995.0  # Low pressure
        }
        
        env_risk = risk_scorer.calculate_environmental_risk(
            location=sample_location,
            environmental_data=env_data
        )
        assert 0.0 <= env_risk <= 1.0
        
        # Test with normal conditions
        normal_env_data = {
            'sea_temperature': 22.0,
            'wave_height': 1.5,
            'wind_speed': 10.0,
            'atmospheric_pressure': 1013.0
        }
        
        normal_risk = risk_scorer.calculate_environmental_risk(
            location=sample_location,
            environmental_data=normal_env_data
        )
        assert 0.0 <= normal_risk <= 1.0
        assert env_risk >= normal_risk  # High conditions should have higher risk
    
    def test_calculate_historical_risk(self, risk_scorer, sample_location, sample_historical_data):
        """Test historical risk calculation."""
        historical_risk = risk_scorer.calculate_historical_risk(
            location=sample_location,
            historical_events=sample_historical_data
        )
        assert 0.0 <= historical_risk <= 1.0
        
        # Test with no historical events
        no_events_risk = risk_scorer.calculate_historical_risk(
            location=sample_location,
            historical_events=[]
        )
        assert 0.0 <= no_events_risk <= 1.0
        assert historical_risk >= no_events_risk  # Events should increase risk
    
    def test_calculate_comprehensive_risk_score(self, risk_scorer, sample_location):
        """Test comprehensive risk score calculation."""
        # Mock data
        mock_data = {
            'historical_events': [
                {
                    'location': (35.7, 139.7),
                    'magnitude': 8.0,
                    'event_date': datetime.now() - timedelta(days=100)
                }
            ]
        }
        
        mock_env_data = {
            'sea_temperature': 25.0,
            'wave_height': 3.0,
            'wind_speed': 15.0,
            'atmospheric_pressure': 1005.0
        }
        
        # Calculate comprehensive risk
        risk_result = risk_scorer.calculate_comprehensive_risk_score(
            location=sample_location,
            data=mock_data,
            environmental_data=mock_env_data,
            hazard_type='tsunami'
        )
        
        # Validate result structure
        assert 'overall_risk_score' in risk_result
        assert 'risk_level' in risk_result
        assert 'component_scores' in risk_result
        assert 'confidence_score' in risk_result
        
        # Validate score ranges
        assert 0.0 <= risk_result['overall_risk_score'] <= 1.0
        assert 0.0 <= risk_result['confidence_score'] <= 1.0
        assert risk_result['risk_level'] in ['Low', 'Medium', 'High', 'Critical']
        
        # Validate component scores
        components = risk_result['component_scores']
        for component in ['temporal_risk', 'spatial_risk', 'environmental_risk', 'historical_risk']:
            assert component in components
            assert 0.0 <= components[component] <= 1.0
    
    def test_determine_risk_level(self, risk_scorer):
        """Test risk level determination."""
        # Test different risk score ranges
        assert risk_scorer.determine_risk_level(0.1) == 'Low'
        assert risk_scorer.determine_risk_level(0.5) == 'Medium'
        assert risk_scorer.determine_risk_level(0.7) == 'High'
        assert risk_scorer.determine_risk_level(0.9) == 'Critical'
        
        # Test boundary conditions
        assert risk_scorer.determine_risk_level(0.0) == 'Low'
        assert risk_scorer.determine_risk_level(1.0) == 'Critical'
    
    def test_batch_risk_assessment(self, risk_scorer, sample_locations):
        """Test batch risk assessment."""
        # Mock data for multiple locations
        mock_data = {
            'historical_events': [
                {
                    'location': (35.7, 139.7),
                    'magnitude': 8.0,
                    'event_date': datetime.now() - timedelta(days=100)
                }
            ]
        }
        
        batch_results = risk_scorer.batch_risk_assessment(
            locations=sample_locations,
            data=mock_data,
            hazard_type='tsunami'
        )
        
        # Validate results
        assert len(batch_results) == len(sample_locations)
        
        for i, result in enumerate(batch_results):
            assert 'location' in result
            assert 'risk_assessment' in result
            assert result['location'] == sample_locations[i]
            
            assessment = result['risk_assessment']
            assert 'overall_risk_score' in assessment
            assert 'risk_level' in assessment
            assert 0.0 <= assessment['overall_risk_score'] <= 1.0
    
    def test_risk_trend_analysis(self, risk_scorer, sample_location):
        """Test risk trend analysis."""
        # Create mock historical risk assessments
        mock_assessments = []
        for i in range(10):
            assessment = {
                'location': sample_location,
                'overall_risk_score': 0.5 + (i * 0.05),  # Increasing trend
                'assessment_time': datetime.now() - timedelta(days=i),
                'hazard_type': 'tsunami'
            }
            mock_assessments.append(assessment)
        
        trend_analysis = risk_scorer.analyze_risk_trends(
            location=sample_location,
            historical_assessments=mock_assessments,
            time_window_days=30
        )
        
        # Validate trend analysis
        assert 'trend_direction' in trend_analysis
        assert 'trend_strength' in trend_analysis
        assert 'average_risk' in trend_analysis
        assert 'risk_volatility' in trend_analysis
        
        assert trend_analysis['trend_direction'] in ['increasing', 'decreasing', 'stable']
        assert 0.0 <= trend_analysis['trend_strength'] <= 1.0
        assert 0.0 <= trend_analysis['average_risk'] <= 1.0
    
    def test_confidence_score_calculation(self, risk_scorer, sample_location):
        """Test confidence score calculation."""
        # Test with comprehensive data
        high_confidence_data = {
            'data_quality': 0.95,
            'data_completeness': 0.90,
            'model_agreement': 0.88,
            'historical_coverage': 0.85
        }
        
        high_confidence = risk_scorer.calculate_confidence_score(
            location=sample_location,
            **high_confidence_data
        )
        
        # Test with poor data
        low_confidence_data = {
            'data_quality': 0.60,
            'data_completeness': 0.40,
            'model_agreement': 0.50,
            'historical_coverage': 0.30
        }
        
        low_confidence = risk_scorer.calculate_confidence_score(
            location=sample_location,
            **low_confidence_data
        )
        
        # Validate confidence scores
        assert 0.0 <= high_confidence <= 1.0
        assert 0.0 <= low_confidence <= 1.0
        assert high_confidence > low_confidence
    
    def test_risk_factor_identification(self, risk_scorer, sample_location):
        """Test risk factor identification."""
        # Mock component scores
        component_scores = {
            'temporal_risk': 0.8,    # High
            'spatial_risk': 0.3,     # Low
            'environmental_risk': 0.7,  # High
            'historical_risk': 0.9   # Very high
        }
        
        risk_factors = risk_scorer.identify_risk_factors(
            location=sample_location,
            component_scores=component_scores,
            hazard_type='tsunami'
        )
        
        # Validate risk factors
        assert isinstance(risk_factors, list)
        assert len(risk_factors) > 0
        
        # High scoring components should be mentioned
        factor_text = ' '.join(risk_factors)
        assert 'historical' in factor_text.lower()  # Highest score
        assert 'temporal' in factor_text.lower()    # Second highest
    
    @patch('src.output_generation.risk_scorer.RiskScorer.calculate_temporal_risk')
    @patch('src.output_generation.risk_scorer.RiskScorer.calculate_spatial_risk')
    def test_mocked_risk_calculation(self, mock_spatial, mock_temporal, risk_scorer, sample_location):
        """Test risk calculation with mocked components."""
        # Mock the individual risk calculations
        mock_temporal.return_value = 0.6
        mock_spatial.return_value = 0.8
        
        # Mock environmental and historical calculations would go here
        with patch.object(risk_scorer, 'calculate_environmental_risk', return_value=0.7), \
             patch.object(risk_scorer, 'calculate_historical_risk', return_value=0.9):
            
            result = risk_scorer.calculate_comprehensive_risk_score(
                location=sample_location,
                hazard_type='tsunami'
            )
            
            # Verify mocked methods were called
            mock_temporal.assert_called_once()
            mock_spatial.assert_called_once()
            
            # Verify result structure
            assert 'overall_risk_score' in result
            assert 'component_scores' in result
    
    def test_edge_cases(self, risk_scorer):
        """Test edge cases and error handling."""
        # Test with invalid location
        with pytest.raises(ValueError):
            risk_scorer.calculate_comprehensive_risk_score(
                location=(91, 181),  # Invalid coordinates
                hazard_type='tsunami'
            )
        
        # Test with invalid hazard type
        with pytest.raises(ValueError):
            risk_scorer.calculate_comprehensive_risk_score(
                location=(35.6762, 139.6503),
                hazard_type='invalid_hazard'
            )
        
        # Test with None data
        result = risk_scorer.calculate_comprehensive_risk_score(
            location=(35.6762, 139.6503),
            data=None,
            hazard_type='tsunami'
        )
        
        # Should handle gracefully
        assert 'overall_risk_score' in result
        assert 0.0 <= result['overall_risk_score'] <= 1.0