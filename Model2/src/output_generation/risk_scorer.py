"""Risk scoring system for ocean hazard prediction."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json

# ML and data processing
from sklearn.preprocessing import MinMaxScaler
import joblib

logger = logging.getLogger(__name__)

class RiskScorer:
    """Comprehensive risk scoring system for ocean hazards."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._load_default_config()
        self.risk_models = {}
        self.scalers = {}
        self.risk_thresholds = self.config.get('risk_thresholds', {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.95
        })
        
    def _load_default_config(self) -> Dict:
        """Load default risk scoring configuration."""
        
        return {
            'risk_factors': {
                'temporal': {
                    'weight': 0.25,
                    'factors': ['time_since_last_event', 'seasonal_pattern', 'weather_forecast']
                },
                'spatial': {
                    'weight': 0.25,
                    'factors': ['distance_to_fault', 'coastal_vulnerability', 'bathymetry']
                },
                'environmental': {
                    'weight': 0.30,
                    'factors': ['sea_temperature', 'wave_height', 'wind_speed', 'atmospheric_pressure']
                },
                'historical': {
                    'weight': 0.20,
                    'factors': ['historical_frequency', 'maximum_magnitude', 'impact_severity']
                }
            },
            'hazard_types': {
                'tsunami': {'base_weight': 1.0, 'urgency_factor': 0.9},
                'storm_surge': {'base_weight': 0.8, 'urgency_factor': 0.7},
                'coastal_flooding': {'base_weight': 0.6, 'urgency_factor': 0.5},
                'erosion': {'base_weight': 0.4, 'urgency_factor': 0.3},
                'oil_spill': {'base_weight': 0.7, 'urgency_factor': 0.8}
            },
            'time_windows': {
                'immediate': 24,    # hours
                'short_term': 168,  # 1 week
                'medium_term': 720, # 1 month
                'long_term': 2160   # 3 months
            }
        }
    
    def calculate_temporal_risk(self, data: pd.DataFrame, location: Tuple[float, float],
                              time_window: str = 'short_term') -> Dict:
        """Calculate temporal risk factors."""
        
        logger.info(f"Calculating temporal risk for location {location}")
        
        temporal_risk = {
            'time_window': time_window,
            'window_hours': self.config['time_windows'][time_window],
            'factors': {}
        }
        
        current_time = datetime.now()
        window_hours = self.config['time_windows'][time_window]
        window_start = current_time - timedelta(hours=window_hours)
        
        # Filter data by time window and location proximity
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            recent_data = data[data['timestamp'] >= window_start]
        else:
            recent_data = data
        
        # Time since last event
        if len(recent_data) > 0 and 'timestamp' in recent_data.columns:
            last_event_time = recent_data['timestamp'].max()
            hours_since_last = (current_time - last_event_time).total_seconds() / 3600
            
            # Risk decreases with time since last event (exponential decay)
            time_risk = np.exp(-hours_since_last / (window_hours * 0.5))
            temporal_risk['factors']['time_since_last_event'] = float(time_risk)
        else:
            temporal_risk['factors']['time_since_last_event'] = 0.1
        
        # Seasonal pattern analysis
        month = current_time.month
        seasonal_risk_pattern = {
            12: 0.8, 1: 0.9, 2: 0.7,  # Winter - higher storm risk
            3: 0.6, 4: 0.5, 5: 0.4,   # Spring - moderate risk
            6: 0.7, 7: 0.8, 8: 0.9,   # Summer - hurricane season
            9: 0.9, 10: 0.8, 11: 0.7  # Fall - continued storm season
        }
        temporal_risk['factors']['seasonal_pattern'] = seasonal_risk_pattern.get(month, 0.5)
        
        # Weather forecast analysis (mock implementation)
        # In real implementation, this would integrate with weather APIs
        weather_indicators = {
            'wind_speed_forecast': np.random.uniform(0.3, 0.8),
            'pressure_drop_forecast': np.random.uniform(0.2, 0.7),
            'storm_probability': np.random.uniform(0.1, 0.6)
        }
        
        weather_risk = np.mean(list(weather_indicators.values()))
        temporal_risk['factors']['weather_forecast'] = float(weather_risk)
        temporal_risk['weather_details'] = weather_indicators
        
        # Calculate overall temporal risk
        factor_weights = [0.4, 0.3, 0.3]  # time_since, seasonal, weather
        factor_values = [
            temporal_risk['factors']['time_since_last_event'],
            temporal_risk['factors']['seasonal_pattern'],
            temporal_risk['factors']['weather_forecast']
        ]
        
        temporal_risk['overall_score'] = float(np.average(factor_values, weights=factor_weights))
        
        return temporal_risk
    
    def calculate_spatial_risk(self, location: Tuple[float, float], 
                             geospatial_data: Dict = None) -> Dict:
        """Calculate spatial risk factors for a location."""
        
        logger.info(f"Calculating spatial risk for location {location}")
        
        lat, lon = location
        spatial_risk = {
            'location': {'latitude': lat, 'longitude': lon},
            'factors': {}
        }
        
        # Coastal vulnerability assessment
        # Distance from coastline (closer = higher risk)
        # This is a simplified calculation - real implementation would use actual coastline data
        coastal_distance_km = abs(lat) * 111  # Rough approximation
        coastal_risk = max(0.1, 1.0 - (coastal_distance_km / 100))  # Risk decreases with distance
        spatial_risk['factors']['coastal_vulnerability'] = float(coastal_risk)
        
        # Bathymetry impact (water depth)
        # Shallower waters can amplify wave effects
        if geospatial_data and 'bathymetry' in geospatial_data:
            depth = abs(geospatial_data['bathymetry'])
            depth_risk = max(0.1, 1.0 - (depth / 1000))  # Risk higher in shallow water
        else:
            depth_risk = 0.5  # Default moderate risk
        spatial_risk['factors']['bathymetry'] = float(depth_risk)
        
        # Tectonic activity proximity
        # Distance to known fault lines or seismic zones
        tectonic_zones = [
            (35.0, 140.0),  # Japan area
            (38.0, -122.0), # California
            (-6.0, 130.0),  # Indonesia
            (36.0, 28.0)    # Mediterranean
        ]
        
        min_distance = float('inf')
        for zone_lat, zone_lon in tectonic_zones:
            distance = np.sqrt((lat - zone_lat)**2 + (lon - zone_lon)**2)
            min_distance = min(min_distance, distance)
        
        # Risk decreases with distance from tectonic zones
        tectonic_risk = max(0.1, 1.0 - (min_distance / 50))
        spatial_risk['factors']['distance_to_fault'] = float(tectonic_risk)
        
        # Population density factor (higher population = higher risk impact)
        # This is a simplified model - real implementation would use actual population data
        population_density_factor = min(1.0, abs(lat) / 90 + abs(lon) / 180)
        spatial_risk['factors']['population_density'] = float(population_density_factor)
        
        # Calculate overall spatial risk
        factor_weights = [0.3, 0.25, 0.25, 0.2]  # coastal, bathymetry, tectonic, population
        factor_values = [
            spatial_risk['factors']['coastal_vulnerability'],
            spatial_risk['factors']['bathymetry'],
            spatial_risk['factors']['distance_to_fault'],
            spatial_risk['factors']['population_density']
        ]
        
        spatial_risk['overall_score'] = float(np.average(factor_values, weights=factor_weights))
        
        return spatial_risk
    
    def calculate_environmental_risk(self, location: Tuple[float, float],
                                   environmental_data: Dict = None) -> Dict:
        """Calculate environmental risk factors."""
        
        logger.info(f"Calculating environmental risk for location {location}")
        
        environmental_risk = {
            'location': {'latitude': location[0], 'longitude': location[1]},
            'factors': {}
        }
        
        # Sea surface temperature anomaly
        if environmental_data and 'sea_temperature' in environmental_data:
            temp = environmental_data['sea_temperature']
            # Higher temperatures can increase storm intensity
            temp_risk = min(1.0, max(0.0, (temp - 20) / 15))  # Risk increases above 20°C
        else:
            temp_risk = 0.4  # Default moderate risk
        environmental_risk['factors']['sea_temperature'] = float(temp_risk)
        
        # Wave height conditions
        if environmental_data and 'wave_height' in environmental_data:
            wave_height = environmental_data['wave_height']
            wave_risk = min(1.0, wave_height / 10)  # Risk increases with wave height
        else:
            wave_risk = 0.3
        environmental_risk['factors']['wave_height'] = float(wave_risk)
        
        # Wind speed conditions
        if environmental_data and 'wind_speed' in environmental_data:
            wind_speed = environmental_data['wind_speed']
            wind_risk = min(1.0, wind_speed / 50)  # Risk increases with wind speed
        else:
            wind_risk = 0.3
        environmental_risk['factors']['wind_speed'] = float(wind_risk)
        
        # Atmospheric pressure
        if environmental_data and 'atmospheric_pressure' in environmental_data:
            pressure = environmental_data['atmospheric_pressure']
            # Lower pressure indicates potential storms
            pressure_risk = max(0.0, (1013 - pressure) / 50)  # Risk increases as pressure drops
        else:
            pressure_risk = 0.2
        environmental_risk['factors']['atmospheric_pressure'] = float(pressure_risk)
        
        # Ocean current strength
        if environmental_data and 'current_speed' in environmental_data:
            current_speed = environmental_data['current_speed']
            current_risk = min(1.0, current_speed / 5)  # Risk increases with current strength
        else:
            current_risk = 0.3
        environmental_risk['factors']['ocean_current'] = float(current_risk)
        
        # Calculate overall environmental risk
        factor_weights = [0.25, 0.2, 0.2, 0.15, 0.2]  # temp, wave, wind, pressure, current
        factor_values = [
            environmental_risk['factors']['sea_temperature'],
            environmental_risk['factors']['wave_height'],
            environmental_risk['factors']['wind_speed'],
            environmental_risk['factors']['atmospheric_pressure'],
            environmental_risk['factors']['ocean_current']
        ]
        
        environmental_risk['overall_score'] = float(np.average(factor_values, weights=factor_weights))
        
        return environmental_risk
    
    def calculate_historical_risk(self, location: Tuple[float, float],
                                historical_data: pd.DataFrame = None) -> Dict:
        """Calculate historical risk based on past events."""
        
        logger.info(f"Calculating historical risk for location {location}")
        
        historical_risk = {
            'location': {'latitude': location[0], 'longitude': location[1]},
            'factors': {}
        }
        
        if historical_data is None or len(historical_data) == 0:
            # Default risk when no historical data available
            historical_risk['factors'] = {
                'historical_frequency': 0.3,
                'maximum_magnitude': 0.3,
                'impact_severity': 0.3
            }
            historical_risk['overall_score'] = 0.3
            return historical_risk
        
        lat, lon = location
        
        # Filter events near the location (within ~100km)
        if 'latitude' in historical_data.columns and 'longitude' in historical_data.columns:
            distance_threshold = 1.0  # Approximately 100km
            nearby_events = historical_data[
                (abs(historical_data['latitude'] - lat) < distance_threshold) &
                (abs(historical_data['longitude'] - lon) < distance_threshold)
            ]
        else:
            nearby_events = historical_data
        
        # Historical frequency analysis
        if len(nearby_events) > 0:
            # Calculate events per year
            if 'timestamp' in nearby_events.columns:
                nearby_events['timestamp'] = pd.to_datetime(nearby_events['timestamp'])
                date_range = (nearby_events['timestamp'].max() - nearby_events['timestamp'].min()).days
                years_covered = max(1, date_range / 365.25)
                events_per_year = len(nearby_events) / years_covered
            else:
                events_per_year = len(nearby_events) / 10  # Assume 10-year data if no timestamps
            
            # Risk increases with frequency
            frequency_risk = min(1.0, events_per_year / 2)  # Normalize to 2 events per year = max risk
        else:
            frequency_risk = 0.1  # Low risk if no historical events
        
        historical_risk['factors']['historical_frequency'] = float(frequency_risk)
        historical_risk['events_analyzed'] = len(nearby_events)
        
        # Maximum magnitude analysis
        if len(nearby_events) > 0 and 'magnitude' in nearby_events.columns:
            max_magnitude = nearby_events['magnitude'].max()
            # Risk increases with maximum observed magnitude
            magnitude_risk = min(1.0, max_magnitude / 10)  # Normalize to magnitude 10 = max risk
        else:
            magnitude_risk = 0.2
        
        historical_risk['factors']['maximum_magnitude'] = float(magnitude_risk)
        
        # Impact severity analysis
        if len(nearby_events) > 0:
            if 'severity' in nearby_events.columns:
                avg_severity = nearby_events['severity'].mean()
                severity_risk = min(1.0, avg_severity / 5)  # Normalize to severity 5 = max risk
            elif 'impact_score' in nearby_events.columns:
                avg_impact = nearby_events['impact_score'].mean()
                severity_risk = min(1.0, avg_impact)
            else:
                severity_risk = 0.3  # Default moderate severity
        else:
            severity_risk = 0.1
        
        historical_risk['factors']['impact_severity'] = float(severity_risk)
        
        # Calculate overall historical risk
        factor_weights = [0.4, 0.3, 0.3]  # frequency, magnitude, severity
        factor_values = [
            historical_risk['factors']['historical_frequency'],
            historical_risk['factors']['maximum_magnitude'],
            historical_risk['factors']['impact_severity']
        ]
        
        historical_risk['overall_score'] = float(np.average(factor_values, weights=factor_weights))
        
        return historical_risk
    
    def calculate_comprehensive_risk_score(self, location: Tuple[float, float],
                                         data: pd.DataFrame = None,
                                         environmental_data: Dict = None,
                                         geospatial_data: Dict = None,
                                         time_window: str = 'short_term',
                                         hazard_type: str = 'tsunami') -> Dict:
        """Calculate comprehensive risk score combining all factors."""
        
        logger.info(f"Calculating comprehensive risk score for {hazard_type} at {location}")
        
        # Calculate individual risk components
        temporal_risk = self.calculate_temporal_risk(data, location, time_window)
        spatial_risk = self.calculate_spatial_risk(location, geospatial_data)
        environmental_risk = self.calculate_environmental_risk(location, environmental_data)
        historical_risk = self.calculate_historical_risk(location, data)
        
        # Get weights from configuration
        risk_weights = self.config['risk_factors']
        
        # Calculate weighted overall risk
        overall_risk = (
            temporal_risk['overall_score'] * risk_weights['temporal']['weight'] +
            spatial_risk['overall_score'] * risk_weights['spatial']['weight'] +
            environmental_risk['overall_score'] * risk_weights['environmental']['weight'] +
            historical_risk['overall_score'] * risk_weights['historical']['weight']
        )
        
        # Apply hazard-specific adjustments
        hazard_config = self.config['hazard_types'].get(hazard_type, {'base_weight': 1.0, 'urgency_factor': 0.5})
        adjusted_risk = overall_risk * hazard_config['base_weight']
        
        # Determine risk level
        risk_level = self._determine_risk_level(adjusted_risk)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(
            temporal_risk, spatial_risk, environmental_risk, historical_risk, data
        )
        
        comprehensive_risk = {
            'location': {'latitude': location[0], 'longitude': location[1]},
            'hazard_type': hazard_type,
            'time_window': time_window,
            'timestamp': datetime.now().isoformat(),
            'risk_score': float(adjusted_risk),
            'risk_level': risk_level,
            'confidence_score': float(confidence),
            'components': {
                'temporal': temporal_risk,
                'spatial': spatial_risk,
                'environmental': environmental_risk,
                'historical': historical_risk
            },
            'urgency_factor': hazard_config['urgency_factor'],
            'recommendations': self._generate_recommendations(adjusted_risk, risk_level, hazard_type)
        }
        
        return comprehensive_risk
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score."""
        
        thresholds = self.risk_thresholds
        
        if risk_score >= thresholds['critical']:
            return 'Critical'
        elif risk_score >= thresholds['high']:
            return 'High'
        elif risk_score >= thresholds['medium']:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_confidence_score(self, temporal_risk: Dict, spatial_risk: Dict,
                                  environmental_risk: Dict, historical_risk: Dict,
                                  data: pd.DataFrame = None) -> float:
        """Calculate confidence score based on data quality and completeness."""
        
        confidence_factors = []
        
        # Data availability confidence
        if data is not None and len(data) > 0:
            data_confidence = min(1.0, len(data) / 100)  # More data = higher confidence
            confidence_factors.append(data_confidence)
        else:
            confidence_factors.append(0.3)  # Low confidence with no data
        
        # Historical data confidence
        events_analyzed = historical_risk.get('events_analyzed', 0)
        historical_confidence = min(1.0, events_analyzed / 50)  # More events = higher confidence
        confidence_factors.append(historical_confidence)
        
        # Environmental data confidence
        env_factors = len(environmental_risk.get('factors', {}))
        env_confidence = min(1.0, env_factors / 5)  # More factors = higher confidence
        confidence_factors.append(env_confidence)
        
        # Temporal analysis confidence
        temp_factors = len(temporal_risk.get('factors', {}))
        temp_confidence = min(1.0, temp_factors / 3)
        confidence_factors.append(temp_confidence)
        
        # Overall confidence is the average of all factors
        overall_confidence = np.mean(confidence_factors)
        
        return overall_confidence
    
    def _generate_recommendations(self, risk_score: float, risk_level: str, 
                                hazard_type: str) -> List[str]:
        """Generate actionable recommendations based on risk assessment."""
        
        recommendations = []
        
        if risk_level == 'Critical':
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Evacuate area if possible",
                "Contact emergency services and local authorities",
                "Monitor emergency broadcasts continuously",
                "Activate emergency response protocols"
            ])
        elif risk_level == 'High':
            recommendations.extend([
                "Prepare for potential evacuation",
                "Secure property and valuable items",
                "Monitor weather and ocean conditions closely",
                "Review emergency response plans"
            ])
        elif risk_level == 'Medium':
            recommendations.extend([
                "Stay informed about developing conditions",
                "Check emergency supplies and equipment",
                "Avoid unnecessary coastal activities",
                "Monitor official warnings and advisories"
            ])
        else:  # Low
            recommendations.extend([
                "Continue normal activities with awareness",
                "Stay informed about ocean conditions",
                "Maintain preparedness for potential changes"
            ])
        
        # Hazard-specific recommendations
        if hazard_type == 'tsunami':
            recommendations.append("Know tsunami evacuation routes and assembly points")
        elif hazard_type == 'storm_surge':
            recommendations.append("Secure boats and waterfront property")
        elif hazard_type == 'coastal_flooding':
            recommendations.append("Check drainage systems and flood barriers")
        elif hazard_type == 'oil_spill':
            recommendations.append("Avoid contact with water and report observations")
        
        return recommendations
    
    def batch_risk_assessment(self, locations: List[Tuple[float, float]],
                            data: pd.DataFrame = None,
                            hazard_type: str = 'tsunami') -> List[Dict]:
        """Perform risk assessment for multiple locations."""
        
        logger.info(f"Performing batch risk assessment for {len(locations)} locations")
        
        results = []
        
        for i, location in enumerate(locations):
            try:
                risk_assessment = self.calculate_comprehensive_risk_score(
                    location, data, hazard_type=hazard_type
                )
                risk_assessment['location_id'] = i
                results.append(risk_assessment)
                
            except Exception as e:
                logger.error(f"Error assessing risk for location {location}: {e}")
                results.append({
                    'location_id': i,
                    'location': {'latitude': location[0], 'longitude': location[1]},
                    'error': str(e),
                    'risk_score': 0.0,
                    'risk_level': 'Unknown'
                })
        
        # Sort by risk score (highest first)
        results.sort(key=lambda x: x.get('risk_score', 0), reverse=True)
        
        return results
    
    def update_risk_thresholds(self, new_thresholds: Dict):
        """Update risk level thresholds."""
        
        self.risk_thresholds.update(new_thresholds)
        logger.info(f"Updated risk thresholds: {self.risk_thresholds}")
    
    def save_risk_model(self, filepath: str):
        """Save risk scoring configuration and models."""
        
        model_data = {
            'config': self.config,
            'risk_thresholds': self.risk_thresholds,
            'scalers': self.scalers,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            joblib.dump(model_data, filepath)
            logger.info(f"Risk scoring model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving risk model: {e}")
    
    def load_risk_model(self, filepath: str):
        """Load risk scoring configuration and models."""
        
        try:
            model_data = joblib.load(filepath)
            
            self.config = model_data.get('config', self.config)
            self.risk_thresholds = model_data.get('risk_thresholds', self.risk_thresholds)
            self.scalers = model_data.get('scalers', {})
            
            logger.info(f"Risk scoring model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading risk model: {e}")
    
    def export_risk_assessment_report(self, risk_assessment: Dict, 
                                    filepath: str = None) -> str:
        """Export risk assessment as detailed report."""
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"risk_assessment_report_{timestamp}.json"
        
        # Add metadata to the report
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'Ocean Hazard Risk Assessment',
                'version': '1.0'
            },
            'risk_assessment': risk_assessment,
            'interpretation': {
                'risk_level_meaning': {
                    'Critical': 'Immediate threat - take action now',
                    'High': 'Significant risk - prepare for potential impact',
                    'Medium': 'Moderate risk - stay informed and prepared',
                    'Low': 'Minimal risk - maintain normal awareness'
                },
                'confidence_interpretation': {
                    'high': '>0.7 - High data quality and reliability',
                    'medium': '0.4-0.7 - Moderate data quality',
                    'low': '<0.4 - Limited data available'
                }
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Risk assessment report exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return ""