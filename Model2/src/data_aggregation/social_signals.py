"""Social signals collector for integrating crowd report data from Model A."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import requests
import asyncio

from config import config

logger = logging.getLogger(__name__)

class SocialSignalsCollector:
    """Collects social media and crowd report signals from Model A."""
    
    def __init__(self):
        self.model_a_api_url = config.get('data_sources.model_a_api', 'http://localhost:8001')
        self.social_data_cache = {}
        self.crowd_reports_cache = {}
    
    async def collect_crowd_reports(self, time_window_hours: int = 24) -> pd.DataFrame:
        """Collect crowd reports from Model A API."""
        try:
            # In a real implementation, this would call Model A's API
            # For now, we'll generate realistic mock data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_window_hours)
            
            reports_data = self._generate_mock_crowd_reports(start_time, end_time)
            
            return pd.DataFrame(reports_data)
            
        except Exception as e:
            logger.error(f"Error collecting crowd reports: {e}")
            return pd.DataFrame()
    
    def _generate_mock_crowd_reports(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Generate mock crowd reports for testing."""
        reports = []
        
        # Generate 20-100 reports in the time window
        num_reports = np.random.randint(20, 100)
        
        # Indian coastal locations
        locations = [
            {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777},
            {'name': 'Chennai', 'lat': 13.0827, 'lon': 80.2707},
            {'name': 'Goa', 'lat': 15.2993, 'lon': 74.1240},
            {'name': 'Visakhapatnam', 'lat': 17.6868, 'lon': 83.2185},
            {'name': 'Kolkata', 'lat': 22.5726, 'lon': 88.3639},
            {'name': 'Kanyakumari', 'lat': 8.0883, 'lon': 77.5385}
        ]
        
        hazard_types = ['high_waves', 'storm_surge', 'coastal_flooding', 'unusual_tide', 'tsunami_like']
        urgency_levels = ['low', 'medium', 'high', 'critical']
        
        for _ in range(num_reports):
            # Random timestamp within the window
            time_delta = np.random.uniform(0, (end_time - start_time).total_seconds())
            report_time = start_time + timedelta(seconds=time_delta)
            
            # Random location
            location = np.random.choice(locations)
            
            # Add some noise to coordinates
            lat_noise = np.random.normal(0, 0.01)
            lon_noise = np.random.normal(0, 0.01)
            
            report = {
                'report_id': f"CR_{int(report_time.timestamp())}_{np.random.randint(1000, 9999)}",
                'timestamp': report_time,
                'latitude': location['lat'] + lat_noise,
                'longitude': location['lon'] + lon_noise,
                'location_name': location['name'],
                'hazard_type': np.random.choice(hazard_types),
                'urgency_level': np.random.choice(urgency_levels),
                'severity_score': round(np.random.uniform(1, 10), 2),
                'confidence_score': round(np.random.uniform(0.3, 1.0), 2),
                'source_type': np.random.choice(['user_report', 'social_media', 'official']),
                'verified': np.random.choice([True, False], p=[0.7, 0.3]),
                'description': f"Report of {np.random.choice(hazard_types)} in {location['name']} area",
                'media_count': np.random.randint(0, 5),
                'engagement_score': round(np.random.exponential(10), 2)
            }
            
            reports.append(report)
        
        return reports
    
    async def collect_social_media_signals(self, keywords: List[str], 
                                         locations: List[Dict]) -> Dict:
        """Collect social media activity signals."""
        try:
            # Mock social media data aggregation
            signals = {}
            
            for location in locations:
                location_signals = self._generate_mock_social_signals(location, keywords)
                signals[location['name']] = location_signals
            
            return signals
            
        except Exception as e:
            logger.error(f"Error collecting social media signals: {e}")
            return {}
    
    def _generate_mock_social_signals(self, location: Dict, keywords: List[str]) -> Dict:
        """Generate mock social media signals for a location."""
        # Base activity level
        base_activity = np.random.randint(50, 500)
        
        # Generate spikes (representing real events)
        spike_probability = 0.1  # 10% chance of spike
        spike_multiplier = np.random.uniform(2, 10) if np.random.random() < spike_probability else 1
        
        return {
            'total_mentions': int(base_activity * spike_multiplier),
            'sentiment_positive': round(np.random.uniform(0.2, 0.8), 3),
            'sentiment_negative': round(np.random.uniform(0.1, 0.6), 3),
            'sentiment_neutral': round(np.random.uniform(0.1, 0.5), 3),
            'urgency_indicators': int(base_activity * spike_multiplier * np.random.uniform(0.05, 0.3)),
            'panic_indicators': int(base_activity * spike_multiplier * np.random.uniform(0.01, 0.1)),
            'help_requests': int(base_activity * spike_multiplier * np.random.uniform(0.02, 0.15)),
            'official_responses': int(base_activity * spike_multiplier * np.random.uniform(0.001, 0.05)),
            'misinformation_flags': int(base_activity * spike_multiplier * np.random.uniform(0.01, 0.08)),
            'keyword_matches': {keyword: np.random.randint(0, int(base_activity * 0.2)) for keyword in keywords},
            'engagement_rate': round(np.random.uniform(0.01, 0.15), 4),
            'share_rate': round(np.random.uniform(0.005, 0.08), 4),
            'geographical_spread': round(np.random.uniform(1, 50), 2),  # km radius
            'temporal_intensity': round(spike_multiplier, 2)
        }
    
    def detect_social_anomalies(self, current_signals: Dict, 
                              historical_baseline: Dict) -> Dict:
        """Detect anomalies in social media activity."""
        anomalies = {}
        
        for location, signals in current_signals.items():
            if location not in historical_baseline:
                continue
                
            baseline = historical_baseline[location]
            location_anomalies = {}
            
            # Check for significant spikes
            metrics_to_check = [
                'total_mentions', 'urgency_indicators', 'panic_indicators',
                'help_requests', 'misinformation_flags'
            ]
            
            for metric in metrics_to_check:
                if metric in signals and metric in baseline:
                    current_value = signals[metric]
                    baseline_mean = baseline.get(f'{metric}_mean', current_value)
                    baseline_std = baseline.get(f'{metric}_std', current_value * 0.5)
                    
                    if baseline_std > 0:
                        z_score = (current_value - baseline_mean) / baseline_std
                        
                        if z_score > 2:  # Significant spike
                            location_anomalies[metric] = {
                                'current_value': current_value,
                                'baseline_mean': baseline_mean,
                                'z_score': round(z_score, 2),
                                'anomaly_type': 'spike',
                                'severity': 'high' if z_score > 3 else 'medium'
                            }
            
            if location_anomalies:
                anomalies[location] = location_anomalies
        
        return anomalies
    
    def calculate_social_risk_indicators(self, signals: Dict) -> Dict:
        """Calculate risk indicators from social signals."""
        risk_indicators = {}
        
        for location, data in signals.items():
            # Calculate composite risk score
            panic_score = min(100, data.get('panic_indicators', 0) * 2)
            urgency_score = min(100, data.get('urgency_indicators', 0) * 1.5)
            help_score = min(100, data.get('help_requests', 0) * 3)
            
            # Sentiment-based risk
            negative_sentiment = data.get('sentiment_negative', 0)
            sentiment_risk = negative_sentiment * 100
            
            # Temporal intensity
            temporal_risk = min(100, data.get('temporal_intensity', 1) * 20)
            
            # Geographic spread risk
            spread_risk = min(100, data.get('geographical_spread', 1) * 2)
            
            # Composite score
            composite_risk = (
                panic_score * 0.3 +
                urgency_score * 0.25 +
                help_score * 0.2 +
                sentiment_risk * 0.15 +
                temporal_risk * 0.1
            )
            
            risk_indicators[location] = {
                'composite_risk_score': round(composite_risk, 2),
                'panic_risk': round(panic_score, 2),
                'urgency_risk': round(urgency_score, 2),
                'help_request_risk': round(help_score, 2),
                'sentiment_risk': round(sentiment_risk, 2),
                'temporal_risk': round(temporal_risk, 2),
                'spread_risk': round(spread_risk, 2),
                'risk_level': self._classify_risk_level(composite_risk),
                'confidence': round(data.get('engagement_rate', 0.05) * 100, 2)
            }
        
        return risk_indicators
    
    def _classify_risk_level(self, score: float) -> str:
        """Classify risk level based on composite score."""
        if score < 20:
            return 'Low'
        elif score < 40:
            return 'Moderate'
        elif score < 60:
            return 'High'
        elif score < 80:
            return 'Very High'
        else:
            return 'Critical'
    
    def aggregate_crowd_intelligence(self, reports_df: pd.DataFrame) -> Dict:
        """Aggregate crowd intelligence from reports."""
        if reports_df.empty:
            return {}
        
        aggregation = {}
        
        # Aggregate by location
        for location in reports_df['location_name'].unique():
            location_reports = reports_df[reports_df['location_name'] == location]
            
            # Calculate aggregated metrics
            total_reports = len(location_reports)
            verified_reports = len(location_reports[location_reports['verified'] == True])
            avg_severity = location_reports['severity_score'].mean()
            avg_confidence = location_reports['confidence_score'].mean()
            
            # Hazard type distribution
            hazard_distribution = location_reports['hazard_type'].value_counts().to_dict()
            
            # Urgency distribution
            urgency_distribution = location_reports['urgency_level'].value_counts().to_dict()
            
            # Recent activity (last 6 hours)
            recent_cutoff = datetime.now() - timedelta(hours=6)
            recent_reports = location_reports[location_reports['timestamp'] > recent_cutoff]
            recent_activity = len(recent_reports)
            
            aggregation[location] = {
                'total_reports': total_reports,
                'verified_reports': verified_reports,
                'verification_rate': round(verified_reports / total_reports, 3) if total_reports > 0 else 0,
                'avg_severity': round(avg_severity, 2),
                'avg_confidence': round(avg_confidence, 3),
                'hazard_distribution': hazard_distribution,
                'urgency_distribution': urgency_distribution,
                'recent_activity_6h': recent_activity,
                'activity_trend': self._calculate_activity_trend(location_reports),
                'dominant_hazard': max(hazard_distribution.items(), key=lambda x: x[1])[0] if hazard_distribution else None,
                'risk_assessment': self._assess_location_risk(location_reports)
            }
        
        return aggregation
    
    def _calculate_activity_trend(self, reports_df: pd.DataFrame) -> str:
        """Calculate activity trend for reports."""
        if len(reports_df) < 2:
            return 'stable'
        
        # Sort by timestamp
        sorted_reports = reports_df.sort_values('timestamp')
        
        # Compare recent half vs older half
        midpoint = len(sorted_reports) // 2
        recent_half = sorted_reports.iloc[midpoint:]
        older_half = sorted_reports.iloc[:midpoint]
        
        recent_avg_time = (datetime.now() - recent_half['timestamp']).dt.total_seconds().mean()
        older_avg_time = (datetime.now() - older_half['timestamp']).dt.total_seconds().mean()
        
        if recent_avg_time < older_avg_time * 0.7:
            return 'increasing'
        elif recent_avg_time > older_avg_time * 1.3:
            return 'decreasing'
        else:
            return 'stable'
    
    def _assess_location_risk(self, reports_df: pd.DataFrame) -> Dict:
        """Assess overall risk for a location based on crowd reports."""
        if reports_df.empty:
            return {'level': 'unknown', 'score': 0}
        
        # Weight factors
        severity_weight = 0.3
        urgency_weight = 0.25
        volume_weight = 0.2
        verification_weight = 0.15
        recency_weight = 0.1
        
        # Calculate components
        avg_severity = reports_df['severity_score'].mean() / 10  # Normalize to 0-1
        
        urgency_scores = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}
        avg_urgency = reports_df['urgency_level'].map(urgency_scores).mean()
        
        volume_score = min(1.0, len(reports_df) / 50)  # Normalize, cap at 50 reports
        
        verification_rate = reports_df['verified'].mean()
        
        # Recency score (more recent = higher score)
        time_diffs = (datetime.now() - reports_df['timestamp']).dt.total_seconds()
        recency_score = 1 - (time_diffs.mean() / (24 * 3600))  # Normalize by 24 hours
        recency_score = max(0, min(1, recency_score))
        
        # Composite risk score
        composite_score = (
            avg_severity * severity_weight +
            avg_urgency * urgency_weight +
            volume_score * volume_weight +
            verification_rate * verification_weight +
            recency_score * recency_weight
        )
        
        # Classify risk level
        if composite_score < 0.2:
            risk_level = 'Low'
        elif composite_score < 0.4:
            risk_level = 'Moderate'
        elif composite_score < 0.6:
            risk_level = 'High'
        elif composite_score < 0.8:
            risk_level = 'Very High'
        else:
            risk_level = 'Critical'
        
        return {
            'level': risk_level,
            'score': round(composite_score * 100, 2),
            'components': {
                'severity': round(avg_severity * 100, 2),
                'urgency': round(avg_urgency * 100, 2),
                'volume': round(volume_score * 100, 2),
                'verification': round(verification_rate * 100, 2),
                'recency': round(recency_score * 100, 2)
            }
        }
    
    def save_social_data(self, reports_df: pd.DataFrame, signals: Dict, filepath_prefix: str):
        """Save social media and crowd report data."""
        try:
            # Save crowd reports
            if not reports_df.empty:
                reports_df.to_csv(f"{filepath_prefix}_crowd_reports.csv", index=False)
            
            # Save social signals
            if signals:
                signals_df = pd.DataFrame.from_dict(signals, orient='index')
                signals_df.to_csv(f"{filepath_prefix}_social_signals.csv")
            
            logger.info(f"Saved social data with prefix {filepath_prefix}")
            
        except Exception as e:
            logger.error(f"Error saving social data: {e}")