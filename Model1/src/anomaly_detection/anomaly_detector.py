import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

# Statistical analysis
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# Time series analysis
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from ..data_ingestion import RawReport
from ..preprocessing import ProcessedReport
from ..nlp_analysis import HazardPrediction
from ..geolocation import GeolocationResult
from ...config import config

@dataclass
class AnomalyAlert:
    """Anomaly detection alert"""
    id: str
    alert_type: str  # 'volume_spike', 'geographic_cluster', 'temporal_anomaly'
    severity: str  # 'low', 'medium', 'high', 'critical'
    location: Optional[str]
    coordinates: Optional[Tuple[float, float]]
    detection_time: datetime
    time_window: str
    message: str
    metrics: Dict[str, Any]
    affected_reports: List[str]
    confidence_score: float

@dataclass 
class SpatialCluster:
    """Spatial cluster of hazard reports"""
    cluster_id: int
    center_coords: Tuple[float, float]
    radius_km: float
    report_count: int
    report_ids: List[str]
    dominant_hazard_type: str
    detection_time: datetime
    confidence: float

@dataclass
class TemporalPattern:
    """Temporal pattern analysis result"""
    pattern_type: str  # 'spike', 'trend', 'seasonal'
    time_window: str
    baseline_value: float
    current_value: float
    deviation_score: float
    statistical_significance: float
    detection_time: datetime

class SpatialAnomalyDetector:
    """Detect spatial anomalies in hazard reports"""
    
    def __init__(self):
        self.clustering_params = {
            'eps_km': config.get('ANOMALY_DETECTION.SPATIAL_CLUSTERING.EPS_KM', 5.0),
            'min_samples': config.get('ANOMALY_DETECTION.SPATIAL_CLUSTERING.MIN_SAMPLES', 3),
            'alert_threshold': config.get('ANOMALY_DETECTION.SPATIAL_CLUSTERING.ALERT_THRESHOLD', 5)
        }
        
        # Storage for recent reports with coordinates
        self.recent_reports = deque(maxlen=1000)
        
        # Historical cluster information
        self.historical_clusters = {}
        
    async def detect_spatial_anomalies(
        self, 
        geolocation_results: List[GeolocationResult],
        time_window_hours: int = 24
    ) -> List[AnomalyAlert]:
        """Detect spatial clustering anomalies"""
        
        # Filter reports with valid coordinates from recent time window
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_reports_with_coords = []
        
        for result in geolocation_results:
            if result.primary_location and result.primary_location.coordinates:
                report_time = result.processing_metadata.get('processed_at', datetime.now())
                if report_time >= cutoff_time:
                    recent_reports_with_coords.append({
                        'report_id': result.report_id,
                        'coordinates': result.primary_location.coordinates,
                        'timestamp': report_time,
                        'location_name': result.primary_location.text
                    })
        
        if len(recent_reports_with_coords) < self.clustering_params['min_samples']:
            return []
        
        # Perform DBSCAN clustering
        coordinates = np.array([r['coordinates'] for r in recent_reports_with_coords])
        
        # Convert lat/lon to approximate kilometers (rough approximation)
        # 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 * cos(lat) km
        lat_scale = 111.0
        lon_scale = 111.0 * np.cos(np.radians(np.mean(coordinates[:, 0])))
        
        scaled_coords = coordinates.copy()
        scaled_coords[:, 0] *= lat_scale
        scaled_coords[:, 1] *= lon_scale
        
        eps_km = self.clustering_params['eps_km']
        min_samples = self.clustering_params['min_samples']
        
        clustering = DBSCAN(eps=eps_km, min_samples=min_samples).fit(scaled_coords)
        
        # Analyze clusters
        anomaly_alerts = []
        unique_clusters = set(clustering.labels_)
        unique_clusters.discard(-1)  # Remove noise points
        
        for cluster_id in unique_clusters:
            cluster_mask = clustering.labels_ == cluster_id
            cluster_reports = [recent_reports_with_coords[i] for i in np.where(cluster_mask)[0]]
            
            if len(cluster_reports) >= self.clustering_params['alert_threshold']:
                # Calculate cluster center
                cluster_coords = np.array([r['coordinates'] for r in cluster_reports])
                center = np.mean(cluster_coords, axis=0)
                
                # Calculate radius
                distances = np.linalg.norm(cluster_coords - center, axis=1)
                radius_deg = np.max(distances)
                radius_km = radius_deg * lat_scale  # Approximate
                
                # Determine dominant location name
                location_names = [r['location_name'] for r in cluster_reports if r['location_name']]
                dominant_location = max(set(location_names), key=location_names.count) if location_names else "Unknown"
                
                # Calculate severity based on cluster size and density
                density = len(cluster_reports) / (np.pi * radius_km**2) if radius_km > 0 else len(cluster_reports)
                
                if len(cluster_reports) >= 10:
                    severity = 'critical'
                elif len(cluster_reports) >= 7:
                    severity = 'high'
                elif len(cluster_reports) >= 5:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                alert = AnomalyAlert(
                    id=f"spatial_{cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_type='geographic_cluster',
                    severity=severity,
                    location=dominant_location,
                    coordinates=tuple(center),
                    detection_time=datetime.now(),
                    time_window=f"{time_window_hours}h",
                    message=f"Spatial cluster detected: {len(cluster_reports)} reports within {radius_km:.1f}km radius near {dominant_location}",
                    metrics={
                        'cluster_size': len(cluster_reports),
                        'radius_km': radius_km,
                        'density_per_km2': density,
                        'cluster_id': int(cluster_id)
                    },
                    affected_reports=[r['report_id'] for r in cluster_reports],
                    confidence_score=min(len(cluster_reports) / 10.0, 1.0)
                )
                
                anomaly_alerts.append(alert)
        
        return anomaly_alerts

class TemporalAnomalyDetector:
    """Detect temporal anomalies in hazard reports"""
    
    def __init__(self):
        self.detection_params = {
            'volume_threshold_std': config.get('ANOMALY_DETECTION.TEMPORAL.VOLUME_THRESHOLD_STD', 2.5),
            'trend_threshold': config.get('ANOMALY_DETECTION.TEMPORAL.TREND_THRESHOLD', 0.05),
            'min_history_days': config.get('ANOMALY_DETECTION.TEMPORAL.MIN_HISTORY_DAYS', 7)
        }
        
        # Time series storage
        self.hourly_counts = defaultdict(lambda: deque(maxlen=24*30))  # 30 days of hourly data
        self.daily_counts = defaultdict(lambda: deque(maxlen=90))      # 90 days of daily data
        
        # Baseline statistics
        self.baseline_stats = {}
        
    async def update_time_series(self, processed_reports: List[ProcessedReport]) -> None:
        """Update time series data with new reports"""
        
        current_time = datetime.now()
        current_hour = current_time.replace(minute=0, second=0, microsecond=0)
        current_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Count reports by hour and day
        hourly_count = defaultdict(int)
        daily_count = defaultdict(int)
        
        for report in processed_reports:
            report_time = report.ingested_at
            report_hour = report_time.replace(minute=0, second=0, microsecond=0)
            report_day = report_time.replace(hour=0, minute=0, second=0, microsecond=0)
            
            hourly_count[report_hour] += 1
            daily_count[report_day] += 1
        
        # Update time series
        for hour, count in hourly_count.items():
            self.hourly_counts['total'].append((hour, count))
        
        for day, count in daily_count.items():
            self.daily_counts['total'].append((day, count))
        
        # Update baseline statistics
        await self._update_baseline_stats()
    
    async def _update_baseline_stats(self) -> None:
        """Update baseline statistics for anomaly detection"""
        
        # Calculate hourly baseline
        if len(self.hourly_counts['total']) >= 24:  # At least 1 day of data
            hourly_values = [count for _, count in self.hourly_counts['total']]
            self.baseline_stats['hourly_mean'] = np.mean(hourly_values)
            self.baseline_stats['hourly_std'] = np.std(hourly_values)
            self.baseline_stats['hourly_median'] = np.median(hourly_values)
        
        # Calculate daily baseline
        if len(self.daily_counts['total']) >= 7:  # At least 1 week of data
            daily_values = [count for _, count in self.daily_counts['total']]
            self.baseline_stats['daily_mean'] = np.mean(daily_values)
            self.baseline_stats['daily_std'] = np.std(daily_values)
            self.baseline_stats['daily_median'] = np.median(daily_values)
    
    async def detect_volume_spikes(
        self, 
        current_reports: List[ProcessedReport]
    ) -> List[AnomalyAlert]:
        """Detect unusual volume spikes in report frequency"""
        
        if not self.baseline_stats:
            return []
        
        alerts = []
        current_time = datetime.now()
        current_hour_count = len(current_reports)
        
        # Hourly volume spike detection
        if 'hourly_mean' in self.baseline_stats and 'hourly_std' in self.baseline_stats:
            hourly_mean = self.baseline_stats['hourly_mean']
            hourly_std = self.baseline_stats['hourly_std']
            
            if hourly_std > 0:
                z_score = (current_hour_count - hourly_mean) / hourly_std
                
                if z_score > self.detection_params['volume_threshold_std']:
                    
                    if z_score > 4.0:
                        severity = 'critical'
                    elif z_score > 3.0:
                        severity = 'high'
                    elif z_score > 2.5:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    alert = AnomalyAlert(
                        id=f"volume_spike_{current_time.strftime('%Y%m%d_%H%M%S')}",
                        alert_type='volume_spike',
                        severity=severity,
                        location=None,
                        coordinates=None,
                        detection_time=current_time,
                        time_window='1h',
                        message=f"Volume spike detected: {current_hour_count} reports vs {hourly_mean:.1f} baseline (Z-score: {z_score:.2f})",
                        metrics={
                            'current_count': current_hour_count,
                            'baseline_mean': hourly_mean,
                            'baseline_std': hourly_std,
                            'z_score': z_score,
                            'statistical_significance': stats.norm.sf(abs(z_score)) * 2
                        },
                        affected_reports=[r.id for r in current_reports],
                        confidence_score=min(z_score / 4.0, 1.0)
                    )
                    
                    alerts.append(alert)
        
        return alerts
    
    async def detect_temporal_patterns(self) -> List[TemporalPattern]:
        """Detect temporal patterns and trends"""
        
        patterns = []
        
        if not STATSMODELS_AVAILABLE:
            return patterns
        
        # Analyze daily trends
        if len(self.daily_counts['total']) >= 14:  # At least 2 weeks
            daily_data = list(self.daily_counts['total'])
            daily_data.sort(key=lambda x: x[0])
            
            dates = [item[0] for item in daily_data]
            values = [item[1] for item in daily_data]
            
            # Create time series
            ts_index = pd.date_range(start=dates[0], end=dates[-1], freq='D')
            ts_data = pd.Series(values, index=dates).reindex(ts_index, fill_value=0)
            
            try:
                # Trend analysis using linear regression
                x_numeric = np.arange(len(ts_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, ts_data.values)
                
                # Detect significant trends
                if abs(slope) > self.detection_params['trend_threshold'] and p_value < 0.05:
                    trend_type = 'increasing' if slope > 0 else 'decreasing'
                    
                    pattern = TemporalPattern(
                        pattern_type='trend',
                        time_window=f"{len(ts_data)}d",
                        baseline_value=ts_data.mean(),
                        current_value=ts_data.iloc[-1],
                        deviation_score=abs(slope),
                        statistical_significance=p_value,
                        detection_time=datetime.now()
                    )
                    
                    patterns.append(pattern)
                
                # Seasonal decomposition (if enough data)
                if len(ts_data) >= 30:
                    try:
                        decomposition = seasonal_decompose(ts_data, model='additive', period=7)
                        
                        # Check for significant seasonal component
                        seasonal_strength = np.std(decomposition.seasonal) / np.std(ts_data)
                        
                        if seasonal_strength > 0.1:  # 10% seasonal variation
                            pattern = TemporalPattern(
                                pattern_type='seasonal',
                                time_window=f"{len(ts_data)}d",
                                baseline_value=ts_data.mean(),
                                current_value=decomposition.seasonal.iloc[-1],
                                deviation_score=seasonal_strength,
                                statistical_significance=0.0,  # Not applicable for seasonal
                                detection_time=datetime.now()
                            )
                            
                            patterns.append(pattern)
                    
                    except Exception as e:
                        print(f"Seasonal decomposition error: {e}")
            
            except Exception as e:
                print(f"Trend analysis error: {e}")
        
        return patterns

class HazardTypeAnomalyDetector:
    """Detect anomalies in hazard type distributions"""
    
    def __init__(self):
        self.hazard_type_history = defaultdict(lambda: deque(maxlen=100))
        self.baseline_distributions = {}
        
    async def update_hazard_types(self, hazard_predictions: List[HazardPrediction]) -> None:
        """Update hazard type distributions"""
        
        current_time = datetime.now()
        
        # Count hazard types
        hazard_counts = defaultdict(int)
        for prediction in hazard_predictions:
            if prediction.predictions:
                top_hazard = max(prediction.predictions, key=lambda x: x['confidence'])
                hazard_counts[top_hazard['hazard_type']] += 1
        
        # Update history
        for hazard_type, count in hazard_counts.items():
            self.hazard_type_history[hazard_type].append((current_time, count))
        
        # Update baseline distributions
        await self._update_hazard_baselines()
    
    async def _update_hazard_baselines(self) -> None:
        """Update baseline hazard type distributions"""
        
        for hazard_type, history in self.hazard_type_history.items():
            if len(history) >= 10:  # Minimum history
                counts = [count for _, count in history]
                self.baseline_distributions[hazard_type] = {
                    'mean': np.mean(counts),
                    'std': np.std(counts),
                    'median': np.median(counts),
                    'percentile_95': np.percentile(counts, 95)
                }
    
    async def detect_hazard_type_anomalies(
        self, 
        current_predictions: List[HazardPrediction]
    ) -> List[AnomalyAlert]:
        """Detect anomalies in hazard type distributions"""
        
        alerts = []
        
        # Count current hazard types
        current_counts = defaultdict(int)
        for prediction in current_predictions:
            if prediction.predictions:
                top_hazard = max(prediction.predictions, key=lambda x: x['confidence'])
                current_counts[top_hazard['hazard_type']] += 1
        
        # Check each hazard type for anomalies
        for hazard_type, current_count in current_counts.items():
            if hazard_type in self.baseline_distributions:
                baseline = self.baseline_distributions[hazard_type]
                
                if baseline['std'] > 0:
                    z_score = (current_count - baseline['mean']) / baseline['std']
                    
                    # Check for significant deviations
                    if abs(z_score) > 2.0:
                        severity = 'high' if abs(z_score) > 3.0 else 'medium'
                        
                        alert = AnomalyAlert(
                            id=f"hazard_type_{hazard_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            alert_type='hazard_type_anomaly',
                            severity=severity,
                            location=None,
                            coordinates=None,
                            detection_time=datetime.now(),
                            time_window='current',
                            message=f"Unusual frequency of {hazard_type} reports: {current_count} vs {baseline['mean']:.1f} baseline",
                            metrics={
                                'hazard_type': hazard_type,
                                'current_count': current_count,
                                'baseline_mean': baseline['mean'],
                                'baseline_std': baseline['std'],
                                'z_score': z_score
                            },
                            affected_reports=[p.report_id for p in current_predictions if p.predictions and max(p.predictions, key=lambda x: x['confidence'])['hazard_type'] == hazard_type],
                            confidence_score=min(abs(z_score) / 3.0, 1.0)
                        )
                        
                        alerts.append(alert)
        
        return alerts

class AnomalyDetectionEngine:
    """Main anomaly detection engine coordinating all detectors"""
    
    def __init__(self):
        self.spatial_detector = SpatialAnomalyDetector()
        self.temporal_detector = TemporalAnomalyDetector()
        self.hazard_type_detector = HazardTypeAnomalyDetector()
        
        # Alert storage
        self.recent_alerts = deque(maxlen=1000)
        
        # Detection statistics
        self.stats = {
            'total_alerts_generated': 0,
            'alerts_by_type': defaultdict(int),
            'alerts_by_severity': defaultdict(int),
            'last_detection_run': None
        }
    
    async def detect_anomalies(
        self,
        processed_reports: List[ProcessedReport],
        hazard_predictions: List[HazardPrediction],
        geolocation_results: List[GeolocationResult]
    ) -> List[AnomalyAlert]:
        """Run comprehensive anomaly detection"""
        
        print(f"Running anomaly detection on {len(processed_reports)} reports...")
        
        all_alerts = []
        
        try:
            # Update time series data
            await self.temporal_detector.update_time_series(processed_reports)
            await self.hazard_type_detector.update_hazard_types(hazard_predictions)
            
            # Run spatial anomaly detection
            spatial_alerts = await self.spatial_detector.detect_spatial_anomalies(geolocation_results)
            all_alerts.extend(spatial_alerts)
            
            # Run temporal anomaly detection
            temporal_alerts = await self.temporal_detector.detect_volume_spikes(processed_reports)
            all_alerts.extend(temporal_alerts)
            
            # Run hazard type anomaly detection
            hazard_type_alerts = await self.hazard_type_detector.detect_hazard_type_anomalies(hazard_predictions)
            all_alerts.extend(hazard_type_alerts)
            
            # Store alerts and update statistics
            for alert in all_alerts:
                self.recent_alerts.append(alert)
                self.stats['total_alerts_generated'] += 1
                self.stats['alerts_by_type'][alert.alert_type] += 1
                self.stats['alerts_by_severity'][alert.severity] += 1
            
            self.stats['last_detection_run'] = datetime.now()
            
            print(f"Anomaly detection complete. Generated {len(all_alerts)} alerts.")
            
            return all_alerts
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return []
    
    async def get_temporal_patterns(self) -> List[TemporalPattern]:
        """Get detected temporal patterns"""
        return await self.temporal_detector.detect_temporal_patterns()
    
    def get_recent_alerts(self, hours: int = 24) -> List[AnomalyAlert]:
        """Get recent alerts within specified time window"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.recent_alerts if alert.detection_time >= cutoff_time]
    
    def get_alerts_by_severity(self, severity: str) -> List[AnomalyAlert]:
        """Get alerts by severity level"""
        return [alert for alert in self.recent_alerts if alert.severity == severity]
    
    def get_alerts_by_type(self, alert_type: str) -> List[AnomalyAlert]:
        """Get alerts by type"""
        return [alert for alert in self.recent_alerts if alert.alert_type == alert_type]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get anomaly detection statistics"""
        return {
            **self.stats,
            'alerts_by_type': dict(self.stats['alerts_by_type']),
            'alerts_by_severity': dict(self.stats['alerts_by_severity'])
        }
    
    def reset_statistics(self) -> None:
        """Reset anomaly detection statistics"""
        self.stats = {
            'total_alerts_generated': 0,
            'alerts_by_type': defaultdict(int),
            'alerts_by_severity': defaultdict(int),
            'last_detection_run': None
        }
        self.recent_alerts.clear()