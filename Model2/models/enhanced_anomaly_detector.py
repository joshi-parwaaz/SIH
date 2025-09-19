"""
Enhanced Anomaly Detection System for Disaster Management
Advanced pattern recognition with geographic clustering and ML-based analysis.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import json
import os
import statistics

logger = logging.getLogger(__name__)


class EnhancedAnomalyDetector:
    """
    Advanced anomaly detection for disaster patterns.
    Combines statistical analysis, geographic clustering, and pattern recognition.
    """
    
    def __init__(self, history_file: str = "enhanced_hazard_history.json"):
        self.history_file = history_file
        self.history_data = self._load_history()
        
        # Enhanced thresholds and parameters
        self.spike_threshold = 2.5  # Statistical significance threshold
        self.geographic_cluster_radius = 100  # km for geographic clustering
        self.time_window_hours = 24  # Primary analysis window
        self.historical_baseline_days = 30  # Days for baseline calculation
        self.confidence_threshold = 0.7  # Minimum confidence for anomaly reporting
        
        # Seasonal and cyclical patterns
        self.seasonal_weights = {
            "cyclone": [0.1, 0.1, 0.2, 0.3, 0.8, 1.0, 1.0, 0.9, 0.8, 0.6, 0.3, 0.2],  # Monthly weights
            "monsoon": [0.2, 0.1, 0.3, 0.4, 0.6, 1.0, 1.0, 0.9, 0.8, 0.5, 0.3, 0.2],
            "tsunami": [1.0] * 12,  # No seasonal pattern
            "earthquake": [1.0] * 12,  # No seasonal pattern
            "flood": [0.3, 0.2, 0.4, 0.5, 0.7, 1.0, 1.0, 0.9, 0.8, 0.6, 0.4, 0.3]
        }
        
    def _load_history(self) -> Dict:
        """Load enhanced historical data."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load enhanced history file: {e}")
        
        return {
            "events": [],
            "geographic_clusters": {},
            "temporal_patterns": {},
            "source_reliability": {},
            "hazard_baselines": {},
            "last_updated": None,
            "anomaly_log": []
        }
    
    def _save_history(self):
        """Save enhanced historical data."""
        try:
            self.history_data["last_updated"] = datetime.utcnow().isoformat()
            with open(self.history_file, 'w') as f:
                json.dump(self.history_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save enhanced history file: {e}")
    
    def _calculate_geographic_distance(self, loc1: str, loc2: str) -> float:
        """
        Calculate approximate distance between two locations.
        Simplified implementation using state/district matching.
        """
        # Simplified geographic distance calculation
        # In production, would use actual coordinates
        
        if loc1 == loc2:
            return 0.0
        
        # Define major regions and their approximate clusters
        regions = {
            "mumbai": ["maharashtra", "mumbai", "thane", "pune"],
            "delhi": ["delhi", "ncr", "gurgaon", "noida"],
            "chennai": ["tamil_nadu", "chennai", "coimbatore"],
            "kolkata": ["west_bengal", "kolkata", "howrah"],
            "bangalore": ["karnataka", "bangalore", "mysore"],
            "kerala": ["kerala", "kochi", "trivandrum", "calicut"],
            "gujarat": ["gujarat", "ahmedabad", "surat", "vadodara"],
            "odisha": ["odisha", "bhubaneswar", "cuttack"],
            "coastal": ["coast", "coastal", "shore", "beach"]
        }
        
        # Find which regions the locations belong to
        loc1_regions = []
        loc2_regions = []
        
        for region, cities in regions.items():
            if any(city in loc1.lower() for city in cities):
                loc1_regions.append(region)
            if any(city in loc2.lower() for city in cities):
                loc2_regions.append(region)
        
        # If they share a region, they're close
        if set(loc1_regions) & set(loc2_regions):
            return 25.0  # Close distance
        elif loc1_regions and loc2_regions:
            return 200.0  # Different regions
        else:
            return 100.0  # Unknown, assume medium distance
    
    def detect_geographic_clusters(self, recent_events: List[Dict]) -> List[Dict]:
        """
        Detect geographic clustering of events.
        
        Args:
            recent_events: List of recent hazard events
            
        Returns:
            List of detected geographic clusters
        """
        if len(recent_events) < 3:
            return []
        
        clusters = []
        location_groups = defaultdict(list)
        
        # Group events by approximate location
        for event in recent_events:
            location = event.get("location", "unknown").lower()
            location_groups[location].append(event)
        
        # Find clusters of events in nearby locations
        for base_location, base_events in location_groups.items():
            if len(base_events) < 2:
                continue
                
            cluster_events = base_events.copy()
            
            # Find nearby locations with events
            for other_location, other_events in location_groups.items():
                if other_location != base_location:
                    distance = self._calculate_geographic_distance(base_location, other_location)
                    if distance <= self.geographic_cluster_radius:
                        cluster_events.extend(other_events)
            
            # If we have a significant cluster
            if len(cluster_events) >= 3:
                # Calculate cluster statistics
                hazard_types = [e.get("hazard_type", "unknown") for e in cluster_events]
                hazard_counter = Counter(hazard_types)
                
                time_span = self._calculate_time_span(cluster_events)
                
                cluster = {
                    "center_location": base_location,
                    "event_count": len(cluster_events),
                    "dominant_hazard": hazard_counter.most_common(1)[0][0],
                    "hazard_distribution": dict(hazard_counter),
                    "time_span_hours": time_span,
                    "density": len(cluster_events) / max(time_span, 1),  # Events per hour
                    "confidence": min(0.95, len(cluster_events) * 0.15)
                }
                
                clusters.append(cluster)
        
        return clusters
    
    def _calculate_time_span(self, events: List[Dict]) -> float:
        """Calculate time span of events in hours."""
        try:
            timestamps = []
            for event in events:
                if "timestamp" in event:
                    ts = datetime.fromisoformat(event["timestamp"].replace('Z', '+00:00'))
                    timestamps.append(ts)
            
            if len(timestamps) < 2:
                return 1.0
                
            time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600
            return max(time_span, 0.1)  # Minimum 0.1 hours
            
        except Exception as e:
            logger.warning(f"Error calculating time span: {e}")
            return 1.0
    
    def detect_temporal_anomalies(self, recent_events: List[Dict]) -> List[Dict]:
        """
        Detect temporal patterns and anomalies.
        
        Args:
            recent_events: List of recent events
            
        Returns:
            List of temporal anomalies detected
        """
        anomalies = []
        
        if len(recent_events) < 5:
            return anomalies
        
        # Analyze event frequency patterns
        hazard_frequency = defaultdict(list)
        
        # Group events by hazard type and extract timestamps
        for event in recent_events:
            hazard_type = event.get("hazard_type", "unknown")
            try:
                timestamp = datetime.fromisoformat(event["timestamp"].replace('Z', '+00:00'))
                hazard_frequency[hazard_type].append(timestamp)
            except:
                continue
        
        # Detect frequency spikes for each hazard type
        for hazard_type, timestamps in hazard_frequency.items():
            if len(timestamps) < 3:
                continue
            
            # Calculate event frequency (events per hour)
            time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600
            current_frequency = len(timestamps) / max(time_span, 1)
            
            # Get historical baseline
            historical_baseline = self._get_historical_baseline(hazard_type)
            
            # Apply seasonal adjustment
            seasonal_factor = self._get_seasonal_factor(hazard_type)
            adjusted_baseline = historical_baseline * seasonal_factor
            
            # Detect anomaly
            if current_frequency > adjusted_baseline * self.spike_threshold:
                anomaly = {
                    "type": "frequency_spike",
                    "hazard_type": hazard_type,
                    "current_frequency": current_frequency,
                    "expected_frequency": adjusted_baseline,
                    "spike_factor": current_frequency / adjusted_baseline if adjusted_baseline > 0 else float('inf'),
                    "confidence": min(0.95, (current_frequency / adjusted_baseline) * 0.1),
                    "time_window_hours": time_span,
                    "event_count": len(timestamps)
                }
                anomalies.append(anomaly)
        
        return anomalies
    
    def _get_historical_baseline(self, hazard_type: str) -> float:
        """Get historical baseline frequency for a hazard type."""
        if "hazard_baselines" not in self.history_data:
            self.history_data["hazard_baselines"] = {}
        
        baseline = self.history_data["hazard_baselines"].get(hazard_type, 0.1)
        return max(baseline, 0.05)  # Minimum baseline
    
    def _get_seasonal_factor(self, hazard_type: str) -> float:
        """Get seasonal adjustment factor."""
        current_month = datetime.now().month - 1  # 0-indexed
        
        if hazard_type in self.seasonal_weights:
            return self.seasonal_weights[hazard_type][current_month]
        else:
            return 1.0  # No seasonal adjustment for unknown hazards
    
    def detect_cross_source_anomalies(self, events_by_source: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Detect anomalies across different data sources.
        
        Args:
            events_by_source: Events grouped by source
            
        Returns:
            List of cross-source anomalies
        """
        anomalies = []
        
        if len(events_by_source) < 2:
            return anomalies
        
        # Analyze source agreement patterns
        hazard_by_source = {}
        for source, events in events_by_source.items():
            hazard_counts = Counter(event.get("hazard_type") for event in events)
            hazard_by_source[source] = hazard_counts
        
        # Find discrepancies between sources
        all_hazard_types = set()
        for counts in hazard_by_source.values():
            all_hazard_types.update(counts.keys())
        
        for hazard_type in all_hazard_types:
            source_counts = {}
            for source, counts in hazard_by_source.items():
                source_counts[source] = counts.get(hazard_type, 0)
            
            if len(source_counts) >= 2:
                values = list(source_counts.values())
                if max(values) > 0:
                    # Calculate coefficient of variation
                    mean_val = statistics.mean(values)
                    if mean_val > 0:
                        std_val = statistics.stdev(values) if len(values) > 1 else 0
                        cv = std_val / mean_val
                        
                        # High variation indicates potential anomaly
                        if cv > 1.0:  # High disagreement threshold
                            anomaly = {
                                "type": "source_disagreement",
                                "hazard_type": hazard_type,
                                "source_counts": source_counts,
                                "coefficient_variation": cv,
                                "confidence": min(0.9, cv * 0.3),
                                "description": f"High disagreement between sources on {hazard_type} reports"
                            }
                            anomalies.append(anomaly)
        
        return anomalies
    
    def analyze_all_anomalies(self, reports: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive anomaly analysis.
        
        Args:
            reports: List of hazard reports
            
        Returns:
            Dict containing all anomaly analysis results
        """
        start_time = datetime.utcnow()
        
        # Convert reports to events format
        events = []
        events_by_source = defaultdict(list)
        
        for report in reports:
            event = {
                "timestamp": start_time.isoformat(),
                "hazard_type": report.get("hazard_type", "unknown"),
                "location": report.get("location", "unknown"),
                "severity": report.get("severity", "medium"),
                "source": report.get("source", "unknown"),
                "confidence": report.get("confidence", 0.5)
            }
            events.append(event)
            events_by_source[event["source"]].append(event)
        
        # Run all anomaly detection methods
        geographic_clusters = self.detect_geographic_clusters(events)
        temporal_anomalies = self.detect_temporal_anomalies(events)
        cross_source_anomalies = self.detect_cross_source_anomalies(events_by_source)
        
        # Update historical data
        self._update_baselines(events)
        self._save_history()
        
        # Compile results
        analysis_result = {
            "analysis_timestamp": start_time.isoformat(),
            "total_events_analyzed": len(events),
            "geographic_clusters": geographic_clusters,
            "temporal_anomalies": temporal_anomalies,
            "cross_source_anomalies": cross_source_anomalies,
            "total_anomalies": len(geographic_clusters) + len(temporal_anomalies) + len(cross_source_anomalies),
            "anomaly_summary": {
                "geographic_clusters": len(geographic_clusters),
                "temporal_spikes": len(temporal_anomalies),
                "source_disagreements": len(cross_source_anomalies)
            },
            "confidence_distribution": {
                "high_confidence": len([a for cluster in [geographic_clusters, temporal_anomalies, cross_source_anomalies] 
                                      for a in cluster if a.get("confidence", 0) >= 0.8]),
                "medium_confidence": len([a for cluster in [geographic_clusters, temporal_anomalies, cross_source_anomalies] 
                                        for a in cluster if 0.5 <= a.get("confidence", 0) < 0.8]),
                "low_confidence": len([a for cluster in [geographic_clusters, temporal_anomalies, cross_source_anomalies] 
                                     for a in cluster if a.get("confidence", 0) < 0.5])
            }
        }
        
        # Log significant anomalies
        if analysis_result["total_anomalies"] > 0:
            self._log_anomalies(analysis_result)
        
        return analysis_result
    
    def _update_baselines(self, events: List[Dict]):
        """Update historical baselines with new data."""
        hazard_counts = Counter(event.get("hazard_type") for event in events)
        
        for hazard_type, count in hazard_counts.items():
            # Simple exponential moving average update
            current_baseline = self._get_historical_baseline(hazard_type)
            current_frequency = count / self.time_window_hours  # events per hour
            
            # Update with 10% weight for new data
            new_baseline = current_baseline * 0.9 + current_frequency * 0.1
            self.history_data["hazard_baselines"][hazard_type] = new_baseline
    
    def _log_anomalies(self, analysis_result: Dict):
        """Log anomalies for historical tracking."""
        if "anomaly_log" not in self.history_data:
            self.history_data["anomaly_log"] = []
        
        log_entry = {
            "timestamp": analysis_result["analysis_timestamp"],
            "total_anomalies": analysis_result["total_anomalies"],
            "summary": analysis_result["anomaly_summary"]
        }
        
        self.history_data["anomaly_log"].append(log_entry)
        
        # Keep only last 100 log entries
        if len(self.history_data["anomaly_log"]) > 100:
            self.history_data["anomaly_log"] = self.history_data["anomaly_log"][-100:]
        
        logger.info(f"Detected {analysis_result['total_anomalies']} anomalies: {analysis_result['anomaly_summary']}")


if __name__ == "__main__":
    # Test the enhanced anomaly detector
    print("🔍 Testing Enhanced Anomaly Detection System")
    print("=" * 60)
    
    # Initialize detector
    detector = EnhancedAnomalyDetector()
    
    # Sample reports for testing
    sample_reports = [
        {"hazard_type": "cyclone", "location": "gujarat", "source": "INCOIS", "confidence": 0.9},
        {"hazard_type": "cyclone", "location": "gujarat", "source": "Twitter", "confidence": 0.8},
        {"hazard_type": "cyclone", "location": "mumbai", "source": "Government", "confidence": 0.95},
        {"hazard_type": "flood", "location": "mumbai", "source": "YouTube", "confidence": 0.7},
        {"hazard_type": "flood", "location": "mumbai", "source": "Google_News", "confidence": 0.85},
        {"hazard_type": "tsunami", "location": "kerala", "source": "INCOIS", "confidence": 0.9},
        {"hazard_type": "earthquake", "location": "delhi", "source": "Twitter", "confidence": 0.6},
    ]
    
    # Run anomaly analysis
    results = detector.analyze_all_anomalies(sample_reports)
    
    # Display results
    print(f"📊 Anomaly Analysis Results:")
    print(f"  - Total events analyzed: {results['total_events_analyzed']}")
    print(f"  - Total anomalies detected: {results['total_anomalies']}")
    print(f"  - Geographic clusters: {results['anomaly_summary']['geographic_clusters']}")
    print(f"  - Temporal spikes: {results['anomaly_summary']['temporal_spikes']}")
    print(f"  - Source disagreements: {results['anomaly_summary']['source_disagreements']}")
    
    # Show detected clusters
    if results['geographic_clusters']:
        print(f"\n🗺️ Geographic Clusters Detected:")
        for i, cluster in enumerate(results['geographic_clusters'], 1):
            print(f"  {i}. {cluster['center_location']}: {cluster['event_count']} events")
            print(f"     Dominant hazard: {cluster['dominant_hazard']}")
            print(f"     Confidence: {cluster['confidence']:.3f}")
    
    # Show temporal anomalies
    if results['temporal_anomalies']:
        print(f"\n⏰ Temporal Anomalies Detected:")
        for i, anomaly in enumerate(results['temporal_anomalies'], 1):
            print(f"  {i}. {anomaly['hazard_type']}: {anomaly['spike_factor']:.1f}x normal frequency")
            print(f"     Confidence: {anomaly['confidence']:.3f}")
    
    print(f"\n" + "=" * 60)
    print("✅ Enhanced Anomaly Detection Test Complete!")