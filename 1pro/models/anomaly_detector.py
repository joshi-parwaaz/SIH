"""
Anomaly Detection Module for Ocean Hazard Analysis.
Detects sudden surges in event types, regions, and patterns.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import statistics
import json
import os

logger = logging.getLogger(__name__)


class HazardAnomalyDetector:
    """Detects anomalies and spikes in hazard patterns."""
    
    def __init__(self, history_file: str = "hazard_history.json"):
        self.history_file = history_file
        self.history_data = self._load_history()
        
        # Anomaly thresholds
        self.spike_threshold = 2.0  # 2x standard deviation
        self.frequency_threshold = 3  # Minimum events to consider
        self.time_window_hours = 24  # Look back window
        
    def _load_history(self) -> Dict:
        """Load historical hazard data from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load history file: {e}")
        
        return {
            "events": [],
            "region_counts": {},
            "hazard_type_counts": {},
            "hourly_patterns": {},
            "last_updated": None
        }
    
    def _save_history(self):
        """Save historical data to file."""
        try:
            self.history_data["last_updated"] = datetime.utcnow().isoformat()
            with open(self.history_file, 'w') as f:
                json.dump(self.history_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save history file: {e}")
    
    def update_history(self, reports: List[Dict]):
        """
        Update historical data with new reports.
        
        Args:
            reports: List of hazard reports from current analysis
        """
        current_time = datetime.utcnow()
        
        for report in reports:
            event = {
                "timestamp": current_time.isoformat(),
                "hazard_type": getattr(report, "hazard_type", "unknown"),
                "location": getattr(report, "location", "unknown"),
                "severity": getattr(report, "severity", "medium"),
                "source": getattr(report, "source", "unknown"),
                "confidence": getattr(report, "confidence", 0.5)
            }
            
            self.history_data["events"].append(event)
            
            # Update region counts
            location = event["location"]
            if location not in self.history_data["region_counts"]:
                self.history_data["region_counts"][location] = []
            self.history_data["region_counts"][location].append(current_time.isoformat())
            
            # Update hazard type counts
            hazard_type = event["hazard_type"]
            if hazard_type not in self.history_data["hazard_type_counts"]:
                self.history_data["hazard_type_counts"][hazard_type] = []
            self.history_data["hazard_type_counts"][hazard_type].append(current_time.isoformat())
            
            # Update hourly patterns
            hour_key = current_time.strftime("%H")
            if hour_key not in self.history_data["hourly_patterns"]:
                self.history_data["hourly_patterns"][hour_key] = 0
            self.history_data["hourly_patterns"][hour_key] += 1
        
        # Clean old events (keep last 30 days)
        self._cleanup_old_events()
        
        # Save updated history
        self._save_history()
    
    def _cleanup_old_events(self):
        """Remove events older than 30 days."""
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        # Filter events
        self.history_data["events"] = [
            event for event in self.history_data["events"]
            if datetime.fromisoformat(event["timestamp"].replace('Z', '+00:00')) > cutoff_date
        ]
        
        # Clean region counts
        for region in self.history_data["region_counts"]:
            self.history_data["region_counts"][region] = [
                timestamp for timestamp in self.history_data["region_counts"][region]
                if datetime.fromisoformat(timestamp.replace('Z', '+00:00')) > cutoff_date
            ]
        
        # Clean hazard type counts
        for hazard_type in self.history_data["hazard_type_counts"]:
            self.history_data["hazard_type_counts"][hazard_type] = [
                timestamp for timestamp in self.history_data["hazard_type_counts"][hazard_type]
                if datetime.fromisoformat(timestamp.replace('Z', '+00:00')) > cutoff_date
            ]
    
    def detect_regional_spikes(self) -> List[Dict]:
        """
        Detect sudden spikes in regional activity.
        
        Returns:
            List of detected regional anomalies
        """
        anomalies = []
        current_time = datetime.utcnow()
        
        for region, timestamps in self.history_data["region_counts"].items():
            if len(timestamps) < self.frequency_threshold:
                continue
            
            # Count events in recent time windows
            recent_events = []
            historical_events = []
            
            for timestamp_str in timestamps:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                hours_ago = (current_time - timestamp).total_seconds() / 3600
                
                if hours_ago <= self.time_window_hours:
                    recent_events.append(timestamp)
                elif hours_ago <= self.time_window_hours * 7:  # Last week for comparison
                    historical_events.append(timestamp)
            
            if len(recent_events) >= self.frequency_threshold and len(historical_events) > 0:
                # Calculate rates
                recent_rate = len(recent_events) / self.time_window_hours
                historical_rate = len(historical_events) / (self.time_window_hours * 6)  # 6 days average
                
                # Check for spike
                if recent_rate > historical_rate * self.spike_threshold:
                    spike_severity = recent_rate / max(historical_rate, 0.1)
                    
                    anomaly = {
                        "type": "regional_spike",
                        "region": region,
                        "recent_events": len(recent_events),
                        "historical_average": historical_rate * self.time_window_hours,
                        "spike_factor": spike_severity,
                        "severity": "high" if spike_severity > 5 else "medium",
                        "confidence": min(0.9, 0.5 + (spike_severity - 2) * 0.1),
                        "detected_at": current_time.isoformat()
                    }
                    anomalies.append(anomaly)
        
        return anomalies
    
    def detect_hazard_type_spikes(self) -> List[Dict]:
        """
        Detect sudden increases in specific hazard types.
        
        Returns:
            List of detected hazard type anomalies
        """
        anomalies = []
        current_time = datetime.utcnow()
        
        for hazard_type, timestamps in self.history_data["hazard_type_counts"].items():
            if len(timestamps) < self.frequency_threshold:
                continue
            
            # Count recent vs historical
            recent_count = 0
            historical_counts = []
            
            # Group by days for trend analysis
            day_counts = defaultdict(int)
            
            for timestamp_str in timestamps:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                hours_ago = (current_time - timestamp).total_seconds() / 3600
                day_key = timestamp.strftime("%Y-%m-%d")
                
                if hours_ago <= self.time_window_hours:
                    recent_count += 1
                elif hours_ago <= self.time_window_hours * 7:  # Last week
                    day_counts[day_key] += 1
            
            if recent_count >= self.frequency_threshold and day_counts:
                historical_average = statistics.mean(day_counts.values()) if day_counts else 0
                
                if recent_count > historical_average * self.spike_threshold:
                    anomaly = {
                        "type": "hazard_type_spike",
                        "hazard_type": hazard_type,
                        "recent_count": recent_count,
                        "historical_average": historical_average,
                        "spike_factor": recent_count / max(historical_average, 0.1),
                        "severity": "high" if recent_count > historical_average * 4 else "medium",
                        "confidence": min(0.9, 0.6 + (recent_count - historical_average) * 0.05),
                        "detected_at": current_time.isoformat()
                    }
                    anomalies.append(anomaly)
        
        return anomalies
    
    def detect_temporal_anomalies(self) -> List[Dict]:
        """
        Detect unusual temporal patterns (e.g., events at unusual hours).
        
        Returns:
            List of temporal anomalies
        """
        anomalies = []
        
        if not self.history_data["hourly_patterns"]:
            return anomalies
        
        # Calculate expected vs actual hourly distribution
        hourly_counts = self.history_data["hourly_patterns"]
        total_events = sum(hourly_counts.values())
        
        if total_events < 20:  # Need sufficient data
            return anomalies
        
        # Expected distribution (roughly uniform with some variation)
        expected_per_hour = total_events / 24
        
        current_hour = datetime.utcnow().strftime("%H")
        current_hour_count = hourly_counts.get(current_hour, 0)
        
        # Check if current hour has unusual activity
        if current_hour_count > expected_per_hour * 3:  # 3x normal
            anomaly = {
                "type": "temporal_anomaly",
                "hour": current_hour,
                "event_count": current_hour_count,
                "expected_count": expected_per_hour,
                "anomaly_factor": current_hour_count / max(expected_per_hour, 1),
                "severity": "medium",
                "confidence": 0.7,
                "detected_at": datetime.utcnow().isoformat()
            }
            anomalies.append(anomaly)
        
        return anomalies
    
    def analyze_all_anomalies(self, current_reports: List[Dict]) -> Dict:
        """
        Comprehensive anomaly analysis.
        
        Args:
            current_reports: Latest hazard reports
            
        Returns:
            Complete anomaly analysis results
        """
        # Update history with current reports
        self.update_history(current_reports)
        
        # Detect all types of anomalies
        regional_spikes = self.detect_regional_spikes()
        hazard_spikes = self.detect_hazard_type_spikes()
        temporal_anomalies = self.detect_temporal_anomalies()
        
        all_anomalies = regional_spikes + hazard_spikes + temporal_anomalies
        
        # Categorize by severity
        high_severity = [a for a in all_anomalies if a.get("severity") == "high"]
        medium_severity = [a for a in all_anomalies if a.get("severity") == "medium"]
        
        analysis = {
            "total_anomalies": len(all_anomalies),
            "high_severity_count": len(high_severity),
            "medium_severity_count": len(medium_severity),
            "anomalies_by_type": {
                "regional_spikes": regional_spikes,
                "hazard_type_spikes": hazard_spikes,
                "temporal_anomalies": temporal_anomalies
            },
            "all_anomalies": all_anomalies,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "alert_level": self._calculate_alert_level(all_anomalies)
        }
        
        return analysis
    
    def _calculate_alert_level(self, anomalies: List[Dict]) -> str:
        """Calculate overall alert level based on anomalies."""
        if not anomalies:
            return "normal"
        
        high_count = sum(1 for a in anomalies if a.get("severity") == "high")
        total_count = len(anomalies)
        
        if high_count >= 2 or total_count >= 5:
            return "critical"
        elif high_count >= 1 or total_count >= 3:
            return "elevated"
        else:
            return "moderate"


if __name__ == "__main__":
    # Test anomaly detection
    detector = HazardAnomalyDetector("test_history.json")
    
    # Simulate some test reports
    test_reports = [
        {
            "hazard_type": "flood",
            "location": "mumbai",
            "severity": "high",
            "source": "twitter",
            "confidence": 0.8
        },
        {
            "hazard_type": "flood", 
            "location": "mumbai",
            "severity": "extreme",
            "source": "incois",
            "confidence": 0.9
        }
    ]
    
    analysis = detector.analyze_all_anomalies(test_reports)
    print(f"Anomaly Analysis Results:")
    print(f"Total anomalies: {analysis['total_anomalies']}")
    print(f"Alert level: {analysis['alert_level']}")
    
    for anomaly in analysis['all_anomalies']:
        print(f"- {anomaly['type']}: {anomaly}")