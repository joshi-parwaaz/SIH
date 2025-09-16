import json
import csv
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import geojson
from collections import defaultdict

from ..data_ingestion import RawReport
from ..preprocessing import ProcessedReport
from ..nlp_analysis import HazardPrediction
from ..geolocation import GeolocationResult, LocationEntity
from ..anomaly_detection import AnomalyAlert
from ..feedback import OperatorFeedback

class GeoJSONExporter:
    """Export data in GeoJSON format for map visualizations"""
    
    @staticmethod
    def export_hazard_points(
        hazard_predictions: List[HazardPrediction],
        geolocation_results: List[GeolocationResult]
    ) -> Dict[str, Any]:
        """Export hazard reports as GeoJSON point features"""
        
        # Create lookup for geolocation results
        geo_lookup = {result.report_id: result for result in geolocation_results}
        
        features = []
        
        for prediction in hazard_predictions:
            if not prediction.is_hazard:
                continue
                
            geo_result = geo_lookup.get(prediction.report_id)
            if not geo_result or not geo_result.primary_location or not geo_result.primary_location.coordinates:
                continue
            
            coordinates = geo_result.primary_location.coordinates
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [coordinates[1], coordinates[0]]  # GeoJSON uses [lon, lat]
                },
                "properties": {
                    "report_id": prediction.report_id,
                    "hazard_type": prediction.hazard_type,
                    "severity": prediction.severity,
                    "urgency": prediction.urgency,
                    "confidence": prediction.confidence,
                    "sentiment": prediction.sentiment,
                    "sentiment_score": prediction.sentiment_score,
                    "misinformation_probability": prediction.misinformation_probability,
                    "location_name": geo_result.primary_location.text,
                    "location_address": geo_result.primary_location.address,
                    "analyzed_at": prediction.processing_metadata.get('analyzed_at', '').isoformat() if prediction.processing_metadata.get('analyzed_at') else None,
                    "marker_color": GeoJSONExporter._get_severity_color(prediction.severity),
                    "marker_size": GeoJSONExporter._get_urgency_size(prediction.urgency)
                }
            }
            
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_features": len(features),
                "export_time": datetime.now().isoformat(),
                "data_type": "hazard_reports"
            }
        }
    
    @staticmethod
    def export_anomaly_zones(anomaly_alerts: List[AnomalyAlert]) -> Dict[str, Any]:
        """Export anomaly alerts as GeoJSON features"""
        
        features = []
        
        for alert in anomaly_alerts:
            if not alert.coordinates:
                continue
            
            # Create different geometry based on alert type
            if alert.alert_type == 'geographic_cluster':
                # Create circle for spatial clusters
                radius_km = alert.metrics.get('radius_km', 1.0)
                center = alert.coordinates
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [center[1], center[0]]
                    },
                    "properties": {
                        "alert_id": alert.id,
                        "alert_type": alert.alert_type,
                        "severity": alert.severity,
                        "message": alert.message,
                        "detection_time": alert.detection_time.isoformat(),
                        "affected_reports_count": len(alert.affected_reports),
                        "cluster_size": alert.metrics.get('cluster_size', 0),
                        "radius_km": radius_km,
                        "confidence_score": alert.confidence_score,
                        "marker_color": GeoJSONExporter._get_alert_color(alert.severity),
                        "stroke_width": GeoJSONExporter._get_alert_stroke_width(alert.severity)
                    }
                }
            else:
                # Point feature for other alert types
                feature = {
                    "type": "Feature", 
                    "geometry": {
                        "type": "Point",
                        "coordinates": [alert.coordinates[1], alert.coordinates[0]]
                    },
                    "properties": {
                        "alert_id": alert.id,
                        "alert_type": alert.alert_type,
                        "severity": alert.severity,
                        "message": alert.message,
                        "detection_time": alert.detection_time.isoformat(),
                        "confidence_score": alert.confidence_score,
                        "marker_color": GeoJSONExporter._get_alert_color(alert.severity)
                    }
                }
            
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_features": len(features),
                "export_time": datetime.now().isoformat(),
                "data_type": "anomaly_alerts"
            }
        }
    
    @staticmethod
    def _get_severity_color(severity: str) -> str:
        """Get color for severity level"""
        color_map = {
            'low': '#32CD32',      # Green
            'medium': '#FFD700',   # Gold
            'high': '#FF4500',     # Red Orange  
            'critical': '#DC143C', # Crimson
            'immediate': '#8B0000' # Dark Red
        }
        return color_map.get(severity.lower(), '#808080')  # Default gray
    
    @staticmethod
    def _get_urgency_size(urgency: str) -> str:
        """Get marker size for urgency level"""
        size_map = {
            'low': 'small',
            'medium': 'medium', 
            'high': 'large',
            'immediate': 'extra-large'
        }
        return size_map.get(urgency.lower(), 'medium')
    
    @staticmethod
    def _get_alert_color(severity: str) -> str:
        """Get color for alert severity"""
        color_map = {
            'low': '#FFA500',      # Orange
            'medium': '#FF6347',   # Tomato
            'high': '#FF0000',     # Red
            'critical': '#8B0000'  # Dark Red
        }
        return color_map.get(severity.lower(), '#808080')
    
    @staticmethod
    def _get_alert_stroke_width(severity: str) -> int:
        """Get stroke width for alert severity"""
        width_map = {
            'low': 2,
            'medium': 3,
            'high': 4, 
            'critical': 5
        }
        return width_map.get(severity.lower(), 2)

class TimeSeriesExporter:
    """Export time series data for charts and trends"""
    
    @staticmethod
    def export_hazard_frequency_over_time(
        hazard_predictions: List[HazardPrediction],
        time_interval: str = 'hour'  # 'hour', 'day', 'week'
    ) -> Dict[str, Any]:
        """Export hazard frequency data over time"""
        
        # Group hazards by time interval
        time_groups = defaultdict(int)
        hazard_type_groups = defaultdict(lambda: defaultdict(int))
        
        for prediction in hazard_predictions:
            if not prediction.is_hazard:
                continue
                
            timestamp = prediction.processing_metadata.get('analyzed_at')
            if not timestamp:
                continue
            
            # Round timestamp to interval
            if time_interval == 'hour':
                time_key = timestamp.replace(minute=0, second=0, microsecond=0)
            elif time_interval == 'day':
                time_key = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            elif time_interval == 'week':
                days_since_monday = timestamp.weekday()
                time_key = timestamp.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
            else:
                time_key = timestamp
            
            time_groups[time_key] += 1
            hazard_type_groups[time_key][prediction.hazard_type] += 1
        
        # Convert to chart-friendly format
        timestamps = sorted(time_groups.keys())
        
        chart_data = {
            "timestamps": [ts.isoformat() for ts in timestamps],
            "total_hazards": [time_groups[ts] for ts in timestamps],
            "hazard_types": {},
            "metadata": {
                "time_interval": time_interval,
                "total_data_points": len(timestamps),
                "date_range": {
                    "start": timestamps[0].isoformat() if timestamps else None,
                    "end": timestamps[-1].isoformat() if timestamps else None
                }
            }
        }
        
        # Add hazard type breakdown
        all_hazard_types = set()
        for type_counts in hazard_type_groups.values():
            all_hazard_types.update(type_counts.keys())
        
        for hazard_type in all_hazard_types:
            chart_data["hazard_types"][hazard_type] = [
                hazard_type_groups[ts].get(hazard_type, 0) for ts in timestamps
            ]
        
        return chart_data
    
    @staticmethod
    def export_sentiment_trends(
        hazard_predictions: List[HazardPrediction],
        time_interval: str = 'day'
    ) -> Dict[str, Any]:
        """Export sentiment analysis trends over time"""
        
        time_groups = defaultdict(list)
        
        for prediction in hazard_predictions:
            if not prediction.is_hazard:
                continue
                
            timestamp = prediction.processing_metadata.get('analyzed_at')
            if not timestamp:
                continue
            
            # Round timestamp to interval
            if time_interval == 'hour':
                time_key = timestamp.replace(minute=0, second=0, microsecond=0)
            elif time_interval == 'day':
                time_key = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                time_key = timestamp
            
            time_groups[time_key].append({
                'sentiment': prediction.sentiment,
                'sentiment_score': prediction.sentiment_score,
                'urgency': prediction.urgency
            })
        
        # Calculate averages and distributions
        timestamps = sorted(time_groups.keys())
        
        chart_data = {
            "timestamps": [ts.isoformat() for ts in timestamps],
            "average_sentiment_score": [],
            "sentiment_distribution": {
                "positive": [],
                "neutral": [], 
                "negative": []
            },
            "urgency_distribution": {
                "low": [],
                "medium": [],
                "high": [],
                "immediate": []
            },
            "metadata": {
                "time_interval": time_interval,
                "total_data_points": len(timestamps)
            }
        }
        
        for ts in timestamps:
            sentiments = time_groups[ts]
            
            # Average sentiment score
            avg_score = sum(s['sentiment_score'] for s in sentiments) / len(sentiments)
            chart_data["average_sentiment_score"].append(avg_score)
            
            # Sentiment distribution
            sentiment_counts = defaultdict(int)
            urgency_counts = defaultdict(int)
            
            for s in sentiments:
                sentiment_counts[s['sentiment']] += 1
                urgency_counts[s['urgency']] += 1
            
            total_sentiments = len(sentiments)
            
            chart_data["sentiment_distribution"]["positive"].append(
                sentiment_counts.get('positive', 0) / total_sentiments * 100
            )
            chart_data["sentiment_distribution"]["neutral"].append(
                sentiment_counts.get('neutral', 0) / total_sentiments * 100
            )
            chart_data["sentiment_distribution"]["negative"].append(
                sentiment_counts.get('negative', 0) / total_sentiments * 100
            )
            
            chart_data["urgency_distribution"]["low"].append(
                urgency_counts.get('low', 0) / total_sentiments * 100
            )
            chart_data["urgency_distribution"]["medium"].append(
                urgency_counts.get('medium', 0) / total_sentiments * 100
            )
            chart_data["urgency_distribution"]["high"].append(
                urgency_counts.get('high', 0) / total_sentiments * 100
            )
            chart_data["urgency_distribution"]["immediate"].append(
                urgency_counts.get('immediate', 0) / total_sentiments * 100
            )
        
        return chart_data

class StatisticsExporter:
    """Export statistical summaries for dashboards"""
    
    @staticmethod
    def export_hazard_summary(
        hazard_predictions: List[HazardPrediction],
        geolocation_results: List[GeolocationResult],
        anomaly_alerts: List[AnomalyAlert]
    ) -> Dict[str, Any]:
        """Export comprehensive hazard summary statistics"""
        
        # Basic counts
        total_reports = len(hazard_predictions)
        hazard_reports = len([p for p in hazard_predictions if p.is_hazard])
        
        # Hazard type distribution
        hazard_type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        urgency_counts = defaultdict(int)
        
        for prediction in hazard_predictions:
            if prediction.is_hazard:
                hazard_type_counts[prediction.hazard_type] += 1
                severity_counts[prediction.severity] += 1
                urgency_counts[prediction.urgency] += 1
        
        # Location coverage
        geo_lookup = {result.report_id: result for result in geolocation_results}
        located_reports = 0
        state_counts = defaultdict(int)
        
        for prediction in hazard_predictions:
            if not prediction.is_hazard:
                continue
                
            geo_result = geo_lookup.get(prediction.report_id)
            if geo_result and geo_result.primary_location and geo_result.primary_location.coordinates:
                located_reports += 1
                
                # Count by state if available
                for location in geo_result.extracted_locations:
                    if location.state:
                        state_counts[location.state] += 1
        
        # Misinformation statistics
        misinformation_reports = len([
            p for p in hazard_predictions 
            if p.is_hazard and p.misinformation_probability > 0.5
        ])
        
        # Anomaly statistics
        anomaly_counts = defaultdict(int)
        for alert in anomaly_alerts:
            anomaly_counts[alert.alert_type] += 1
        
        return {
            "overview": {
                "total_reports_processed": total_reports,
                "hazard_reports_detected": hazard_reports,
                "hazard_detection_rate": (hazard_reports / total_reports * 100) if total_reports > 0 else 0,
                "reports_with_location": located_reports,
                "location_extraction_rate": (located_reports / hazard_reports * 100) if hazard_reports > 0 else 0,
                "misinformation_flagged": misinformation_reports,
                "misinformation_rate": (misinformation_reports / hazard_reports * 100) if hazard_reports > 0 else 0,
                "anomaly_alerts_generated": len(anomaly_alerts)
            },
            "hazard_types": dict(hazard_type_counts),
            "severity_distribution": dict(severity_counts),
            "urgency_distribution": dict(urgency_counts),
            "geographic_distribution": dict(state_counts),
            "anomaly_types": dict(anomaly_counts),
            "generated_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def export_performance_metrics(
        processing_times: List[float],
        model_accuracies: Dict[str, float]
    ) -> Dict[str, Any]:
        """Export performance metrics"""
        
        if processing_times:
            avg_processing_time = sum(processing_times) / len(processing_times)
            max_processing_time = max(processing_times)
            min_processing_time = min(processing_times)
        else:
            avg_processing_time = max_processing_time = min_processing_time = 0
        
        return {
            "processing_performance": {
                "average_processing_time_ms": avg_processing_time,
                "max_processing_time_ms": max_processing_time,
                "min_processing_time_ms": min_processing_time,
                "total_requests_processed": len(processing_times)
            },
            "model_accuracies": model_accuracies,
            "system_health": {
                "status": "healthy" if avg_processing_time < 5000 else "degraded",  # 5 second threshold
                "performance_grade": "A" if avg_processing_time < 1000 else "B" if avg_processing_time < 3000 else "C"
            },
            "generated_at": datetime.now().isoformat()
        }

class CSVExporter:
    """Export data in CSV format for analysis"""
    
    @staticmethod
    def export_hazard_reports_csv(
        hazard_predictions: List[HazardPrediction],
        geolocation_results: List[GeolocationResult],
        output_file: str
    ) -> str:
        """Export hazard reports to CSV file"""
        
        geo_lookup = {result.report_id: result for result in geolocation_results}
        
        # Prepare data rows
        rows = []
        for prediction in hazard_predictions:
            geo_result = geo_lookup.get(prediction.report_id)
            
            row = {
                'report_id': prediction.report_id,
                'is_hazard': prediction.is_hazard,
                'hazard_type': prediction.hazard_type,
                'confidence': prediction.confidence,
                'severity': prediction.severity,
                'urgency': prediction.urgency,
                'sentiment': prediction.sentiment,
                'sentiment_score': prediction.sentiment_score,
                'misinformation_probability': prediction.misinformation_probability,
                'analyzed_at': prediction.processing_metadata.get('analyzed_at', ''),
                'location_name': '',
                'location_address': '',
                'latitude': '',
                'longitude': '',
                'location_confidence': ''
            }
            
            if geo_result and geo_result.primary_location:
                row.update({
                    'location_name': geo_result.primary_location.text or '',
                    'location_address': geo_result.primary_location.address or '',
                    'latitude': geo_result.primary_location.coordinates[0] if geo_result.primary_location.coordinates else '',
                    'longitude': geo_result.primary_location.coordinates[1] if geo_result.primary_location.coordinates else '',
                    'location_confidence': geo_result.confidence_score
                })
            
            rows.append(row)
        
        # Write to CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
        
        return output_file

class VisualizationDataExporter:
    """Main class for exporting data for visualization"""
    
    def __init__(self):
        self.geojson_exporter = GeoJSONExporter()
        self.timeseries_exporter = TimeSeriesExporter()
        self.stats_exporter = StatisticsExporter()
        self.csv_exporter = CSVExporter()
    
    def export_for_dashboard(
        self,
        hazard_predictions: List[HazardPrediction],
        geolocation_results: List[GeolocationResult],
        anomaly_alerts: List[AnomalyAlert]
    ) -> Dict[str, Any]:
        """Export comprehensive data package for dashboard"""
        
        return {
            "map_data": {
                "hazard_points": self.geojson_exporter.export_hazard_points(
                    hazard_predictions, geolocation_results
                ),
                "anomaly_zones": self.geojson_exporter.export_anomaly_zones(anomaly_alerts)
            },
            "time_series": {
                "hazard_frequency_hourly": self.timeseries_exporter.export_hazard_frequency_over_time(
                    hazard_predictions, 'hour'
                ),
                "hazard_frequency_daily": self.timeseries_exporter.export_hazard_frequency_over_time(
                    hazard_predictions, 'day'
                ),
                "sentiment_trends": self.timeseries_exporter.export_sentiment_trends(
                    hazard_predictions, 'day'
                )
            },
            "statistics": self.stats_exporter.export_hazard_summary(
                hazard_predictions, geolocation_results, anomaly_alerts
            ),
            "export_metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "data_version": "1.0",
                "total_reports": len(hazard_predictions),
                "export_format": "dashboard_package"
            }
        }
    
    def export_for_api_response(
        self,
        hazard_predictions: List[HazardPrediction],
        geolocation_results: List[GeolocationResult],
        format_type: str = 'json'
    ) -> Dict[str, Any]:
        """Export data formatted for API responses"""
        
        if format_type == 'geojson':
            return self.geojson_exporter.export_hazard_points(
                hazard_predictions, geolocation_results
            )
        else:
            # Default JSON format
            geo_lookup = {result.report_id: result for result in geolocation_results}
            
            formatted_reports = []
            for prediction in hazard_predictions:
                geo_result = geo_lookup.get(prediction.report_id)
                
                report_data = {
                    "report_id": prediction.report_id,
                    "hazard_analysis": {
                        "is_hazard": prediction.is_hazard,
                        "hazard_type": prediction.hazard_type,
                        "confidence": prediction.confidence,
                        "severity": prediction.severity,
                        "urgency": prediction.urgency,
                        "sentiment": prediction.sentiment,
                        "sentiment_score": prediction.sentiment_score,
                        "misinformation_probability": prediction.misinformation_probability
                    },
                    "location": None,
                    "processed_at": prediction.processing_metadata.get('analyzed_at', '').isoformat() if prediction.processing_metadata.get('analyzed_at') else None
                }
                
                if geo_result and geo_result.primary_location:
                    report_data["location"] = {
                        "name": geo_result.primary_location.text,
                        "address": geo_result.primary_location.address,
                        "coordinates": geo_result.primary_location.coordinates,
                        "confidence": geo_result.confidence_score
                    }
                
                formatted_reports.append(report_data)
            
            return {
                "reports": formatted_reports,
                "metadata": {
                    "total_reports": len(formatted_reports),
                    "hazard_reports": len([r for r in formatted_reports if r["hazard_analysis"]["is_hazard"]]),
                    "export_time": datetime.now().isoformat()
                }
            }