"""
Visualization Data Generator for Map-based Dashboard.
Generates lat/long coordinates and visualization metadata for frontend maps.
"""

import logging
from typing import Dict, List, Tuple, Optional
import re
import json
from datetime import datetime
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

logger = logging.getLogger(__name__)


class VisualizationDataGenerator:
    """Generates coordinates and visualization data for dashboard maps."""
    
    def __init__(self):
        self.geocoder = Nominatim(user_agent="ocean_hazard_analyzer")
        self.location_cache = {}
        
        # Predefined coordinates for common Indian coastal locations
        self.predefined_locations = {
            "mumbai": {"lat": 19.0760, "lng": 72.8777, "state": "Maharashtra"},
            "chennai": {"lat": 13.0827, "lng": 80.2707, "state": "Tamil Nadu"},
            "kolkata": {"lat": 22.5726, "lng": 88.3639, "state": "West Bengal"},
            "kochi": {"lat": 9.9312, "lng": 76.2673, "state": "Kerala"},
            "visakhapatnam": {"lat": 17.6868, "lng": 83.2185, "state": "Andhra Pradesh"},
            "goa": {"lat": 15.2993, "lng": 74.1240, "state": "Goa"},
            "bhubaneswar": {"lat": 20.2961, "lng": 85.8245, "state": "Odisha"},
            "thiruvananthapuram": {"lat": 8.5241, "lng": 76.9366, "state": "Kerala"},
            "mangalore": {"lat": 12.9141, "lng": 74.8560, "state": "Karnataka"},
            "pondicherry": {"lat": 11.9416, "lng": 79.8083, "state": "Puducherry"},
            "andaman": {"lat": 11.7401, "lng": 92.6586, "state": "Andaman and Nicobar Islands"},
            "gujarat": {"lat": 23.0225, "lng": 72.5714, "state": "Gujarat"},
            "dwarka": {"lat": 22.2394, "lng": 68.9678, "state": "Gujarat"},
            "karwar": {"lat": 14.8140, "lng": 74.1296, "state": "Karnataka"},
            "uttarakhand": {"lat": 30.0668, "lng": 79.0193, "state": "Uttarakhand"},
            "dehradun": {"lat": 30.3165, "lng": 78.0322, "state": "Uttarakhand"},
            "haridwar": {"lat": 29.9457, "lng": 78.1642, "state": "Uttarakhand"},
            "kedarnath": {"lat": 30.7346, "lng": 79.0669, "state": "Uttarakhand"},
            "rishikesh": {"lat": 30.0869, "lng": 78.2676, "state": "Uttarakhand"},
            "rameswaram": {"lat": 9.2876, "lng": 79.3129, "state": "Tamil Nadu"},
            "diu": {"lat": 20.7144, "lng": 70.9876, "state": "Daman and Diu"}
        }
        
        # Hazard type styling for visualization
        self.hazard_styles = {
            "flood": {
                "color": "#2196F3",
                "icon": "flood",
                "priority": 9,
                "radius_multiplier": 1.5
            },
            "cyclone": {
                "color": "#FF5722", 
                "icon": "hurricane",
                "priority": 10,
                "radius_multiplier": 2.0
            },
            "tsunami": {
                "color": "#9C27B0",
                "icon": "tsunami", 
                "priority": 10,
                "radius_multiplier": 3.0
            },
            "storm": {
                "color": "#607D8B",
                "icon": "storm",
                "priority": 7,
                "radius_multiplier": 1.2
            },
            "landslide": {
                "color": "#795548",
                "icon": "landslide",
                "priority": 8,
                "radius_multiplier": 1.3
            },
            "earthquake": {
                "color": "#FF9800",
                "icon": "earthquake", 
                "priority": 9,
                "radius_multiplier": 2.5
            },
            "coastal_erosion": {
                "color": "#FFC107",
                "icon": "erosion",
                "priority": 5,
                "radius_multiplier": 1.0
            },
            "algal_bloom": {
                "color": "#4CAF50", 
                "icon": "algae",
                "priority": 4,
                "radius_multiplier": 1.1
            },
            "oil_spill": {
                "color": "#000000",
                "icon": "oil_spill",
                "priority": 8,
                "radius_multiplier": 1.4
            },
            "extreme_weather": {
                "color": "#E91E63",
                "icon": "weather",
                "priority": 6,
                "radius_multiplier": 1.2
            }
        }
        
        # Severity multipliers
        self.severity_multipliers = {
            "low": 0.5,
            "medium": 1.0,
            "high": 1.5,
            "extreme": 2.0,
            "critical": 2.5
        }
    
    def extract_coordinates_from_text(self, text: str) -> List[Tuple[float, float]]:
        """
        Extract coordinate patterns from text.
        
        Args:
            text: Input text that may contain coordinates
            
        Returns:
            List of (latitude, longitude) tuples
        """
        coordinates = []
        
        # Pattern for decimal degrees (e.g., "19.0760, 72.8777")
        decimal_pattern = r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)'
        matches = re.findall(decimal_pattern, text)
        
        for match in matches:
            try:
                lat, lng = float(match[0]), float(match[1])
                # Basic validation for Indian subcontinent
                if 6 <= lat <= 37 and 68 <= lng <= 97:
                    coordinates.append((lat, lng))
            except ValueError:
                continue
        
        return coordinates
    
    def geocode_location(self, location_name: str) -> Optional[Dict]:
        """
        Get coordinates for a location name.
        
        Args:
            location_name: Name of location to geocode
            
        Returns:
            Dictionary with lat, lng, and metadata
        """
        # Check cache first
        location_key = location_name.lower().strip()
        if location_key in self.location_cache:
            return self.location_cache[location_key]
        
        # Check predefined locations
        if location_key in self.predefined_locations:
            result = self.predefined_locations[location_key]
            self.location_cache[location_key] = result
            return result
        
        # Try geocoding
        try:
            time.sleep(0.1)  # Rate limiting
            location = self.geocoder.geocode(f"{location_name}, India", timeout=5)
            
            if location:
                result = {
                    "lat": location.latitude,
                    "lng": location.longitude,
                    "display_name": location.address,
                    "state": self._extract_state(location.address)
                }
                self.location_cache[location_key] = result
                return result
        
        except (GeocoderTimedOut, Exception) as e:
            logger.warning(f"Geocoding failed for {location_name}: {e}")
        
        return None
    
    def _extract_state(self, address: str) -> str:
        """Extract state name from geocoded address."""
        indian_states = [
            "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
            "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
            "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
            "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
            "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
            "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli",
            "Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep",
            "Puducherry"
        ]
        
        for state in indian_states:
            if state in address:
                return state
        
        return "Unknown"
    
    def generate_map_markers(self, hazard_reports: List[Dict]) -> List[Dict]:
        """
        Generate map markers from hazard reports.
        
        Args:
            hazard_reports: List of hazard reports with location data
            
        Returns:
            List of map markers with coordinates and styling
        """
        markers = []
        
        for report in hazard_reports:
            marker_data = self._create_marker_from_report(report)
            if marker_data:
                markers.append(marker_data)
        
        # Sort by priority (highest first)
        markers.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        return markers
    
    def _create_marker_from_report(self, report: Dict) -> Optional[Dict]:
        """Create a single map marker from a hazard report."""
        # Extract location
        location = getattr(report, "location", "") if hasattr(report, 'location') else report.get("location", "")
        location = location.lower()
        coordinates = None
        
        # Try to get coordinates
        if hasattr(report, 'coordinates') or "coordinates" in report:
            coordinates = getattr(report, "coordinates", None) or report.get("coordinates")
        elif location:
            coord_result = self.geocode_location(location)
            if coord_result:
                coordinates = {"lat": coord_result["lat"], "lng": coord_result["lng"]}
        
        if not coordinates:
            return None
        
        # Get hazard type and styling
        hazard_type = (getattr(report, "hazard_type", "unknown") if hasattr(report, 'hazard_type') 
                      else report.get("hazard_type", "unknown")).lower()
        style = self.hazard_styles.get(hazard_type, {
            "color": "#9E9E9E",
            "icon": "warning",
            "priority": 1,
            "radius_multiplier": 1.0
        })
        
        # Calculate marker size based on severity
        severity = (getattr(report, "severity", "medium") if hasattr(report, 'severity') 
                   else report.get("severity", "medium")).lower()
        severity_mult = self.severity_multipliers.get(severity, 1.0)
        base_radius = 20
        radius = base_radius * style["radius_multiplier"] * severity_mult
        
        # Create marker
        marker = {
            "id": f"{hazard_type}_{location}_{int(datetime.utcnow().timestamp())}",
            "lat": coordinates["lat"],
            "lng": coordinates["lng"],
            "hazard_type": hazard_type,
            "location": location,
            "severity": severity,
            "confidence": getattr(report, "confidence", 0.5) if hasattr(report, 'confidence') else report.get("confidence", 0.5),
            "source": getattr(report, "source", "unknown") if hasattr(report, 'source') else report.get("source", "unknown"),
            "timestamp": getattr(report, "timestamp", datetime.utcnow().isoformat()) if hasattr(report, 'timestamp') else report.get("timestamp", datetime.utcnow().isoformat()),
            "styling": {
                "color": style["color"],
                "icon": style["icon"],
                "radius": radius,
                "opacity": min(0.9, 0.3 + (getattr(report, "confidence", 0.5) if hasattr(report, 'confidence') else report.get("confidence", 0.5)) * 0.6),
                "stroke_width": 2 if severity in ["high", "extreme", "critical"] else 1
            },
            "priority": style["priority"],
            "popup_data": {
                "title": f"{hazard_type.title()} Alert",
                "location": location.title(),
                "severity": severity.title(),
                "confidence": f"{(getattr(report, 'confidence', 0.5) if hasattr(report, 'confidence') else report.get('confidence', 0.5))*100:.1f}%",
                "source": (getattr(report, "source", "unknown") if hasattr(report, 'source') else report.get("source", "unknown")).title(),
                "description": getattr(report, "description", "No description available") if hasattr(report, 'description') else report.get("description", "No description available"),
                "timestamp": getattr(report, "timestamp", datetime.utcnow().isoformat()) if hasattr(report, 'timestamp') else report.get("timestamp", datetime.utcnow().isoformat())
            }
        }
        
        return marker
    
    def generate_heat_map_data(self, hazard_reports: List[Dict]) -> List[Dict]:
        """
        Generate heatmap data points for intensity visualization.
        
        Args:
            hazard_reports: List of hazard reports
            
        Returns:
            List of heatmap points with weights
        """
        heat_points = []
        
        for report in hazard_reports:
            location = (getattr(report, "location", "") if hasattr(report, 'location') else report.get("location", "")).lower()
            coordinates = None
            
            # Get coordinates
            if hasattr(report, 'coordinates') or "coordinates" in report:
                coordinates = getattr(report, "coordinates", None) or report.get("coordinates")
            elif location:
                coord_result = self.geocode_location(location)
                if coord_result:
                    coordinates = {"lat": coord_result["lat"], "lng": coord_result["lng"]}
            
            if coordinates:
                # Calculate intensity weight
                severity = (getattr(report, "severity", "medium") if hasattr(report, 'severity') else report.get("severity", "medium")).lower()
                confidence = getattr(report, "confidence", 0.5) if hasattr(report, 'confidence') else report.get("confidence", 0.5)
                
                # Base weight calculation
                severity_weight = self.severity_multipliers.get(severity, 1.0)
                weight = severity_weight * confidence * 100
                
                heat_point = {
                    "lat": coordinates["lat"],
                    "lng": coordinates["lng"], 
                    "weight": weight,
                    "hazard_type": getattr(report, "hazard_type", "unknown") if hasattr(report, 'hazard_type') else report.get("hazard_type", "unknown"),
                    "severity": severity
                }
                heat_points.append(heat_point)
        
        return heat_points
    
    def generate_cluster_data(self, hazard_reports: List[Dict]) -> List[Dict]:
        """
        Generate clustered view of hazards by region.
        
        Args:
            hazard_reports: List of hazard reports
            
        Returns:
            List of cluster data
        """
        from collections import defaultdict
        
        clusters = defaultdict(list)
        
        # Group by location
        for report in hazard_reports:
            location = (getattr(report, "location", "unknown") if hasattr(report, 'location') else report.get("location", "unknown")).lower()
            clusters[location].append(report)
        
        cluster_data = []
        
        for location, reports in clusters.items():
            if len(reports) < 2:  # Only cluster if multiple reports
                continue
            
            coord_result = self.geocode_location(location)
            if not coord_result:
                continue
            
            # Analyze cluster
            hazard_types = [r.get("hazard_type", "unknown") for r in reports]
            severities = [r.get("severity", "medium") for r in reports]
            
            # Most common hazard type
            most_common_hazard = max(set(hazard_types), key=hazard_types.count)
            
            # Highest severity
            severity_order = ["low", "medium", "high", "extreme", "critical"]
            max_severity = max(severities, key=lambda x: severity_order.index(x) if x in severity_order else 0)
            
            cluster = {
                "id": f"cluster_{location}",
                "lat": coord_result["lat"],
                "lng": coord_result["lng"],
                "location": location.title(),
                "report_count": len(reports),
                "primary_hazard": most_common_hazard,
                "max_severity": max_severity,
                "hazard_types": list(set(hazard_types)),
                "avg_confidence": sum(r.get("confidence", 0.5) for r in reports) / len(reports),
                "reports": reports
            }
            
            cluster_data.append(cluster)
        
        return cluster_data
    
    def generate_visualization_package(self, hazard_reports: List[Dict], anomaly_data: Dict = None) -> Dict:
        """
        Generate complete visualization data package for frontend.
        
        Args:
            hazard_reports: List of hazard reports
            anomaly_data: Optional anomaly detection results
            
        Returns:
            Complete visualization data package
        """
        markers = self.generate_map_markers(hazard_reports)
        heat_map = self.generate_heat_map_data(hazard_reports)
        clusters = self.generate_cluster_data(hazard_reports)
        
        # Calculate bounds for map centering
        if markers:
            lats = [m["lat"] for m in markers]
            lngs = [m["lng"] for m in markers]
            bounds = {
                "north": max(lats),
                "south": min(lats),
                "east": max(lngs),
                "west": min(lngs)
            }
            center = {
                "lat": sum(lats) / len(lats),
                "lng": sum(lngs) / len(lngs)
            }
        else:
            # Default to India center
            bounds = {"north": 37, "south": 6, "east": 97, "west": 68}
            center = {"lat": 20, "lng": 77}
        
        package = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_reports": len(hazard_reports),
                "total_markers": len(markers),
                "total_clusters": len(clusters),
                "map_bounds": bounds,
                "map_center": center
            },
            "markers": markers,
            "heat_map_data": heat_map,
            "clusters": clusters,
            "legend": {
                "hazard_types": self.hazard_styles,
                "severity_colors": {
                    "low": "#4CAF50",
                    "medium": "#FF9800", 
                    "high": "#F44336",
                    "extreme": "#9C27B0",
                    "critical": "#000000"
                }
            },
            "layer_options": {
                "show_markers": True,
                "show_heatmap": True,
                "show_clusters": True,
                "default_zoom": 6
            }
        }
        
        # Add anomaly overlays if provided
        if anomaly_data:
            package["anomaly_overlays"] = self._generate_anomaly_overlays(anomaly_data)
        
        return package
    
    def _generate_anomaly_overlays(self, anomaly_data: Dict) -> List[Dict]:
        """Generate visualization overlays for anomalies."""
        overlays = []
        
        for anomaly in anomaly_data.get("all_anomalies", []):
            if anomaly["type"] == "regional_spike":
                # Create overlay for regional spike
                coord_result = self.geocode_location(anomaly["region"])
                if coord_result:
                    overlay = {
                        "type": "alert_zone",
                        "lat": coord_result["lat"],
                        "lng": coord_result["lng"],
                        "radius": 50000,  # 50km radius
                        "color": "#FF5722",
                        "opacity": 0.3,
                        "stroke_color": "#D32F2F",
                        "stroke_weight": 3,
                        "alert_level": anomaly["severity"],
                        "message": f"Regional spike detected: {anomaly['region']}"
                    }
                    overlays.append(overlay)
        
        return overlays


if __name__ == "__main__":
    # Test visualization data generation
    visualizer = VisualizationDataGenerator()
    
    test_reports = [
        {
            "hazard_type": "flood",
            "location": "mumbai",
            "severity": "high",
            "confidence": 0.8,
            "source": "twitter"
        },
        {
            "hazard_type": "cyclone",
            "location": "chennai", 
            "severity": "extreme",
            "confidence": 0.9,
            "source": "incois"
        }
    ]
    
    viz_package = visualizer.generate_visualization_package(test_reports)
    
    print("Generated visualization package:")
    print(f"- {len(viz_package['markers'])} markers")
    print(f"- {len(viz_package['heat_map_data'])} heat points")
    print(f"- {len(viz_package['clusters'])} clusters")
    print(f"- Map center: {viz_package['metadata']['map_center']}")