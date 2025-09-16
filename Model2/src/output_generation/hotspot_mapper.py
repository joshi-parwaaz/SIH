"""Hotspot mapping and visualization system."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json

# Visualization and mapping
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Geospatial processing
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, MultiPolygon
    import folium
    from folium import plugins
    HAS_GEO_LIBS = True
except ImportError:
    HAS_GEO_LIBS = False
    gpd = None
    Point = None
    Polygon = None
    folium = None

logger = logging.getLogger(__name__)

class HotspotMapper:
    """Hotspot mapping and visualization for ocean hazard predictions."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._load_default_config()
        self.hotspot_data = {}
        self.map_cache = {}
        
    def _load_default_config(self) -> Dict:
        """Load default mapping configuration."""
        
        return {
            'map_settings': {
                'default_zoom': 8,
                'tile_layer': 'OpenStreetMap',
                'width': 1000,
                'height': 600
            },
            'hotspot_criteria': {
                'min_risk_score': 0.6,
                'min_events': 3,
                'time_window_hours': 168,  # 1 week
                'spatial_radius_km': 50
            },
            'risk_colors': {
                'Critical': '#8B0000',  # Dark red
                'High': '#FF0000',      # Red
                'Medium': '#FFA500',    # Orange
                'Low': '#FFFF00',       # Yellow
                'Unknown': '#808080'    # Gray
            },
            'visualization_options': {
                'show_historical_events': True,
                'show_risk_contours': True,
                'show_coastal_lines': True,
                'show_population_centers': True
            }
        }
    
    def identify_risk_hotspots(self, risk_assessments: List[Dict]) -> List[Dict]:
        """Identify hotspots from risk assessment data."""
        
        logger.info(f"Identifying hotspots from {len(risk_assessments)} risk assessments")
        
        hotspots = []
        criteria = self.config['hotspot_criteria']
        
        # Filter assessments that meet hotspot criteria
        high_risk_assessments = [
            assessment for assessment in risk_assessments
            if assessment.get('risk_score', 0) >= criteria['min_risk_score']
        ]
        
        if len(high_risk_assessments) == 0:
            logger.info("No locations meet hotspot criteria")
            return []
        
        # Group nearby high-risk locations
        hotspot_groups = self._group_nearby_locations(high_risk_assessments)
        
        # Create hotspot objects
        for group_id, group_assessments in enumerate(hotspot_groups):
            if len(group_assessments) < criteria['min_events']:
                continue
            
            hotspot = self._create_hotspot_from_group(group_assessments, group_id)
            hotspots.append(hotspot)
        
        # Sort hotspots by severity
        hotspots.sort(key=lambda x: x['max_risk_score'], reverse=True)
        
        logger.info(f"Identified {len(hotspots)} hotspots")
        
        return hotspots
    
    def _group_nearby_locations(self, assessments: List[Dict], 
                              radius_km: float = None) -> List[List[Dict]]:
        """Group nearby locations using spatial clustering."""
        
        if radius_km is None:
            radius_km = self.config['hotspot_criteria']['spatial_radius_km']
        
        # Convert radius to approximate degrees
        radius_deg = radius_km / 111  # Rough conversion: 1 degree ≈ 111 km
        
        groups = []
        unassigned = assessments.copy()
        
        while unassigned:
            # Start new group with first unassigned location
            seed = unassigned.pop(0)
            current_group = [seed]
            
            # Find all locations within radius of seed
            seed_lat = seed['location']['latitude']
            seed_lon = seed['location']['longitude']
            
            # Check remaining unassigned locations
            i = 0
            while i < len(unassigned):
                location = unassigned[i]
                lat = location['location']['latitude']
                lon = location['location']['longitude']
                
                # Calculate distance
                distance = np.sqrt((lat - seed_lat)**2 + (lon - seed_lon)**2)
                
                if distance <= radius_deg:
                    current_group.append(unassigned.pop(i))
                else:
                    i += 1
            
            groups.append(current_group)
        
        return groups
    
    def _create_hotspot_from_group(self, group_assessments: List[Dict], 
                                 group_id: int) -> Dict:
        """Create hotspot object from grouped assessments."""
        
        # Calculate centroid
        latitudes = [a['location']['latitude'] for a in group_assessments]
        longitudes = [a['location']['longitude'] for a in group_assessments]
        
        centroid_lat = np.mean(latitudes)
        centroid_lon = np.mean(longitudes)
        
        # Calculate statistics
        risk_scores = [a.get('risk_score', 0) for a in group_assessments]
        max_risk = max(risk_scores)
        avg_risk = np.mean(risk_scores)
        
        # Determine overall risk level
        risk_levels = [a.get('risk_level', 'Unknown') for a in group_assessments]
        level_priority = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1, 'Unknown': 0}
        overall_risk_level = max(risk_levels, key=lambda x: level_priority.get(x, 0))
        
        # Calculate spatial extent
        lat_range = max(latitudes) - min(latitudes)
        lon_range = max(longitudes) - min(longitudes)
        radius_km = max(lat_range, lon_range) * 111 / 2  # Approximate radius
        
        # Collect hazard types
        hazard_types = list(set(a.get('hazard_type', 'unknown') for a in group_assessments))
        
        hotspot = {
            'hotspot_id': group_id,
            'centroid': {
                'latitude': float(centroid_lat),
                'longitude': float(centroid_lon)
            },
            'extent': {
                'min_latitude': float(min(latitudes)),
                'max_latitude': float(max(latitudes)),
                'min_longitude': float(min(longitudes)),
                'max_longitude': float(max(longitudes)),
                'radius_km': float(radius_km)
            },
            'risk_statistics': {
                'max_risk_score': float(max_risk),
                'avg_risk_score': float(avg_risk),
                'overall_risk_level': overall_risk_level,
                'num_locations': len(group_assessments)
            },
            'hazard_types': hazard_types,
            'identified_at': datetime.now().isoformat(),
            'constituent_assessments': group_assessments
        }
        
        return hotspot
    
    def create_risk_heatmap(self, risk_assessments: List[Dict], 
                          output_file: str = None) -> str:
        """Create risk heatmap visualization."""
        
        logger.info("Creating risk heatmap")
        
        if not risk_assessments:
            logger.warning("No risk assessments provided for heatmap")
            return ""
        
        # Extract coordinates and risk scores
        latitudes = []
        longitudes = []
        risk_scores = []
        risk_levels = []
        
        for assessment in risk_assessments:
            if 'location' in assessment:
                latitudes.append(assessment['location']['latitude'])
                longitudes.append(assessment['location']['longitude'])
                risk_scores.append(assessment.get('risk_score', 0))
                risk_levels.append(assessment.get('risk_level', 'Unknown'))
        
        if not latitudes:
            logger.warning("No valid location data found")
            return ""
        
        # Create matplotlib figure
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with risk-based coloring
        risk_colors = [self.config['risk_colors'].get(level, '#808080') for level in risk_levels]
        
        scatter = plt.scatter(longitudes, latitudes, c=risk_scores, 
                            s=[score * 100 + 20 for score in risk_scores],
                            cmap='YlOrRd', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Risk Score', rotation=270, labelpad=15)
        
        # Customize plot
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Ocean Hazard Risk Heatmap')
        plt.grid(True, alpha=0.3)
        
        # Add risk level legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=level)
            for level, color in self.config['risk_colors'].items()
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        
        # Save plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"risk_heatmap_{timestamp}.png"
        
        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Risk heatmap saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving heatmap: {e}")
            return ""
    
    def create_interactive_map(self, risk_assessments: List[Dict] = None,
                             hotspots: List[Dict] = None,
                             output_file: str = None) -> str:
        """Create interactive Folium map with risk data."""
        
        if not HAS_GEO_LIBS:
            logger.warning("Folium not available. Cannot create interactive map.")
            return ""
        
        logger.info("Creating interactive map")
        
        # Determine map center
        if risk_assessments:
            latitudes = [a['location']['latitude'] for a in risk_assessments if 'location' in a]
            longitudes = [a['location']['longitude'] for a in risk_assessments if 'location' in a]
        elif hotspots:
            latitudes = [h['centroid']['latitude'] for h in hotspots]
            longitudes = [h['centroid']['longitude'] for h in hotspots]
        else:
            latitudes = [0]
            longitudes = [0]
        
        if latitudes and longitudes:
            center_lat = np.mean(latitudes)
            center_lon = np.mean(longitudes)
        else:
            center_lat, center_lon = 0, 0
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=self.config['map_settings']['default_zoom'],
            tiles=self.config['map_settings']['tile_layer']
        )
        
        # Add risk assessment points
        if risk_assessments:
            self._add_risk_points_to_map(m, risk_assessments)
        
        # Add hotspot areas
        if hotspots:
            self._add_hotspots_to_map(m, hotspots)
        
        # Add map controls
        folium.plugins.Fullscreen().add_to(m)
        folium.plugins.MeasureControl().add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"interactive_map_{timestamp}.html"
        
        try:
            m.save(output_file)
            logger.info(f"Interactive map saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving interactive map: {e}")
            return ""
    
    def _add_risk_points_to_map(self, map_obj, risk_assessments: List[Dict]):
        """Add risk assessment points to Folium map."""
        
        # Create feature groups for different risk levels
        risk_groups = {}
        for level in ['Critical', 'High', 'Medium', 'Low', 'Unknown']:
            risk_groups[level] = folium.FeatureGroup(name=f"{level} Risk")
        
        for assessment in risk_assessments:
            if 'location' not in assessment:
                continue
            
            lat = assessment['location']['latitude']
            lon = assessment['location']['longitude']
            risk_score = assessment.get('risk_score', 0)
            risk_level = assessment.get('risk_level', 'Unknown')
            hazard_type = assessment.get('hazard_type', 'Unknown')
            
            # Choose color and size based on risk
            color = self.config['risk_colors'].get(risk_level, '#808080')
            radius = max(5, risk_score * 15)
            
            # Create popup content
            popup_html = f"""
            <div style='width: 250px'>
                <h4>Risk Assessment</h4>
                <p><strong>Location:</strong> {lat:.4f}, {lon:.4f}</p>
                <p><strong>Hazard Type:</strong> {hazard_type}</p>
                <p><strong>Risk Level:</strong> <span style='color: {color}'>{risk_level}</span></p>
                <p><strong>Risk Score:</strong> {risk_score:.3f}</p>
                <p><strong>Confidence:</strong> {assessment.get('confidence_score', 0):.3f}</p>
                <p><strong>Time Window:</strong> {assessment.get('time_window', 'Unknown')}</p>
            </div>
            """
            
            # Add marker to appropriate risk group
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                color='black',
                fillColor=color,
                fillOpacity=0.7,
                weight=1
            ).add_to(risk_groups[risk_level])
        
        # Add all risk groups to map
        for group in risk_groups.values():
            group.add_to(map_obj)
    
    def _add_hotspots_to_map(self, map_obj, hotspots: List[Dict]):
        """Add hotspot areas to Folium map."""
        
        hotspot_group = folium.FeatureGroup(name="Hotspots")
        
        for hotspot in hotspots:
            centroid = hotspot['centroid']
            extent = hotspot['extent']
            risk_stats = hotspot['risk_statistics']
            
            # Create hotspot boundary
            bounds = [
                [extent['min_latitude'], extent['min_longitude']],
                [extent['max_latitude'], extent['min_longitude']],
                [extent['max_latitude'], extent['max_longitude']],
                [extent['min_latitude'], extent['max_longitude']]
            ]
            
            # Choose color based on risk level
            risk_level = risk_stats['overall_risk_level']
            color = self.config['risk_colors'].get(risk_level, '#808080')
            
            # Create popup content for hotspot
            popup_html = f"""
            <div style='width: 300px'>
                <h4>Risk Hotspot #{hotspot['hotspot_id']}</h4>
                <p><strong>Risk Level:</strong> <span style='color: {color}'>{risk_level}</span></p>
                <p><strong>Max Risk Score:</strong> {risk_stats['max_risk_score']:.3f}</p>
                <p><strong>Avg Risk Score:</strong> {risk_stats['avg_risk_score']:.3f}</p>
                <p><strong>Locations:</strong> {risk_stats['num_locations']}</p>
                <p><strong>Hazard Types:</strong> {', '.join(hotspot['hazard_types'])}</p>
                <p><strong>Radius:</strong> {extent['radius_km']:.1f} km</p>
            </div>
            """
            
            # Add hotspot polygon
            folium.Polygon(
                locations=bounds,
                popup=folium.Popup(popup_html, max_width=350),
                color=color,
                fillColor=color,
                fillOpacity=0.3,
                weight=2
            ).add_to(hotspot_group)
            
            # Add center marker
            folium.Marker(
                location=[centroid['latitude'], centroid['longitude']],
                popup=folium.Popup(popup_html, max_width=350),
                icon=folium.Icon(color='red', icon='warning-sign')
            ).add_to(hotspot_group)
        
        hotspot_group.add_to(map_obj)
    
    def create_risk_contour_map(self, risk_assessments: List[Dict],
                              grid_resolution: int = 50,
                              output_file: str = None) -> str:
        """Create contour map showing risk distribution."""
        
        logger.info("Creating risk contour map")
        
        if len(risk_assessments) < 4:
            logger.warning("Need at least 4 points for contour interpolation")
            return ""
        
        # Extract coordinates and risk scores
        latitudes = np.array([a['location']['latitude'] for a in risk_assessments if 'location' in a])
        longitudes = np.array([a['location']['longitude'] for a in risk_assessments if 'location' in a])
        risk_scores = np.array([a.get('risk_score', 0) for a in risk_assessments if 'location' in a])
        
        if len(latitudes) == 0:
            return ""
        
        # Create grid for interpolation
        lat_min, lat_max = latitudes.min(), latitudes.max()
        lon_min, lon_max = longitudes.min(), longitudes.max()
        
        # Add padding
        lat_padding = (lat_max - lat_min) * 0.1
        lon_padding = (lon_max - lon_min) * 0.1
        
        lat_grid = np.linspace(lat_min - lat_padding, lat_max + lat_padding, grid_resolution)
        lon_grid = np.linspace(lon_min - lon_padding, lon_max + lon_padding, grid_resolution)
        
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Interpolate risk scores
        from scipy.interpolate import griddata
        
        points = np.column_stack((longitudes, latitudes))
        risk_grid = griddata(points, risk_scores, (lon_mesh, lat_mesh), method='cubic')
        
        # Create contour plot
        plt.figure(figsize=(12, 8))
        
        # Create filled contours
        contour_levels = np.linspace(0, 1, 11)
        contours = plt.contourf(lon_mesh, lat_mesh, risk_grid, 
                               levels=contour_levels, cmap='YlOrRd', alpha=0.8)
        
        # Add contour lines
        plt.contour(lon_mesh, lat_mesh, risk_grid, levels=contour_levels, 
                   colors='black', alpha=0.3, linewidths=0.5)
        
        # Add original data points
        plt.scatter(longitudes, latitudes, c=risk_scores, s=50, 
                   cmap='YlOrRd', edgecolors='black', linewidth=1, zorder=5)
        
        # Add colorbar
        cbar = plt.colorbar(contours)
        cbar.set_label('Risk Score', rotation=270, labelpad=15)
        
        # Customize plot
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Ocean Hazard Risk Contour Map')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"risk_contour_map_{timestamp}.png"
        
        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Risk contour map saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving contour map: {e}")
            return ""
    
    def generate_hotspot_report(self, hotspots: List[Dict], 
                              output_file: str = None) -> str:
        """Generate comprehensive hotspot analysis report."""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"hotspot_report_{timestamp}.json"
        
        # Create summary statistics
        if hotspots:
            risk_scores = [h['risk_statistics']['max_risk_score'] for h in hotspots]
            num_locations = [h['risk_statistics']['num_locations'] for h in hotspots]
            
            summary = {
                'total_hotspots': len(hotspots),
                'max_risk_score': float(max(risk_scores)),
                'avg_risk_score': float(np.mean(risk_scores)),
                'total_high_risk_locations': sum(num_locations),
                'avg_locations_per_hotspot': float(np.mean(num_locations))
            }
        else:
            summary = {
                'total_hotspots': 0,
                'max_risk_score': 0.0,
                'avg_risk_score': 0.0,
                'total_high_risk_locations': 0,
                'avg_locations_per_hotspot': 0.0
            }
        
        # Risk level distribution
        risk_levels = [h['risk_statistics']['overall_risk_level'] for h in hotspots]
        risk_distribution = {level: risk_levels.count(level) for level in set(risk_levels)}
        
        # Hazard type analysis
        all_hazard_types = []
        for hotspot in hotspots:
            all_hazard_types.extend(hotspot['hazard_types'])
        
        hazard_distribution = {hazard: all_hazard_types.count(hazard) for hazard in set(all_hazard_types)}
        
        # Create comprehensive report
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'Hotspot Analysis Report',
                'version': '1.0'
            },
            'summary_statistics': summary,
            'risk_level_distribution': risk_distribution,
            'hazard_type_distribution': hazard_distribution,
            'detailed_hotspots': hotspots,
            'analysis': {
                'highest_risk_hotspot': max(hotspots, key=lambda x: x['risk_statistics']['max_risk_score']) if hotspots else None,
                'largest_hotspot': max(hotspots, key=lambda x: x['risk_statistics']['num_locations']) if hotspots else None,
                'recommendations': self._generate_hotspot_recommendations(hotspots)
            }
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Hotspot report saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving hotspot report: {e}")
            return ""
    
    def _generate_hotspot_recommendations(self, hotspots: List[Dict]) -> List[str]:
        """Generate recommendations based on hotspot analysis."""
        
        recommendations = []
        
        if not hotspots:
            recommendations.append("No hotspots identified. Continue monitoring.")
            return recommendations
        
        # Critical hotspots
        critical_hotspots = [h for h in hotspots if h['risk_statistics']['overall_risk_level'] == 'Critical']
        if critical_hotspots:
            recommendations.append(f"URGENT: {len(critical_hotspots)} critical risk hotspots require immediate attention")
        
        # High risk hotspots
        high_risk_hotspots = [h for h in hotspots if h['risk_statistics']['overall_risk_level'] == 'High']
        if high_risk_hotspots:
            recommendations.append(f"High priority: {len(high_risk_hotspots)} high-risk hotspots need monitoring")
        
        # Large hotspots
        large_hotspots = [h for h in hotspots if h['risk_statistics']['num_locations'] > 10]
        if large_hotspots:
            recommendations.append(f"Wide area impact: {len(large_hotspots)} hotspots affect multiple locations")
        
        # Hazard-specific recommendations
        all_hazards = set()
        for hotspot in hotspots:
            all_hazards.update(hotspot['hazard_types'])
        
        if 'tsunami' in all_hazards:
            recommendations.append("Tsunami hotspots detected - review evacuation routes")
        if 'storm_surge' in all_hazards:
            recommendations.append("Storm surge hotspots detected - secure coastal infrastructure")
        
        # General recommendations
        recommendations.extend([
            "Deploy additional monitoring in hotspot areas",
            "Prepare emergency response resources for high-risk zones",
            "Increase public awareness in affected regions"
        ])
        
        return recommendations
    
    def update_hotspot_tracking(self, new_hotspots: List[Dict]):
        """Update hotspot tracking with new data."""
        
        timestamp = datetime.now().isoformat()
        
        # Store new hotspots
        self.hotspot_data[timestamp] = {
            'hotspots': new_hotspots,
            'summary': {
                'count': len(new_hotspots),
                'max_risk': max([h['risk_statistics']['max_risk_score'] for h in new_hotspots]) if new_hotspots else 0,
                'critical_count': len([h for h in new_hotspots if h['risk_statistics']['overall_risk_level'] == 'Critical'])
            }
        }
        
        logger.info(f"Updated hotspot tracking with {len(new_hotspots)} hotspots")
    
    def export_mapping_data(self, output_format: str = 'geojson', 
                          output_file: str = None) -> str:
        """Export mapping data in various formats."""
        
        if not HAS_GEO_LIBS and output_format == 'geojson':
            logger.warning("GeoPandas not available. Cannot export GeoJSON.")
            return ""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"mapping_data_{timestamp}.{output_format}"
        
        try:
            if output_format == 'json':
                # Export as regular JSON
                export_data = {
                    'hotspot_data': self.hotspot_data,
                    'config': self.config,
                    'exported_at': datetime.now().isoformat()
                }
                
                with open(output_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                    
            elif output_format == 'geojson' and HAS_GEO_LIBS:
                # Export as GeoJSON
                features = []
                
                for timestamp, data in self.hotspot_data.items():
                    for hotspot in data['hotspots']:
                        feature = {
                            'type': 'Feature',
                            'geometry': {
                                'type': 'Point',
                                'coordinates': [
                                    hotspot['centroid']['longitude'],
                                    hotspot['centroid']['latitude']
                                ]
                            },
                            'properties': {
                                'hotspot_id': hotspot['hotspot_id'],
                                'risk_level': hotspot['risk_statistics']['overall_risk_level'],
                                'risk_score': hotspot['risk_statistics']['max_risk_score'],
                                'timestamp': timestamp
                            }
                        }
                        features.append(feature)
                
                geojson_data = {
                    'type': 'FeatureCollection',
                    'features': features
                }
                
                with open(output_file, 'w') as f:
                    json.dump(geojson_data, f, indent=2)
            
            logger.info(f"Mapping data exported to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting mapping data: {e}")
            return ""