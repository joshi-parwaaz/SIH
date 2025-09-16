"""Geospatial data collector for coastal vulnerability and geographical features."""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from typing import Dict, List, Tuple, Optional
import logging
import requests
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)

class GeospatialDataCollector:
    """Collects and processes geospatial data for coastal areas."""
    
    def __init__(self):
        self.coastal_zones = None
        self.vulnerability_index = None
        self.population_data = None
        
    def load_coastal_boundaries(self, shapefile_path: str = None) -> gpd.GeoDataFrame:
        """Load coastal boundary data."""
        try:
            if shapefile_path and Path(shapefile_path).exists():
                self.coastal_zones = gpd.read_file(shapefile_path)
                logger.info(f"Loaded coastal boundaries from {shapefile_path}")
            else:
                # Create sample coastal zones for Indian coastline
                self.coastal_zones = self._create_sample_coastal_zones()
                logger.info("Created sample coastal zones")
            
            return self.coastal_zones
            
        except Exception as e:
            logger.error(f"Error loading coastal boundaries: {e}")
            return gpd.GeoDataFrame()
    
    def _create_sample_coastal_zones(self) -> gpd.GeoDataFrame:
        """Create sample coastal zones for major Indian coastal areas."""
        zones_data = {
            'zone_id': ['WC001', 'WC002', 'EC001', 'EC002', 'EC003', 'SC001'],
            'zone_name': ['Mumbai Coast', 'Goa Coast', 'Chennai Coast', 'Visakhapatnam Coast', 'Kolkata Coast', 'Kanyakumari Coast'],
            'state': ['Maharashtra', 'Goa', 'Tamil Nadu', 'Andhra Pradesh', 'West Bengal', 'Tamil Nadu'],
            'coast_type': ['urban', 'tourist', 'urban', 'industrial', 'delta', 'rural'],
            'latitude': [19.0760, 15.2993, 13.0827, 17.6868, 22.5726, 8.0883],
            'longitude': [72.8777, 74.1240, 80.2707, 83.2185, 88.3639, 77.5385],
            'coastline_length_km': [150, 105, 180, 95, 200, 120],
            'elevation_avg_m': [14, 60, 6, 45, 3, 30],
            'population_density': [20000, 350, 26000, 5000, 24000, 800]
        }
        
        # Create point geometries for each zone center
        geometries = [Point(lon, lat) for lat, lon in zip(zones_data['latitude'], zones_data['longitude'])]
        
        gdf = gpd.GeoDataFrame(zones_data, geometry=geometries, crs='EPSG:4326')
        return gdf
    
    def calculate_coastal_vulnerability_index(self) -> pd.DataFrame:
        """Calculate Coastal Vulnerability Index (CVI) for each zone."""
        if self.coastal_zones is None or self.coastal_zones.empty:
            logger.warning("No coastal zones data available")
            return pd.DataFrame()
        
        cvi_data = []
        
        for _, zone in self.coastal_zones.iterrows():
            # CVI components (normalized 1-5 scale, 5 = highest vulnerability)
            
            # Geomorphology (coast type vulnerability)
            geomorph_score = self._get_geomorphology_score(zone.get('coast_type', 'urban'))
            
            # Shoreline change rate (simplified)
            shoreline_score = np.random.uniform(1, 5)  # Would be calculated from historical data
            
            # Coastal slope
            slope_score = self._get_slope_score(zone.get('elevation_avg_m', 10))
            
            # Relative sea level change
            sea_level_score = 3.0  # Assumed constant for simplification
            
            # Mean wave height (would come from buoy data)
            wave_score = np.random.uniform(2, 4)
            
            # Mean tide range
            tide_score = np.random.uniform(1, 3)
            
            # Calculate CVI using geometric mean
            cvi = np.sqrt(geomorph_score * shoreline_score * slope_score * 
                         sea_level_score * wave_score * tide_score)
            
            # Normalize to 0-100 scale
            cvi_normalized = min(100, (cvi / 5.0) * 100)
            
            cvi_data.append({
                'zone_id': zone['zone_id'],
                'zone_name': zone['zone_name'],
                'geomorphology_score': geomorph_score,
                'shoreline_score': shoreline_score,
                'slope_score': slope_score,
                'sea_level_score': sea_level_score,
                'wave_score': wave_score,
                'tide_score': tide_score,
                'cvi_raw': cvi,
                'cvi_normalized': round(cvi_normalized, 2),
                'vulnerability_class': self._get_vulnerability_class(cvi_normalized)
            })
        
        self.vulnerability_index = pd.DataFrame(cvi_data)
        return self.vulnerability_index
    
    def _get_geomorphology_score(self, coast_type: str) -> float:
        """Get geomorphology vulnerability score based on coast type."""
        type_scores = {
            'rocky': 1.0,
            'cliff': 1.5,
            'beach': 3.0,
            'delta': 5.0,
            'urban': 4.0,
            'industrial': 3.5,
            'tourist': 3.0,
            'rural': 2.5
        }
        return type_scores.get(coast_type, 3.0)
    
    def _get_slope_score(self, elevation: float) -> float:
        """Get coastal slope vulnerability score."""
        if elevation > 30:
            return 1.0  # Low vulnerability
        elif elevation > 20:
            return 2.0
        elif elevation > 10:
            return 3.0
        elif elevation > 5:
            return 4.0
        else:
            return 5.0  # High vulnerability
    
    def _get_vulnerability_class(self, cvi_score: float) -> str:
        """Classify vulnerability based on CVI score."""
        if cvi_score < 20:
            return 'Very Low'
        elif cvi_score < 40:
            return 'Low'
        elif cvi_score < 60:
            return 'Moderate'
        elif cvi_score < 80:
            return 'High'
        else:
            return 'Very High'
    
    def get_population_exposure(self) -> pd.DataFrame:
        """Calculate population exposure in coastal zones."""
        if self.coastal_zones is None:
            return pd.DataFrame()
        
        exposure_data = []
        
        for _, zone in self.coastal_zones.iterrows():
            # Buffer zone analysis (simplified)
            population_1km = zone.get('population_density', 1000) * 3.14  # Rough circle area
            population_5km = zone.get('population_density', 1000) * 0.5 * 78.5  # Scaled down
            population_10km = zone.get('population_density', 1000) * 0.3 * 314  # Scaled down
            
            exposure_data.append({
                'zone_id': zone['zone_id'],
                'zone_name': zone['zone_name'],
                'population_1km': int(population_1km),
                'population_5km': int(population_5km),
                'population_10km': int(population_10km),
                'critical_infrastructure': self._estimate_infrastructure(zone),
                'economic_assets': self._estimate_economic_assets(zone)
            })
        
        return pd.DataFrame(exposure_data)
    
    def _estimate_infrastructure(self, zone) -> Dict:
        """Estimate critical infrastructure in the zone."""
        coast_type = zone.get('coast_type', 'rural')
        
        if coast_type == 'urban':
            return {
                'hospitals': np.random.randint(5, 20),
                'schools': np.random.randint(50, 200),
                'power_plants': np.random.randint(1, 5),
                'ports': np.random.randint(1, 3)
            }
        elif coast_type == 'industrial':
            return {
                'hospitals': np.random.randint(2, 8),
                'schools': np.random.randint(10, 50),
                'power_plants': np.random.randint(2, 8),
                'ports': np.random.randint(1, 5)
            }
        else:
            return {
                'hospitals': np.random.randint(1, 5),
                'schools': np.random.randint(5, 30),
                'power_plants': np.random.randint(0, 2),
                'ports': np.random.randint(0, 2)
            }
    
    def _estimate_economic_assets(self, zone) -> Dict:
        """Estimate economic assets value in the zone."""
        coast_type = zone.get('coast_type', 'rural')
        population = zone.get('population_density', 1000)
        
        if coast_type == 'urban':
            asset_value_per_person = 50000  # USD
        elif coast_type == 'industrial':
            asset_value_per_person = 75000
        elif coast_type == 'tourist':
            asset_value_per_person = 40000
        else:
            asset_value_per_person = 15000
        
        total_assets = population * asset_value_per_person
        
        return {
            'total_asset_value_usd': total_assets,
            'residential_value': total_assets * 0.6,
            'commercial_value': total_assets * 0.25,
            'industrial_value': total_assets * 0.15
        }
    
    def find_nearest_zones(self, lat: float, lon: float, radius_km: float = 50) -> gpd.GeoDataFrame:
        """Find coastal zones within specified radius of a point."""
        if self.coastal_zones is None:
            return gpd.GeoDataFrame()
        
        point = Point(lon, lat)
        
        # Calculate distances (simplified using degrees)
        distances = []
        for _, zone in self.coastal_zones.iterrows():
            zone_point = zone.geometry
            # Approximate distance calculation
            distance = ((lat - zone_point.y)**2 + (lon - zone_point.x)**2)**0.5 * 111  # Rough km conversion
            distances.append(distance)
        
        self.coastal_zones['distance_km'] = distances
        nearby_zones = self.coastal_zones[self.coastal_zones['distance_km'] <= radius_km].copy()
        
        return nearby_zones.sort_values('distance_km')
    
    def get_land_use_features(self, zone_id: str) -> Dict:
        """Get land use features for a specific zone."""
        # Simplified land use classification
        land_use_types = ['urban', 'agricultural', 'forest', 'water', 'barren', 'wetland']
        
        # Generate realistic land use percentages
        if zone_id.startswith('WC'):  # West Coast
            land_use = {
                'urban': np.random.uniform(30, 70),
                'agricultural': np.random.uniform(10, 30),
                'forest': np.random.uniform(5, 25),
                'water': np.random.uniform(5, 15),
                'barren': np.random.uniform(0, 10),
                'wetland': np.random.uniform(0, 15)
            }
        else:  # East Coast
            land_use = {
                'urban': np.random.uniform(20, 60),
                'agricultural': np.random.uniform(20, 40),
                'forest': np.random.uniform(5, 20),
                'water': np.random.uniform(5, 20),
                'barren': np.random.uniform(0, 15),
                'wetland': np.random.uniform(5, 25)
            }
        
        # Normalize to 100%
        total = sum(land_use.values())
        land_use = {k: round(v/total * 100, 2) for k, v in land_use.items()}
        
        return land_use
    
    def calculate_exposure_index(self, zone_id: str) -> Dict:
        """Calculate comprehensive exposure index for a zone."""
        if self.coastal_zones is None:
            return {}
        
        zone = self.coastal_zones[self.coastal_zones['zone_id'] == zone_id]
        if zone.empty:
            return {}
        
        zone_data = zone.iloc[0]
        
        # Get various exposure components
        population_exposure = self.get_population_exposure()
        zone_pop = population_exposure[population_exposure['zone_id'] == zone_id]
        
        vulnerability = self.vulnerability_index
        zone_vuln = vulnerability[vulnerability['zone_id'] == zone_id] if vulnerability is not None else None
        
        land_use = self.get_land_use_features(zone_id)
        
        # Calculate composite exposure index
        exposure_components = {
            'population_exposure': zone_pop.iloc[0]['population_10km'] if not zone_pop.empty else 0,
            'vulnerability_score': zone_vuln.iloc[0]['cvi_normalized'] if zone_vuln is not None and not zone_vuln.empty else 50,
            'urban_percentage': land_use.get('urban', 0),
            'infrastructure_density': sum(zone_pop.iloc[0]['critical_infrastructure'].values()) if not zone_pop.empty else 0,
            'economic_value': zone_pop.iloc[0]['economic_assets']['total_asset_value_usd'] if not zone_pop.empty else 0
        }
        
        # Weighted composite score (0-100)
        weights = {
            'population_exposure': 0.3,
            'vulnerability_score': 0.25,
            'urban_percentage': 0.2,
            'infrastructure_density': 0.15,
            'economic_value': 0.1
        }
        
        # Normalize components to 0-100 scale
        normalized_pop = min(100, exposure_components['population_exposure'] / 10000)
        normalized_infra = min(100, exposure_components['infrastructure_density'] / 100)
        normalized_econ = min(100, exposure_components['economic_value'] / 1000000)
        
        composite_score = (
            weights['population_exposure'] * normalized_pop +
            weights['vulnerability_score'] * exposure_components['vulnerability_score'] +
            weights['urban_percentage'] * exposure_components['urban_percentage'] +
            weights['infrastructure_density'] * normalized_infra +
            weights['economic_value'] * normalized_econ
        )
        
        return {
            'zone_id': zone_id,
            'exposure_index': round(composite_score, 2),
            'components': exposure_components,
            'risk_class': self._get_exposure_class(composite_score)
        }
    
    def _get_exposure_class(self, score: float) -> str:
        """Classify exposure based on composite score."""
        if score < 20:
            return 'Low'
        elif score < 40:
            return 'Moderate'
        elif score < 60:
            return 'High'
        elif score < 80:
            return 'Very High'
        else:
            return 'Extreme'
    
    def save_geospatial_data(self, output_dir: str):
        """Save all geospatial data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            if self.coastal_zones is not None:
                self.coastal_zones.to_file(output_path / "coastal_zones.shp")
                
            if self.vulnerability_index is not None:
                self.vulnerability_index.to_csv(output_path / "vulnerability_index.csv", index=False)
                
            logger.info(f"Saved geospatial data to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving geospatial data: {e}")