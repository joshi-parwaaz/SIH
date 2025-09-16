"""Spatial feature extraction for hazard prediction models."""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class SpatialFeatureExtractor:
    """Extracts spatial features for geographic hazard prediction."""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
    
    def extract_coordinate_features(self, coordinates: pd.DataFrame) -> pd.DataFrame:
        """Extract basic coordinate-based features."""
        df = pd.DataFrame()
        
        # Ensure we have latitude and longitude columns
        if 'latitude' not in coordinates.columns or 'longitude' not in coordinates.columns:
            raise ValueError("Input must contain 'latitude' and 'longitude' columns")
        
        df['latitude'] = coordinates['latitude']
        df['longitude'] = coordinates['longitude']
        
        # Coordinate transformations
        df['lat_rad'] = np.radians(df['latitude'])
        df['lon_rad'] = np.radians(df['longitude'])
        
        # Cartesian coordinates (for distance calculations)
        R = 6371  # Earth's radius in km
        df['x_cartesian'] = R * np.cos(df['lat_rad']) * np.cos(df['lon_rad'])
        df['y_cartesian'] = R * np.cos(df['lat_rad']) * np.sin(df['lon_rad'])
        df['z_cartesian'] = R * np.sin(df['lat_rad'])
        
        # Distance from equator
        df['distance_from_equator'] = np.abs(df['latitude'])
        
        # Distance from prime meridian
        df['distance_from_prime_meridian'] = np.abs(df['longitude'])
        
        return df
    
    def extract_coastal_features(self, coordinates: pd.DataFrame, 
                               coastline_data: Optional[gpd.GeoDataFrame] = None) -> pd.DataFrame:
        """Extract features related to coastal proximity and characteristics."""
        df = pd.DataFrame()
        
        if coastline_data is not None:
            # Calculate distance to nearest coastline
            distances_to_coast = []
            coastal_orientations = []
            
            for _, row in coordinates.iterrows():
                point = Point(row['longitude'], row['latitude'])
                
                # Find nearest coastline segment
                min_distance = float('inf')
                nearest_orientation = 0
                
                for _, coast_segment in coastline_data.iterrows():
                    distance = point.distance(coast_segment.geometry)
                    if distance < min_distance:
                        min_distance = distance
                        # Calculate orientation (simplified)
                        nearest_orientation = self._calculate_orientation(coast_segment.geometry)
                
                distances_to_coast.append(min_distance * 111)  # Convert to km
                coastal_orientations.append(nearest_orientation)
            
            df['distance_to_coast'] = distances_to_coast
            df['coastal_orientation'] = coastal_orientations
        else:
            # Use simplified coastal proximity estimation
            df['distance_to_coast'] = self._estimate_coastal_distance(coordinates)
            df['coastal_orientation'] = np.random.uniform(0, 360, len(coordinates))  # Placeholder
        
        # Coastal exposure indicators
        df['is_coastal'] = (df['distance_to_coast'] <= 10).astype(int)  # Within 10km
        df['is_very_coastal'] = (df['distance_to_coast'] <= 1).astype(int)  # Within 1km
        
        # Coastal vulnerability based on distance
        df['coastal_vulnerability'] = np.exp(-df['distance_to_coast'] / 20)  # Exponential decay
        
        return df
    
    def _estimate_coastal_distance(self, coordinates: pd.DataFrame) -> np.ndarray:
        """Estimate distance to coast using simplified model for Indian coastline."""
        distances = []
        
        # Simplified Indian coastline boundaries
        west_coast_lon = 73.0  # Approximate western coastline
        east_coast_lon = 80.0  # Approximate eastern coastline
        south_coast_lat = 8.0  # Southern tip
        
        for _, row in coordinates.iterrows():
            lat, lon = row['latitude'], row['longitude']
            
            # Calculate distance to nearest coast
            if lon < west_coast_lon:  # West of west coast
                dist_to_west = (west_coast_lon - lon) * 111 * np.cos(np.radians(lat))
                distances.append(dist_to_west)
            elif lon > east_coast_lon:  # East of east coast
                dist_to_east = (lon - east_coast_lon) * 111 * np.cos(np.radians(lat))
                distances.append(dist_to_east)
            elif lat < south_coast_lat:  # South of southern coast
                dist_to_south = (south_coast_lat - lat) * 111
                distances.append(dist_to_south)
            else:
                # Inland - estimate based on terrain
                inland_distance = min(
                    abs(lon - west_coast_lon) * 111 * np.cos(np.radians(lat)),
                    abs(lon - east_coast_lon) * 111 * np.cos(np.radians(lat))
                )
                distances.append(inland_distance)
        
        return np.array(distances)
    
    def _calculate_orientation(self, geometry) -> float:
        """Calculate orientation of a coastline segment."""
        coords = list(geometry.coords)
        if len(coords) >= 2:
            x1, y1 = coords[0]
            x2, y2 = coords[-1]
            angle = np.arctan2(y2 - y1, x2 - x1)
            return np.degrees(angle) % 360
        return 0
    
    def extract_elevation_features(self, coordinates: pd.DataFrame,
                                 elevation_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Extract elevation and topographic features."""
        df = pd.DataFrame()
        
        if elevation_data is not None and 'elevation' in elevation_data.columns:
            df['elevation'] = elevation_data['elevation']
        else:
            # Estimate elevation based on distance from coast and latitude
            coastal_dist = self._estimate_coastal_distance(coordinates)
            df['elevation'] = np.maximum(0, coastal_dist * 0.5 + np.random.normal(0, 10, len(coordinates)))
        
        # Elevation-based features
        df['elevation_log'] = np.log1p(df['elevation'])
        df['is_lowland'] = (df['elevation'] <= 10).astype(int)
        df['is_highland'] = (df['elevation'] >= 100).astype(int)
        
        # Topographic vulnerability (lower elevation = higher vulnerability)
        df['topographic_vulnerability'] = 1 / (1 + df['elevation'] / 10)
        
        # Slope estimation (simplified)
        if len(coordinates) > 1:
            df['estimated_slope'] = np.abs(np.gradient(df['elevation']))
        else:
            df['estimated_slope'] = 0
        
        return df
    
    def extract_distance_features(self, coordinates: pd.DataFrame,
                                reference_points: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """Extract distance features to important reference points."""
        df = pd.DataFrame()
        
        for ref_name, (ref_lat, ref_lon) in reference_points.items():
            distances = []
            
            for _, row in coordinates.iterrows():
                distance = self._haversine_distance(
                    row['latitude'], row['longitude'], ref_lat, ref_lon
                )
                distances.append(distance)
            
            df[f'distance_to_{ref_name}'] = distances
            df[f'log_distance_to_{ref_name}'] = np.log1p(distances)
        
        return df
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points."""
        R = 6371  # Earth's radius in km
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def extract_clustering_features(self, coordinates: pd.DataFrame,
                                  eps: float = 0.1, min_samples: int = 5) -> pd.DataFrame:
        """Extract spatial clustering features."""
        df = pd.DataFrame()
        
        if len(coordinates) < min_samples:
            # Not enough points for clustering
            df['cluster_id'] = -1
            df['cluster_size'] = 0
            df['is_cluster_core'] = 0
            df['distance_to_cluster_center'] = 0
            return df
        
        # Normalize coordinates for clustering
        coords_normalized = self.scaler.fit_transform(
            coordinates[['latitude', 'longitude']]
        )
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(coords_normalized)
        
        df['cluster_id'] = cluster_labels
        
        # Calculate cluster-based features
        cluster_sizes = []
        is_core_points = []
        distances_to_centers = []
        
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise point
                cluster_sizes.append(0)
                is_core_points.append(0)
                distances_to_centers.append(np.inf)
            else:
                # Cluster size
                cluster_size = np.sum(cluster_labels == label)
                cluster_sizes.append(cluster_size)
                
                # Core point check
                is_core = i in clustering.core_sample_indices_
                is_core_points.append(int(is_core))
                
                # Distance to cluster center
                cluster_points = coords_normalized[cluster_labels == label]
                cluster_center = np.mean(cluster_points, axis=0)
                distance_to_center = np.linalg.norm(
                    coords_normalized[i] - cluster_center
                )
                distances_to_centers.append(distance_to_center)
        
        df['cluster_size'] = cluster_sizes
        df['is_cluster_core'] = is_core_points
        df['distance_to_cluster_center'] = distances_to_centers
        
        # Cluster density
        df['cluster_density'] = df['cluster_size'] / (df['distance_to_cluster_center'] + 1e-8)
        
        return df
    
    def extract_neighborhood_features(self, coordinates: pd.DataFrame,
                                    radius_km: float = 50) -> pd.DataFrame:
        """Extract neighborhood-based spatial features."""
        df = pd.DataFrame()
        
        neighbor_counts = []
        avg_neighbor_distances = []
        neighbor_densities = []
        
        for i, row_i in coordinates.iterrows():
            distances_to_others = []
            
            for j, row_j in coordinates.iterrows():
                if i != j:
                    distance = self._haversine_distance(
                        row_i['latitude'], row_i['longitude'],
                        row_j['latitude'], row_j['longitude']
                    )
                    if distance <= radius_km:
                        distances_to_others.append(distance)
            
            # Neighborhood statistics
            neighbor_count = len(distances_to_others)
            avg_distance = np.mean(distances_to_others) if distances_to_others else 0
            density = neighbor_count / (np.pi * radius_km**2)  # Points per km²
            
            neighbor_counts.append(neighbor_count)
            avg_neighbor_distances.append(avg_distance)
            neighbor_densities.append(density)
        
        df['neighbor_count'] = neighbor_counts
        df['avg_neighbor_distance'] = avg_neighbor_distances
        df['neighborhood_density'] = neighbor_densities
        
        # Neighborhood isolation indicator
        df['is_isolated'] = (df['neighbor_count'] == 0).astype(int)
        df['is_dense_area'] = (df['neighborhood_density'] > np.median(df['neighborhood_density'])).astype(int)
        
        return df
    
    def extract_geometric_features(self, coordinates: pd.DataFrame) -> pd.DataFrame:
        """Extract geometric features from coordinate patterns."""
        df = pd.DataFrame()
        
        if len(coordinates) < 3:
            # Not enough points for geometric features
            return pd.DataFrame(index=coordinates.index)
        
        # Centroid of all points
        centroid_lat = coordinates['latitude'].mean()
        centroid_lon = coordinates['longitude'].mean()
        
        # Distance from centroid
        distances_from_centroid = []
        for _, row in coordinates.iterrows():
            distance = self._haversine_distance(
                row['latitude'], row['longitude'],
                centroid_lat, centroid_lon
            )
            distances_from_centroid.append(distance)
        
        df['distance_from_centroid'] = distances_from_centroid
        
        # Angular position relative to centroid
        angles_from_centroid = []
        for _, row in coordinates.iterrows():
            angle = np.arctan2(
                row['latitude'] - centroid_lat,
                row['longitude'] - centroid_lon
            )
            angles_from_centroid.append(np.degrees(angle) % 360)
        
        df['angle_from_centroid'] = angles_from_centroid
        
        # Convert to cyclical features
        angle_rad = np.radians(df['angle_from_centroid'])
        df['angle_sin'] = np.sin(angle_rad)
        df['angle_cos'] = np.cos(angle_rad)
        
        # Spatial spread indicators
        lat_range = coordinates['latitude'].max() - coordinates['latitude'].min()
        lon_range = coordinates['longitude'].max() - coordinates['longitude'].min()
        
        df['latitude_spread'] = lat_range
        df['longitude_spread'] = lon_range
        df['spatial_spread'] = np.sqrt(lat_range**2 + lon_range**2)
        
        return df
    
    def create_spatial_feature_matrix(self, coordinates: pd.DataFrame,
                                    coastline_data: Optional[gpd.GeoDataFrame] = None,
                                    elevation_data: Optional[pd.DataFrame] = None,
                                    reference_points: Optional[Dict] = None) -> pd.DataFrame:
        """Create comprehensive spatial feature matrix."""
        
        logger.info("Creating spatial feature matrix...")
        
        # Basic coordinate features
        coord_features = self.extract_coordinate_features(coordinates)
        
        # Coastal features
        coastal_features = self.extract_coastal_features(coordinates, coastline_data)
        
        # Elevation features
        elevation_features = self.extract_elevation_features(coordinates, elevation_data)
        
        # Combine basic features
        feature_matrix = pd.concat([
            coord_features, coastal_features, elevation_features
        ], axis=1)
        
        # Distance features to reference points
        if reference_points:
            distance_features = self.extract_distance_features(coordinates, reference_points)
            feature_matrix = pd.concat([feature_matrix, distance_features], axis=1)
        
        # Clustering features (if enough points)
        if len(coordinates) >= 10:
            clustering_features = self.extract_clustering_features(coordinates)
            feature_matrix = pd.concat([feature_matrix, clustering_features], axis=1)
        
        # Neighborhood features
        if len(coordinates) >= 5:
            neighborhood_features = self.extract_neighborhood_features(coordinates)
            feature_matrix = pd.concat([feature_matrix, neighborhood_features], axis=1)
        
        # Geometric features
        if len(coordinates) >= 3:
            geometric_features = self.extract_geometric_features(coordinates)
            feature_matrix = pd.concat([feature_matrix, geometric_features], axis=1)
        
        # Store feature names
        self.feature_names = feature_matrix.columns.tolist()
        
        logger.info(f"Created {len(self.feature_names)} spatial features")
        
        return feature_matrix