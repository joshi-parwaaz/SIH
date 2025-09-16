"""Spatial clustering for hotspot identification."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

# Clustering
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Spatial analysis
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    from scipy.spatial import ConvexHull
except ImportError:
    gpd = None
    Point = None
    Polygon = None
    ConvexHull = None

import joblib

logger = logging.getLogger(__name__)

class SpatialClustering:
    """Spatial clustering for hotspot identification and risk area mapping."""
    
    def __init__(self):
        self.clustering_models = {}
        self.scaler = StandardScaler()
        self.cluster_results = {}
        
    def prepare_spatial_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare spatial data for clustering."""
        
        # Ensure required columns exist
        required_cols = ['latitude', 'longitude']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Remove invalid coordinates
        data_clean = data[
            (data['latitude'].between(-90, 90)) & 
            (data['longitude'].between(-180, 180))
        ].copy()
        
        # Remove duplicates
        data_clean = data_clean.drop_duplicates(subset=['latitude', 'longitude'])
        
        return data_clean
    
    def perform_dbscan_clustering(self, data: pd.DataFrame, 
                                eps: float = 0.5, min_samples: int = 5,
                                use_risk_weights: bool = True) -> Dict:
        """Perform DBSCAN clustering to identify spatial hotspots."""
        
        logger.info(f"Performing DBSCAN clustering with eps={eps}, min_samples={min_samples}")
        
        # Prepare coordinates
        coords = data[['latitude', 'longitude']].values
        
        # Add risk weighting if available and requested
        if use_risk_weights and 'risk_score' in data.columns:
            # Weight coordinates by risk score
            risk_weights = data['risk_score'].values.reshape(-1, 1)
            coords_weighted = np.column_stack([coords, risk_weights])
            
            # Scale the features
            coords_scaled = self.scaler.fit_transform(coords_weighted)
        else:
            # Scale only coordinates
            coords_scaled = self.scaler.fit_transform(coords)
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(coords_scaled)
        
        # Store results
        results = {
            'model_type': 'DBSCAN',
            'cluster_labels': cluster_labels.tolist(),
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_noise': list(cluster_labels).count(-1),
            'eps': eps,
            'min_samples': min_samples,
            'coordinates': coords.tolist()
        }
        
        # Calculate cluster statistics
        if results['n_clusters'] > 0:
            cluster_stats = self._calculate_cluster_statistics(data, cluster_labels)
            results['cluster_statistics'] = cluster_stats
            
            # Calculate silhouette score if more than one cluster
            if results['n_clusters'] > 1:
                # Remove noise points for silhouette calculation
                non_noise_mask = cluster_labels != -1
                if np.sum(non_noise_mask) > 1:
                    silhouette = silhouette_score(
                        coords_scaled[non_noise_mask], 
                        cluster_labels[non_noise_mask]
                    )
                    results['silhouette_score'] = silhouette
        
        # Store model
        self.clustering_models['dbscan'] = {
            'model': dbscan,
            'scaler': self.scaler,
            'params': {'eps': eps, 'min_samples': min_samples}
        }
        
        logger.info(f"DBSCAN completed. Found {results['n_clusters']} clusters, {results['n_noise']} noise points")
        
        return results
    
    def perform_kmeans_clustering(self, data: pd.DataFrame, 
                                n_clusters: int = 5) -> Dict:
        """Perform K-means clustering for spatial regions."""
        
        logger.info(f"Performing K-means clustering with {n_clusters} clusters")
        
        # Prepare coordinates
        coords = data[['latitude', 'longitude']].values
        coords_scaled = self.scaler.fit_transform(coords)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords_scaled)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(coords_scaled, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(coords_scaled, cluster_labels)
        
        # Store results
        results = {
            'model_type': 'K-Means',
            'cluster_labels': cluster_labels.tolist(),
            'n_clusters': n_clusters,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'inertia': inertia,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'coordinates': coords.tolist()
        }
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(data, cluster_labels)
        results['cluster_statistics'] = cluster_stats
        
        # Store model
        self.clustering_models['kmeans'] = {
            'model': kmeans,
            'scaler': self.scaler,
            'params': {'n_clusters': n_clusters}
        }
        
        logger.info(f"K-means completed. Silhouette score: {silhouette:.3f}")
        
        return results
    
    def perform_hierarchical_clustering(self, data: pd.DataFrame,
                                      n_clusters: int = 5,
                                      linkage: str = 'ward') -> Dict:
        """Perform hierarchical clustering."""
        
        logger.info(f"Performing hierarchical clustering with {n_clusters} clusters, linkage={linkage}")
        
        # Prepare coordinates
        coords = data[['latitude', 'longitude']].values
        coords_scaled = self.scaler.fit_transform(coords)
        
        # Perform hierarchical clustering
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage=linkage
        )
        cluster_labels = hierarchical.fit_predict(coords_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(coords_scaled, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(coords_scaled, cluster_labels)
        
        # Store results
        results = {
            'model_type': 'Hierarchical',
            'cluster_labels': cluster_labels.tolist(),
            'n_clusters': n_clusters,
            'linkage': linkage,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'coordinates': coords.tolist()
        }
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(data, cluster_labels)
        results['cluster_statistics'] = cluster_stats
        
        # Store model
        self.clustering_models['hierarchical'] = {
            'model': hierarchical,
            'scaler': self.scaler,
            'params': {'n_clusters': n_clusters, 'linkage': linkage}
        }
        
        logger.info(f"Hierarchical clustering completed. Silhouette score: {silhouette:.3f}")
        
        return results
    
    def _calculate_cluster_statistics(self, data: pd.DataFrame, 
                                    cluster_labels: np.ndarray) -> Dict:
        """Calculate statistics for each cluster."""
        
        cluster_stats = {}
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points in DBSCAN
                continue
                
            cluster_mask = cluster_labels == label
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            # Basic statistics
            stats = {
                'size': len(cluster_data),
                'center_lat': cluster_data['latitude'].mean(),
                'center_lon': cluster_data['longitude'].mean(),
                'lat_std': cluster_data['latitude'].std(),
                'lon_std': cluster_data['longitude'].std(),
                'lat_range': cluster_data['latitude'].max() - cluster_data['latitude'].min(),
                'lon_range': cluster_data['longitude'].max() - cluster_data['longitude'].min()
            }
            
            # Risk statistics if available
            if 'risk_score' in cluster_data.columns:
                stats.update({
                    'avg_risk_score': cluster_data['risk_score'].mean(),
                    'max_risk_score': cluster_data['risk_score'].max(),
                    'min_risk_score': cluster_data['risk_score'].min(),
                    'risk_std': cluster_data['risk_score'].std()
                })
            
            # Temporal statistics if timestamp available
            if 'timestamp' in cluster_data.columns:
                cluster_data['timestamp'] = pd.to_datetime(cluster_data['timestamp'])
                stats.update({
                    'first_event': cluster_data['timestamp'].min().isoformat(),
                    'last_event': cluster_data['timestamp'].max().isoformat(),
                    'time_span_hours': (cluster_data['timestamp'].max() - 
                                      cluster_data['timestamp'].min()).total_seconds() / 3600
                })
            
            # Calculate cluster density (points per square degree)
            area = stats['lat_range'] * stats['lon_range']
            stats['density'] = stats['size'] / max(area, 0.001)  # Avoid division by zero
            
            cluster_stats[int(label)] = stats
        
        return cluster_stats
    
    def identify_hotspots(self, clustering_results: Dict, 
                         hotspot_criteria: Dict = None) -> List[Dict]:
        """Identify hotspots based on clustering results and criteria."""
        
        if hotspot_criteria is None:
            hotspot_criteria = {
                'min_size': 5,
                'min_density': 1.0,
                'min_risk_score': 0.6
            }
        
        hotspots = []
        cluster_stats = clustering_results.get('cluster_statistics', {})
        
        for cluster_id, stats in cluster_stats.items():
            # Check if cluster meets hotspot criteria
            is_hotspot = True
            
            # Size criterion
            if stats['size'] < hotspot_criteria.get('min_size', 5):
                is_hotspot = False
            
            # Density criterion
            if stats['density'] < hotspot_criteria.get('min_density', 1.0):
                is_hotspot = False
            
            # Risk score criterion (if available)
            if 'avg_risk_score' in stats:
                if stats['avg_risk_score'] < hotspot_criteria.get('min_risk_score', 0.6):
                    is_hotspot = False
            
            if is_hotspot:
                hotspot = {
                    'cluster_id': cluster_id,
                    'center_latitude': stats['center_lat'],
                    'center_longitude': stats['center_lon'],
                    'size': stats['size'],
                    'density': stats['density'],
                    'risk_level': self._classify_risk_level(stats),
                    'radius_km': self._estimate_cluster_radius(stats),
                    'statistics': stats
                }
                hotspots.append(hotspot)
        
        # Sort hotspots by risk level and size
        hotspots.sort(key=lambda x: (x['risk_level'], x['size']), reverse=True)
        
        logger.info(f"Identified {len(hotspots)} hotspots from {len(cluster_stats)} clusters")
        
        return hotspots
    
    def _classify_risk_level(self, stats: Dict) -> str:
        """Classify risk level based on cluster statistics."""
        
        # Base risk assessment on multiple factors
        risk_factors = []
        
        # Size factor
        if stats['size'] > 20:
            risk_factors.append(2)
        elif stats['size'] > 10:
            risk_factors.append(1)
        else:
            risk_factors.append(0)
        
        # Density factor
        if stats['density'] > 5:
            risk_factors.append(2)
        elif stats['density'] > 2:
            risk_factors.append(1)
        else:
            risk_factors.append(0)
        
        # Risk score factor (if available)
        if 'avg_risk_score' in stats:
            if stats['avg_risk_score'] > 0.8:
                risk_factors.append(2)
            elif stats['avg_risk_score'] > 0.6:
                risk_factors.append(1)
            else:
                risk_factors.append(0)
        
        # Calculate overall risk
        avg_risk = np.mean(risk_factors)
        
        if avg_risk > 1.5:
            return 'High'
        elif avg_risk > 0.8:
            return 'Medium'
        else:
            return 'Low'
    
    def _estimate_cluster_radius(self, stats: Dict) -> float:
        """Estimate cluster radius in kilometers."""
        
        # Approximate radius based on standard deviation
        lat_radius_km = stats.get('lat_std', 0) * 111  # 1 degree ≈ 111 km
        lon_radius_km = stats.get('lon_std', 0) * 111 * np.cos(np.radians(stats['center_lat']))
        
        return max(lat_radius_km, lon_radius_km)
    
    def create_hotspot_polygons(self, hotspots: List[Dict]) -> List[Dict]:
        """Create polygon boundaries for hotspots."""
        
        if Point is None or Polygon is None:
            logger.warning("Shapely not available. Cannot create polygons.")
            return hotspots
        
        enhanced_hotspots = []
        
        for hotspot in hotspots:
            # Create circular polygon around hotspot center
            center_lat = hotspot['center_latitude']
            center_lon = hotspot['center_longitude']
            radius_km = hotspot['radius_km']
            
            # Convert radius to degrees (approximate)
            radius_deg_lat = radius_km / 111
            radius_deg_lon = radius_km / (111 * np.cos(np.radians(center_lat)))
            
            # Create polygon points
            angles = np.linspace(0, 2*np.pi, 32)
            polygon_points = []
            
            for angle in angles:
                lat = center_lat + radius_deg_lat * np.sin(angle)
                lon = center_lon + radius_deg_lon * np.cos(angle)
                polygon_points.append((lon, lat))
            
            polygon = Polygon(polygon_points)
            
            # Enhanced hotspot with polygon
            enhanced_hotspot = hotspot.copy()
            enhanced_hotspot['polygon'] = polygon
            enhanced_hotspot['polygon_coordinates'] = list(polygon.exterior.coords)
            
            enhanced_hotspots.append(enhanced_hotspot)
        
        return enhanced_hotspots
    
    def predict_cluster(self, model_name: str, coordinates: np.ndarray) -> np.ndarray:
        """Predict cluster for new coordinates."""
        
        if model_name not in self.clustering_models:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self.clustering_models[model_name]
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Scale coordinates
        coords_scaled = scaler.transform(coordinates)
        
        # Predict clusters
        if hasattr(model, 'predict'):
            cluster_labels = model.predict(coords_scaled)
        else:
            # For DBSCAN, we need to use a different approach
            # This is a simplified version - in practice, you might want to 
            # implement a more sophisticated prediction method
            cluster_labels = np.full(len(coordinates), -1)  # Mark as noise
        
        return cluster_labels
    
    def optimize_clustering_parameters(self, data: pd.DataFrame, 
                                     method: str = 'dbscan') -> Dict:
        """Optimize clustering parameters using grid search."""
        
        logger.info(f"Optimizing parameters for {method} clustering")
        
        coords = data[['latitude', 'longitude']].values
        coords_scaled = self.scaler.fit_transform(coords)
        
        best_params = {}
        best_score = -1
        
        if method == 'dbscan':
            eps_values = np.arange(0.1, 2.0, 0.2)
            min_samples_values = [3, 5, 10, 15]
            
            for eps in eps_values:
                for min_samples in min_samples_values:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(coords_scaled)
                    
                    # Calculate score (negative of noise ratio + silhouette if applicable)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    noise_ratio = list(labels).count(-1) / len(labels)
                    
                    if n_clusters > 1:
                        non_noise_mask = labels != -1
                        if np.sum(non_noise_mask) > 1:
                            silhouette = silhouette_score(coords_scaled[non_noise_mask], labels[non_noise_mask])
                            score = silhouette - noise_ratio
                        else:
                            score = -noise_ratio
                    else:
                        score = -noise_ratio
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps, 'min_samples': min_samples}
        
        elif method == 'kmeans':
            k_values = range(2, min(20, len(coords) // 2))
            
            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(coords_scaled)
                
                silhouette = silhouette_score(coords_scaled, labels)
                
                if silhouette > best_score:
                    best_score = silhouette
                    best_params = {'n_clusters': k}
        
        logger.info(f"Best parameters for {method}: {best_params} (score: {best_score:.3f})")
        
        return {
            'method': method,
            'best_params': best_params,
            'best_score': best_score
        }
    
    def save_models(self, filepath_prefix: str):
        """Save clustering models."""
        
        for model_name, model_info in self.clustering_models.items():
            filepath = f"{filepath_prefix}_{model_name}_clustering.pkl"
            
            try:
                joblib.dump(model_info, filepath)
                logger.info(f"Saved clustering model {model_name} to {filepath}")
                
            except Exception as e:
                logger.error(f"Error saving clustering model {model_name}: {e}")
    
    def load_models(self, filepath_prefix: str):
        """Load clustering models from disk."""
        
        import os
        import glob
        
        pattern = f"{filepath_prefix}_*_clustering.pkl"
        model_files = glob.glob(pattern)
        
        for filepath in model_files:
            try:
                model_name = os.path.basename(filepath).replace(f"{os.path.basename(filepath_prefix)}_", "").replace("_clustering.pkl", "")
                model_info = joblib.load(filepath)
                self.clustering_models[model_name] = model_info
                logger.info(f"Loaded clustering model {model_name} from {filepath}")
                
            except Exception as e:
                logger.error(f"Error loading clustering model from {filepath}: {e}")