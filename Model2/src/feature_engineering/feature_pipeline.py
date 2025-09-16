"""Feature pipeline for comprehensive feature engineering."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import joblib

from .temporal_features import TemporalFeatureExtractor
from .spatial_features import SpatialFeatureExtractor
from .environmental_features import EnvironmentalFeatureExtractor

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """Comprehensive feature engineering pipeline for hazard prediction."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Feature extractors
        self.temporal_extractor = TemporalFeatureExtractor()
        self.spatial_extractor = SpatialFeatureExtractor()
        self.environmental_extractor = EnvironmentalFeatureExtractor()
        
        # Preprocessing components
        self.scaler = None
        self.feature_selector = None
        self.feature_names = []
        self.selected_features = []
        
        # Pipeline state
        self.is_fitted = False
        
    def create_features(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Create comprehensive feature matrix from input data."""
        
        logger.info("Starting comprehensive feature engineering...")
        
        feature_matrices = []
        
        # Extract temporal features
        if 'timestamps' in data:
            timestamps = pd.to_datetime(data['timestamps'])
            
            temporal_features = self.temporal_extractor.create_temporal_feature_matrix(
                timestamps=timestamps,
                event_data=data.get('historical_events'),
                target_series=data.get('target_series')
            )
            
            if not temporal_features.empty:
                feature_matrices.append(temporal_features)
                logger.info(f"Created {len(temporal_features.columns)} temporal features")
        
        # Extract spatial features
        if 'coordinates' in data:
            coordinates = data['coordinates']
            
            spatial_features = self.spatial_extractor.create_spatial_feature_matrix(
                coordinates=coordinates,
                coastline_data=data.get('coastline_data'),
                elevation_data=data.get('elevation_data'),
                reference_points=data.get('reference_points')
            )
            
            if not spatial_features.empty:
                feature_matrices.append(spatial_features)
                logger.info(f"Created {len(spatial_features.columns)} spatial features")
        
        # Extract environmental features
        environmental_features = self.environmental_extractor.create_environmental_feature_matrix(
            weather_data=data.get('weather_data'),
            ocean_data=data.get('ocean_data'),
            timestamps=data.get('timestamps')
        )
        
        if not environmental_features.empty:
            feature_matrices.append(environmental_features)
            logger.info(f"Created {len(environmental_features.columns)} environmental features")
        
        # Extract social/crowd features if available
        if 'social_signals' in data:
            social_features = self._extract_social_features(data['social_signals'])
            if not social_features.empty:
                feature_matrices.append(social_features)
                logger.info(f"Created {len(social_features.columns)} social features")
        
        # Combine all feature matrices
        if feature_matrices:
            feature_matrix = pd.concat(feature_matrices, axis=1)
            
            # Handle any remaining alignment issues
            feature_matrix = self._align_features(feature_matrix)
            
            # Store feature names
            self.feature_names = feature_matrix.columns.tolist()
            
            logger.info(f"Total features created: {len(self.feature_names)}")
            
            return feature_matrix
        else:
            logger.warning("No features could be created from input data")
            return pd.DataFrame()
    
    def _extract_social_features(self, social_signals: Dict) -> pd.DataFrame:
        """Extract features from social media and crowd report signals."""
        features_list = []
        
        for location, signals in social_signals.items():
            # Basic social metrics
            feature_row = {
                f'social_{location}_total_mentions': signals.get('total_mentions', 0),
                f'social_{location}_sentiment_positive': signals.get('sentiment_positive', 0),
                f'social_{location}_sentiment_negative': signals.get('sentiment_negative', 0),
                f'social_{location}_urgency_indicators': signals.get('urgency_indicators', 0),
                f'social_{location}_panic_indicators': signals.get('panic_indicators', 0),
                f'social_{location}_help_requests': signals.get('help_requests', 0),
                f'social_{location}_engagement_rate': signals.get('engagement_rate', 0),
                f'social_{location}_temporal_intensity': signals.get('temporal_intensity', 1),
                f'social_{location}_geographical_spread': signals.get('geographical_spread', 0)
            }
            
            # Derived features
            total_mentions = signals.get('total_mentions', 1)
            feature_row.update({
                f'social_{location}_urgency_ratio': signals.get('urgency_indicators', 0) / max(total_mentions, 1),
                f'social_{location}_panic_ratio': signals.get('panic_indicators', 0) / max(total_mentions, 1),
                f'social_{location}_help_ratio': signals.get('help_requests', 0) / max(total_mentions, 1),
                f'social_{location}_sentiment_score': signals.get('sentiment_positive', 0) - signals.get('sentiment_negative', 0)
            })
            
            features_list.append(feature_row)
        
        if features_list:
            # Aggregate across locations
            social_df = pd.DataFrame(features_list)
            
            # Create aggregated features
            aggregated_features = {
                'social_total_mentions_sum': social_df[[col for col in social_df.columns if 'total_mentions' in col]].sum(axis=1),
                'social_avg_sentiment_positive': social_df[[col for col in social_df.columns if 'sentiment_positive' in col]].mean(axis=1),
                'social_avg_sentiment_negative': social_df[[col for col in social_df.columns if 'sentiment_negative' in col]].mean(axis=1),
                'social_max_urgency_ratio': social_df[[col for col in social_df.columns if 'urgency_ratio' in col]].max(axis=1),
                'social_max_panic_ratio': social_df[[col for col in social_df.columns if 'panic_ratio' in col]].max(axis=1),
                'social_total_help_requests': social_df[[col for col in social_df.columns if 'help_requests' in col]].sum(axis=1),
                'social_max_temporal_intensity': social_df[[col for col in social_df.columns if 'temporal_intensity' in col]].max(axis=1)
            }
            
            agg_df = pd.DataFrame([aggregated_features])
            
            return pd.concat([social_df, agg_df], axis=1)
        
        return pd.DataFrame()
    
    def _align_features(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Align and clean feature matrix."""
        
        # Remove duplicate columns
        feature_matrix = feature_matrix.loc[:, ~feature_matrix.columns.duplicated()]
        
        # Handle infinite values
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values
        numeric_columns = feature_matrix.select_dtypes(include=[np.number]).columns
        
        # Fill missing values with appropriate strategies
        for col in numeric_columns:
            if feature_matrix[col].isna().sum() > 0:
                if feature_matrix[col].isna().sum() / len(feature_matrix) > 0.5:
                    # If more than 50% missing, fill with 0
                    feature_matrix[col] = feature_matrix[col].fillna(0)
                else:
                    # Otherwise, use forward fill then backward fill
                    feature_matrix[col] = feature_matrix[col].fillna(method='ffill').fillna(method='bfill')
                    # If still NaN, fill with median
                    feature_matrix[col] = feature_matrix[col].fillna(feature_matrix[col].median())
        
        return feature_matrix
    
    def fit_preprocessing(self, feature_matrix: pd.DataFrame, target: pd.Series = None,
                         scaler_type: str = 'standard') -> 'FeaturePipeline':
        """Fit preprocessing components on training data."""
        
        logger.info("Fitting preprocessing components...")
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Fit scaler on numeric columns
        numeric_columns = feature_matrix.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            self.scaler.fit(feature_matrix[numeric_columns])
        
        # Feature selection if target is provided
        if target is not None and len(numeric_columns) > 0:
            self._fit_feature_selection(feature_matrix[numeric_columns], target)
        
        self.is_fitted = True
        logger.info("Preprocessing components fitted successfully")
        
        return self
    
    def _fit_feature_selection(self, features: pd.DataFrame, target: pd.Series):
        """Fit feature selection based on target correlation."""
        
        # Remove constant features
        constant_features = features.columns[features.var() == 0]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
            logger.info(f"Removed {len(constant_features)} constant features")
        
        # Remove highly correlated features
        correlation_matrix = features.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_triangle.columns 
                            if any(upper_triangle[column] > 0.95)]
        
        if len(high_corr_features) > 0:
            features = features.drop(columns=high_corr_features)
            logger.info(f"Removed {len(high_corr_features)} highly correlated features")
        
        # Select top K features based on mutual information
        n_features_to_select = min(len(features.columns), 
                                 self.config.get('max_features', 100))
        
        if len(features.columns) > n_features_to_select:
            # Align features and target
            aligned_features, aligned_target = features.align(target, axis=0, join='inner')
            
            if len(aligned_features) > 0:
                self.feature_selector = SelectKBest(
                    score_func=mutual_info_regression,
                    k=n_features_to_select
                )
                self.feature_selector.fit(aligned_features, aligned_target)
                
                selected_mask = self.feature_selector.get_support()
                self.selected_features = aligned_features.columns[selected_mask].tolist()
                
                logger.info(f"Selected {len(self.selected_features)} features out of {len(features.columns)}")
        else:
            self.selected_features = features.columns.tolist()
    
    def transform(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted preprocessing components."""
        
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transformation")
        
        transformed_matrix = feature_matrix.copy()
        
        # Apply scaling to numeric columns
        numeric_columns = transformed_matrix.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0 and self.scaler is not None:
            transformed_matrix[numeric_columns] = self.scaler.transform(transformed_matrix[numeric_columns])
        
        # Apply feature selection
        if self.selected_features:
            # Keep only selected features that exist in the current matrix
            available_selected_features = [f for f in self.selected_features if f in transformed_matrix.columns]
            transformed_matrix = transformed_matrix[available_selected_features]
        
        return transformed_matrix
    
    def fit_transform(self, feature_matrix: pd.DataFrame, target: pd.Series = None,
                     scaler_type: str = 'standard') -> pd.DataFrame:
        """Fit preprocessing and transform features in one step."""
        
        self.fit_preprocessing(feature_matrix, target, scaler_type)
        return self.transform(feature_matrix)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores from selection."""
        
        if self.feature_selector is None:
            return pd.DataFrame()
        
        scores = self.feature_selector.scores_
        feature_names = self.feature_selector.feature_names_in_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_score': scores,
            'selected': self.feature_selector.get_support()
        }).sort_values('importance_score', ascending=False)
        
        return importance_df
    
    def save_pipeline(self, filepath: str):
        """Save fitted pipeline to disk."""
        
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        pipeline_data = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'config': self.config
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """Load fitted pipeline from disk."""
        
        pipeline_data = joblib.load(filepath)
        
        self.scaler = pipeline_data['scaler']
        self.feature_selector = pipeline_data['feature_selector']
        self.feature_names = pipeline_data['feature_names']
        self.selected_features = pipeline_data['selected_features']
        self.config = pipeline_data.get('config', {})
        
        self.is_fitted = True
        logger.info(f"Pipeline loaded from {filepath}")
    
    def create_feature_summary(self) -> Dict:
        """Create summary of feature engineering process."""
        
        summary = {
            'total_features_created': len(self.feature_names),
            'selected_features_count': len(self.selected_features),
            'temporal_features': len([f for f in self.feature_names if any(kw in f for kw in ['hour', 'day', 'month', 'season', 'lag', 'rolling'])]),
            'spatial_features': len([f for f in self.feature_names if any(kw in f for kw in ['latitude', 'longitude', 'distance', 'coastal', 'elevation'])]),
            'environmental_features': len([f for f in self.feature_names if any(kw in f for kw in ['temperature', 'pressure', 'wind', 'wave', 'sea'])]),
            'social_features': len([f for f in self.feature_names if 'social' in f]),
            'preprocessing_applied': self.is_fitted,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None
        }
        
        if self.selected_features:
            summary['feature_reduction_ratio'] = len(self.selected_features) / len(self.feature_names)
        
        return summary