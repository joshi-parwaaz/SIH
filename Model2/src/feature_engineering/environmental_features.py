"""Environmental feature extraction for hazard prediction models."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from scipy import signal
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class EnvironmentalFeatureExtractor:
    """Extracts environmental features from sensor and weather data."""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
    
    def extract_weather_features(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from weather data."""
        df = pd.DataFrame()
        
        required_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'rainfall']
        
        # Basic weather features
        for col in required_columns:
            if col in weather_data.columns:
                df[col] = weather_data[col]
                
                # Derived features
                df[f'{col}_squared'] = weather_data[col] ** 2
                df[f'{col}_log'] = np.log1p(np.abs(weather_data[col]))
                
                # Moving averages
                for window in [3, 6, 12, 24]:
                    df[f'{col}_ma_{window}'] = weather_data[col].rolling(window).mean()
                    df[f'{col}_std_{window}'] = weather_data[col].rolling(window).std()
                
                # Rate of change
                df[f'{col}_change_1h'] = weather_data[col].diff(1)
                df[f'{col}_change_6h'] = weather_data[col].diff(6)
                df[f'{col}_change_24h'] = weather_data[col].diff(24)
        
        # Composite indices
        if all(col in weather_data.columns for col in ['temperature', 'humidity']):
            df['heat_index'] = self._calculate_heat_index(
                weather_data['temperature'], weather_data['humidity']
            )
        
        if all(col in weather_data.columns for col in ['temperature', 'wind_speed']):
            df['wind_chill'] = self._calculate_wind_chill(
                weather_data['temperature'], weather_data['wind_speed']
            )
        
        if 'pressure' in weather_data.columns:
            df['pressure_tendency'] = weather_data['pressure'].diff(3)  # 3-hour pressure change
            df['pressure_anomaly'] = weather_data['pressure'] - 1013.25  # Standard atmosphere
        
        # Weather pattern indicators
        if 'wind_speed' in weather_data.columns:
            df['is_calm'] = (weather_data['wind_speed'] < 2).astype(int)
            df['is_windy'] = (weather_data['wind_speed'] > 15).astype(int)
            df['is_storm_force'] = (weather_data['wind_speed'] > 25).astype(int)
        
        if 'rainfall' in weather_data.columns:
            df['is_raining'] = (weather_data['rainfall'] > 0).astype(int)
            df['is_heavy_rain'] = (weather_data['rainfall'] > 10).astype(int)
            df['rainfall_intensity'] = pd.cut(weather_data['rainfall'], 
                                            bins=[0, 2.5, 10, 50, float('inf')],
                                            labels=[0, 1, 2, 3])
        
        return df
    
    def extract_ocean_features(self, ocean_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from ocean/buoy sensor data."""
        df = pd.DataFrame()
        
        ocean_columns = ['sea_level', 'wave_height', 'wave_period', 'sea_temperature', 
                        'current_speed', 'tide_level']
        
        for col in ocean_columns:
            if col in ocean_data.columns:
                df[col] = ocean_data[col]
                
                # Statistical features
                for window in [3, 6, 12, 24]:
                    df[f'{col}_mean_{window}'] = ocean_data[col].rolling(window).mean()
                    df[f'{col}_max_{window}'] = ocean_data[col].rolling(window).max()
                    df[f'{col}_min_{window}'] = ocean_data[col].rolling(window).min()
                    df[f'{col}_range_{window}'] = df[f'{col}_max_{window}'] - df[f'{col}_min_{window}']
                
                # Trend features
                df[f'{col}_trend_3h'] = ocean_data[col].diff(3)
                df[f'{col}_trend_6h'] = ocean_data[col].diff(6)
                
                # Acceleration (second derivative)
                df[f'{col}_acceleration'] = ocean_data[col].diff().diff()
        
        # Wave-related composite features
        if 'wave_height' in ocean_data.columns and 'wave_period' in ocean_data.columns:
            df['wave_steepness'] = ocean_data['wave_height'] / (ocean_data['wave_period'] ** 2)
            df['wave_energy'] = ocean_data['wave_height'] ** 2 * ocean_data['wave_period']
        
        # Sea level anomalies
        if 'sea_level' in ocean_data.columns:
            df['sea_level_anomaly'] = ocean_data['sea_level'] - ocean_data['sea_level'].rolling(24*7).mean()
            df['extreme_sea_level'] = (np.abs(df['sea_level_anomaly']) > 2 * ocean_data['sea_level'].rolling(24*7).std()).astype(int)
        
        # Current-related features
        if 'current_speed' in ocean_data.columns:
            df['strong_current'] = (ocean_data['current_speed'] > ocean_data['current_speed'].quantile(0.9)).astype(int)
        
        # Tide-related features
        if 'tide_level' in ocean_data.columns:
            df['tide_rate'] = ocean_data['tide_level'].diff()
            df['is_high_tide'] = (ocean_data['tide_level'] > ocean_data['tide_level'].rolling(24).mean()).astype(int)
            df['is_spring_tide'] = self._identify_spring_tides(ocean_data['tide_level'])
        
        return df
    
    def extract_atmospheric_features(self, atmospheric_data: pd.DataFrame) -> pd.DataFrame:
        """Extract atmospheric stability and convection features."""
        df = pd.DataFrame()
        
        if 'temperature' in atmospheric_data.columns and 'pressure' in atmospheric_data.columns:
            # Atmospheric stability indicators
            df['potential_temperature'] = atmospheric_data['temperature'] * (1000 / atmospheric_data['pressure']) ** 0.286
            
            # Convective available potential energy (simplified)
            df['cape_index'] = (atmospheric_data['temperature'] - atmospheric_data['temperature'].rolling(24).mean()) * atmospheric_data['humidity'] / 100
        
        if 'wind_speed' in atmospheric_data.columns and 'wind_direction' in atmospheric_data.columns:
            # Wind shear indicators
            df['wind_shear'] = atmospheric_data['wind_speed'].diff()
            df['wind_direction_change'] = self._calculate_wind_direction_change(atmospheric_data['wind_direction'])
            
            # Wind components
            wind_rad = np.radians(atmospheric_data['wind_direction'])
            df['wind_u'] = atmospheric_data['wind_speed'] * np.cos(wind_rad)
            df['wind_v'] = atmospheric_data['wind_speed'] * np.sin(wind_rad)
        
        if 'humidity' in atmospheric_data.columns:
            df['moisture_flux'] = atmospheric_data['humidity'] * atmospheric_data.get('wind_speed', 1)
        
        return df
    
    def extract_extreme_event_indicators(self, environmental_data: pd.DataFrame) -> pd.DataFrame:
        """Extract indicators for extreme weather/ocean events."""
        df = pd.DataFrame()
        
        # Multi-parameter extreme event detection
        if 'wind_speed' in environmental_data.columns:
            wind_threshold_95 = environmental_data['wind_speed'].quantile(0.95)
            df['extreme_wind'] = (environmental_data['wind_speed'] > wind_threshold_95).astype(int)
        
        if 'wave_height' in environmental_data.columns:
            wave_threshold_95 = environmental_data['wave_height'].quantile(0.95)
            df['extreme_waves'] = (environmental_data['wave_height'] > wave_threshold_95).astype(int)
        
        if 'rainfall' in environmental_data.columns:
            rain_threshold_95 = environmental_data['rainfall'].quantile(0.95)
            df['extreme_rainfall'] = (environmental_data['rainfall'] > rain_threshold_95).astype(int)
        
        if 'pressure' in environmental_data.columns:
            pressure_threshold_low = environmental_data['pressure'].quantile(0.05)
            df['extreme_low_pressure'] = (environmental_data['pressure'] < pressure_threshold_low).astype(int)
        
        # Compound events
        extreme_cols = [col for col in df.columns if col.startswith('extreme_')]
        if len(extreme_cols) > 1:
            df['compound_extreme_count'] = df[extreme_cols].sum(axis=1)
            df['is_compound_extreme'] = (df['compound_extreme_count'] >= 2).astype(int)
        
        # Persistence of extreme conditions
        for col in extreme_cols:
            df[f'{col}_persistence'] = df[col].rolling(6).sum()  # 6-hour persistence
        
        return df
    
    def extract_seasonal_environmental_features(self, environmental_data: pd.DataFrame, 
                                              timestamps: pd.Series) -> pd.DataFrame:
        """Extract seasonal environmental features."""
        df = pd.DataFrame()
        
        timestamps = pd.to_datetime(timestamps)
        
        # Seasonal patterns
        month = timestamps.dt.month
        df['is_monsoon'] = month.isin([6, 7, 8, 9]).astype(int)
        df['is_cyclone_season'] = month.isin([4, 5, 6, 10, 11, 12]).astype(int)
        df['is_winter'] = month.isin([12, 1, 2]).astype(int)
        
        # Day/night patterns
        hour = timestamps.dt.hour
        df['is_daytime'] = hour.between(6, 18).astype(int)
        df['is_dawn_dusk'] = hour.isin([5, 6, 18, 19]).astype(int)
        
        # Seasonal anomalies
        for col in environmental_data.select_dtypes(include=[np.number]).columns:
            monthly_means = environmental_data.groupby(month)[col].transform('mean')
            df[f'{col}_seasonal_anomaly'] = environmental_data[col] - monthly_means
            
            # Seasonal percentile rank
            monthly_ranks = environmental_data.groupby(month)[col].rank(pct=True)
            df[f'{col}_seasonal_percentile'] = monthly_ranks
        
        return df
    
    def extract_cross_correlation_features(self, environmental_data: pd.DataFrame,
                                         max_lag: int = 12) -> pd.DataFrame:
        """Extract cross-correlation features between environmental variables."""
        df = pd.DataFrame()
        
        numeric_cols = environmental_data.select_dtypes(include=[np.number]).columns
        
        # Cross-correlations with different lags
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if col1 != col2:
                    # Calculate cross-correlation at different lags
                    series1 = environmental_data[col1].dropna()
                    series2 = environmental_data[col2].dropna()
                    
                    if len(series1) > max_lag and len(series2) > max_lag:
                        cross_corr = signal.correlate(series1, series2, mode='full')
                        lags = signal.correlation_lags(len(series1), len(series2), mode='full')
                        
                        # Find peak correlation and its lag
                        peak_idx = np.argmax(np.abs(cross_corr))
                        peak_lag = lags[peak_idx]
                        peak_corr = cross_corr[peak_idx]
                        
                        df[f'{col1}_{col2}_peak_corr'] = peak_corr
                        df[f'{col1}_{col2}_peak_lag'] = peak_lag
                        
                        # Correlation at specific lags
                        for lag in [0, 3, 6, 12]:
                            if abs(lag) < len(lags):
                                lag_idx = np.where(lags == lag)[0]
                                if len(lag_idx) > 0:
                                    df[f'{col1}_{col2}_corr_lag_{lag}'] = cross_corr[lag_idx[0]]
        
        return df
    
    def _calculate_heat_index(self, temperature: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index from temperature and humidity."""
        T = temperature
        RH = humidity
        
        # Simplified heat index formula
        HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
        
        # Apply full formula for high temperatures
        mask = T >= 80
        if mask.any():
            c1, c2, c3, c4, c5, c6, c7, c8, c9 = [
                -42.379, 2.04901523, 10.14333127, -0.22475541, -0.00683783,
                -0.05481717, 0.00122874, 0.00085282, -0.00000199
            ]
            
            HI_full = (c1 + c2*T + c3*RH + c4*T*RH + c5*T**2 + c6*RH**2 + 
                      c7*T**2*RH + c8*T*RH**2 + c9*T**2*RH**2)
            
            HI[mask] = HI_full[mask]
        
        return HI
    
    def _calculate_wind_chill(self, temperature: pd.Series, wind_speed: pd.Series) -> pd.Series:
        """Calculate wind chill from temperature and wind speed."""
        T = temperature
        V = wind_speed
        
        # Wind chill formula (for T <= 50°F and V >= 3 mph)
        mask = (T <= 50) & (V >= 3)
        wind_chill = T.copy()
        
        if mask.any():
            wind_chill[mask] = (35.74 + 0.6215*T[mask] - 35.75*(V[mask]**0.16) + 
                               0.4275*T[mask]*(V[mask]**0.16))
        
        return wind_chill
    
    def _identify_spring_tides(self, tide_level: pd.Series) -> pd.Series:
        """Identify spring tide periods (simplified)."""
        # Calculate tidal range over rolling window
        tidal_range = tide_level.rolling(24).max() - tide_level.rolling(24).min()
        spring_threshold = tidal_range.quantile(0.75)
        
        return (tidal_range > spring_threshold).astype(int)
    
    def _calculate_wind_direction_change(self, wind_direction: pd.Series) -> pd.Series:
        """Calculate wind direction change accounting for circular nature."""
        direction_diff = wind_direction.diff()
        
        # Handle circular nature of wind direction
        direction_diff = np.where(direction_diff > 180, direction_diff - 360, direction_diff)
        direction_diff = np.where(direction_diff < -180, direction_diff + 360, direction_diff)
        
        return pd.Series(np.abs(direction_diff), index=wind_direction.index)
    
    def create_environmental_feature_matrix(self, weather_data: Optional[pd.DataFrame] = None,
                                          ocean_data: Optional[pd.DataFrame] = None,
                                          timestamps: Optional[pd.Series] = None) -> pd.DataFrame:
        """Create comprehensive environmental feature matrix."""
        
        logger.info("Creating environmental feature matrix...")
        
        feature_matrices = []
        
        # Weather features
        if weather_data is not None:
            weather_features = self.extract_weather_features(weather_data)
            feature_matrices.append(weather_features)
            
            # Atmospheric features
            atmospheric_features = self.extract_atmospheric_features(weather_data)
            feature_matrices.append(atmospheric_features)
        
        # Ocean features
        if ocean_data is not None:
            ocean_features = self.extract_ocean_features(ocean_data)
            feature_matrices.append(ocean_features)
        
        # Combined environmental data for extreme event detection
        if weather_data is not None and ocean_data is not None:
            combined_data = pd.concat([weather_data, ocean_data], axis=1)
        elif weather_data is not None:
            combined_data = weather_data
        elif ocean_data is not None:
            combined_data = ocean_data
        else:
            combined_data = pd.DataFrame()
        
        if not combined_data.empty:
            # Extreme event indicators
            extreme_features = self.extract_extreme_event_indicators(combined_data)
            feature_matrices.append(extreme_features)
            
            # Seasonal features
            if timestamps is not None:
                seasonal_features = self.extract_seasonal_environmental_features(combined_data, timestamps)
                feature_matrices.append(seasonal_features)
            
            # Cross-correlation features
            if len(combined_data.columns) > 1:
                cross_corr_features = self.extract_cross_correlation_features(combined_data)
                if not cross_corr_features.empty:
                    # Broadcast to match the length of other features
                    cross_corr_features = pd.DataFrame(
                        np.tile(cross_corr_features.values, (len(combined_data), 1)),
                        columns=cross_corr_features.columns,
                        index=combined_data.index
                    )
                    feature_matrices.append(cross_corr_features)
        
        # Combine all feature matrices
        if feature_matrices:
            feature_matrix = pd.concat(feature_matrices, axis=1)
        else:
            feature_matrix = pd.DataFrame()
        
        # Store feature names
        self.feature_names = feature_matrix.columns.tolist()
        
        logger.info(f"Created {len(self.feature_names)} environmental features")
        
        return feature_matrix