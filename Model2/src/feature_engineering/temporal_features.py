"""Temporal feature extraction for hazard prediction models."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class TemporalFeatureExtractor:
    """Extracts temporal features for time series prediction."""
    
    def __init__(self):
        self.feature_names = []
        self.cyclical_features = ['hour', 'day_of_week', 'month', 'day_of_year']
    
    def extract_basic_temporal_features(self, timestamps: pd.Series) -> pd.DataFrame:
        """Extract basic temporal features from timestamps."""
        df = pd.DataFrame()
        
        # Ensure timestamps are datetime objects
        timestamps = pd.to_datetime(timestamps)
        
        # Basic time components
        df['year'] = timestamps.dt.year
        df['month'] = timestamps.dt.month
        df['day'] = timestamps.dt.day
        df['hour'] = timestamps.dt.hour
        df['minute'] = timestamps.dt.minute
        df['day_of_week'] = timestamps.dt.dayofweek
        df['day_of_year'] = timestamps.dt.dayofyear
        df['week_of_year'] = timestamps.dt.isocalendar().week
        
        # Derived features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Seasonal features
        df['season'] = df['month'].apply(self._get_season_numeric)
        df['monsoon_season'] = df['month'].apply(self._get_monsoon_season)
        df['cyclone_season'] = df['month'].apply(self._get_cyclone_season)
        
        return df
    
    def extract_cyclical_features(self, timestamps: pd.Series) -> pd.DataFrame:
        """Extract cyclical features using sine/cosine transformations."""
        df = pd.DataFrame()
        timestamps = pd.to_datetime(timestamps)
        
        # Hour of day (24-hour cycle)
        hour_angle = 2 * np.pi * timestamps.dt.hour / 24
        df['hour_sin'] = np.sin(hour_angle)
        df['hour_cos'] = np.cos(hour_angle)
        
        # Day of week (7-day cycle)
        dow_angle = 2 * np.pi * timestamps.dt.dayofweek / 7
        df['dow_sin'] = np.sin(dow_angle)
        df['dow_cos'] = np.cos(dow_angle)
        
        # Month (12-month cycle)
        month_angle = 2 * np.pi * timestamps.dt.month / 12
        df['month_sin'] = np.sin(month_angle)
        df['month_cos'] = np.cos(month_angle)
        
        # Day of year (365-day cycle)
        doy_angle = 2 * np.pi * timestamps.dt.dayofyear / 365
        df['doy_sin'] = np.sin(doy_angle)
        df['doy_cos'] = np.cos(doy_angle)
        
        return df
    
    def extract_lag_features(self, data: pd.DataFrame, target_column: str, 
                           lags: List[int]) -> pd.DataFrame:
        """Extract lag features for time series data."""
        lag_df = pd.DataFrame()
        
        for lag in lags:
            lag_df[f'{target_column}_lag_{lag}'] = data[target_column].shift(lag)
        
        return lag_df
    
    def extract_rolling_features(self, data: pd.DataFrame, target_column: str,
                               windows: List[int]) -> pd.DataFrame:
        """Extract rolling window statistical features."""
        rolling_df = pd.DataFrame()
        
        for window in windows:
            # Rolling statistics
            rolling_df[f'{target_column}_rolling_mean_{window}'] = data[target_column].rolling(window).mean()
            rolling_df[f'{target_column}_rolling_std_{window}'] = data[target_column].rolling(window).std()
            rolling_df[f'{target_column}_rolling_min_{window}'] = data[target_column].rolling(window).min()
            rolling_df[f'{target_column}_rolling_max_{window}'] = data[target_column].rolling(window).max()
            rolling_df[f'{target_column}_rolling_median_{window}'] = data[target_column].rolling(window).median()
            
            # Rolling percentiles
            rolling_df[f'{target_column}_rolling_q25_{window}'] = data[target_column].rolling(window).quantile(0.25)
            rolling_df[f'{target_column}_rolling_q75_{window}'] = data[target_column].rolling(window).quantile(0.75)
            
            # Rolling skewness and kurtosis
            rolling_df[f'{target_column}_rolling_skew_{window}'] = data[target_column].rolling(window).skew()
            rolling_df[f'{target_column}_rolling_kurt_{window}'] = data[target_column].rolling(window).kurt()
        
        return rolling_df
    
    def extract_time_since_features(self, event_timestamps: pd.Series, 
                                  reference_timestamps: pd.Series) -> pd.DataFrame:
        """Extract time-since-last-event features."""
        time_since_df = pd.DataFrame()
        
        event_timestamps = pd.to_datetime(event_timestamps)
        reference_timestamps = pd.to_datetime(reference_timestamps)
        
        # Time since last event in various units
        time_diffs = []
        
        for ref_time in reference_timestamps:
            # Find most recent event before reference time
            previous_events = event_timestamps[event_timestamps <= ref_time]
            
            if len(previous_events) > 0:
                last_event = previous_events.max()
                time_diff = (ref_time - last_event).total_seconds()
            else:
                time_diff = np.inf  # No previous events
            
            time_diffs.append(time_diff)
        
        time_diffs = np.array(time_diffs)
        
        # Convert to various time units
        time_since_df['time_since_last_event_hours'] = time_diffs / 3600
        time_since_df['time_since_last_event_days'] = time_diffs / (3600 * 24)
        time_since_df['time_since_last_event_weeks'] = time_diffs / (3600 * 24 * 7)
        
        # Log transform for better distribution
        time_since_df['log_time_since_last_event'] = np.log1p(time_diffs / 3600)
        
        # Binary indicators for recent events
        time_since_df['event_within_1h'] = (time_diffs <= 3600).astype(int)
        time_since_df['event_within_6h'] = (time_diffs <= 6 * 3600).astype(int)
        time_since_df['event_within_24h'] = (time_diffs <= 24 * 3600).astype(int)
        time_since_df['event_within_7d'] = (time_diffs <= 7 * 24 * 3600).astype(int)
        
        return time_since_df
    
    def extract_trend_features(self, data: pd.DataFrame, target_column: str,
                             windows: List[int]) -> pd.DataFrame:
        """Extract trend and momentum features."""
        trend_df = pd.DataFrame()
        
        for window in windows:
            if len(data) >= window:
                # Linear trend slope over window
                trend_slopes = []
                
                for i in range(len(data)):
                    if i >= window - 1:
                        window_data = data[target_column].iloc[i-window+1:i+1]
                        x = np.arange(len(window_data))
                        
                        if len(window_data) > 1 and not window_data.isna().all():
                            slope, _, _, _, _ = stats.linregress(x, window_data.fillna(method='ffill'))
                            trend_slopes.append(slope)
                        else:
                            trend_slopes.append(np.nan)
                    else:
                        trend_slopes.append(np.nan)
                
                trend_df[f'{target_column}_trend_slope_{window}'] = trend_slopes
                
                # Momentum features
                change_col = f'{target_column}_change_{window}'
                trend_df[change_col] = data[target_column].diff(window)
                trend_df[f'{target_column}_momentum_{window}'] = trend_df[change_col] / (data[target_column].shift(window) + 1e-8)
                
                # Acceleration (second derivative)
                trend_df[f'{target_column}_acceleration_{window}'] = trend_df[change_col].diff()
        
        return trend_df
    
    def extract_frequency_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Extract frequency domain features using FFT."""
        freq_df = pd.DataFrame()
        
        # Remove NaN values for FFT
        clean_data = data[target_column].dropna()
        
        if len(clean_data) >= 8:  # Minimum length for meaningful FFT
            # Compute FFT
            fft_values = np.fft.fft(clean_data.values)
            fft_magnitude = np.abs(fft_values)
            fft_freq = np.fft.fftfreq(len(clean_data))
            
            # Extract dominant frequencies
            # Find peaks in frequency domain
            peak_indices = []
            for i in range(1, len(fft_magnitude) // 2):  # Only positive frequencies
                if (fft_magnitude[i] > fft_magnitude[i-1] and 
                    fft_magnitude[i] > fft_magnitude[i+1] and 
                    fft_magnitude[i] > np.mean(fft_magnitude) * 2):
                    peak_indices.append(i)
            
            # Features from FFT
            freq_df = pd.DataFrame(index=data.index)
            freq_df['spectral_energy'] = np.sum(fft_magnitude[:len(fft_magnitude)//2])
            freq_df['dominant_frequency'] = fft_freq[np.argmax(fft_magnitude[:len(fft_magnitude)//2])] if len(fft_magnitude) > 0 else 0
            freq_df['num_peaks'] = len(peak_indices)
            freq_df['spectral_centroid'] = np.sum(fft_freq[:len(fft_freq)//2] * fft_magnitude[:len(fft_magnitude)//2]) / (np.sum(fft_magnitude[:len(fft_magnitude)//2]) + 1e-8)
            
            # Replicate to match original data length
            freq_df = freq_df.reindex(data.index, method='ffill')
        else:
            # Default values if insufficient data
            freq_df = pd.DataFrame(index=data.index)
            freq_df['spectral_energy'] = 0
            freq_df['dominant_frequency'] = 0
            freq_df['num_peaks'] = 0
            freq_df['spectral_centroid'] = 0
        
        return freq_df
    
    def extract_recurrence_features(self, event_data: pd.DataFrame, 
                                   timestamp_column: str) -> pd.DataFrame:
        """Extract features related to event recurrence patterns."""
        recurrence_df = pd.DataFrame()
        
        if event_data.empty:
            return recurrence_df
        
        event_data = event_data.copy()
        event_data[timestamp_column] = pd.to_datetime(event_data[timestamp_column])
        event_data = event_data.sort_values(timestamp_column)
        
        # Inter-event intervals
        intervals = event_data[timestamp_column].diff().dt.total_seconds() / 3600  # Hours
        
        recurrence_df = pd.DataFrame(index=event_data.index)
        recurrence_df['inter_event_interval'] = intervals
        recurrence_df['avg_inter_event_interval'] = intervals.expanding().mean()
        recurrence_df['std_inter_event_interval'] = intervals.expanding().std()
        
        # Recurrence statistics
        recurrence_df['events_last_24h'] = event_data[timestamp_column].apply(
            lambda x: len(event_data[(event_data[timestamp_column] >= x - timedelta(hours=24)) & 
                                   (event_data[timestamp_column] <= x)])
        )
        
        recurrence_df['events_last_week'] = event_data[timestamp_column].apply(
            lambda x: len(event_data[(event_data[timestamp_column] >= x - timedelta(days=7)) & 
                                   (event_data[timestamp_column] <= x)])
        )
        
        recurrence_df['events_last_month'] = event_data[timestamp_column].apply(
            lambda x: len(event_data[(event_data[timestamp_column] >= x - timedelta(days=30)) & 
                                   (event_data[timestamp_column] <= x)])
        )
        
        return recurrence_df
    
    def _get_season_numeric(self, month: int) -> int:
        """Convert month to numeric season."""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn
    
    def _get_monsoon_season(self, month: int) -> int:
        """Identify monsoon season for India."""
        if month in [6, 7, 8, 9]:  # June to September
            return 1  # Southwest monsoon
        elif month in [10, 11, 12]:  # October to December
            return 2  # Northeast monsoon
        else:
            return 0  # Non-monsoon
    
    def _get_cyclone_season(self, month: int) -> int:
        """Identify cyclone season for Indian Ocean."""
        if month in [4, 5, 6] or month in [10, 11, 12]:
            return 1  # Peak cyclone season
        else:
            return 0  # Low cyclone season
    
    def create_temporal_feature_matrix(self, timestamps: pd.Series,
                                     event_data: Optional[pd.DataFrame] = None,
                                     target_series: Optional[pd.Series] = None) -> pd.DataFrame:
        """Create comprehensive temporal feature matrix."""
        
        logger.info("Creating temporal feature matrix...")
        
        # Basic temporal features
        basic_features = self.extract_basic_temporal_features(timestamps)
        
        # Cyclical features
        cyclical_features = self.extract_cyclical_features(timestamps)
        
        # Combine basic features
        feature_matrix = pd.concat([basic_features, cyclical_features], axis=1)
        
        # Add time-since-event features if event data provided
        if event_data is not None and 'timestamp' in event_data.columns:
            time_since_features = self.extract_time_since_features(
                event_data['timestamp'], timestamps
            )
            feature_matrix = pd.concat([feature_matrix, time_since_features], axis=1)
        
        # Add target-based features if target series provided
        if target_series is not None:
            # Lag features
            lag_features = self.extract_lag_features(
                pd.DataFrame({'target': target_series}), 'target', [1, 2, 3, 6, 12, 24]
            )
            
            # Rolling features
            rolling_features = self.extract_rolling_features(
                pd.DataFrame({'target': target_series}), 'target', [3, 6, 12, 24]
            )
            
            # Trend features
            trend_features = self.extract_trend_features(
                pd.DataFrame({'target': target_series}), 'target', [3, 6, 12]
            )
            
            feature_matrix = pd.concat([
                feature_matrix, lag_features, rolling_features, trend_features
            ], axis=1)
        
        # Store feature names
        self.feature_names = feature_matrix.columns.tolist()
        
        logger.info(f"Created {len(self.feature_names)} temporal features")
        
        return feature_matrix