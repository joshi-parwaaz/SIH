"""Historical events data collector for hazard prediction."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)

class HistoricalEventsCollector:
    """Collects and processes historical hazard event data."""
    
    def __init__(self):
        self.data_path = config.get('data_sources.historical_data.path', 'data/historical_events.csv')
        self.events_df = None
    
    def load_historical_data(self) -> pd.DataFrame:
        """Load historical hazard events from various sources."""
        try:
            # Load from CSV file if exists
            if Path(self.data_path).exists():
                self.events_df = pd.read_csv(self.data_path)
                logger.info(f"Loaded {len(self.events_df)} historical events from {self.data_path}")
            else:
                # Create sample data structure if file doesn't exist
                self.events_df = self._create_sample_data()
                logger.warning(f"Historical data file not found. Created sample data structure.")
            
            return self._preprocess_events()
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample historical events data structure."""
        sample_data = {
            'event_id': ['E001', 'E002', 'E003', 'E004', 'E005'],
            'date': ['2023-06-15', '2023-07-22', '2023-08-10', '2023-09-05', '2023-10-12'],
            'latitude': [19.0760, 15.2993, 13.0827, 11.1271, 9.9252],
            'longitude': [72.8777, 74.1240, 80.2707, 75.8333, 78.1198],
            'location': ['Mumbai', 'Goa', 'Chennai', 'Mysore', 'Bangalore'],
            'hazard_type': ['storm_surge', 'high_waves', 'tsunami', 'coastal_flooding', 'storm_surge'],
            'severity': [8, 6, 9, 7, 5],
            'impact_score': [85, 60, 95, 70, 50],
            'casualties': [12, 0, 45, 8, 2],
            'economic_damage': [50000000, 5000000, 200000000, 15000000, 3000000],
            'duration_hours': [18, 12, 6, 24, 15],
            'affected_population': [150000, 20000, 500000, 80000, 25000]
        }
        return pd.DataFrame(sample_data)
    
    def _preprocess_events(self) -> pd.DataFrame:
        """Preprocess historical events data."""
        if self.events_df is None or self.events_df.empty:
            return pd.DataFrame()
        
        # Convert date column to datetime
        self.events_df['date'] = pd.to_datetime(self.events_df['date'])
        
        # Add temporal features
        self.events_df['year'] = self.events_df['date'].dt.year
        self.events_df['month'] = self.events_df['date'].dt.month
        self.events_df['day_of_year'] = self.events_df['date'].dt.dayofyear
        self.events_df['season'] = self.events_df['month'].apply(self._get_season)
        
        # Ensure required columns exist
        required_columns = ['latitude', 'longitude', 'hazard_type', 'severity', 'impact_score']
        for col in required_columns:
            if col not in self.events_df.columns:
                logger.warning(f"Missing column {col} in historical data")
                self.events_df[col] = 0
        
        return self.events_df
    
    def _get_season(self, month: int) -> str:
        """Get season from month number."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def get_events_by_location(self, lat: float, lon: float, radius_km: float = 50) -> pd.DataFrame:
        """Get historical events within specified radius of a location."""
        if self.events_df is None or self.events_df.empty:
            return pd.DataFrame()
        
        # Calculate distance using haversine formula (simplified)
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth's radius in km
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c
        
        self.events_df['distance'] = self.events_df.apply(
            lambda row: haversine_distance(lat, lon, row['latitude'], row['longitude']),
            axis=1
        )
        
        return self.events_df[self.events_df['distance'] <= radius_km].copy()
    
    def get_events_by_timeframe(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical events within specified timeframe."""
        if self.events_df is None or self.events_df.empty:
            return pd.DataFrame()
        
        mask = (self.events_df['date'] >= start_date) & (self.events_df['date'] <= end_date)
        return self.events_df[mask].copy()
    
    def get_event_frequency(self, hazard_type: str = None, location: str = None) -> Dict:
        """Calculate event frequency statistics."""
        if self.events_df is None or self.events_df.empty:
            return {}
        
        df = self.events_df.copy()
        
        if hazard_type:
            df = df[df['hazard_type'] == hazard_type]
        if location:
            df = df[df['location'] == location]
        
        if df.empty:
            return {}
        
        # Calculate various frequency metrics
        total_events = len(df)
        date_range = (df['date'].max() - df['date'].min()).days
        events_per_year = total_events / (date_range / 365.25) if date_range > 0 else 0
        
        severity_stats = df['severity'].describe().to_dict()
        seasonal_distribution = df.groupby('season').size().to_dict()
        
        return {
            'total_events': total_events,
            'events_per_year': events_per_year,
            'date_range_days': date_range,
            'severity_stats': severity_stats,
            'seasonal_distribution': seasonal_distribution,
            'average_impact': df['impact_score'].mean() if 'impact_score' in df.columns else 0
        }
    
    def get_recurrence_patterns(self) -> Dict:
        """Analyze recurrence patterns in historical data."""
        if self.events_df is None or self.events_df.empty:
            return {}
        
        patterns = {}
        
        # Monthly patterns
        monthly_counts = self.events_df.groupby('month').size()
        patterns['monthly_distribution'] = monthly_counts.to_dict()
        
        # Seasonal patterns
        seasonal_counts = self.events_df.groupby('season').size()
        patterns['seasonal_distribution'] = seasonal_counts.to_dict()
        
        # Location-based patterns
        if 'location' in self.events_df.columns:
            location_counts = self.events_df.groupby('location').size()
            patterns['location_distribution'] = location_counts.to_dict()
        
        # Hazard type patterns
        hazard_counts = self.events_df.groupby('hazard_type').size()
        patterns['hazard_type_distribution'] = hazard_counts.to_dict()
        
        return patterns
    
    def save_processed_data(self, output_path: str = None):
        """Save processed historical data."""
        if self.events_df is None or self.events_df.empty:
            logger.warning("No data to save")
            return
        
        if output_path is None:
            output_path = self.data_path.replace('.csv', '_processed.csv')
        
        try:
            self.events_df.to_csv(output_path, index=False)
            logger.info(f"Saved processed historical data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")