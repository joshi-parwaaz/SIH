"""Real-time sensor data collector for environmental monitoring."""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import asyncio
import aiohttp
from time import sleep

from config import config

logger = logging.getLogger(__name__)

class SensorDataCollector:
    """Collects real-time environmental and sensor data."""
    
    def __init__(self):
        self.weather_api_key = config.get('data_sources.weather_api.api_key')
        self.weather_base_url = config.get('data_sources.weather_api.base_url')
        self.buoy_api_url = config.get('data_sources.sensor_data.buoy_api')
        self.refresh_interval = config.get('data_sources.sensor_data.refresh_interval', 300)
        
        self.current_data = {}
        self.historical_sensor_data = []
    
    async def collect_weather_data(self, locations: List[Dict]) -> Dict:
        """Collect weather data for multiple locations."""
        weather_data = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for location in locations:
                task = self._fetch_weather_for_location(session, location)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching weather for {locations[i]}: {result}")
                else:
                    weather_data[locations[i]['name']] = result
        
        return weather_data
    
    async def _fetch_weather_for_location(self, session: aiohttp.ClientSession, location: Dict) -> Dict:
        """Fetch weather data for a specific location."""
        if not self.weather_api_key:
            # Return mock data if no API key
            return self._generate_mock_weather_data(location)
        
        try:
            params = {
                'lat': location['lat'],
                'lon': location['lon'],
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            
            url = f"{self.weather_base_url}/weather"
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_weather_data(data)
                else:
                    logger.error(f"Weather API error: {response.status}")
                    return self._generate_mock_weather_data(location)
                    
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._generate_mock_weather_data(location)
    
    def _parse_weather_data(self, data: Dict) -> Dict:
        """Parse weather API response."""
        return {
            'timestamp': datetime.now(),
            'temperature': data.get('main', {}).get('temp', 0),
            'humidity': data.get('main', {}).get('humidity', 0),
            'pressure': data.get('main', {}).get('pressure', 0),
            'wind_speed': data.get('wind', {}).get('speed', 0),
            'wind_direction': data.get('wind', {}).get('deg', 0),
            'visibility': data.get('visibility', 0) / 1000,  # Convert to km
            'weather_condition': data.get('weather', [{}])[0].get('main', 'Clear'),
            'rainfall': data.get('rain', {}).get('1h', 0) if 'rain' in data else 0,
            'cloudiness': data.get('clouds', {}).get('all', 0)
        }
    
    def _generate_mock_weather_data(self, location: Dict) -> Dict:
        """Generate mock weather data for testing."""
        base_temp = 25 + np.random.normal(0, 5)
        
        return {
            'timestamp': datetime.now(),
            'temperature': round(base_temp, 2),
            'humidity': round(60 + np.random.normal(0, 20), 2),
            'pressure': round(1013 + np.random.normal(0, 10), 2),
            'wind_speed': round(np.random.exponential(5), 2),
            'wind_direction': round(np.random.uniform(0, 360), 2),
            'visibility': round(np.random.uniform(5, 20), 2),
            'weather_condition': np.random.choice(['Clear', 'Cloudy', 'Rain', 'Storm']),
            'rainfall': round(np.random.exponential(2) if np.random.random() > 0.7 else 0, 2),
            'cloudiness': round(np.random.uniform(0, 100), 2)
        }
    
    async def collect_buoy_data(self, buoy_ids: List[str]) -> Dict:
        """Collect data from ocean buoys."""
        buoy_data = {}
        
        for buoy_id in buoy_ids:
            try:
                data = await self._fetch_buoy_data(buoy_id)
                buoy_data[buoy_id] = data
            except Exception as e:
                logger.error(f"Error fetching buoy data for {buoy_id}: {e}")
                buoy_data[buoy_id] = self._generate_mock_buoy_data()
        
        return buoy_data
    
    async def _fetch_buoy_data(self, buoy_id: str) -> Dict:
        """Fetch data from a specific buoy."""
        # Since we don't have real buoy API, generate mock data
        return self._generate_mock_buoy_data()
    
    def _generate_mock_buoy_data(self) -> Dict:
        """Generate mock buoy sensor data."""
        return {
            'timestamp': datetime.now(),
            'sea_level': round(np.random.normal(0, 0.5), 3),  # meters relative to mean
            'wave_height': round(np.random.exponential(1.5), 2),  # meters
            'wave_period': round(np.random.normal(8, 2), 2),  # seconds
            'wave_direction': round(np.random.uniform(0, 360), 2),  # degrees
            'sea_temperature': round(np.random.normal(20, 3), 2),  # celsius
            'salinity': round(np.random.normal(35, 2), 2),  # ppt
            'current_speed': round(np.random.exponential(0.5), 3),  # m/s
            'current_direction': round(np.random.uniform(0, 360), 2),  # degrees
            'tide_level': round(np.random.normal(1.5, 0.8), 2)  # meters
        }
    
    def collect_coastal_sensors(self, sensor_locations: List[Dict]) -> Dict:
        """Collect data from coastal monitoring sensors."""
        sensor_data = {}
        
        for location in sensor_locations:
            sensor_id = location.get('id', f"sensor_{location['lat']}_{location['lon']}")
            sensor_data[sensor_id] = {
                'timestamp': datetime.now(),
                'location': location,
                'water_level': round(np.random.normal(2, 0.5), 2),
                'flow_rate': round(np.random.exponential(1), 2),
                'turbidity': round(np.random.exponential(10), 2),
                'ph_level': round(np.random.normal(7.8, 0.3), 2),
                'dissolved_oxygen': round(np.random.normal(8, 1), 2),
                'beach_erosion_rate': round(np.random.normal(0, 0.1), 3)
            }
        
        return sensor_data
    
    def aggregate_sensor_data(self, weather_data: Dict, buoy_data: Dict, 
                            coastal_data: Dict) -> pd.DataFrame:
        """Aggregate all sensor data into a unified format."""
        aggregated_data = []
        
        timestamp = datetime.now()
        
        # Process weather data
        for location, data in weather_data.items():
            record = {
                'timestamp': timestamp,
                'location': location,
                'data_type': 'weather',
                'temperature': data.get('temperature'),
                'humidity': data.get('humidity'),
                'pressure': data.get('pressure'),
                'wind_speed': data.get('wind_speed'),
                'rainfall': data.get('rainfall'),
                'visibility': data.get('visibility')
            }
            aggregated_data.append(record)
        
        # Process buoy data
        for buoy_id, data in buoy_data.items():
            record = {
                'timestamp': timestamp,
                'location': buoy_id,
                'data_type': 'buoy',
                'sea_level': data.get('sea_level'),
                'wave_height': data.get('wave_height'),
                'wave_period': data.get('wave_period'),
                'sea_temperature': data.get('sea_temperature'),
                'current_speed': data.get('current_speed'),
                'tide_level': data.get('tide_level')
            }
            aggregated_data.append(record)
        
        # Process coastal sensor data
        for sensor_id, data in coastal_data.items():
            record = {
                'timestamp': timestamp,
                'location': sensor_id,
                'data_type': 'coastal',
                'water_level': data.get('water_level'),
                'flow_rate': data.get('flow_rate'),
                'turbidity': data.get('turbidity'),
                'beach_erosion_rate': data.get('beach_erosion_rate')
            }
            aggregated_data.append(record)
        
        return pd.DataFrame(aggregated_data)
    
    def detect_anomalies(self, current_data: Dict, historical_data: List[Dict]) -> Dict:
        """Detect anomalies in current sensor readings."""
        anomalies = {}
        
        if not historical_data:
            return anomalies
        
        # Convert historical data to DataFrame for analysis
        hist_df = pd.DataFrame(historical_data)
        
        for key, value in current_data.items():
            if key in hist_df.columns and pd.api.types.is_numeric_dtype(hist_df[key]):
                # Calculate Z-score
                mean_val = hist_df[key].mean()
                std_val = hist_df[key].std()
                
                if std_val > 0:
                    z_score = abs((value - mean_val) / std_val)
                    
                    if z_score > 3:  # 3-sigma rule
                        anomalies[key] = {
                            'current_value': value,
                            'historical_mean': mean_val,
                            'z_score': z_score,
                            'severity': 'high' if z_score > 4 else 'medium'
                        }
        
        return anomalies
    
    def get_data_quality_score(self, data: Dict) -> float:
        """Calculate data quality score based on completeness and validity."""
        if not data:
            return 0.0
        
        total_fields = len(data)
        valid_fields = 0
        
        for key, value in data.items():
            if value is not None and not pd.isna(value):
                # Check for reasonable ranges
                if key == 'temperature' and -50 <= value <= 60:
                    valid_fields += 1
                elif key == 'humidity' and 0 <= value <= 100:
                    valid_fields += 1
                elif key == 'pressure' and 900 <= value <= 1100:
                    valid_fields += 1
                elif key == 'wind_speed' and 0 <= value <= 200:
                    valid_fields += 1
                elif key == 'wave_height' and 0 <= value <= 30:
                    valid_fields += 1
                else:
                    valid_fields += 1
        
        return valid_fields / total_fields if total_fields > 0 else 0.0
    
    def save_sensor_data(self, data: pd.DataFrame, filepath: str):
        """Save sensor data to file."""
        try:
            data.to_csv(filepath, index=False)
            logger.info(f"Saved sensor data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving sensor data: {e}")