import yaml
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration management for Model A"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as file:
            self._config = yaml.safe_load(file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'DATABASE.MONGODB_URL')"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self._config.get('DATABASE', {})
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get specific model configuration"""
        return self._config.get('MODELS', {}).get(model_name, {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self._config.get('API', {})
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return self._config.get('LANGUAGES', {}).get('SUPPORTED', ['en'])
    
    @property
    def mongodb_url(self) -> str:
        """Get MongoDB connection URL"""
        return self.get('DATABASE.MONGODB_URL', 'mongodb://localhost:27017')
    
    @property
    def database_name(self) -> str:
        """Get database name"""
        return self.get('DATABASE.DATABASE_NAME', 'ocean_hazard_platform')

# Global config instance
config = Config()