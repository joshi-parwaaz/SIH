"""Configuration module for Model B - Ocean Hazard Prediction Model."""

import yaml
import os
from pathlib import Path

class Config:
    """Configuration class for loading and managing application settings."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Load environment variables
        self._load_env_variables()
    
    def _load_env_variables(self):
        """Load environment variables and replace placeholders in config."""
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                return os.getenv(env_var, obj)
            else:
                return obj
        
        self.config = replace_env_vars(self.config)
    
    def get(self, key: str, default=None):
        """Get configuration value by key using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def app(self):
        return self.config.get('app', {})
    
    @property
    def api(self):
        return self.config.get('api', {})
    
    @property
    def database(self):
        return self.config.get('database', {})
    
    @property
    def prediction(self):
        return self.config.get('prediction', {})
    
    @property
    def models(self):
        return self.config.get('models', {})
    
    @property
    def features(self):
        return self.config.get('features', {})
    
    @property
    def data_sources(self):
        return self.config.get('data_sources', {})
    
    @property
    def logging(self):
        return self.config.get('logging', {})

# Global config instance
config = Config()