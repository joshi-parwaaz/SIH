from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass
import asyncio

@dataclass
class RawReport:
    """Raw report data structure"""
    id: str
    source: str  # 'twitter', 'facebook', 'youtube', 'user_app', 'government'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
    location: Optional[Dict[str, Any]] = None  # lat, lon, address
    media_urls: Optional[List[str]] = None
    language: Optional[str] = None

class DataSource(ABC):
    """Abstract base class for all data sources"""
    
    @abstractmethod
    async def fetch_data(self, **kwargs) -> List[RawReport]:
        """Fetch data from the source"""
        pass
    
    @abstractmethod
    async def setup(self) -> None:
        """Setup the data source (authentication, etc.)"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass

class DataIngestionManager:
    """Manages all data sources and coordinates data ingestion"""
    
    def __init__(self):
        self.sources: Dict[str, DataSource] = {}
        self.is_running = False
    
    def register_source(self, name: str, source: DataSource) -> None:
        """Register a new data source"""
        self.sources[name] = source
    
    async def setup_all_sources(self) -> None:
        """Setup all registered data sources"""
        for name, source in self.sources.items():
            try:
                await source.setup()
                print(f"✓ Successfully setup {name}")
            except Exception as e:
                print(f"✗ Failed to setup {name}: {e}")
    
    async def fetch_all_data(self) -> List[RawReport]:
        """Fetch data from all sources"""
        all_reports = []
        
        tasks = []
        for name, source in self.sources.items():
            tasks.append(self._fetch_from_source(name, source))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                print(f"Error fetching data: {result}")
            elif isinstance(result, list):
                all_reports.extend(result)
        
        return all_reports
    
    async def _fetch_from_source(self, name: str, source: DataSource) -> List[RawReport]:
        """Fetch data from a single source with error handling"""
        try:
            return await source.fetch_data()
        except Exception as e:
            print(f"Error fetching from {name}: {e}")
            return []
    
    async def start_continuous_ingestion(self, interval: int = 300) -> AsyncGenerator[List[RawReport], None]:
        """Start continuous data ingestion"""
        self.is_running = True
        
        while self.is_running:
            try:
                reports = await self.fetch_all_data()
                print(f"Fetched {len(reports)} reports at {datetime.now()}")
                
                # Here you would typically send reports to the processing pipeline
                # For now, we'll just yield them
                yield reports
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"Error in continuous ingestion: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def stop_ingestion(self) -> None:
        """Stop continuous data ingestion"""
        self.is_running = False
    
    async def cleanup_all_sources(self) -> None:
        """Cleanup all data sources"""
        for name, source in self.sources.items():
            try:
                await source.cleanup()
                print(f"✓ Cleaned up {name}")
            except Exception as e:
                print(f"✗ Failed to cleanup {name}: {e}")