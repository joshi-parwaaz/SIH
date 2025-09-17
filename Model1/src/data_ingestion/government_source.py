import aiohttp
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from bs4 import BeautifulSoup
import json
import re
from ..data_ingestion import DataSource, RawReport
from config import config

class GovernmentAlertsSource(DataSource):
    """Government alerts data source for official hazard warnings"""
    
    def __init__(self):
        self.session = None
        self.incois_url = config.get('DATA_SOURCES.GOVERNMENT_ALERTS.INCOIS_URL')
        
        # Government data sources
        self.sources = {
            'incois': {
                'url': 'https://incois.gov.in/portal/datainfo/mb.jsp',
                'parser': self._parse_incois_alerts
            },
            'imd': {
                'url': 'https://mausam.imd.gov.in/responsive/cyclone_warning.php',
                'parser': self._parse_imd_alerts
            },
            'ndma': {
                'url': 'https://ndma.gov.in/en/media-public-awareness/warnings.html',
                'parser': self._parse_ndma_alerts
            }
        }
    
    async def setup(self) -> None:
        """Setup HTTP session for government data fetching"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'User-Agent': 'Ocean Hazard Platform Bot 1.0'
                }
            )
            print("Government alerts source setup completed")
        except Exception as e:
            raise Exception(f"Failed to setup government alerts source: {e}")
    
    async def fetch_data(self, **kwargs) -> List[RawReport]:
        """Fetch alerts from all government sources"""
        all_reports = []
        
        for source_name, source_config in self.sources.items():
            try:
                reports = await self._fetch_from_source(source_name, source_config)
                all_reports.extend(reports)
            except Exception as e:
                print(f"Error fetching from {source_name}: {e}")
        
        print(f"Fetched {len(all_reports)} government alerts")
        return all_reports
    
    async def _fetch_from_source(self, source_name: str, source_config: Dict) -> List[RawReport]:
        """Fetch data from a specific government source"""
        reports = []
        
        try:
            async with self.session.get(source_config['url']) as response:
                if response.status == 200:
                    content = await response.text()
                    reports = await source_config['parser'](content, source_name)
                else:
                    print(f"HTTP {response.status} for {source_name}")
        
        except Exception as e:
            print(f"Error fetching {source_name}: {e}")
        
        return reports
    
    async def _parse_incois_alerts(self, html_content: str, source_name: str) -> List[RawReport]:
        """Parse INCOIS alerts from HTML content"""
        reports = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for alert tables or divs (this is a simplified parser)
            alert_elements = soup.find_all(['div', 'table', 'tr'], class_=re.compile(r'alert|warning|bulletin', re.I))
            
            for element in alert_elements:
                text = element.get_text(strip=True)
                
                # Filter for ocean hazard-related content
                if self._is_ocean_hazard_related(text):
                    report = RawReport(
                        id=f"incois_{hash(text)}_{int(datetime.now().timestamp())}",
                        source="incois",
                        content=text,
                        timestamp=datetime.now(),
                        metadata={
                            'source_url': self.sources[source_name]['url'],
                            'element_type': element.name,
                            'element_class': element.get('class', [])
                        },
                        language='en'
                    )
                    reports.append(report)
        
        except Exception as e:
            print(f"Error parsing INCOIS alerts: {e}")
        
        return reports
    
    async def _parse_imd_alerts(self, html_content: str, source_name: str) -> List[RawReport]:
        """Parse IMD (India Meteorological Department) alerts"""
        reports = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for cyclone and weather warnings
            warning_elements = soup.find_all(['div', 'p', 'td'], string=re.compile(r'cyclone|storm|warning|alert', re.I))
            
            for element in warning_elements:
                parent = element.parent
                if parent:
                    text = parent.get_text(strip=True)
                    
                    if self._is_ocean_hazard_related(text) and len(text) > 50:
                        report = RawReport(
                            id=f"imd_{hash(text)}_{int(datetime.now().timestamp())}",
                            source="imd",
                            content=text,
                            timestamp=datetime.now(),
                            metadata={
                                'source_url': self.sources[source_name]['url'],
                                'warning_type': 'meteorological'
                            },
                            language='en'
                        )
                        reports.append(report)
        
        except Exception as e:
            print(f"Error parsing IMD alerts: {e}")
        
        return reports
    
    async def _parse_ndma_alerts(self, html_content: str, source_name: str) -> List[RawReport]:
        """Parse NDMA (National Disaster Management Authority) alerts"""
        reports = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for disaster warnings and advisories
            alert_elements = soup.find_all(['div', 'article', 'section'], class_=re.compile(r'warning|alert|advisory', re.I))
            
            for element in alert_elements:
                text = element.get_text(strip=True)
                
                if self._is_ocean_hazard_related(text) and len(text) > 100:
                    # Try to extract date from the content
                    date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
                    timestamp = datetime.now()
                    
                    if date_match:
                        try:
                            date_str = date_match.group()
                            # Parse date (simplified - in production, use proper date parsing)
                            timestamp = datetime.now()  # For now, use current time
                        except:
                            pass
                    
                    report = RawReport(
                        id=f"ndma_{hash(text)}_{int(datetime.now().timestamp())}",
                        source="ndma",
                        content=text,
                        timestamp=timestamp,
                        metadata={
                            'source_url': self.sources[source_name]['url'],
                            'alert_type': 'disaster_management'
                        },
                        language='en'
                    )
                    reports.append(report)
        
        except Exception as e:
            print(f"Error parsing NDMA alerts: {e}")
        
        return reports
    
    def _is_ocean_hazard_related(self, text: str) -> bool:
        """Check if text is related to ocean hazards"""
        ocean_keywords = [
            'tsunami', 'storm surge', 'cyclone', 'hurricane', 'typhoon',
            'coastal flood', 'high waves', 'tidal', 'sea level', 'maritime',
            'oceanic', 'coastal', 'beach', 'shore', 'bay', 'gulf',
            'port', 'harbour', 'marina', 'lighthouse', 'estuary'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in ocean_keywords)
    
    async def cleanup(self) -> None:
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()
            print("Government alerts source cleaned up")

class RSSFeedSource(DataSource):
    """RSS feed source for government alerts"""
    
    def __init__(self):
        self.session = None
        self.rss_feeds = [
            'https://incois.gov.in/portal/rss/warning.xml',
            'https://mausam.imd.gov.in/backend/assets/rss/cyclone.xml',
            # Add more RSS feeds as needed
        ]
    
    async def setup(self) -> None:
        """Setup RSS feed reader"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        print("RSS feed source setup completed")
    
    async def fetch_data(self, **kwargs) -> List[RawReport]:
        """Fetch data from RSS feeds"""
        reports = []
        
        for feed_url in self.rss_feeds:
            try:
                async with self.session.get(feed_url) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        feed_reports = self._parse_rss_feed(xml_content, feed_url)
                        reports.extend(feed_reports)
            except Exception as e:
                print(f"Error fetching RSS feed {feed_url}: {e}")
        
        return reports
    
    def _parse_rss_feed(self, xml_content: str, feed_url: str) -> List[RawReport]:
        """Parse RSS feed XML content"""
        reports = []
        
        try:
            soup = BeautifulSoup(xml_content, 'xml')
            items = soup.find_all('item')
            
            for item in items:
                title = item.find('title')
                description = item.find('description')
                pub_date = item.find('pubDate')
                link = item.find('link')
                
                if title and description:
                    content = f"{title.get_text()}\n\n{description.get_text()}"
                    
                    # Parse publication date
                    timestamp = datetime.now()
                    if pub_date:
                        try:
                            # RFC 2822 date format parsing
                            from email.utils import parsedate_to_datetime
                            timestamp = parsedate_to_datetime(pub_date.get_text())
                        except:
                            pass
                    
                    report = RawReport(
                        id=f"rss_{hash(content)}_{int(timestamp.timestamp())}",
                        source="rss_feed",
                        content=content,
                        timestamp=timestamp,
                        metadata={
                            'feed_url': feed_url,
                            'original_link': link.get_text() if link else None
                        },
                        language='en'
                    )
                    reports.append(report)
        
        except Exception as e:
            print(f"Error parsing RSS feed: {e}")
        
        return reports
    
    async def cleanup(self) -> None:
        """Cleanup RSS feed resources"""
        if self.session:
            await self.session.close()
            print("RSS feed source cleaned up")