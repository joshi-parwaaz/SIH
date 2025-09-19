"""
Real Government Sources Scraper
Scrapes ACTUAL government websites for disaster alerts and weather bulletins.
Provides real-time information from official Indian government sources.

Sources:
- IMD (India Meteorological Department) - mausam.imd.gov.in
- NDMA (National Disaster Management Authority) - ndma.gov.in
- State Emergency Management - various state portals
- INCOIS integration for marine bulletins
"""

import requests
import re
import json
import time
from datetime import datetime, timedelta
from urllib.parse import urljoin, quote_plus
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class RealGovernmentScraper:
    """Real government sources scraper using web scraping techniques."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Official government sources
        self.imd_urls = [
            'https://mausam.imd.gov.in',
            'https://mausam.imd.gov.in/imd_latest/contents/cyclone.php',
            'https://mausam.imd.gov.in/imd_latest/contents/all-india-weather-summary.php',
            'https://nwp.imd.gov.in/bias/pbias.php'
        ]
        
        self.ndma_urls = [
            'https://ndma.gov.in',
            'https://ndma.gov.in/en/alerts-warnings.html',
            'https://ndma.gov.in/en/media-public-awareness.html'
        ]
        
        self.state_emergency_urls = [
            'https://www.gujaratindia.gov.in',  # Gujarat Emergency
            'https://kerala.gov.in',            # Kerala Disaster Management
            'https://odisha.gov.in',            # Odisha Emergency
            'https://www.tn.gov.in',            # Tamil Nadu Emergency
            'https://wb.gov.in'                 # West Bengal Emergency
        ]
    
    def scrape_imd_weather_data(self, max_results=5):
        """Scrape real IMD weather bulletins and alerts."""
        bulletins = []
        
        for url in self.imd_urls[:2]:  # Limit requests
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for weather bulletins, alerts, and forecasts
                for element in soup.find_all(['div', 'p', 'td'], text=True):
                    text = element.get_text(strip=True)
                    
                    # Filter for weather/disaster related content
                    if any(keyword in text.lower() for keyword in [
                        'cyclone', 'depression', 'rainfall', 'weather warning', 
                        'alert', 'forecast', 'monsoon', 'temperature', 'wind'
                    ]) and len(text) > 50:
                        
                        # Clean and format the text
                        clean_text = re.sub(r'\s+', ' ', text)
                        if len(clean_text) > 100:  # Substantial content
                            bulletins.append({
                                'title': clean_text[:100] + '...' if len(clean_text) > 100 else clean_text,
                                'content': clean_text,
                                'source': 'IMD Official',
                                'url': url,
                                'timestamp': datetime.now().isoformat()
                            })
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error scraping IMD URL {url}: {e}")
        
        logger.info(f"Found {len(bulletins)} real IMD bulletins")
        return bulletins[:max_results]
    
    def scrape_ndma_alerts(self, max_results=5):
        """Scrape real NDMA disaster alerts and advisories."""
        alerts = []
        
        for url in self.ndma_urls[:2]:  # Limit requests
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for disaster alerts and advisories
                for element in soup.find_all(['div', 'p', 'h3', 'h4'], text=True):
                    text = element.get_text(strip=True)
                    
                    # Filter for disaster management content
                    if any(keyword in text.lower() for keyword in [
                        'alert', 'warning', 'advisory', 'disaster', 'emergency',
                        'evacuation', 'rescue', 'relief', 'preparedness', 'cyclone',
                        'flood', 'tsunami', 'earthquake', 'ndrf', 'sdrf'
                    ]) and len(text) > 30:
                        
                        # Clean and format the text
                        clean_text = re.sub(r'\s+', ' ', text)
                        if len(clean_text) > 50:  # Substantial content
                            alerts.append({
                                'title': clean_text[:80] + '...' if len(clean_text) > 80 else clean_text,
                                'content': clean_text,
                                'source': 'NDMA Official',
                                'url': url,
                                'timestamp': datetime.now().isoformat()
                            })
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error scraping NDMA URL {url}: {e}")
        
        logger.info(f"Found {len(alerts)} real NDMA alerts")
        return alerts[:max_results]
    
    def scrape_state_emergency_info(self, max_results=5):
        """Scrape state government emergency information."""
        state_info = []
        
        for url in self.state_emergency_urls[:3]:  # Limit requests
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for emergency/disaster related content
                for element in soup.find_all(['div', 'p', 'span'], text=True):
                    text = element.get_text(strip=True)
                    
                    # Filter for emergency content
                    if any(keyword in text.lower() for keyword in [
                        'emergency', 'disaster', 'alert', 'warning', 'cyclone',
                        'flood', 'rainfall', 'rescue', 'relief', 'evacuation'
                    ]) and len(text) > 40:
                        
                        # Clean and format the text
                        clean_text = re.sub(r'\s+', ' ', text)
                        if len(clean_text) > 60:  # Substantial content
                            state_name = url.split('.')[0].split('//')[-1].title()
                            state_info.append({
                                'title': clean_text[:90] + '...' if len(clean_text) > 90 else clean_text,
                                'content': clean_text,
                                'source': f'{state_name} Government',
                                'url': url,
                                'timestamp': datetime.now().isoformat()
                            })
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error scraping state URL {url}: {e}")
        
        logger.info(f"Found {len(state_info)} real state emergency info")
        return state_info[:max_results]
    
    def search_government_news_via_google(self, query="disaster alert india government", max_results=5):
        """Search for recent government disaster news via Google."""
        try:
            # Search for government disaster news
            search_query = f"site:gov.in OR site:imd.gov.in OR site:ndma.gov.in {query}"
            google_url = f"https://www.google.com/search?q={quote_plus(search_query)}&tbm=nws"
            
            response = self.session.get(google_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = []
            
            # Find news results
            for result in soup.find_all('div', class_='g')[:max_results]:
                title_elem = result.find('h3')
                link_elem = result.find('a')
                
                if title_elem and link_elem:
                    title = title_elem.get_text(strip=True)
                    href = link_elem.get('href', '')
                    
                    if 'gov.in' in href:  # Only government sources
                        news_items.append({
                            'title': title,
                            'content': title,  # Use title as content for now
                            'source': 'Government News',
                            'url': href,
                            'timestamp': datetime.now().isoformat()
                        })
            
            logger.info(f"Found {len(news_items)} government news items")
            return news_items
            
        except Exception as e:
            logger.error(f"Error searching government news: {e}")
            return []


def fetch_all_government_sources():
    """
    Fetch real government disaster and weather information.
    Main function for production pipeline integration.
    """
    try:
        scraper = RealGovernmentScraper()
        
        # Scrape from all real government sources
        imd_bulletins = scraper.scrape_imd_weather_data(max_results=3)
        ndma_alerts = scraper.scrape_ndma_alerts(max_results=3)
        state_info = scraper.scrape_state_emergency_info(max_results=3)
        gov_news = scraper.search_government_news_via_google(max_results=2)
        
        # Combine all sources
        all_sources = imd_bulletins + ndma_alerts + state_info + gov_news
        
        # Format for backward compatibility
        formatted_results = []
        for item in all_sources:
            if isinstance(item, dict):
                formatted_text = f"[{item['source']}] {item['title']}"
                if 'url' in item:
                    formatted_text += f" - {item['url']}"
                formatted_results.append(formatted_text)
        
        # Also return structured format
        result = {
            "imd": [item['title'] for item in imd_bulletins],
            "ndma": [item['title'] for item in ndma_alerts], 
            "state_authorities": [item['title'] for item in state_info],
            "regional_services": [item['title'] for item in gov_news],
            "all_combined": [{"title": item, "description": item, "content": item} for item in formatted_results],
            "total_sources": 4,
            "total_alerts": len(formatted_results),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Retrieved {len(formatted_results)} real government alerts")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching real government sources: {e}")
        return {
            "all_combined": [],
            "total_alerts": 0,
            "imd": [],
            "ndma": [],
            "state_authorities": [],
            "regional_services": []
        }


# For backward compatibility - return just the formatted list
def fetch_government_alerts():
    """Simple function that returns list of government alert strings."""
    result = fetch_all_government_sources()
    return [item['title'] for item in result.get('all_combined', [])]


if __name__ == "__main__":
    # Test the REAL government scraper
    print("=== Testing REAL Government Sources Scraper ===")
    
    result = fetch_all_government_sources()
    
    print(f"Retrieved {result['total_alerts']} real government alerts:")
    print(f"- IMD Bulletins: {len(result['imd'])}")
    print(f"- NDMA Alerts: {len(result['ndma'])}")
    print(f"- State Info: {len(result['state_authorities'])}")
    print(f"- Gov News: {len(result['regional_services'])}")
    
    print(f"\nFirst 5 real alerts:")
    for i, alert_data in enumerate(result['all_combined'][:5], 1):
        print(f"{i}. {alert_data['title']}")
        print("   ✅ REAL scraped content")
    
    print("\n=== REAL Government Scraper Test Completed ===")