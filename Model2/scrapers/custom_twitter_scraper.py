"""
WORKING Disaster News Scraper - Production Version
Based on successful test results, this scraper actually gets REAL data.
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import time
import random
import logging

logger = logging.getLogger(__name__)

class WorkingDisasterScraper:
    """Disaster scraper that actually works and returns real data."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Proven working sources
        self.working_sources = {
            'thehindu': {
                'url': 'https://www.thehindu.com',
                'selectors': ['h2', 'h3', '.story-title', '.title', 'a'],
                'name': 'The Hindu'
            },
            'indiatoday': {
                'url': 'https://www.indiatoday.in', 
                'selectors': ['h2', 'h3', '.story-title', '.headline', 'a'],
                'name': 'India Today'
            },
            'imd': {
                'url': 'https://mausam.imd.gov.in',
                'selectors': ['p', 'div', 'span', 'td'],
                'name': 'IMD Official'
            }
        }
        
        # Keywords for disaster content
        self.disaster_keywords = [
            'alert', 'warning', 'flood', 'cyclone', 'storm', 'emergency', 
            'disaster', 'rescue', 'tsunami', 'earthquake', 'weather warning',
            'heavy rain', 'evacuation', 'relief', 'ndrf', 'red alert',
            'orange alert', 'yellow alert', 'rainfall warning', 'wind warning'
        ]
    
    def scrape_news_source(self, source_key, max_results=5):
        """Scrape a specific news source for disaster-related content."""
        source = self.working_sources[source_key]
        articles = []
        
        try:
            logger.info(f"🔍 Scraping {source['name']}...")
            response = self.session.get(source['url'], timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract all potential headlines/content
            all_content = []
            for selector in source['selectors']:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    if 20 <= len(text) <= 300:  # Reasonable content length
                        all_content.append(text)
            
            # Filter STRICTLY for disaster/weather-related content ONLY
            disaster_content = []
            for content in all_content:
                content_lower = content.lower()
                # First check for disaster keywords
                if any(keyword in content_lower for keyword in self.disaster_keywords):
                    # Double check - make sure it's actually disaster/weather related and NOT politics/business
                    if any(strong_keyword in content_lower for strong_keyword in [
                        'weather', 'rain', 'flood', 'cyclone', 'alert', 'warning',
                        'temperature', 'disaster', 'emergency', 'imd', 'evacuation', 
                        'rescue', 'rainfall', 'wind speed', 'storm', 'tsunami',
                        'earthquake', 'landslide', 'heat wave', 'cold wave', 'victims'
                    ]) and not any(exclude_keyword in content_lower for exclude_keyword in [
                        'trade deal', 'defence pact', 'mea', 'minister', 'policy',
                        'gst', 'tax', 'economy', 'business', 'stock', 'market',
                        'company', 'corporate', 'finance', 'investment', 'profit',
                        'saudi', 'pact', 'agreement', 'intensify efforts', 'u.s.'
                    ]):
                        disaster_content.append(content)
            
            # Only add if we found disaster/weather content - NO FALLBACK
            if disaster_content:
                logger.info(f"✅ Found {len(disaster_content)} disaster-related items")
                for content in disaster_content[:max_results]:
                    articles.append({
                        'title': content,
                        'content': content,
                        'source': source['name'],
                        'url': source['url'],
                        'timestamp': datetime.now().isoformat(),
                        'type': 'disaster_alert',
                        'priority': 'high'
                    })
            else:
                logger.info(f"ℹ️ No disaster/weather content found in {source['name']} - skipping general news")
            
            time.sleep(random.uniform(1, 3))  # Rate limiting
            
        except Exception as e:
            logger.error(f"❌ Failed to scrape {source['name']}: {e}")
        
        return articles
    
    def scrape_imd_weather(self, max_results=3):
        """Scrape IMD for weather warnings and alerts using WORKING extraction method."""
        weather_data = []
        
        try:
            logger.info("🌦️ Scraping IMD weather data...")
            imd_url = 'https://mausam.imd.gov.in'
            response = self.session.get(imd_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Use the SAME successful extraction method as government scraper
            for element in soup.find_all(['div', 'p', 'td'], string=True):
                text = element.get_text(strip=True)
                
                # Filter for weather/disaster related content (same as government scraper)
                if any(keyword in text.lower() for keyword in [
                    'cyclone', 'depression', 'rainfall', 'weather warning', 
                    'alert', 'forecast', 'monsoon', 'temperature', 'wind'
                ]) and len(text) > 50:
                    
                    # Clean and format the text (same as government scraper)
                    clean_text = re.sub(r'\s+', ' ', text)
                    if len(clean_text) > 100:  # Substantial content
                        weather_data.append({
                            'title': clean_text[:150] + '...' if len(clean_text) > 150 else clean_text,
                            'content': clean_text,
                            'source': 'IMD Official',
                            'url': imd_url,
                            'timestamp': datetime.now().isoformat(),
                            'type': 'weather_alert',
                            'priority': 'high'
                        })
            
            logger.info(f"✅ Found {len(weather_data)} weather data items")
            
        except Exception as e:
            logger.error(f"❌ IMD scraping failed: {e}")
        
        return weather_data
    
    def get_all_disaster_alerts(self):
        """Get all disaster alerts from working sources."""
        all_alerts = []
        
        # Scrape news sources
        for source_key in ['thehindu', 'indiatoday']:
            alerts = self.scrape_news_source(source_key, max_results=3)
            all_alerts.extend(alerts)
        
        # Scrape weather data
        weather_alerts = self.scrape_imd_weather(max_results=3)
        all_alerts.extend(weather_alerts)
        
        # Sort by priority (disaster alerts first)
        all_alerts.sort(key=lambda x: 0 if x['priority'] == 'high' else 1)
        
        logger.info(f"✅ Total alerts retrieved: {len(all_alerts)}")
        return all_alerts


# Backward compatibility functions
def fetch_twitter_alerts():
    """Fetch disaster alerts from working news sources (Twitter replacement)."""
    scraper = WorkingDisasterScraper()
    alerts = scraper.get_all_disaster_alerts()
    
    # Return as simple strings for compatibility
    result = []
    for alert in alerts:
        if alert['type'] == 'disaster_alert':
            result.append(f"[{alert['source']}] {alert['title']}")
    
    return result[:5]  # Limit to 5 alerts

def fetch_youtube_alerts():
    """Fetch disaster alerts from working sources (YouTube replacement)."""
    scraper = WorkingDisasterScraper()
    alerts = scraper.get_all_disaster_alerts()
    
    result = []
    for alert in alerts:
        result.append(f"[{alert['source']}] {alert['title']}")
    
    return result[:3]

def fetch_google_news_alerts():
    """Fetch news alerts from working sources."""
    scraper = WorkingDisasterScraper()
    alerts = scraper.get_all_disaster_alerts()
    
    result = []
    for alert in alerts:
        if alert['source'] in ['The Hindu', 'India Today']:
            result.append(f"[{alert['source']}] {alert['title']}")
    
    return result[:5]

def fetch_government_alerts():
    """Fetch government weather alerts."""
    scraper = WorkingDisasterScraper()
    weather_alerts = scraper.scrape_imd_weather(max_results=5)
    
    result = []
    for alert in weather_alerts:
        result.append(f"[{alert['source']}] {alert['title']}")
    
    return result

def fetch_incois_alerts():
    """Fetch INCOIS-style alerts from IMD."""
    scraper = WorkingDisasterScraper()
    weather_alerts = scraper.scrape_imd_weather(max_results=3)
    
    result = []
    for alert in weather_alerts:
        # Format as marine/coastal alerts
        result.append(f"COASTAL ALERT: {alert['title']}")
    
    return result


if __name__ == "__main__":
    print("🚀 Testing WORKING Disaster Scraper...")
    
    scraper = WorkingDisasterScraper()
    all_alerts = scraper.get_all_disaster_alerts()
    
    print(f"\n📊 RESULTS:")
    print(f"Total alerts retrieved: {len(all_alerts)}")
    
    disaster_alerts = [a for a in all_alerts if a['priority'] == 'high']
    general_alerts = [a for a in all_alerts if a['priority'] == 'low']
    
    print(f"🚨 High priority (disaster) alerts: {len(disaster_alerts)}")
    print(f"📰 General alerts: {len(general_alerts)}")
    
    if all_alerts:
        print(f"\n✅ SUCCESS! Sample alerts:")
        for i, alert in enumerate(all_alerts[:5], 1):
            print(f"{i}. [{alert['source']}] {alert['title'][:80]}...")
            print(f"   Type: {alert['type']} | Priority: {alert['priority']}")
    else:
        print(f"\n❌ No alerts retrieved")
    
    # Test compatibility functions
    print(f"\n🔄 Testing backward compatibility:")
    print(f"Twitter replacement: {len(fetch_twitter_alerts())} alerts")
    print(f"YouTube replacement: {len(fetch_youtube_alerts())} alerts") 
    print(f"Google News replacement: {len(fetch_google_news_alerts())} alerts")
    print(f"Government alerts: {len(fetch_government_alerts())} alerts")
    print(f"INCOIS replacement: {len(fetch_incois_alerts())} alerts")
    
    print(f"\n=== WORKING Scraper Test Complete ===")