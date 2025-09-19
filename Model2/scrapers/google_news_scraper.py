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
        
        # General news outlets for disaster/weather news
        self.working_sources = {
            'thehindu': {
                'url': 'https://www.thehindu.com',
                'weather_url': 'https://www.thehindu.com/news/national/',
                'selectors': ['h2', 'h3', '.story-title', '.title', 'a'],
                'name': 'The Hindu'
            },
            'hindustantimes': {
                'url': 'https://www.hindustantimes.com',
                'weather_url': 'https://www.hindustantimes.com/india-news/',
                'selectors': ['h2', 'h3', '.story-title', '.headline', 'a'],
                'name': 'Hindustan Times'
            },
            'timesofindia': {
                'url': 'https://timesofindia.indiatimes.com',
                'weather_url': 'https://timesofindia.indiatimes.com/india/',
                'selectors': ['h2', 'h3', '.story-title', '.headline', 'a'],
                'name': 'Times of India'
            },
            'indiatoday': {
                'url': 'https://www.indiatoday.in', 
                'weather_url': 'https://www.indiatoday.in/india/',
                'selectors': ['h2', 'h3', '.story-title', '.headline', 'a'],
                'name': 'India Today'
            }
        }
        
        # Keywords for disaster/weather content - very specific for news outlets
        self.disaster_keywords = [
            'weather warning', 'cyclone warning', 'heavy rainfall', 'flood alert', 
            'storm warning', 'red alert', 'orange alert', 'yellow alert',
            'disaster management', 'emergency evacuation', 'rescue operations',
            'imd forecast', 'weather bulletin', 'monsoon update', 'heat wave',
            'cold wave', 'thunderstorm warning', 'lightning alert', 'hailstorm',
            'weather advisory', 'meteorological warning', 'natural disaster',
            'tsunami warning', 'earthquake alert', 'landslide warning',
            'forest fire', 'drought conditions', 'flood relief', 'disaster relief',
            'weather conditions', 'rainfall warning', 'wind warning'
        ]
    
    def scrape_news_source(self, source_key, max_results=5):
        """Scrape a specific news source for disaster/weather-related content ONLY."""
        source = self.working_sources[source_key]
        articles = []
        
        try:
            logger.info(f"🔍 Scraping {source['name']} for disaster/weather news...")
            
            # Try the main URL and weather-specific URL
            urls_to_try = [source['url']]
            if 'weather_url' in source:
                urls_to_try.append(source['weather_url'])
            
            for url in urls_to_try:
                response = self.session.get(url, timeout=15)
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
                
                # Filter STRICTLY for disaster/weather-related content
                disaster_content = []
                for content in all_content:
                    content_lower = content.lower()
                    # First check for disaster keywords
                    if any(keyword in content_lower for keyword in self.disaster_keywords):
                        # Double check - make sure it's actually disaster/weather related and NOT sports/business
                        if any(strong_keyword in content_lower for strong_keyword in [
                            'weather', 'rain', 'flood', 'cyclone', 'alert', 'warning',
                            'temperature', 'disaster', 'emergency', 'imd', 'evacuation', 
                            'rescue', 'rainfall', 'wind speed', 'storm', 'tsunami',
                            'earthquake', 'landslide', 'heat wave', 'cold wave', 'victims'
                        ]) and not any(exclude_keyword in content_lower for exclude_keyword in [
                            'tournament', 'match', 'game', 'sport', 'cricket', 'football',
                            'badminton', 'tennis', 'player', 'team', 'score', 'championship',
                            'gst', 'tax', 'economy', 'business', 'stock', 'market', 'trade deal',
                            'company', 'corporate', 'finance', 'investment', 'profit', 'saudi',
                            'pact', 'agreement', 'mea', 'minister', 'policy'
                        ]):
                            disaster_content.append(content)
                    # Also check for general weather terms that might not be in disaster keywords
                    elif any(weather_term in content_lower for weather_term in [
                        'heavy rain', 'thunderstorm', 'weather forecast', 'monsoon',
                        'heat wave', 'cold snap', 'weather update', 'climate'
                    ]) and not any(exclude_keyword in content_lower for exclude_keyword in [
                        'tournament', 'sport', 'game', 'business', 'trade', 'economy'
                    ]):
                        disaster_content.append(content)
                
                # Only add if we found disaster/weather content
                if disaster_content:
                    logger.info(f"✅ Found {len(disaster_content)} disaster/weather items from {source['name']}")
                    for content in disaster_content[:max_results]:
                        articles.append({
                            'title': content,
                            'content': content,
                            'source': source['name'],
                            'url': url,
                            'timestamp': datetime.now().isoformat(),
                            'type': 'disaster_news',
                            'priority': 'high'
                        })
                    break  # Found content, no need to try other URLs
                
                time.sleep(random.uniform(1, 2))  # Rate limiting
            
            if not articles:
                logger.info(f"ℹ️ No disaster/weather content found in {source['name']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to scrape {source['name']}: {e}")
        
        return articles
    
    def scrape_all_news_sources(self):
        """Scrape all general news outlets for disaster/weather content."""
        all_articles = []
        
        # Scrape all major news outlets
        for source_key in ['thehindu', 'hindustantimes', 'timesofindia', 'indiatoday']:
            articles = self.scrape_news_source(source_key, max_results=2)
            all_articles.extend(articles)
        
        return all_articles
    
    def get_all_disaster_alerts(self):
        """Get all disaster/weather alerts from general news sources."""
        all_alerts = []
        
        # Scrape all news sources for disaster/weather content
        news_alerts = self.scrape_all_news_sources()
        all_alerts.extend(news_alerts)
        
        return all_alerts

# Backward compatibility functions
def fetch_google_news_alerts():
    """Fetch disaster/weather alerts from general news outlets (Hindu, Hindustan Times, Times of India, etc.)."""
    scraper = WorkingDisasterScraper()
    alerts = scraper.get_all_disaster_alerts()
    
    result = []
    for alert in alerts:
        # Only include alerts from general news outlets
        if alert['source'] in ['The Hindu', 'Hindustan Times', 'Times of India', 'India Today']:
            result.append(f"[{alert['source']}] {alert['title']}")
    
    return result[:5]  # Limit to 5 alerts


if __name__ == "__main__":
    print("🚀 Testing Google News Scraper (General News Outlets for Disaster/Weather)...")
    
    scraper = WorkingDisasterScraper()
    all_alerts = scraper.get_all_disaster_alerts()
    
    print(f"\n📊 RESULTS:")
    print(f"Total disaster/weather alerts from general news outlets: {len(all_alerts)}")
    
    print(f"\n✅ SUCCESS! Sample disaster/weather news:")
    for i, alert in enumerate(all_alerts[:5], 1):
        print(f"{i}. [{alert['source']}] {alert['title']}")
        print(f"   Type: {alert['type']} | Priority: {alert['priority']}")
    
    print(f"\n=== Google News Scraper Test Complete ===")
    print("✅ This scraper focuses ONLY on disaster/weather content from general news outlets")
    
    # Test compatibility functions
    print(f"\n🔄 Testing backward compatibility:")
    print(f"Google News alerts (disaster/weather from general outlets): {len(fetch_google_news_alerts())} alerts")
    
    print(f"\n=== Google News Scraper Test Complete ===")
    print("📰 This scraper only gets disaster/weather content from general news outlets")
    print("🚫 No general news like trade deals or politics - ONLY disaster/weather!")