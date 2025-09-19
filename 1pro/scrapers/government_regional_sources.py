"""
Regional News Sources and Government API Integration.
Fetches alerts from IMD, NDMA, and regional news sources.
"""

import requests
import logging
from datetime import datetime
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import json

logger = logging.getLogger(__name__)


class GovernmentAPIClient:
    """Client for various government disaster management APIs."""
    
    def __init__(self):
        self.imd_base_url = "https://mausam.imd.gov.in"
        self.ndma_base_url = "https://ndma.gov.in"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_imd_alerts(self) -> List[Dict]:
        """
        Fetch weather alerts from India Meteorological Department.
        
        Returns:
            List[Dict]: IMD weather alerts
        """
        try:
            alerts = []
            
            # IMD RSS feeds and APIs
            imd_feeds = [
                "https://mausam.imd.gov.in/imd_latest/contents/all_india_forcast_bulletin.xml",
                "https://mausam.imd.gov.in/imd_latest/contents/cyclone.xml",
                "https://mausam.imd.gov.in/imd_latest/contents/rainfall_bulletin.xml"
            ]
            
            for feed_url in imd_feeds:
                try:
                    response = self.session.get(feed_url, timeout=10)
                    if response.status_code == 200:
                        # Parse XML
                        root = ET.fromstring(response.content)
                        
                        # Extract items from RSS feed
                        for item in root.findall(".//item"):
                            title = item.find("title")
                            description = item.find("description")
                            pub_date = item.find("pubDate")
                            
                            if title is not None and title.text:
                                alert = {
                                    "source": "IMD",
                                    "title": title.text.strip(),
                                    "description": description.text.strip() if description is not None else "",
                                    "published": pub_date.text.strip() if pub_date is not None else "",
                                    "url": feed_url,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                                alerts.append(alert)
                                
                except Exception as e:
                    logger.warning(f"Error fetching IMD feed {feed_url}: {e}")
                    continue
            
            logger.info(f"Fetched {len(alerts)} alerts from IMD")
            return alerts
            
        except Exception as e:
            logger.error(f"Error in IMD alert fetching: {e}")
            return []
    
    def fetch_ndma_alerts(self) -> List[Dict]:
        """
        Fetch disaster alerts from National Disaster Management Authority.
        
        Returns:
            List[Dict]: NDMA disaster alerts
        """
        try:
            alerts = []
            
            # NDMA might have different endpoints
            ndma_urls = [
                "https://ndma.gov.in/en/alerts",
                "https://ndma.gov.in/en/disaster-data-statistics"
            ]
            
            for url in ndma_urls:
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        # Parse HTML for alert content
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Look for alert-related content
                        alert_elements = soup.find_all(['div', 'article', 'section'], 
                                                     class_=lambda x: x and ('alert' in x.lower() or 'warning' in x.lower()))
                        
                        for element in alert_elements:
                            text_content = element.get_text(strip=True)
                            if len(text_content) > 50:  # Filter out short snippets
                                alert = {
                                    "source": "NDMA",
                                    "content": text_content[:500],  # Limit content length
                                    "url": url,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                                alerts.append(alert)
                                
                except Exception as e:
                    logger.warning(f"Error fetching NDMA content from {url}: {e}")
                    continue
            
            logger.info(f"Fetched {len(alerts)} alerts from NDMA")
            return alerts
            
        except Exception as e:
            logger.error(f"Error in NDMA alert fetching: {e}")
            return []


class RegionalNewsClient:
    """Client for regional news sources covering local disasters."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_regional_news(self, region: str = "uttarakhand") -> List[Dict]:
        """
        Fetch news from regional sources.
        
        Args:
            region (str): Region to focus on (e.g., "uttarakhand", "kerala", "odisha")
            
        Returns:
            List[Dict]: Regional news articles about disasters
        """
        try:
            news_articles = []
            
            # Regional news sources
            regional_sources = {
                "uttarakhand": [
                    "https://www.amarujala.com/uttarakhand",
                    "https://www.jagran.com/uttarakhand/",
                    "https://www.hindustantimes.com/cities/dehradun-news/",
                ],
                "kerala": [
                    "https://www.mathrubhumi.com/news/kerala",
                    "https://www.manoramaonline.com/news/kerala.html",
                ],
                "odisha": [
                    "https://www.sambadenglish.com/",
                    "https://odishatv.in/news/odisha",
                ],
                "general": [
                    "https://www.ndtv.com/topic/natural-disasters",
                    "https://indianexpress.com/section/india/",
                    "https://timesofindia.indiatimes.com/topic/Disasters"
                ]
            }
            
            sources = regional_sources.get(region.lower(), regional_sources["general"])
            
            for source_url in sources:
                try:
                    response = self.session.get(source_url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Look for disaster-related keywords in headlines
                        disaster_keywords = [
                            'flood', 'landslide', 'rain', 'disaster', 'alert', 
                            'warning', 'cyclone', 'storm', 'evacuation', 'rescue',
                            'blocked', 'damage', 'emergency'
                        ]
                        
                        # Find headlines and article links
                        headlines = soup.find_all(['h1', 'h2', 'h3', 'h4'], 
                                                string=lambda text: text and any(keyword in text.lower() for keyword in disaster_keywords))
                        
                        for headline in headlines[:5]:  # Limit to 5 per source
                            headline_text = headline.get_text(strip=True)
                            
                            # Try to find associated article content
                            article_content = ""
                            parent = headline.find_parent(['article', 'div', 'section'])
                            if parent:
                                article_content = parent.get_text(strip=True)[:300]
                            
                            article = {
                                "source": f"Regional News ({source_url})",
                                "headline": headline_text,
                                "content": article_content,
                                "region": region,
                                "url": source_url,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                            news_articles.append(article)
                            
                except Exception as e:
                    logger.warning(f"Error fetching regional news from {source_url}: {e}")
                    continue
            
            logger.info(f"Fetched {len(news_articles)} regional news articles for {region}")
            return news_articles
            
        except Exception as e:
            logger.error(f"Error in regional news fetching: {e}")
            return []
    
    def fetch_uttarakhand_specific_news(self) -> List[Dict]:
        """
        Fetch Uttarakhand-specific disaster news for current situation.
        
        Returns:
            List[Dict]: Uttarakhand disaster news
        """
        try:
            # Use Google News search for current Uttarakhand news
            search_queries = [
                "Uttarakhand flood heavy rain landslide",
                "Uttarakhand roads blocked disaster",
                "Dehradun Haridwar Rishikesh flood",
                "Chamoli Uttarkashi landslide"
            ]
            
            news_results = []
            
            for query in search_queries:
                try:
                    # Google News search URL
                    news_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
                    
                    response = self.session.get(news_url, timeout=10)
                    if response.status_code == 200:
                        # Parse RSS feed
                        root = ET.fromstring(response.content)
                        
                        for item in root.findall(".//item")[:3]:  # Limit to 3 per query
                            title = item.find("title")
                            description = item.find("description")
                            pub_date = item.find("pubDate")
                            link = item.find("link")
                            
                            if title is not None:
                                news_item = {
                                    "source": "Google News (Uttarakhand)",
                                    "title": title.text,
                                    "description": description.text if description is not None else "",
                                    "published": pub_date.text if pub_date is not None else "",
                                    "url": link.text if link is not None else "",
                                    "search_query": query,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                                news_results.append(news_item)
                                
                except Exception as e:
                    logger.warning(f"Error with Google News query '{query}': {e}")
                    continue
            
            logger.info(f"Fetched {len(news_results)} Uttarakhand-specific news items")
            return news_results
            
        except Exception as e:
            logger.error(f"Error in Uttarakhand news fetching: {e}")
            return []


def fetch_all_government_sources() -> Dict[str, List[Dict]]:
    """
    Fetch from all government and regional sources.
    
    Returns:
        Dict: Organized results from all sources
    """
    gov_client = GovernmentAPIClient()
    news_client = RegionalNewsClient()
    
    all_sources = {
        "imd_alerts": gov_client.fetch_imd_alerts(),
        "ndma_alerts": gov_client.fetch_ndma_alerts(),
        "regional_news": news_client.fetch_regional_news("uttarakhand"),
        "uttarakhand_news": news_client.fetch_uttarakhand_specific_news()
    }
    
    # Flatten for easier processing
    all_items = []
    for source_type, items in all_sources.items():
        for item in items:
            item["source_type"] = source_type
            all_items.append(item)
    
    all_sources["all_combined"] = all_items
    
    total_items = len(all_items)
    logger.info(f"Total items fetched from all government/regional sources: {total_items}")
    
    return all_sources


if __name__ == "__main__":
    # Test the enhanced sources
    print("Testing Government and Regional News Sources...")
    
    results = fetch_all_government_sources()
    
    print(f"\nResults Summary:")
    for source_type, items in results.items():
        if source_type != "all_combined":
            print(f"- {source_type}: {len(items)} items")
    
    print(f"\nSample results:")
    for item in results["all_combined"][:3]:
        print(f"- {item.get('source', 'Unknown')}: {item.get('title', item.get('headline', 'No title'))[:80]}...")