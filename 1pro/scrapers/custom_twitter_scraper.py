"""
Custom Twitter scraper that works without API keys or snscrape.
Uses Twitter's public search interface and RSS feeds.
"""

import requests
import re
import json
import logging
from typing import List, Dict
from datetime import datetime, timedelta
from urllib.parse import quote
import time
import random

logger = logging.getLogger(__name__)


class CustomTwitterScraper:
    """Custom Twitter scraper using public endpoints."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
    def search_tweets_nitter(self, query: str, max_results: int = 10) -> List[str]:
        """
        Search tweets using Nitter (alternative Twitter frontend).
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of tweets to fetch
            
        Returns:
            List[str]: List of tweet texts
        """
        try:
            # Nitter instances (public Twitter frontends)
            nitter_instances = [
                "https://nitter.net",
                "https://nitter.it", 
                "https://nitter.fdn.fr",
                "https://nitter.1d4.us",
                "https://nitter.kavin.rocks"
            ]
            
            tweets = []
            
            for instance in nitter_instances:
                try:
                    # Add India filter and recent filter
                    search_url = f"{instance}/search?f=tweets&q={quote(query)}%20lang%3Aen%20place%3AIndia&since=1d"
                    
                    response = self.session.get(search_url, timeout=15)
                    if response.status_code == 200:
                        tweets_found = self._parse_nitter_response(response.text)
                        tweets.extend(tweets_found)
                        
                        if len(tweets) >= max_results:
                            break
                            
                        # Add small delay between requests
                        time.sleep(random.uniform(1, 3))
                        
                except Exception as e:
                    logger.warning(f"Nitter instance {instance} failed: {e}")
                    continue
            
            # Limit results and remove duplicates
            unique_tweets = list(dict.fromkeys(tweets))[:max_results]
            logger.info(f"Found {len(unique_tweets)} unique tweets via Nitter")
            return unique_tweets
            
        except Exception as e:
            logger.error(f"Error in Nitter search: {e}")
            return []
    
    def _parse_nitter_response(self, html_content: str) -> List[str]:
        """Parse tweet content from Nitter HTML."""
        tweets = []
        
        try:
            # Look for tweet content in Nitter's HTML structure
            tweet_patterns = [
                r'<div class="tweet-content[^"]*"[^>]*>(.*?)</div>',
                r'<p class="tweet-content[^"]*"[^>]*>(.*?)</p>',
                r'class="tweet-text"[^>]*>(.*?)</div>'
            ]
            
            for pattern in tweet_patterns:
                matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    # Clean HTML tags and extract text
                    clean_text = re.sub(r'<[^>]+>', '', match)
                    clean_text = clean_text.strip()
                    
                    if len(clean_text) > 20 and clean_text not in tweets:
                        tweets.append(clean_text)
            
        except Exception as e:
            logger.warning(f"Error parsing Nitter response: {e}")
        
        return tweets
    
    def search_tweets_rss(self, query: str, max_results: int = 10) -> List[str]:
        """
        Search tweets using RSS search (backup method).
        
        Args:
            query (str): Search query
            max_results (int): Maximum results
            
        Returns:
            List[str]: Tweet texts
        """
        try:
            # Twitter RSS search (alternative approach)
            rss_queries = [
                f"site:twitter.com {query} India",
                f"from twitter.com {query} flood OR disaster OR alert"
            ]
            
            tweets = []
            
            for rss_query in rss_queries:
                try:
                    # Use Google search for Twitter content
                    search_url = f"https://www.google.com/search?q={quote(rss_query)}&tbm=nws&tbs=qdr:d"
                    
                    response = self.session.get(search_url, timeout=10)
                    if response.status_code == 200:
                        # Extract Twitter-related news mentions
                        twitter_mentions = re.findall(
                            r'twitter\.com[^"]*"[^>]*>([^<]+)', 
                            response.text
                        )
                        
                        for mention in twitter_mentions:
                            clean_mention = re.sub(r'[^\w\s]', ' ', mention).strip()
                            if len(clean_mention) > 20:
                                tweets.append(clean_mention)
                        
                        if len(tweets) >= max_results:
                            break
                            
                except Exception as e:
                    logger.warning(f"RSS search failed for query {rss_query}: {e}")
                    continue
            
            unique_tweets = list(dict.fromkeys(tweets))[:max_results]
            logger.info(f"Found {len(unique_tweets)} tweets via RSS search")
            return unique_tweets
            
        except Exception as e:
            logger.error(f"Error in RSS search: {e}")
            return []
    
    def search_tweets(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Main search method that tries multiple approaches.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of tweets to fetch
            
        Returns:
            List[Dict]: List of tweet dictionaries with text and metadata
        """
        tweets = []
        
        # Try Nitter first (more reliable for recent tweets)
        try:
            logger.info(f"Trying Nitter search for: {query}")
            nitter_tweets = self.search_tweets_nitter(query, max_results)
            
            # Convert to dict format
            for tweet_text in nitter_tweets:
                tweets.append({
                    'text': tweet_text,
                    'source': 'nitter',
                    'username': 'unknown',
                    'timestamp': datetime.now().isoformat()
                })
                
            if len(tweets) >= max_results:
                return tweets[:max_results]
                
        except Exception as e:
            logger.warning(f"Nitter search failed: {e}")
        
        # If we don't have enough tweets, try RSS backup
        if len(tweets) < max_results:
            try:
                logger.info(f"Trying RSS search for: {query}")
                rss_tweets = self.search_tweets_rss(query, max_results - len(tweets))
                
                # Convert to dict format and add
                for tweet_text in rss_tweets:
                    tweets.append({
                        'text': tweet_text,
                        'source': 'rss',
                        'username': 'unknown',
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.warning(f"RSS search failed: {e}")
        
        # Filter for disaster-related content
        disaster_keywords = [
            'flood', 'flooding', 'tsunami', 'cyclone', 'storm', 'hurricane',
            'landslide', 'earthquake', 'disaster', 'emergency', 'evacuation',
            'rescue', 'alert', 'warning', 'damage', 'blocked', 'stranded',
            'rain', 'water', 'heavy', 'monsoon'
        ]
        
        filtered_tweets = []
        for tweet in tweets:
            tweet_text_lower = tweet['text'].lower()
            if any(keyword in tweet_text_lower for keyword in disaster_keywords):
                filtered_tweets.append(tweet)
        
        # Remove duplicates based on text
        unique_tweets = []
        seen_texts = set()
        for tweet in filtered_tweets:
            text_key = tweet['text'][:100]  # Use first 100 chars as key
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_tweets.append(tweet)
        
        logger.info(f"Found {len(unique_tweets)} relevant tweets for query: {query}")
        return unique_tweets[:max_results]
    
    def get_trending_disaster_topics(self) -> List[str]:
        """
        Get trending disaster-related topics from multiple sources.
        
        Returns:
            List[str]: Trending disaster topics
        """
        try:
            trending_topics = []
            
            # Check Google Trends for disaster keywords
            disaster_keywords = [
                "flood India", "cyclone India", "landslide India",
                "earthquake India", "tsunami India", "storm India"
            ]
            
            for keyword in disaster_keywords:
                try:
                    # Simple Google search to check recent activity
                    search_url = f"https://www.google.com/search?q={quote(keyword)}&tbm=nws&tbs=qdr:h"
                    response = self.session.get(search_url, timeout=10)
                    
                    if response.status_code == 200:
                        # Count mentions in recent news
                        news_count = len(re.findall(r'<h3[^>]*>', response.text))
                        if news_count > 5:  # If more than 5 recent news items
                            trending_topics.append(keyword)
                            
                except Exception as e:
                    logger.warning(f"Trend check failed for {keyword}: {e}")
                    continue
            
            logger.info(f"Found {len(trending_topics)} trending disaster topics")
            return trending_topics
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []


def fetch_twitter_alerts(query="flood OR tsunami OR cyclone", max_results=10):
    """
    Main function to fetch Twitter alerts using custom scraper.
    
    Args:
        query (str): Search query for tweets
        max_results (int): Maximum number of tweets to fetch
        
    Returns:
        list: List of tweet contents
    """
    try:
        scraper = CustomTwitterScraper()
        
        # Enhance query with India-specific terms
        enhanced_query = f"{query} (India OR Mumbai OR Delhi OR Chennai OR Kolkata OR Bangalore)"
        
        tweets = []
        
        # Try Nitter first (more reliable)
        logger.info(f"Searching tweets via Nitter for: {enhanced_query}")
        nitter_tweets = scraper.search_tweets_nitter(enhanced_query, max_results)
        tweets.extend(nitter_tweets)
        
        # If we don't have enough tweets, try RSS backup
        if len(tweets) < max_results // 2:
            logger.info("Trying RSS backup method...")
            rss_tweets = scraper.search_tweets_rss(enhanced_query, max_results - len(tweets))
            tweets.extend(rss_tweets)
        
        # Filter for disaster-related content
        disaster_tweets = []
        disaster_keywords = [
            'flood', 'flooding', 'tsunami', 'cyclone', 'storm', 'hurricane',
            'landslide', 'earthquake', 'disaster', 'emergency', 'evacuation',
            'rescue', 'alert', 'warning', 'damage', 'blocked', 'stranded'
        ]
        
        for tweet in tweets:
            tweet_lower = tweet.lower()
            if any(keyword in tweet_lower for keyword in disaster_keywords):
                disaster_tweets.append(tweet)
        
        # Remove duplicates and limit results
        unique_disaster_tweets = list(dict.fromkeys(disaster_tweets))[:max_results]
        
        logger.info(f"Successfully fetched {len(unique_disaster_tweets)} disaster-related tweets")
        return unique_disaster_tweets
        
    except Exception as e:
        logger.error(f"Error in custom Twitter scraping: {e}")
        return []


if __name__ == "__main__":
    # Test the custom scraper
    print("Testing custom Twitter scraper...")
    
    test_queries = [
        "flood Mumbai",
        "cyclone Tamil Nadu", 
        "landslide Uttarakhand"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: {query}")
        results = fetch_twitter_alerts(query, 3)
        
        for i, tweet in enumerate(results, 1):
            print(f"{i}. {tweet}")
        
        if not results:
            print("No tweets found")