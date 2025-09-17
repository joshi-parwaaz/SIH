import tweepy
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta
from ..data_ingestion import DataSource, RawReport
from config import config

class TwitterDataSource(DataSource):
    """Twitter data source for hazard-related tweets"""
    
    def __init__(self):
        self.client = None
        self.api_key = config.get('DATA_SOURCES.SOCIAL_MEDIA.TWITTER.API_KEY')
        self.api_secret = config.get('DATA_SOURCES.SOCIAL_MEDIA.TWITTER.API_SECRET')
        self.access_token = config.get('DATA_SOURCES.SOCIAL_MEDIA.TWITTER.ACCESS_TOKEN')
        self.access_secret = config.get('DATA_SOURCES.SOCIAL_MEDIA.TWITTER.ACCESS_SECRET')
        
        # Hazard-related keywords in multiple languages
        self.keywords = [
            # English
            'tsunami', 'flood', 'storm surge', 'high waves', 'coastal flood',
            'sea level rise', 'tidal wave', 'cyclone', 'hurricane',
            
            # Hindi
            'सुनामी', 'बाढ़', 'तूफान', 'ऊंची लहरें', 'समुद्री तूफान',
            
            # Tamil
            'சுனாமி', 'வெள்ளம்', 'புயல்', 'உயர் அலைகள்',
            
            # Telugu
            'సునామీ', 'వరద', 'తుఫాను', 'ఎత్తైన తరంగాలు',
            
            # Malayalam
            'സുനാമി', 'വെള്ളപ്പൊക്കം', 'കൊടുങ്കാറ്റ്', 'ഉയർന്ന തിരമാലകൾ',
            
            # Kannada
            'ಸುನಾಮಿ', 'ಪ್ರವಾಹ', 'ಚಂಡಮಾರುತ', 'ಎತ್ತರದ ಅಲೆಗಳು',
            
            # Location-based keywords
            'coastal area', 'beach', 'marina', 'port', 'lighthouse',
            'समुद्री तट', 'समुद्र तट', 'बंदरगाह'
        ]
    
    async def setup(self) -> None:
        """Setup Twitter API client"""
        try:
            # Setup Twitter API v2 client
            self.client = tweepy.Client(
                bearer_token=None,  # You'll need to add bearer token to config
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_secret,
                wait_on_rate_limit=True
            )
            
            # Test the connection
            user = self.client.get_me()
            print(f"Twitter API connected successfully as {user.data.username}")
            
        except Exception as e:
            raise Exception(f"Failed to setup Twitter API: {e}")
    
    async def fetch_data(self, **kwargs) -> List[RawReport]:
        """Fetch recent tweets about ocean hazards"""
        reports = []
        
        # Create search query with keywords
        query = ' OR '.join([f'"{keyword}"' for keyword in self.keywords[:10]])  # Limit to avoid query length issues
        query += ' -is:retweet lang:en OR lang:hi'  # Exclude retweets, include English and Hindi
        
        try:
            # Search for recent tweets
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=100,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'lang', 'geo'],
                user_fields=['location', 'verified'],
                expansions=['author_id', 'geo.place_id']
            )
            
            if tweets.data:
                for tweet in tweets.data:
                    # Extract user info
                    user_info = {}
                    if tweets.includes and 'users' in tweets.includes:
                        user = next((u for u in tweets.includes['users'] if u.id == tweet.author_id), None)
                        if user:
                            user_info = {
                                'username': user.username,
                                'location': getattr(user, 'location', None),
                                'verified': getattr(user, 'verified', False)
                            }
                    
                    # Extract location info
                    location_info = None
                    if hasattr(tweet, 'geo') and tweet.geo:
                        if tweets.includes and 'places' in tweets.includes:
                            place = next((p for p in tweets.includes['places'] if p.id == tweet.geo['place_id']), None)
                            if place:
                                location_info = {
                                    'place_name': place.full_name,
                                    'country': place.country,
                                    'place_type': place.place_type
                                }
                    
                    # Create report
                    report = RawReport(
                        id=f"twitter_{tweet.id}",
                        source="twitter",
                        content=tweet.text,
                        timestamp=tweet.created_at,
                        metadata={
                            'user_info': user_info,
                            'public_metrics': tweet.public_metrics,
                            'tweet_id': tweet.id,
                            'author_id': tweet.author_id
                        },
                        location=location_info,
                        language=getattr(tweet, 'lang', None)
                    )
                    
                    reports.append(report)
            
            print(f"Fetched {len(reports)} tweets from Twitter")
            
        except Exception as e:
            print(f"Error fetching Twitter data: {e}")
            
        return reports
    
    async def cleanup(self) -> None:
        """Cleanup Twitter client"""
        if self.client:
            self.client = None
            print("Twitter client cleaned up")

class TwitterStreamSource(DataSource):
    """Real-time Twitter stream for hazard monitoring"""
    
    def __init__(self):
        self.stream = None
        self.api_key = config.get('DATA_SOURCES.SOCIAL_MEDIA.TWITTER.API_KEY')
        self.api_secret = config.get('DATA_SOURCES.SOCIAL_MEDIA.TWITTER.API_SECRET')
        self.access_token = config.get('DATA_SOURCES.SOCIAL_MEDIA.TWITTER.ACCESS_TOKEN')
        self.access_secret = config.get('DATA_SOURCES.SOCIAL_MEDIA.TWITTER.ACCESS_SECRET')
        self.collected_tweets = []
    
    async def setup(self) -> None:
        """Setup Twitter streaming"""
        try:
            # This would require implementing a custom StreamingClient
            # For now, we'll use the regular client
            print("Twitter streaming setup completed")
        except Exception as e:
            raise Exception(f"Failed to setup Twitter streaming: {e}")
    
    async def fetch_data(self, **kwargs) -> List[RawReport]:
        """Get collected streaming data"""
        reports = self.collected_tweets.copy()
        self.collected_tweets.clear()
        return reports
    
    async def cleanup(self) -> None:
        """Cleanup streaming resources"""
        if self.stream:
            self.stream.disconnect()
            print("Twitter stream disconnected")