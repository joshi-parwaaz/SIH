"""
Twitter scraper using snscrape (works without API keys).
Filtered to India with geographic constraints.
"""

import logging
import traceback

logger = logging.getLogger(__name__)

try:
    import snscrape.modules.twitter as sntwitter
    SNSCRAPE_AVAILABLE = True
except ImportError:
    logger.warning("snscrape not available. Twitter scraping disabled.")
    SNSCRAPE_AVAILABLE = False


def fetch_twitter_alerts(query="flood OR tsunami OR cyclone", max_results=10):
    """
    Scrape tweets related to ocean hazards in India.
    
    Args:
        query (str): Search query for tweets
        max_results (int): Maximum number of tweets to fetch
        
    Returns:
        list: List of tweet contents
    """
    if not SNSCRAPE_AVAILABLE:
        logger.warning("snscrape not available, returning empty list")
        return []
    
    try:
        results = []
        
        # Geographic filter for India (approximate center and radius)
        search_query = f"{query} lang:en geocode:20.5937,78.9629,1000km"
        
        logger.info(f"Searching Twitter with query: {search_query}")
        
        for i, tweet in enumerate(
            sntwitter.TwitterSearchScraper(search_query).get_items()
        ):
            if i >= max_results:
                break
                
            # Filter out retweets and very short tweets
            if (hasattr(tweet, 'content') and 
                tweet.content and 
                not tweet.content.startswith('RT @') and
                len(tweet.content.strip()) > 20):
                
                results.append(tweet.content.strip())
        
        logger.info(f"Fetched {len(results)} tweets from Twitter")
        return results
        
    except Exception as e:
        logger.error(f"Error fetching Twitter data: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []


def fetch_twitter_alerts_keywords():
    """
    Fetch tweets using multiple specific keywords for better coverage.
    
    Returns:
        list: Combined list of tweet contents from multiple searches
    """
    if not SNSCRAPE_AVAILABLE:
        return []
    
    keywords = [
        "tsunami India",
        "flood India",
        "cyclone India",
        "storm surge India",
        "ocean warning India"
    ]
    
    all_tweets = []
    
    for keyword in keywords:
        tweets = fetch_twitter_alerts(keyword, max_results=5)
        all_tweets.extend(tweets)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tweets = []
    for tweet in all_tweets:
        if tweet not in seen:
            seen.add(tweet)
            unique_tweets.append(tweet)
    
    return unique_tweets[:20]  # Return max 20 tweets


if __name__ == "__main__":
    # Test the scraper
    alerts = fetch_twitter_alerts()
    print(f"Found {len(alerts)} tweets:")
    for alert in alerts[:3]:  # Show first 3
        print(f"- {alert[:100]}...")  # Truncate for display