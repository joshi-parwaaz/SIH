"""
Enhanced Twitter scraper with better geographic coverage and search terms.
Includes inland hazards like landslides and regional coverage.
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


def fetch_regional_hazards(region="India", max_results=20):
    """
    Enhanced search for regional hazards including inland disasters.
    
    Args:
        region (str): Geographic region to search
        max_results (int): Maximum number of tweets to fetch
        
    Returns:
        list: List of tweet contents
    """
    if not SNSCRAPE_AVAILABLE:
        logger.warning("snscrape not available, returning empty list")
        return []
    
    try:
        results = []
        
        # Enhanced search queries for better coverage
        search_queries = [
            # Ocean hazards (original)
            f"flood OR tsunami OR cyclone lang:en {region}",
            # Inland hazards (NEW)
            f"landslide OR \"heavy rain\" OR \"roads blocked\" lang:en {region}",
            # Regional specific (NEW)
            f"Uttarakhand flood OR landslide OR \"heavy rain\" lang:en",
            f"\"disaster alert\" OR \"weather warning\" lang:en {region}",
            # Emergency terms (NEW)  
            f"evacuation OR rescue OR emergency lang:en {region}",
        ]
        
        for query in search_queries:
            logger.info(f"Searching Twitter with enhanced query: {query}")
            
            try:
                for i, tweet in enumerate(
                    sntwitter.TwitterSearchScraper(query).get_items()
                ):
                    if len(results) >= max_results:
                        break
                        
                    # Filter out retweets and very short tweets
                    if (hasattr(tweet, 'content') and 
                        tweet.content and 
                        not tweet.content.startswith('RT @') and
                        len(tweet.content.strip()) > 20):
                        
                        results.append(tweet.content.strip())
                        
                # Don't exceed total limit
                if len(results) >= max_results:
                    break
                    
            except Exception as e:
                logger.warning(f"Error with query '{query}': {e}")
                continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for result in results:
            if result not in seen:
                seen.add(result)
                unique_results.append(result)
        
        logger.info(f"Fetched {len(unique_results)} unique tweets from enhanced search")
        return unique_results[:max_results]
        
    except Exception as e:
        logger.error(f"Error in enhanced Twitter search: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []


def fetch_uttarakhand_specific():
    """
    Specific search for current Uttarakhand situation.
    
    Returns:
        list: Uttarakhand-specific hazard tweets
    """
    if not SNSCRAPE_AVAILABLE:
        return []
    
    try:
        results = []
        
        # Very specific searches for current situation
        uttarakhand_queries = [
            "Uttarakhand flood lang:en",
            "Uttarakhand landslide lang:en", 
            "Uttarakhand \"heavy rain\" lang:en",
            "Uttarakhand \"roads blocked\" lang:en",
            "Dehradun flood lang:en",
            "Chamoli landslide lang:en",
        ]
        
        for query in uttarakhand_queries:
            logger.info(f"Uttarakhand-specific search: {query}")
            
            try:
                for i, tweet in enumerate(
                    sntwitter.TwitterSearchScraper(query).get_items()
                ):
                    if i >= 5:  # Limit per query
                        break
                        
                    if (hasattr(tweet, 'content') and 
                        tweet.content and 
                        not tweet.content.startswith('RT @') and
                        len(tweet.content.strip()) > 20):
                        
                        results.append(tweet.content.strip())
                        
            except Exception as e:
                logger.warning(f"Error with Uttarakhand query '{query}': {e}")
                continue
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for result in results:
            if result not in seen:
                seen.add(result)
                unique_results.append(result)
        
        logger.info(f"Fetched {len(unique_results)} Uttarakhand-specific tweets")
        return unique_results
        
    except Exception as e:
        logger.error(f"Error in Uttarakhand-specific search: {e}")
        return []


if __name__ == "__main__":
    # Test the enhanced scraper
    print("Testing enhanced regional hazard detection...")
    
    print("\n1. Enhanced general search:")
    general_results = fetch_regional_hazards("India", 10)
    for i, result in enumerate(general_results, 1):
        print(f"{i}. {result[:100]}...")
    
    print(f"\n2. Uttarakhand-specific search:")
    uk_results = fetch_uttarakhand_specific()
    for i, result in enumerate(uk_results, 1):
        print(f"{i}. {result[:100]}...")