"""
YouTube scraper using YouTube Data API v3.
Requires free API key from Google Cloud Console.
"""

import requests
import os
import logging

logger = logging.getLogger(__name__)

API_KEY = os.getenv("YOUTUBE_API_KEY")


def fetch_youtube_alerts(query="flood India", max_results=5):
    """
    Fetch YouTube videos related to ocean hazards in India.
    
    Args:
        query (str): Search query for videos
        max_results (int): Maximum number of videos to fetch
        
    Returns:
        list: List of video titles and descriptions
    """
    if not API_KEY:
        logger.warning("YouTube API key not found. Set YOUTUBE_API_KEY environment variable.")
        return []
    
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "regionCode": "IN",
            "relevanceLanguage": "en",
            "order": "date",  # Get most recent videos
            "key": API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"YouTube API returned status code {response.status_code}")
            return []
        
        data = response.json()
        
        if "error" in data:
            logger.error(f"YouTube API error: {data['error']}")
            return []
        
        results = []
        items = data.get("items", [])
        
        for item in items:
            snippet = item.get("snippet", {})
            title = snippet.get("title", "")
            description = snippet.get("description", "")
            
            # Combine title and description for better context
            if title:
                results.append(title.strip())
            
            if description and len(description.strip()) > 20:
                # Take first 200 chars of description to avoid too much noise
                desc_preview = description.strip()[:200]
                if len(description) > 200:
                    desc_preview += "..."
                results.append(desc_preview)
        
        logger.info(f"Fetched {len(results)} items from YouTube")
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching YouTube data: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in YouTube scraper: {e}")
        return []


def fetch_youtube_alerts_multiple_queries():
    """
    Fetch YouTube videos using multiple specific queries for better coverage.
    
    Returns:
        list: Combined list of video titles/descriptions from multiple searches
    """
    if not API_KEY:
        return []
    
    queries = [
        "tsunami warning India",
        "flood alert India",
        "cyclone India news",
        "storm surge India",
        "ocean hazard India"
    ]
    
    all_results = []
    
    for query in queries:
        results = fetch_youtube_alerts(query, max_results=3)
        all_results.extend(results)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for result in all_results:
        if result not in seen:
            seen.add(result)
            unique_results.append(result)
    
    return unique_results[:15]  # Return max 15 items


if __name__ == "__main__":
    # Test the scraper
    alerts = fetch_youtube_alerts()
    print(f"Found {len(alerts)} YouTube items:")
    for alert in alerts[:3]:  # Show first 3
        print(f"- {alert}")