"""
INCOIS XML feed scraper for ocean hazard alerts.
Fetches alerts directly from INCOIS XML feeds (public, no API key required).
Always India-specific, no filtering overhead.
"""

import requests
from xml.etree import ElementTree
import logging

logger = logging.getLogger(__name__)


def fetch_incois_alerts():
    """
    Fetch alerts from INCOIS XML feed.
    
    Returns:
        list: List of alert titles/descriptions
    """
    try:
        url = "https://incois.gov.in/portal/tsunami.xml"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"INCOIS API returned status code {response.status_code}")
            return []
        
        root = ElementTree.fromstring(response.content)
        alerts = []
        
        # Extract titles and descriptions from XML
        for item in root.findall(".//item"):
            title = item.find("title")
            description = item.find("description")
            
            if title is not None and title.text:
                alerts.append(title.text.strip())
            
            if description is not None and description.text:
                alerts.append(description.text.strip())
        
        logger.info(f"Fetched {len(alerts)} alerts from INCOIS")
        return alerts
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching INCOIS data: {e}")
        return []
    except ElementTree.ParseError as e:
        logger.error(f"Error parsing INCOIS XML: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in INCOIS scraper: {e}")
        return []


if __name__ == "__main__":
    # Test the scraper
    alerts = fetch_incois_alerts()
    print(f"Found {len(alerts)} alerts:")
    for alert in alerts[:5]:  # Show first 5
        print(f"- {alert}")