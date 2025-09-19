"""
Reverse Image Search and Misinformation Detection Module.
Uses Google Reverse Image Search API and TinEye to verify image authenticity.
"""

import requests
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import base64
import io

logger = logging.getLogger(__name__)

class ReverseImageChecker:
    """Handles reverse image search and misinformation detection."""
    
    def __init__(self):
        self.google_api_key = None  # Would need Google Custom Search API key
        self.tineye_api_key = None  # Would need TinEye API key
        self.image_cache = {}  # Cache results to avoid repeat searches
        
    def analyze_image_authenticity(self, image_url: str) -> Dict:
        """
        Analyze image for potential misinformation.
        
        Args:
            image_url (str): URL of image to analyze
            
        Returns:
            Dict: Analysis results with authenticity score
        """
        try:
            # Check cache first
            image_hash = self._get_image_hash(image_url)
            if image_hash in self.image_cache:
                logger.info(f"Using cached result for image: {image_url[:50]}...")
                return self.image_cache[image_hash]
            
            result = {
                "image_url": image_url,
                "authenticity_score": 0.5,  # Default neutral
                "is_likely_misinformation": False,
                "reverse_search_results": [],
                "first_appearance": None,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "confidence": 0.5
            }
            
            # Perform reverse image search
            search_results = self._reverse_image_search(image_url)
            result["reverse_search_results"] = search_results
            
            # Analyze results for misinformation indicators
            analysis = self._analyze_search_results(search_results)
            result.update(analysis)
            
            # Cache the result
            self.image_cache[image_hash] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image authenticity: {e}")
            return {
                "image_url": image_url,
                "authenticity_score": 0.5,
                "is_likely_misinformation": False,
                "error": str(e),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    def _get_image_hash(self, image_url: str) -> str:
        """Generate hash for image caching."""
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                return hashlib.md5(response.content).hexdigest()
        except:
            pass
        return hashlib.md5(image_url.encode()).hexdigest()
    
    def _reverse_image_search(self, image_url: str) -> List[Dict]:
        """
        Perform reverse image search using available APIs.
        
        Args:
            image_url (str): Image URL to search
            
        Returns:
            List[Dict]: Search results from various engines
        """
        results = []
        
        # Google Reverse Image Search (requires API key)
        if self.google_api_key:
            google_results = self._google_reverse_search(image_url)
            results.extend(google_results)
        
        # TinEye Search (requires API key)
        if self.tineye_api_key:
            tineye_results = self._tineye_search(image_url)
            results.extend(tineye_results)
        
        # Fallback: Basic URL pattern analysis
        if not results:
            results = self._basic_image_analysis(image_url)
        
        return results
    
    def _google_reverse_search(self, image_url: str) -> List[Dict]:
        """Google Custom Search API reverse image search."""
        # This would require Google Custom Search API key
        # For now, return mock results showing the concept
        logger.info("Google reverse image search would be performed here")
        return [
            {
                "source": "google",
                "title": "Similar image found",
                "url": "https://example.com/news-article",
                "date": "2024-01-15",
                "context": "Used in news article about different event"
            }
        ]
    
    def _tineye_search(self, image_url: str) -> List[Dict]:
        """TinEye API reverse image search."""
        # This would require TinEye API key
        logger.info("TinEye reverse image search would be performed here")
        return [
            {
                "source": "tineye",
                "first_seen": "2023-06-10",
                "total_matches": 15,
                "oldest_match_url": "https://example.com/original-source"
            }
        ]
    
    def _basic_image_analysis(self, image_url: str) -> List[Dict]:
        """
        Basic analysis without external APIs.
        Analyzes URL patterns and metadata.
        """
        analysis = []
        
        # Check for suspicious URL patterns
        suspicious_domains = [
            "temp-image", "fake-news", "viral-content", 
            "clickbait", "shortened-url"
        ]
        
        is_suspicious = any(domain in image_url.lower() for domain in suspicious_domains)
        
        # Check if image is very recent (could indicate rapid viral spread)
        try:
            response = requests.head(image_url, timeout=5)
            last_modified = response.headers.get('Last-Modified')
            if last_modified:
                # Parse date and check if very recent
                analysis.append({
                    "source": "metadata",
                    "finding": f"Image last modified: {last_modified}",
                    "suspicious": False
                })
        except:
            pass
        
        if is_suspicious:
            analysis.append({
                "source": "url_analysis",
                "finding": "URL contains suspicious patterns",
                "suspicious": True
            })
        
        return analysis
    
    def _analyze_search_results(self, search_results: List[Dict]) -> Dict:
        """
        Analyze reverse search results for misinformation indicators.
        
        Args:
            search_results: Results from reverse image searches
            
        Returns:
            Dict: Analysis summary with misinformation indicators
        """
        analysis = {
            "authenticity_score": 0.7,  # Start with neutral-positive
            "is_likely_misinformation": False,
            "confidence": 0.6,
            "red_flags": [],
            "positive_indicators": []
        }
        
        if not search_results:
            analysis["red_flags"].append("No reverse search results found")
            analysis["authenticity_score"] = 0.3
            analysis["confidence"] = 0.4
            return analysis
        
        # Look for red flags
        for result in search_results:
            if result.get("suspicious"):
                analysis["red_flags"].append(result.get("finding", "Suspicious pattern detected"))
                analysis["authenticity_score"] -= 0.2
            
            # Check for context mismatches
            if "different event" in result.get("context", "").lower():
                analysis["red_flags"].append("Image used in different context previously")
                analysis["authenticity_score"] -= 0.3
        
        # Positive indicators
        if len(search_results) > 0:
            analysis["positive_indicators"].append("Image found in reverse search")
            
        # Final determination
        if analysis["authenticity_score"] < 0.4:
            analysis["is_likely_misinformation"] = True
            analysis["confidence"] = 0.8
        
        # Clamp score between 0 and 1
        analysis["authenticity_score"] = max(0.0, min(1.0, analysis["authenticity_score"]))
        
        return analysis


def check_text_with_images(text: str) -> Dict:
    """
    Enhanced misinformation check that includes image analysis.
    
    Args:
        text (str): Text content that might contain image URLs
        
    Returns:
        Dict: Combined text and image misinformation analysis
    """
    checker = ReverseImageChecker()
    
    # Extract image URLs from text (simple regex)
    import re
    image_urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.(?:jpg|jpeg|png|gif|bmp)', text)
    
    result = {
        "text_analysis": {
            "has_images": len(image_urls) > 0,
            "image_count": len(image_urls),
            "image_urls": image_urls
        },
        "image_analyses": [],
        "overall_misinformation_score": 0.0,
        "is_likely_misinformation": False
    }
    
    # Analyze each image
    total_authenticity = 0.0
    for url in image_urls:
        image_analysis = checker.analyze_image_authenticity(url)
        result["image_analyses"].append(image_analysis)
        total_authenticity += image_analysis.get("authenticity_score", 0.5)
    
    # Calculate overall score
    if image_urls:
        avg_authenticity = total_authenticity / len(image_urls)
        result["overall_misinformation_score"] = 1.0 - avg_authenticity
        result["is_likely_misinformation"] = avg_authenticity < 0.4
    
    return result


if __name__ == "__main__":
    # Test the reverse image checker
    checker = ReverseImageChecker()
    
    test_image = "https://example.com/disaster-image.jpg"
    result = checker.analyze_image_authenticity(test_image)
    
    print("Image Authenticity Analysis:")
    print(f"Authenticity Score: {result['authenticity_score']}")
    print(f"Likely Misinformation: {result['is_likely_misinformation']}")
    print(f"Confidence: {result['confidence']}")