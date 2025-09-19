"""
Enhanced Reverse Image System for Misinformation Detection
Advanced image verification with confidence scoring and source credibility analysis.
"""

import logging
import hashlib
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class EnhancedReverseImageChecker:
    """
    Advanced reverse image checking and misinformation detection system.
    Uses multiple verification methods and confidence scoring.
    """
    
    def __init__(self, cache_file: str = "image_verification_cache.json"):
        self.cache_file = cache_file
        self.verification_cache = self._load_cache()
        
        # Source credibility scores (0.0 to 1.0)
        self.source_credibility = {
            "government": 0.95,
            "official": 0.9,
            "news_agency": 0.85,
            "verified_social": 0.7,
            "social_media": 0.4,
            "unknown": 0.3,
            "suspicious": 0.1
        }
        
        # Known misinformation patterns
        self.misinformation_patterns = [
            r"fake.*alert",
            r"old.*video",
            r"misleading.*image",
            r"unverified.*claim",
            r"rumor.*spread",
            r"false.*information",
            r"clickbait",
            r"sensational.*headline"
        ]
        
        # Trusted domain patterns
        self.trusted_domains = [
            "incois.gov.in", "imd.gov.in", "ndma.gov.in",
            "pib.gov.in", "mha.gov.in", "timesofindia.com",
            "hindustantimes.com", "thehindu.com", "indianexpress.com",
            "ndtv.com", "news18.com", "zee5.com", "india.gov.in"
        ]
        
        # Image hash database for known misinformation
        self.known_fake_hashes = set()
        
    def _load_cache(self) -> Dict:
        """Load verification cache from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load verification cache: {e}")
        
        return {
            "verified_images": {},
            "misinformation_database": {},
            "source_reliability_scores": {},
            "last_updated": None
        }
    
    def _save_cache(self):
        """Save verification cache to file."""
        try:
            self.verification_cache["last_updated"] = datetime.utcnow().isoformat()
            with open(self.cache_file, 'w') as f:
                json.dump(self.verification_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save verification cache: {e}")
    
    def _extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text content."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return urls
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content verification."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _analyze_source_credibility(self, url: str) -> Tuple[float, str]:
        """
        Analyze the credibility of a source URL or source name.
        
        Args:
            url: URL to analyze or source name
            
        Returns:
            Tuple of (credibility_score, source_type)
        """
        try:
            # Handle source names directly (for pipeline integration)
            source_lower = url.lower()
            
            # Check for known trusted sources
            if any(source in source_lower for source in ["incois", "imd", "ndma", "government"]):
                return 0.95, "government"
            
            if any(source in source_lower for source in ["times", "hindu", "express", "ndtv", "news"]):
                return 0.85, "trusted_news"
            
            if "youtube" in source_lower:
                return 0.7, "video_platform"
            
            if "twitter" in source_lower:
                return 0.6, "social_media"
            
            # If it's a URL, parse the domain
            if url.startswith('http'):
                domain = urlparse(url).netloc.lower()
                
                # Check against trusted domains
                for trusted_domain in self.trusted_domains:
                    if trusted_domain in domain:
                        return 0.9, "trusted_news"
                
                # Government domains
                if any(gov_pattern in domain for gov_pattern in [".gov.in", ".nic.in", ".india.gov"]):
                    return 0.95, "government"
                
                # Educational institutions
                if any(edu_pattern in domain for edu_pattern in [".edu", ".ac.in", ".iit", ".iim"]):
                    return 0.8, "educational"
                
                # Social media platforms
                if any(social in domain for social in ["twitter.com", "facebook.com", "instagram.com", "youtube.com"]):
                    return 0.6, "social_media"
                
                # News domains (heuristic)
                if any(news_indicator in domain for news_indicator in ["news", "times", "express", "hindu", "tv"]):
                    return 0.75, "news_media"
            
            # Unknown sources
            return 0.5, "unknown"
            
        except Exception as e:
            logger.warning(f"Error analyzing source credibility: {e}")
            return 0.4, "error"
    
    def _detect_misinformation_patterns(self, text: str) -> List[Dict]:
        """
        Detect potential misinformation patterns in text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of detected misinformation indicators
        """
        text_lower = text.lower()
        detected_patterns = []
        
        for pattern in self.misinformation_patterns:
            if re.search(pattern, text_lower):
                detected_patterns.append({
                    "pattern": pattern,
                    "confidence": 0.7,
                    "description": f"Detected potential misinformation pattern: {pattern}"
                })
        
        # Additional heuristic checks
        
        # Check for sensational language
        sensational_words = ["shocking", "unbelievable", "exclusive", "breaking", "viral", "must watch"]
        sensational_count = sum(1 for word in sensational_words if word in text_lower)
        if sensational_count >= 2:
            detected_patterns.append({
                "pattern": "sensational_language",
                "confidence": 0.6,
                "description": f"High use of sensational language ({sensational_count} indicators)"
            })
        
        # Check for urgency manipulation
        urgency_words = ["urgent", "immediately", "act now", "warning", "danger"]
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        if urgency_count >= 2:
            detected_patterns.append({
                "pattern": "urgency_manipulation",
                "confidence": 0.5,
                "description": f"Potential urgency manipulation ({urgency_count} indicators)"
            })
        
        # Check for lack of specific details
        if len(text) > 100 and not any(indicator in text_lower for indicator in ["time", "date", "location", "official", "confirm"]):
            detected_patterns.append({
                "pattern": "vague_content",
                "confidence": 0.4,
                "description": "Content lacks specific verifiable details"
            })
        
        return detected_patterns
    
    def _simulate_reverse_image_search(self, image_url: str) -> Dict:
        """
        Simulate reverse image search results.
        In production, this would integrate with actual reverse search APIs.
        """
        # Simulate search results based on URL patterns
        domain = urlparse(image_url).netloc.lower() if image_url.startswith('http') else "unknown"
        
        # Simulate realistic reverse search results
        if any(trusted in domain for trusted in self.trusted_domains):
            return {
                "first_appearance": (datetime.now() - timedelta(hours=2)).isoformat(),
                "occurrence_count": 1,
                "trusted_sources": 1,
                "suspicious_sources": 0,
                "oldest_occurrence": (datetime.now() - timedelta(hours=2)).isoformat(),
                "consistency_score": 0.9
            }
        else:
            # Simulate older, possibly misused image
            return {
                "first_appearance": (datetime.now() - timedelta(days=30)).isoformat(),
                "occurrence_count": 5,
                "trusted_sources": 0,
                "suspicious_sources": 2,
                "oldest_occurrence": (datetime.now() - timedelta(days=365)).isoformat(),
                "consistency_score": 0.3
            }
    
    def verify_content_authenticity(self, text: str, source: str = "unknown") -> Dict[str, Any]:
        """
        Comprehensive content authenticity verification.
        
        Args:
            text: Text content to verify
            source: Source of the content
            
        Returns:
            Dict containing verification results
        """
        # Generate content hash for caching
        content_hash = self._generate_content_hash(text)
        
        # Check cache first
        if content_hash in self.verification_cache.get("verified_images", {}):
            cached_result = self.verification_cache["verified_images"][content_hash]
            logger.info(f"Using cached verification result for content")
            return cached_result
        
        # Extract URLs from text
        urls = self._extract_urls_from_text(text)
        
        # Analyze source credibility
        source_credibility, source_type = self._analyze_source_credibility(source)
        
        # If no URLs found, use source name for analysis
        if not urls:
            source_credibility, source_type = self._analyze_source_credibility(source)
        
        # Detect misinformation patterns
        misinformation_indicators = self._detect_misinformation_patterns(text)
        
        # Simulate reverse image search for URLs
        image_analysis = {}
        if urls:
            image_analysis = self._simulate_reverse_image_search(urls[0])
        
        # Calculate overall authenticity score
        authenticity_score = self._calculate_authenticity_score(
            source_credibility, misinformation_indicators, image_analysis
        )
        
        # Determine if content is likely misinformation
        is_likely_misinformation = authenticity_score < 0.4
        
        # Generate verification result
        verification_result = {
            "content_hash": content_hash,
            "text_length": len(text),
            "source": source,
            "source_credibility": source_credibility,
            "source_type": source_type,
            "authenticity_score": authenticity_score,
            "is_likely_misinformation": is_likely_misinformation,
            "confidence": min(0.95, max(0.3, authenticity_score + 0.1)),
            "misinformation_indicators": misinformation_indicators,
            "image_analysis": image_analysis,
            "urls_found": urls,
            "verification_timestamp": datetime.utcnow().isoformat(),
            "risk_level": self._determine_risk_level(authenticity_score),
            "recommendations": self._generate_recommendations(authenticity_score, misinformation_indicators)
        }
        
        # Cache the result
        self.verification_cache["verified_images"][content_hash] = verification_result
        self._save_cache()
        
        return verification_result
    
    def _calculate_authenticity_score(self, source_credibility: float, 
                                    misinformation_indicators: List[Dict], 
                                    image_analysis: Dict) -> float:
        """Calculate overall authenticity score."""
        # Start with source credibility as base score
        score = source_credibility
        
        # Penalize for misinformation indicators
        for indicator in misinformation_indicators:
            penalty = indicator.get("confidence", 0.5) * 0.2
            score -= penalty
        
        # Factor in image analysis if available
        if image_analysis:
            consistency_score = image_analysis.get("consistency_score", 0.5)
            trusted_sources = image_analysis.get("trusted_sources", 0)
            suspicious_sources = image_analysis.get("suspicious_sources", 0)
            
            # Boost for trusted sources
            if trusted_sources > 0:
                score += 0.1 * trusted_sources
            
            # Penalize for suspicious sources
            if suspicious_sources > 0:
                score -= 0.15 * suspicious_sources
            
            # Factor in consistency
            score = score * 0.7 + consistency_score * 0.3
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))
    
    def _determine_risk_level(self, authenticity_score: float) -> str:
        """Determine risk level based on authenticity score."""
        if authenticity_score >= 0.8:
            return "low"
        elif authenticity_score >= 0.5:
            return "medium"
        elif authenticity_score >= 0.3:
            return "high"
        else:
            return "critical"
    
    def _generate_recommendations(self, authenticity_score: float, 
                                misinformation_indicators: List[Dict]) -> List[str]:
        """Generate recommendations based on verification results."""
        recommendations = []
        
        if authenticity_score < 0.4:
            recommendations.append("⚠️ High risk of misinformation - verify with official sources")
        
        if authenticity_score < 0.6:
            recommendations.append("🔍 Cross-reference with multiple trusted sources")
        
        if len(misinformation_indicators) > 0:
            recommendations.append("📋 Content contains potential misinformation patterns")
        
        if authenticity_score >= 0.8:
            recommendations.append("✅ Content appears authentic from reliable source")
        
        recommendations.append("🏛️ Always verify disaster information with official authorities")
        
        return recommendations
    
    def batch_verify_content(self, content_list: List[Tuple[str, str]]) -> List[Dict]:
        """
        Batch verification of multiple content items.
        
        Args:
            content_list: List of (text, source) tuples
            
        Returns:
            List of verification results
        """
        results = []
        
        for text, source in content_list:
            try:
                result = self.verify_content_authenticity(text, source)
                results.append(result)
            except Exception as e:
                logger.error(f"Error verifying content from {source}: {e}")
                # Add error result
                results.append({
                    "source": source,
                    "authenticity_score": 0.3,
                    "is_likely_misinformation": True,
                    "confidence": 0.1,
                    "error": str(e),
                    "risk_level": "unknown"
                })
        
        return results
    
    def get_verification_statistics(self) -> Dict[str, Any]:
        """Get statistics about verification cache and performance."""
        cache_data = self.verification_cache.get("verified_images", {})
        
        if not cache_data:
            return {"total_verifications": 0}
        
        # Calculate statistics
        authenticity_scores = [item.get("authenticity_score", 0) for item in cache_data.values()]
        misinformation_count = sum(1 for item in cache_data.values() if item.get("is_likely_misinformation", False))
        
        risk_levels = [item.get("risk_level", "unknown") for item in cache_data.values()]
        risk_distribution = {level: risk_levels.count(level) for level in ["low", "medium", "high", "critical"]}
        
        return {
            "total_verifications": len(cache_data),
            "misinformation_detected": misinformation_count,
            "average_authenticity_score": sum(authenticity_scores) / len(authenticity_scores) if authenticity_scores else 0,
            "risk_distribution": risk_distribution,
            "cache_size_mb": os.path.getsize(self.cache_file) / (1024*1024) if os.path.exists(self.cache_file) else 0
        }


def enhanced_misinformation_check(text: str, source: str = "unknown") -> Tuple[bool, float, Dict]:
    """
    Enhanced misinformation check function for pipeline integration.
    
    Args:
        text: Text content to check
        source: Source of the content
        
    Returns:
        Tuple of (is_misinformation, confidence, verification_details)
    """
    checker = EnhancedReverseImageChecker()
    result = checker.verify_content_authenticity(text, source)
    
    return (
        result["is_likely_misinformation"],
        result["confidence"],
        result
    )


if __name__ == "__main__":
    # Test the enhanced reverse image system
    print("🔍 Testing Enhanced Reverse Image System")
    print("=" * 60)
    
    # Initialize checker
    checker = EnhancedReverseImageChecker()
    
    # Test cases
    test_contents = [
        # Likely authentic content
        ("[INCOIS] Tsunami watch cancelled for Kerala coast. Sea levels normal.", "INCOIS"),
        ("[Times of India] Official evacuation orders for Gujarat coastal areas", "Google_News"),
        ("[IMD] Weather warning: Heavy rainfall expected in next 24 hours", "Government_Sources"),
        
        # Potentially suspicious content
        ("SHOCKING! Fake tsunami alert spreads panic - viral video is old footage", "social_media"),
        ("URGENT! Unverified claims about earthquake - MUST WATCH NOW!", "unknown"),
        ("Exclusive breaking news: Secret government cover-up of disaster", "suspicious_blog")
    ]
    
    print("🧪 Testing content verification:")
    
    for i, (text, source) in enumerate(test_contents, 1):
        print(f"\nTest {i}: {source}")
        print(f"Content: {text[:60]}...")
        
        result = checker.verify_content_authenticity(text, source)
        
        risk_emoji = {"low": "✅", "medium": "⚠️", "high": "🚨", "critical": "💀"}.get(result["risk_level"], "❓")
        
        print(f"Result: {risk_emoji} {result['risk_level'].upper()} risk")
        print(f"Authenticity: {result['authenticity_score']:.3f}")
        print(f"Misinformation: {'YES' if result['is_likely_misinformation'] else 'NO'}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        if result['misinformation_indicators']:
            print(f"Indicators: {len(result['misinformation_indicators'])} patterns detected")
    
    # Test batch verification
    print(f"\n📦 Testing batch verification:")
    batch_results = checker.batch_verify_content(test_contents)
    
    misinformation_count = sum(1 for r in batch_results if r.get("is_likely_misinformation", False))
    avg_authenticity = sum(r.get("authenticity_score", 0) for r in batch_results) / len(batch_results)
    
    print(f"  - Batch size: {len(test_contents)}")
    print(f"  - Misinformation detected: {misinformation_count}")
    print(f"  - Average authenticity: {avg_authenticity:.3f}")
    
    # Get statistics
    stats = checker.get_verification_statistics()
    print(f"\n📊 Verification Statistics:")
    print(f"  - Total verifications: {stats['total_verifications']}")
    print(f"  - Cache size: {stats.get('cache_size_mb', 0):.2f} MB")
    
    print(f"\n" + "=" * 60)
    print("✅ Enhanced Reverse Image System Test Complete!")