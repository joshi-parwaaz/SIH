"""
Multilingual relevance classifier for ocean hazard detection.
Uses HuggingFace transformers for zero-shot classification.
"""

import logging
import traceback

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers not available. Using fallback keyword-based classification.")
    TRANSFORMERS_AVAILABLE = False

# Global classifier instance (loaded once for efficiency)
_classifier = None


def _get_classifier():
    """Get or initialize the HuggingFace classifier."""
    global _classifier
    
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    if _classifier is None:
        try:
            # Use multilingual model for better coverage of Indian content
            _classifier = pipeline(
                "zero-shot-classification", 
                model="joeddav/xlm-roberta-large-xnli"
            )
            logger.info("Initialized multilingual relevance classifier")
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return None
    
    return _classifier


def is_relevant_ml(text: str) -> bool:
    """
    Use ML model to determine if text is relevant to ocean hazards.
    
    Args:
        text (str): Text to classify
        
    Returns:
        bool: True if relevant to ocean hazards
    """
    classifier = _get_classifier()
    
    if not classifier:
        # Fallback to keyword-based classification
        return is_relevant_keywords(text)
    
    try:
        labels = [
            "ocean hazard warning", 
            "natural disaster alert",
            "emergency announcement",
            "irrelevant content",
            "general news"
        ]
        
        result = classifier(text, labels)
        
        # Consider relevant if top prediction is hazard-related
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        
        hazard_labels = {"ocean hazard warning", "natural disaster alert", "emergency announcement"}
        is_hazard = top_label in hazard_labels and top_score > 0.5
        
        logger.debug(f"ML Classification: {top_label} ({top_score:.3f}) -> {'relevant' if is_hazard else 'irrelevant'}")
        return is_hazard
        
    except Exception as e:
        logger.error(f"Error in ML classification: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Fallback to keyword-based
        return is_relevant_keywords(text)


def is_relevant_keywords(text: str) -> bool:
    """
    Fallback keyword-based relevance detection.
    
    Args:
        text (str): Text to classify
        
    Returns:
        bool: True if relevant to ocean hazards
    """
    if not text or len(text.strip()) < 10:
        return False
    
    text_lower = text.lower()
    
    # Ocean hazard keywords
    hazard_keywords = [
        "tsunami", "flood", "flooding", "cyclone", "hurricane", "typhoon",
        "storm surge", "tidal wave", "sea level", "coastal flooding",
        "ocean warning", "marine warning", "coastal alert", "evacuation",
        "high tide", "storm warning", "weather alert", "emergency",
        "disaster", "warning", "alert", "advisory"
    ]
    
    # Location keywords (India-specific)
    location_keywords = [
        "india", "indian", "bengal", "arabian sea", "bay of bengal",
        "kerala", "tamil nadu", "andhra pradesh", "odisha", "gujarat",
        "mumbai", "chennai", "kolkata", "kochi", "visakhapatnam",
        "coastal", "shore", "beach", "port"
    ]
    
    # Check for hazard keywords
    has_hazard = any(keyword in text_lower for keyword in hazard_keywords)
    
    # Boost relevance if location is mentioned
    has_location = any(keyword in text_lower for keyword in location_keywords)
    
    # More likely to be relevant if both hazard and location are mentioned
    if has_hazard and has_location:
        return True
    
    # Still relevant if strong hazard keywords are present
    strong_hazards = ["tsunami", "cyclone", "flood", "storm surge", "evacuation", "emergency"]
    has_strong_hazard = any(keyword in text_lower for keyword in strong_hazards)
    
    return has_strong_hazard


def is_relevant(text: str) -> bool:
    """
    Main function to determine if text is relevant to ocean hazards.
    Uses ML model if available, falls back to keywords.
    
    Args:
        text (str): Text to classify
        
    Returns:
        bool: True if relevant to ocean hazards
    """
    if TRANSFORMERS_AVAILABLE:
        return is_relevant_ml(text)
    else:
        return is_relevant_keywords(text)


if __name__ == "__main__":
    # Test the classifier
    test_texts = [
        "Tsunami warning issued for coastal areas of Tamil Nadu",
        "Heavy rainfall expected in Mumbai today",
        "Cricket match cancelled due to weather",
        "Cyclone Vardah approaching Andhra Pradesh coast",
        "Stock market closes higher today"
    ]
    
    for text in test_texts:
        relevant = is_relevant(text)
        print(f"{'✓' if relevant else '✗'} {text}")