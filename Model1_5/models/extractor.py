"""
Information extraction from hazard-related text.
Extracts hazard type, severity, urgency, sentiment, and misinformation indicators.
"""

import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def extract_hazard_type(text: str) -> str:
    """
    Extract the type of hazard from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Detected hazard type
    """
    text_lower = text.lower()
    
    # Define hazard patterns with priority (most specific first)
    hazard_patterns = {
        "tsunami": [
            r"tsunami", r"tidal wave", r"seismic sea wave"
        ],
        "cyclone": [
            r"cyclone", r"hurricane", r"typhoon", r"tropical storm"
        ],
        "flood": [
            r"flood", r"flooding", r"inundation", r"waterlogging",
            r"coastal flooding", r"flash flood"
        ],
        "storm_surge": [
            r"storm surge", r"surge", r"high tide", r"tidal surge"
        ],
        "heavy_rain": [
            r"heavy rain", r"rainfall", r"monsoon", r"downpour"
        ],
        "rough_sea": [
            r"rough sea", r"high waves", r"choppy waters", r"marine warning"
        ]
    }
    
    for hazard_type, patterns in hazard_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return hazard_type
    
    return "unknown"


def extract_severity(text: str) -> str:
    """
    Extract severity level from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Severity level (low, medium, high, extreme)
    """
    text_lower = text.lower()
    
    # Extreme severity indicators
    extreme_indicators = [
        r"extreme", r"catastrophic", r"devastating", r"massive",
        r"unprecedented", r"severe", r"deadly", r"life-threatening"
    ]
    
    # High severity indicators
    high_indicators = [
        r"high", r"dangerous", r"serious", r"major", r"significant",
        r"heavy", r"intense", r"strong", r"powerful"
    ]
    
    # Low severity indicators
    low_indicators = [
        r"low", r"minor", r"light", r"mild", r"weak", r"minimal"
    ]
    
    # Check for extreme first
    for pattern in extreme_indicators:
        if re.search(pattern, text_lower):
            return "extreme"
    
    # Check for high
    for pattern in high_indicators:
        if re.search(pattern, text_lower):
            return "high"
    
    # Check for low
    for pattern in low_indicators:
        if re.search(pattern, text_lower):
            return "low"
    
    # Default to medium if no specific indicators
    return "medium"


def extract_urgency(text: str) -> str:
    """
    Extract urgency level from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Urgency level (low, medium, high, immediate)
    """
    text_lower = text.lower()
    
    # Immediate urgency indicators
    immediate_indicators = [
        r"immediate", r"urgent", r"emergency", r"now", r"asap",
        r"evacuate", r"evacuation", r"alert", r"warning"
    ]
    
    # High urgency indicators
    high_indicators = [
        r"soon", r"quickly", r"rapidly", r"fast", r"hurry",
        r"watch", r"advisory", r"prepare"
    ]
    
    # Low urgency indicators
    low_indicators = [
        r"later", r"eventually", r"monitor", r"observe"
    ]
    
    # Check for immediate
    for pattern in immediate_indicators:
        if re.search(pattern, text_lower):
            return "immediate"
    
    # Check for high
    for pattern in high_indicators:
        if re.search(pattern, text_lower):
            return "high"
    
    # Check for low
    for pattern in low_indicators:
        if re.search(pattern, text_lower):
            return "low"
    
    return "medium"


def extract_sentiment(text: str) -> str:
    """
    Extract emotional sentiment from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Sentiment (panic, concern, calm, neutral)
    """
    text_lower = text.lower()
    
    # Panic indicators
    panic_indicators = [
        r"panic", r"terror", r"fear", r"scared", r"frightened",
        r"help", r"save", r"rescue", r"disaster", r"catastrophe"
    ]
    
    # Concern indicators
    concern_indicators = [
        r"worry", r"concerned", r"anxious", r"alert", r"caution",
        r"careful", r"prepare", r"ready"
    ]
    
    # Calm indicators
    calm_indicators = [
        r"calm", r"safe", r"secure", r"controlled", r"managed",
        r"stable", r"normal", r"routine"
    ]
    
    # Check for panic
    for pattern in panic_indicators:
        if re.search(pattern, text_lower):
            return "panic"
    
    # Check for concern
    for pattern in concern_indicators:
        if re.search(pattern, text_lower):
            return "concern"
    
    # Check for calm
    for pattern in calm_indicators:
        if re.search(pattern, text_lower):
            return "calm"
    
    return "neutral"


def detect_misinformation(text: str) -> bool:
    """
    Simple misinformation detection based on patterns.
    
    Args:
        text (str): Input text
        
    Returns:
        bool: True if potential misinformation detected
    """
    text_lower = text.lower()
    
    # Misinformation indicators
    misinfo_patterns = [
        r"fake", r"hoax", r"rumor", r"rumour", r"false",
        r"unverified", r"unconfirmed", r"alleged",
        r"they say", r"i heard", r"someone told me",
        r"whatsapp forward", r"viral message"
    ]
    
    # Exaggeration patterns
    exaggeration_patterns = [
        r"shocking", r"unbelievable", r"incredible", r"amazing",
        r"you won't believe", r"must see", r"viral"
    ]
    
    for pattern in misinfo_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Count exaggeration patterns
    exaggeration_count = sum(
        1 for pattern in exaggeration_patterns 
        if re.search(pattern, text_lower)
    )
    
    # If multiple exaggeration patterns, likely misinformation
    return exaggeration_count >= 2


def extract_location(text: str) -> str:
    """
    Extract location information from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Detected location or "unknown"
    """
    text_lower = text.lower()
    
    # Indian coastal locations
    locations = {
        "kerala": [r"kerala", r"kochi", r"trivandrum", r"calicut"],
        "tamil_nadu": [r"tamil nadu", r"chennai", r"madras", r"kanyakumari"],
        "andhra_pradesh": [r"andhra pradesh", r"visakhapatnam", r"vijayawada"],
        "odisha": [r"odisha", r"orissa", r"bhubaneswar", r"puri"],
        "west_bengal": [r"west bengal", r"kolkata", r"calcutta"],
        "gujarat": [r"gujarat", r"ahmedabad", r"surat", r"rajkot"],
        "maharashtra": [r"maharashtra", r"mumbai", r"bombay", r"pune"],
        "karnataka": [r"karnataka", r"bangalore", r"mangalore"],
        "goa": [r"goa", r"panaji"]
    }
    
    for state, patterns in locations.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return state
    
    # Check for general coastal terms
    coastal_terms = [r"coast", r"coastal", r"shore", r"beach", r"port"]
    for term in coastal_terms:
        if re.search(term, text_lower):
            return "coastal_area"
    
    return "unknown"


def extract_info(text: str) -> Dict[str, Any]:
    """
    Main function to extract all information from text.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Extracted information
    """
    try:
        if not text or len(text.strip()) < 5:
            return {
                "hazard_type": "unknown",
                "severity": "low",
                "urgency": "low",
                "sentiment": "neutral",
                "misinformation": False,
                "location": "unknown"
            }
        
        info = {
            "hazard_type": extract_hazard_type(text),
            "severity": extract_severity(text),
            "urgency": extract_urgency(text),
            "sentiment": extract_sentiment(text),
            "misinformation": detect_misinformation(text),
            "location": extract_location(text)
        }
        
        logger.debug(f"Extracted info: {info}")
        return info
        
    except Exception as e:
        logger.error(f"Error extracting info: {e}")
        return {
            "hazard_type": "unknown",
            "severity": "medium",
            "urgency": "medium",
            "sentiment": "neutral",
            "misinformation": False,
            "location": "unknown"
        }


if __name__ == "__main__":
    # Test the extractor
    test_texts = [
        "Tsunami warning issued for coastal areas of Tamil Nadu - evacuate immediately!",
        "Heavy rainfall expected in Mumbai today, minor flooding possible",
        "Cyclone Vardah approaching Andhra Pradesh coast with high intensity winds",
        "SHOCKING: Fake tsunami alert creates panic in Kerala - don't believe WhatsApp forwards!"
    ]
    
    for text in test_texts:
        info = extract_info(text)
        print(f"\nText: {text}")
        print(f"Info: {info}")