import torch
import asyncio
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# Transformers
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModel,
    pipeline
)

# Text processing
from textblob import TextBlob

from ..preprocessing import ProcessedReport
from config import config

@dataclass
class HazardPrediction:
    """Hazard prediction result"""
    is_hazard: bool
    hazard_type: str
    confidence: float
    severity: str
    urgency: str
    sentiment: str
    sentiment_score: float
    misinformation_probability: float
    entities: List[Dict[str, Any]]
    processing_metadata: Dict[str, Any]

class HazardClassifier:
    """Multilingual hazard classification model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hazard categories
        self.hazard_types = [
            'tsunami', 'flood', 'storm_surge', 'high_waves', 
            'cyclone', 'coastal_erosion', 'sea_level_rise', 'other'
        ]
        
        # Severity levels
        self.severity_levels = ['low', 'moderate', 'high', 'severe']
        
        # Urgency levels
        self.urgency_levels = ['low', 'medium', 'high', 'immediate']
        
        # Load model configuration
        self.model_config = config.get_model_config('CLASSIFICATION_MODEL')
    
    async def load_model(self):
        """Load the hazard classification model"""
        try:
            model_name = self.model_config.get('NAME', 'xlm-roberta-base')
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # For now, we'll use a general classification model
            # In production, you'd fine-tune this on hazard-specific data
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.hazard_types)
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Hazard classifier loaded: {model_name}")
            
        except Exception as e:
            print(f"Error loading hazard classifier: {e}")
            # Fallback to rule-based classification
            self.model = None
            self.tokenizer = None
    
    async def classify_hazard(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """Classify if text contains hazard information"""
        
        if self.model and self.tokenizer:
            return await self._classify_with_model(text, language)
        else:
            return await self._classify_with_rules(text, language)
    
    async def _classify_with_model(self, text: str, language: str) -> Dict[str, Any]:
        """Classify using transformer model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Get top prediction
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            hazard_type = self.hazard_types[predicted_class]
            is_hazard = hazard_type != 'other' and confidence > 0.5
            
            return {
                'is_hazard': is_hazard,
                'hazard_type': hazard_type,
                'confidence': confidence,
                'all_probabilities': {
                    self.hazard_types[i]: probabilities[0][i].item()
                    for i in range(len(self.hazard_types))
                }
            }
            
        except Exception as e:
            print(f"Model classification error: {e}")
            return await self._classify_with_rules(text, language)
    
    async def _classify_with_rules(self, text: str, language: str) -> Dict[str, Any]:
        """Rule-based hazard classification as fallback"""
        text_lower = text.lower()
        
        # Hazard keywords by type
        hazard_keywords = {
            'tsunami': ['tsunami', 'tidal wave', 'seismic wave', 'सुनामी', 'சுனாமி', 'సునామీ'],
            'flood': ['flood', 'flooding', 'inundation', 'waterlogging', 'बाढ़', 'വെള്ളപ്പൊക്കം'],
            'storm_surge': ['storm surge', 'surge', 'storm tide', 'तूफानी लहर'],
            'high_waves': ['high waves', 'large waves', 'big waves', 'rough sea', 'ऊंची लहरें'],
            'cyclone': ['cyclone', 'hurricane', 'typhoon', 'चक्रवात', 'புயல்'],
            'coastal_erosion': ['erosion', 'coastal erosion', 'beach erosion', 'shore erosion']
        }
        
        # Check for hazard keywords
        detected_hazards = []
        max_confidence = 0.0
        
        for hazard_type, keywords in hazard_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Calculate confidence based on keyword match
                    confidence = 0.7 + (len(keyword) / 20)  # Longer keywords get higher confidence
                    detected_hazards.append((hazard_type, confidence))
                    max_confidence = max(max_confidence, confidence)
        
        if detected_hazards:
            # Return the hazard type with highest confidence
            best_hazard = max(detected_hazards, key=lambda x: x[1])
            return {
                'is_hazard': True,
                'hazard_type': best_hazard[0],
                'confidence': min(best_hazard[1], 1.0),
                'detected_hazards': detected_hazards
            }
        else:
            return {
                'is_hazard': False,
                'hazard_type': 'none',
                'confidence': 0.1,
                'detected_hazards': []
            }

class SentimentAnalyzer:
    """Multilingual sentiment analysis for hazard reports"""
    
    def __init__(self):
        self.sentiment_pipeline = None
        
    async def load_model(self):
        """Load sentiment analysis model"""
        try:
            # Use a multilingual sentiment model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
            print("Sentiment analyzer loaded")
            
        except Exception as e:
            print(f"Error loading sentiment analyzer: {e}")
            self.sentiment_pipeline = None
    
    async def analyze_sentiment(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """Analyze sentiment of text"""
        
        if self.sentiment_pipeline:
            return await self._analyze_with_model(text)
        else:
            return await self._analyze_with_textblob(text)
    
    async def _analyze_with_model(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using transformer model"""
        try:
            results = self.sentiment_pipeline(text)
            result = results[0]
            
            # Convert to our format
            sentiment_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral',
                'POSITIVE': 'positive'
            }
            
            sentiment = sentiment_mapping.get(result['label'], 'neutral')
            confidence = result['score']
            
            # Calculate urgency based on sentiment and confidence
            urgency = self._calculate_urgency(sentiment, confidence, text)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'urgency': urgency,
                'raw_result': result
            }
            
        except Exception as e:
            print(f"Model sentiment analysis error: {e}")
            return await self._analyze_with_textblob(text)
    
    async def _analyze_with_textblob(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob as fallback"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Convert polarity to sentiment
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            confidence = min(abs(polarity) + 0.5, 1.0)
            urgency = self._calculate_urgency(sentiment, confidence, text)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'urgency': urgency,
                'polarity': polarity
            }
            
        except Exception as e:
            print(f"TextBlob sentiment analysis error: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'urgency': 'medium'
            }
    
    def _calculate_urgency(self, sentiment: str, confidence: float, text: str) -> str:
        """Calculate urgency based on sentiment and text content"""
        urgency_keywords = {
            'immediate': ['emergency', 'urgent', 'immediate', 'help', 'rescue', 'danger'],
            'high': ['warning', 'alert', 'serious', 'severe', 'critical'],
            'medium': ['concern', 'caution', 'watch', 'monitor'],
            'low': ['minor', 'slight', 'small', 'calm']
        }
        
        text_lower = text.lower()
        
        # Check for urgency keywords
        for level, keywords in urgency_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return level
        
        # Default urgency based on sentiment
        if sentiment == 'negative' and confidence > 0.7:
            return 'high'
        elif sentiment == 'negative':
            return 'medium'
        else:
            return 'low'

class MisinformationDetector:
    """Detect potential misinformation in hazard reports"""
    
    def __init__(self):
        self.suspicious_patterns = [
            # Exaggerated claims
            r'(\d+)\s*(thousand|million|billion)\s*(dead|killed|missing)',
            r'(completely|totally|entirely)\s*(destroyed|devastated|wiped out)',
            
            # Unverified sources
            r'(heard from|someone said|they say|rumors)',
            r'(facebook said|whatsapp message|forwarded)',
            
            # Absolute statements without evidence
            r'(definitely|certainly|absolutely)\s*(will happen|going to happen)',
            r'(government hiding|cover-up|conspiracy)'
        ]
        
        self.credibility_indicators = {
            'positive': [
                'official', 'confirmed', 'verified', 'authorities', 'government',
                'meteorological', 'seismic', 'satellite', 'observed', 'recorded'
            ],
            'negative': [
                'rumor', 'unconfirmed', 'alleged', 'claimed', 'supposedly',
                'viral', 'forwarded', 'whatsapp', 'facebook post'
            ]
        }
    
    async def detect_misinformation(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential misinformation"""
        
        risk_score = 0.0
        flags = []
        
        text_lower = text.lower()
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text_lower):
                risk_score += 0.3
                flags.append(f"Suspicious pattern: {pattern}")
        
        # Check credibility indicators
        positive_indicators = sum(1 for indicator in self.credibility_indicators['positive'] 
                                if indicator in text_lower)
        negative_indicators = sum(1 for indicator in self.credibility_indicators['negative'] 
                                if indicator in text_lower)
        
        # Adjust risk score based on indicators
        risk_score += negative_indicators * 0.2
        risk_score -= positive_indicators * 0.1
        
        # Check source credibility
        source = metadata.get('source', '')
        if source in ['twitter', 'facebook', 'youtube']:
            risk_score += 0.1
        elif source in ['incois', 'imd', 'ndma']:
            risk_score -= 0.2
        
        # Check for verification status
        if metadata.get('verification_status') == 'verified':
            risk_score -= 0.3
        
        # Normalize risk score
        risk_score = max(0.0, min(1.0, risk_score))
        
        # Determine risk level
        if risk_score > 0.7:
            risk_level = 'high'
        elif risk_score > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'misinformation_probability': risk_score,
            'risk_level': risk_level,
            'flags': flags,
            'credibility_score': 1.0 - risk_score
        }

class NLPAnalysisEngine:
    """Main NLP analysis engine combining all components"""
    
    def __init__(self):
        self.hazard_classifier = HazardClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.misinformation_detector = MisinformationDetector()
        
        self.is_loaded = False
        
        self.stats = {
            'total_analyzed': 0,
            'hazards_detected': 0,
            'high_urgency_reports': 0,
            'misinformation_flagged': 0
        }
    
    async def load_models(self):
        """Load all NLP models"""
        print("Loading NLP models...")
        
        await self.hazard_classifier.load_model()
        await self.sentiment_analyzer.load_model()
        
        self.is_loaded = True
        print("All NLP models loaded successfully")
    
    async def analyze_report(self, processed_report: ProcessedReport) -> HazardPrediction:
        """Analyze a processed report and generate predictions"""
        
        if not self.is_loaded:
            await self.load_models()
        
        text = processed_report.normalized_content
        language = processed_report.detected_language
        
        try:
            # Hazard classification
            hazard_result = await self.hazard_classifier.classify_hazard(text, language)
            
            # Sentiment analysis
            sentiment_result = await self.sentiment_analyzer.analyze_sentiment(text, language)
            
            # Misinformation detection
            misinformation_result = await self.misinformation_detector.detect_misinformation(
                text, processed_report.metadata
            )
            
            # Determine severity based on text content and sentiment
            severity = self._determine_severity(text, sentiment_result, hazard_result)
            
            # Create prediction result
            prediction = HazardPrediction(
                is_hazard=hazard_result['is_hazard'],
                hazard_type=hazard_result['hazard_type'],
                confidence=hazard_result['confidence'],
                severity=severity,
                urgency=sentiment_result['urgency'],
                sentiment=sentiment_result['sentiment'],
                sentiment_score=sentiment_result['confidence'],
                misinformation_probability=misinformation_result['misinformation_probability'],
                entities=[],  # Will be filled by NER component
                processing_metadata={
                    'analyzed_at': datetime.now(),
                    'language': language,
                    'text_length': len(text),
                    'analysis_version': '1.0'
                }
            )
            
            # Update statistics
            self.stats['total_analyzed'] += 1
            if prediction.is_hazard:
                self.stats['hazards_detected'] += 1
            if prediction.urgency in ['high', 'immediate']:
                self.stats['high_urgency_reports'] += 1
            if prediction.misinformation_probability > 0.5:
                self.stats['misinformation_flagged'] += 1
            
            return prediction
            
        except Exception as e:
            print(f"Error analyzing report {processed_report.id}: {e}")
            
            # Return default prediction on error
            return HazardPrediction(
                is_hazard=False,
                hazard_type='unknown',
                confidence=0.0,
                severity='unknown',
                urgency='low',
                sentiment='neutral',
                sentiment_score=0.5,
                misinformation_probability=0.5,
                entities=[],
                processing_metadata={
                    'analyzed_at': datetime.now(),
                    'error': str(e),
                    'analysis_version': '1.0'
                }
            )
    
    def _determine_severity(self, text: str, sentiment_result: Dict, hazard_result: Dict) -> str:
        """Determine severity level based on text analysis"""
        
        severity_keywords = {
            'severe': ['catastrophic', 'devastating', 'massive', 'enormous', 'extreme'],
            'high': ['serious', 'major', 'significant', 'large', 'dangerous'],
            'moderate': ['moderate', 'considerable', 'noticeable', 'medium'],
            'low': ['minor', 'small', 'slight', 'little', 'low']
        }
        
        text_lower = text.lower()
        
        # Check for severity keywords
        for level, keywords in severity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return level
        
        # Determine based on sentiment and hazard confidence
        if sentiment_result['sentiment'] == 'negative' and hazard_result['confidence'] > 0.8:
            return 'high'
        elif sentiment_result['sentiment'] == 'negative' and hazard_result['confidence'] > 0.6:
            return 'moderate'
        elif hazard_result['confidence'] > 0.7:
            return 'moderate'
        else:
            return 'low'
    
    async def analyze_batch(self, processed_reports: List[ProcessedReport]) -> List[HazardPrediction]:
        """Analyze a batch of processed reports"""
        print(f"Analyzing batch of {len(processed_reports)} reports...")
        
        tasks = [self.analyze_report(report) for report in processed_reports]
        predictions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_predictions = []
        for result in predictions:
            if isinstance(result, HazardPrediction):
                valid_predictions.append(result)
            else:
                print(f"Analysis error: {result}")
        
        print(f"Successfully analyzed {len(valid_predictions)} reports")
        return valid_predictions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset analysis statistics"""
        self.stats = {
            'total_analyzed': 0,
            'hazards_detected': 0,
            'high_urgency_reports': 0,
            'misinformation_flagged': 0
        }