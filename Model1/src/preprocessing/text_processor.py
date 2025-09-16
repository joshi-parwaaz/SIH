import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import unicodedata

# Language detection
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # For consistent results

# Text processing
import emoji
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

from ..data_ingestion import RawReport
from ...config import config

@dataclass
class ProcessedReport:
    """Processed report data structure"""
    id: str
    source: str
    original_content: str
    cleaned_content: str
    normalized_content: str
    detected_language: str
    confidence_score: float
    timestamp: datetime
    metadata: Dict[str, Any]
    location: Optional[Dict[str, Any]] = None
    media_urls: Optional[List[str]] = None
    processing_info: Optional[Dict[str, Any]] = None

class LanguageDetector:
    """Language detection for multilingual text"""
    
    def __init__(self):
        self.supported_languages = config.get_supported_languages()
        
        # Language code mapping
        self.language_map = {
            'en': 'english',
            'hi': 'hindi',
            'ta': 'tamil',
            'te': 'telugu',
            'ml': 'malayalam',
            'kn': 'kannada',
            'gu': 'gujarati',
            'mr': 'marathi',
            'bn': 'bengali',
            'or': 'oriya'
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of text
        Returns: (language_code, confidence_score)
        """
        if not text or len(text.strip()) < 3:
            return 'en', 0.5
        
        try:
            # Use langdetect library
            detected_lang = detect(text)
            
            # Map to supported language or default to English
            if detected_lang in self.supported_languages:
                return detected_lang, 0.8
            else:
                # Try to detect Indian languages by script
                script_lang = self._detect_by_script(text)
                if script_lang:
                    return script_lang, 0.7
                
                return 'en', 0.6
                
        except Exception as e:
            print(f"Language detection error: {e}")
            # Fallback: detect by script/characters
            script_lang = self._detect_by_script(text)
            return script_lang if script_lang else 'en', 0.5
    
    def _detect_by_script(self, text: str) -> Optional[str]:
        """Detect language by Unicode script/characters"""
        
        # Devanagari script (Hindi, Marathi, etc.)
        if any('\u0900' <= char <= '\u097F' for char in text):
            return 'hi'
        
        # Tamil script
        if any('\u0B80' <= char <= '\u0BFF' for char in text):
            return 'ta'
        
        # Telugu script
        if any('\u0C00' <= char <= '\u0C7F' for char in text):
            return 'te'
        
        # Malayalam script
        if any('\u0D00' <= char <= '\u0D7F' for char in text):
            return 'ml'
        
        # Kannada script
        if any('\u0C80' <= char <= '\u0CFF' for char in text):
            return 'kn'
        
        # Gujarati script
        if any('\u0A80' <= char <= '\u0AFF' for char in text):
            return 'gu'
        
        # Bengali script
        if any('\u0980' <= char <= '\u09FF' for char in text):
            return 'bn'
        
        # Oriya script
        if any('\u0B00' <= char <= '\u0B7F' for char in text):
            return 'or'
        
        return None

class TextNormalizer:
    """Text normalization and cleaning"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        self.email_pattern = re.compile(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b')
        self.phone_pattern = re.compile(r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b')
        
        # Indian phone number patterns
        self.indian_phone_pattern = re.compile(r'\\b(?:\\+91|91)?[6-9]\\d{9}\\b')
        
        # Load stopwords for supported languages
        self.stopwords = {}
        for lang_code in config.get_supported_languages():
            if lang_code in ['en', 'hi']:
                try:
                    lang_name = 'english' if lang_code == 'en' else 'hindi'
                    self.stopwords[lang_code] = set(stopwords.words(lang_name))
                except:
                    self.stopwords[lang_code] = set()
            else:
                self.stopwords[lang_code] = set()
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing noise and normalizing"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle emojis - convert to text descriptions
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Remove URLs
        text = self.url_pattern.sub(' [URL] ', text)
        
        # Replace mentions and hashtags
        text = self.mention_pattern.sub(' [MENTION] ', text)
        text = self.hashtag_pattern.sub(' [HASHTAG] ', text)
        
        # Remove email addresses and phone numbers
        text = self.email_pattern.sub(' [EMAIL] ', text)
        text = self.phone_pattern.sub(' [PHONE] ', text)
        text = self.indian_phone_pattern.sub(' [PHONE] ', text)
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text).strip()
        
        # Remove very short words (< 2 characters) except important ones
        words = text.split()
        important_short_words = {'is', 'in', 'at', 'on', 'to', 'of', 'me', 'my', 'we'}
        words = [word for word in words if len(word) >= 2 or word in important_short_words]
        
        return ' '.join(words)
    
    def normalize_text(self, text: str, language: str = 'en') -> str:
        """Normalize text for specific language"""
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = self.clean_text(text)
        
        # Language-specific normalization
        if language == 'en':
            return self._normalize_english(cleaned)
        elif language == 'hi':
            return self._normalize_hindi(cleaned)
        else:
            return self._normalize_generic(cleaned, language)
    
    def _normalize_english(self, text: str) -> str:
        """English-specific normalization"""
        # Handle contractions
        contractions = {
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove stopwords (optional - keep for context)
        # words = word_tokenize(text)
        # words = [word for word in words if word not in self.stopwords.get('en', set())]
        # text = ' '.join(words)
        
        return text
    
    def _normalize_hindi(self, text: str) -> str:
        """Hindi-specific normalization"""
        # Normalize Hindi characters
        hindi_normalizations = {
            'ऱ': 'र',
            'ऴ': 'ल',
            'क़': 'क',
            'ख़': 'ख',
            'ग़': 'ग',
            'ज़': 'ज',
            'ड़': 'ड',
            'ढ़': 'ढ',
            'फ़': 'फ'
        }
        
        for char, normalized in hindi_normalizations.items():
            text = text.replace(char, normalized)
        
        return text
    
    def _normalize_generic(self, text: str, language: str) -> str:
        """Generic normalization for other languages"""
        # Basic normalization
        return text
    
    def remove_noise(self, text: str) -> str:
        """Remove noise like excessive punctuation, repeated characters"""
        if not text:
            return ""
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove repeated characters (but keep double letters that are valid)
        text = re.sub(r'([a-zA-Z])\\1{2,}', r'\\1\\1', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\\s+', ' ', text).strip()
        
        return text

class CodeMixedTextHandler:
    """Handle code-mixed text (e.g., Hindi-English, Tamil-English)"""
    
    def __init__(self):
        self.romanized_patterns = {
            'hi': {
                'aur': 'और',
                'hai': 'है',
                'main': 'मैं',
                'kya': 'क्या',
                'mein': 'में',
                'se': 'से',
                'ko': 'को',
                'ka': 'का',
                'ki': 'की',
                'ke': 'के'
            }
        }
    
    def handle_code_mixing(self, text: str, primary_language: str) -> str:
        """Handle code-mixed text"""
        if not text or primary_language not in self.romanized_patterns:
            return text
        
        words = text.split()
        processed_words = []
        
        for word in words:
            # Check if word is in romanized pattern
            if word.lower() in self.romanized_patterns[primary_language]:
                processed_words.append(self.romanized_patterns[primary_language][word.lower()])
            else:
                processed_words.append(word)
        
        return ' '.join(processed_words)

class PreprocessingPipeline:
    """Main preprocessing pipeline for raw reports"""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.text_normalizer = TextNormalizer()
        self.code_mixed_handler = CodeMixedTextHandler()
        
        self.stats = {
            'total_processed': 0,
            'language_distribution': {},
            'processing_errors': 0
        }
    
    async def process_report(self, raw_report: RawReport) -> ProcessedReport:
        """Process a single raw report"""
        try:
            # Detect language
            detected_lang, confidence = self.language_detector.detect_language(raw_report.content)
            
            # Clean and normalize text
            cleaned_content = self.text_normalizer.clean_text(raw_report.content)
            
            # Handle code-mixing
            if confidence < 0.7:  # Low confidence might indicate code-mixing
                cleaned_content = self.code_mixed_handler.handle_code_mixing(
                    cleaned_content, detected_lang
                )
            
            # Normalize for the detected language
            normalized_content = self.text_normalizer.normalize_text(
                cleaned_content, detected_lang
            )
            
            # Remove noise
            normalized_content = self.text_normalizer.remove_noise(normalized_content)
            
            # Create processed report
            processed_report = ProcessedReport(
                id=raw_report.id,
                source=raw_report.source,
                original_content=raw_report.content,
                cleaned_content=cleaned_content,
                normalized_content=normalized_content,
                detected_language=detected_lang,
                confidence_score=confidence,
                timestamp=raw_report.timestamp,
                metadata=raw_report.metadata,
                location=raw_report.location,
                media_urls=raw_report.media_urls,
                processing_info={
                    'processed_at': datetime.now(),
                    'text_length_original': len(raw_report.content),
                    'text_length_processed': len(normalized_content),
                    'processing_version': '1.0'
                }
            )
            
            # Update statistics
            self.stats['total_processed'] += 1
            self.stats['language_distribution'][detected_lang] = (
                self.stats['language_distribution'].get(detected_lang, 0) + 1
            )
            
            return processed_report
            
        except Exception as e:
            print(f"Error processing report {raw_report.id}: {e}")
            self.stats['processing_errors'] += 1
            
            # Return a minimal processed report on error
            return ProcessedReport(
                id=raw_report.id,
                source=raw_report.source,
                original_content=raw_report.content,
                cleaned_content=raw_report.content,
                normalized_content=raw_report.content,
                detected_language='en',
                confidence_score=0.0,
                timestamp=raw_report.timestamp,
                metadata=raw_report.metadata,
                location=raw_report.location,
                media_urls=raw_report.media_urls,
                processing_info={
                    'processed_at': datetime.now(),
                    'processing_error': str(e),
                    'processing_version': '1.0'
                }
            )
    
    async def process_batch(self, raw_reports: List[RawReport]) -> List[ProcessedReport]:
        """Process a batch of raw reports"""
        print(f"Processing batch of {len(raw_reports)} reports...")
        
        tasks = [self.process_report(report) for report in raw_reports]
        processed_reports = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_reports = []
        for result in processed_reports:
            if isinstance(result, ProcessedReport):
                valid_reports.append(result)
            else:
                print(f"Processing error: {result}")
                self.stats['processing_errors'] += 1
        
        print(f"Successfully processed {len(valid_reports)} reports")
        return valid_reports
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset preprocessing statistics"""
        self.stats = {
            'total_processed': 0,
            'language_distribution': {},
            'processing_errors': 0
        }