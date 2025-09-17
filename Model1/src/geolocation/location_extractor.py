import asyncio
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# NLP and geolocation libraries
import spacy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from fuzzywuzzy import fuzz, process

# For NER
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from ..preprocessing import ProcessedReport
from ..nlp_analysis import HazardPrediction
from config import config

@dataclass
class LocationEntity:
    """Location entity extracted from text"""
    text: str
    label: str  # LOCATION, GPE, FACILITY
    start_char: int
    end_char: int
    confidence: float
    coordinates: Optional[Tuple[float, float]] = None
    address: Optional[str] = None
    place_type: Optional[str] = None
    country: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None

@dataclass
class GeolocationResult:
    """Result of geolocation extraction"""
    report_id: str
    extracted_locations: List[LocationEntity]
    primary_location: Optional[LocationEntity]
    confidence_score: float
    processing_metadata: Dict[str, Any]

class IndianLocationDatabase:
    """Database of Indian locations for fuzzy matching"""
    
    def __init__(self):
        # Major Indian coastal cities and landmarks
        self.coastal_locations = {
            # Tamil Nadu
            'chennai': {'lat': 13.0827, 'lon': 80.2707, 'state': 'Tamil Nadu', 'type': 'city'},
            'kanyakumari': {'lat': 8.0883, 'lon': 77.5385, 'state': 'Tamil Nadu', 'type': 'city'},
            'rameswaram': {'lat': 9.2876, 'lon': 79.3129, 'state': 'Tamil Nadu', 'type': 'city'},
            'mahabalipuram': {'lat': 12.6208, 'lon': 80.1982, 'state': 'Tamil Nadu', 'type': 'heritage'},
            'puducherry': {'lat': 11.9416, 'lon': 79.8083, 'state': 'Puducherry', 'type': 'city'},
            
            # Kerala
            'kochi': {'lat': 9.9312, 'lon': 76.2673, 'state': 'Kerala', 'type': 'city'},
            'thiruvananthapuram': {'lat': 8.5241, 'lon': 76.9366, 'state': 'Kerala', 'type': 'city'},
            'kozhikode': {'lat': 11.2588, 'lon': 75.7804, 'state': 'Kerala', 'type': 'city'},
            'kollam': {'lat': 8.8932, 'lon': 76.6141, 'state': 'Kerala', 'type': 'city'},
            'alappuzha': {'lat': 9.4981, 'lon': 76.3388, 'state': 'Kerala', 'type': 'city'},
            
            # Karnataka
            'mangalore': {'lat': 12.9141, 'lon': 74.8560, 'state': 'Karnataka', 'type': 'city'},
            'karwar': {'lat': 14.8137, 'lon': 74.1290, 'state': 'Karnataka', 'type': 'city'},
            
            # Goa
            'panaji': {'lat': 15.4909, 'lon': 73.8278, 'state': 'Goa', 'type': 'city'},
            'margao': {'lat': 15.2832, 'lon': 73.9862, 'state': 'Goa', 'type': 'city'},
            
            # Maharashtra
            'mumbai': {'lat': 19.0760, 'lon': 72.8777, 'state': 'Maharashtra', 'type': 'city'},
            'pune': {'lat': 18.5204, 'lon': 73.8567, 'state': 'Maharashtra', 'type': 'city'},
            'ratnagiri': {'lat': 16.9944, 'lon': 73.3000, 'state': 'Maharashtra', 'type': 'city'},
            
            # Gujarat
            'ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'state': 'Gujarat', 'type': 'city'},
            'surat': {'lat': 21.1702, 'lon': 72.8311, 'state': 'Gujarat', 'type': 'city'},
            'gandhinagar': {'lat': 23.2156, 'lon': 72.6369, 'state': 'Gujarat', 'type': 'city'},
            'bhuj': {'lat': 23.2420, 'lon': 69.6669, 'state': 'Gujarat', 'type': 'city'},
            'porbandar': {'lat': 21.6417, 'lon': 69.6293, 'state': 'Gujarat', 'type': 'city'},
            
            # Andhra Pradesh & Telangana
            'visakhapatnam': {'lat': 17.6868, 'lon': 83.2185, 'state': 'Andhra Pradesh', 'type': 'city'},
            'vijayawada': {'lat': 16.5062, 'lon': 80.6480, 'state': 'Andhra Pradesh', 'type': 'city'},
            'hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'state': 'Telangana', 'type': 'city'},
            
            # Odisha
            'bhubaneswar': {'lat': 20.2961, 'lon': 85.8245, 'state': 'Odisha', 'type': 'city'},
            'puri': {'lat': 19.8135, 'lon': 85.8312, 'state': 'Odisha', 'type': 'city'},
            'cuttack': {'lat': 20.4625, 'lon': 85.8828, 'state': 'Odisha', 'type': 'city'},
            
            # West Bengal
            'kolkata': {'lat': 22.5726, 'lon': 88.3639, 'state': 'West Bengal', 'type': 'city'},
            'digha': {'lat': 21.6273, 'lon': 87.5338, 'state': 'West Bengal', 'type': 'beach'},
            
            # Andaman & Nicobar
            'port blair': {'lat': 11.6234, 'lon': 92.7265, 'state': 'Andaman & Nicobar', 'type': 'city'},
            
            # Lakshadweep
            'kavaratti': {'lat': 10.5593, 'lon': 72.6420, 'state': 'Lakshadweep', 'type': 'island'}
        }
        
        # Coastal landmarks and features
        self.landmarks = {
            'marina beach': {'lat': 13.0493, 'lon': 80.2824, 'state': 'Tamil Nadu', 'type': 'beach'},
            'juhu beach': {'lat': 19.0990, 'lon': 72.8262, 'state': 'Maharashtra', 'type': 'beach'},
            'ganga sagar': {'lat': 21.6273, 'lon': 88.0985, 'state': 'West Bengal', 'type': 'beach'},
            'mandarmani beach': {'lat': 21.6581, 'lon': 87.7664, 'state': 'West Bengal', 'type': 'beach'},
            'puri beach': {'lat': 19.8135, 'lon': 85.8312, 'state': 'Odisha', 'type': 'beach'},
            'kovalam beach': {'lat': 8.4004, 'lon': 76.9787, 'state': 'Kerala', 'type': 'beach'},
            'baga beach': {'lat': 15.5557, 'lon': 73.7516, 'state': 'Goa', 'type': 'beach'},
            'calangute beach': {'lat': 15.5443, 'lon': 73.7554, 'state': 'Goa', 'type': 'beach'},
            'lighthouse': {'lat': 0, 'lon': 0, 'state': 'Generic', 'type': 'landmark'}
        }
        
        # Combine all locations
        self.all_locations = {**self.coastal_locations, **self.landmarks}
        
        # Create search index
        self.location_names = list(self.all_locations.keys())
    
    def find_similar_locations(self, query: str, threshold: int = 70) -> List[Tuple[str, int, Dict]]:
        """Find similar location names using fuzzy matching"""
        matches = process.extract(query.lower(), self.location_names, limit=5)
        
        results = []
        for match_name, score in matches:
            if score >= threshold:
                location_data = self.all_locations[match_name]
                results.append((match_name, score, location_data))
        
        return results

class NamedEntityRecognizer:
    """Named Entity Recognition for location extraction"""
    
    def __init__(self):
        self.ner_pipeline = None
        self.spacy_nlp = None
        
        # Location entity types
        self.location_labels = ['LOC', 'GPE', 'FACILITY', 'LOCATION', 'PERSON']  # PERSON sometimes misclassified locations
    
    async def load_models(self):
        """Load NER models"""
        try:
            # Load transformer-based NER
            self.ner_pipeline = pipeline(
                "ner",
                model="xlm-roberta-large-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load spaCy model for additional NER
            try:
                self.spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy English model not found. Installing...")
                # In production, ensure models are pre-installed
                self.spacy_nlp = None
            
            print("NER models loaded successfully")
            
        except Exception as e:
            print(f"Error loading NER models: {e}")
            self.ner_pipeline = None
            self.spacy_nlp = None
    
    async def extract_entities(self, text: str, language: str = 'en') -> List[LocationEntity]:
        """Extract location entities from text"""
        entities = []
        
        # Use transformer-based NER
        if self.ner_pipeline:
            transformer_entities = await self._extract_with_transformer(text)
            entities.extend(transformer_entities)
        
        # Use spaCy NER as additional source
        if self.spacy_nlp:
            spacy_entities = await self._extract_with_spacy(text)
            entities.extend(spacy_entities)
        
        # Rule-based extraction for specific patterns
        rule_entities = await self._extract_with_rules(text)
        entities.extend(rule_entities)
        
        # Remove duplicates and merge overlapping entities
        entities = self._merge_entities(entities)
        
        return entities
    
    async def _extract_with_transformer(self, text: str) -> List[LocationEntity]:
        """Extract entities using transformer model"""
        entities = []
        
        try:
            ner_results = self.ner_pipeline(text)
            
            for result in ner_results:
                if result['entity_group'] in self.location_labels:
                    entity = LocationEntity(
                        text=result['word'],
                        label=result['entity_group'],
                        start_char=result['start'],
                        end_char=result['end'],
                        confidence=result['score']
                    )
                    entities.append(entity)
        
        except Exception as e:
            print(f"Transformer NER error: {e}")
        
        return entities
    
    async def _extract_with_spacy(self, text: str) -> List[LocationEntity]:
        """Extract entities using spaCy"""
        entities = []
        
        try:
            doc = self.spacy_nlp(text)
            
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC', 'FAC']:  # Geographic, Location, Facility
                    entity = LocationEntity(
                        text=ent.text,
                        label=ent.label_,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        confidence=0.8  # spaCy doesn't provide confidence scores
                    )
                    entities.append(entity)
        
        except Exception as e:
            print(f"spaCy NER error: {e}")
        
        return entities
    
    async def _extract_with_rules(self, text: str) -> List[LocationEntity]:
        """Extract entities using rule-based patterns"""
        entities = []
        
        # Common location patterns
        patterns = [
            # "in [location]", "at [location]", "near [location]"
            r'(?:in|at|near|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            
            # "Beach", "Port", "Harbor", etc.
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:beach|port|harbor|lighthouse|station|airport)',
            
            # State/City patterns
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z][a-z]+)',
            
            # Coordinates pattern
            r'(\d+\.\d+)[°]?\s*[NS],?\s*(\d+\.\d+)[°]?\s*[EW]'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                location_text = match.group(1) if match.lastindex >= 1 else match.group(0)
                
                entity = LocationEntity(
                    text=location_text.strip(),
                    label='LOCATION',
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.6  # Lower confidence for rule-based
                )
                entities.append(entity)
        
        return entities
    
    def _merge_entities(self, entities: List[LocationEntity]) -> List[LocationEntity]:
        """Merge overlapping entities and remove duplicates"""
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda x: x.start_char)
        
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            # Check for overlap
            if (next_entity.start_char <= current.end_char and 
                next_entity.end_char >= current.start_char):
                
                # Merge entities - keep the one with higher confidence
                if next_entity.confidence > current.confidence:
                    current = next_entity
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        
        # Remove duplicates based on text similarity
        final_entities = []
        for entity in merged:
            is_duplicate = False
            for existing in final_entities:
                if fuzz.ratio(entity.text.lower(), existing.text.lower()) > 85:
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        final_entities.remove(existing)
                        final_entities.append(entity)
                    break
            
            if not is_duplicate:
                final_entities.append(entity)
        
        return final_entities

class GeocodingService:
    """Geocoding service to resolve locations to coordinates"""
    
    def __init__(self):
        self.nominatim = Nominatim(
            user_agent=config.get('GEOLOCATION.NOMINATIM_USER_AGENT', 'ocean_hazard_platform')
        )
        self.location_db = IndianLocationDatabase()
        
        # Cache for geocoding results
        self.geocoding_cache = {}
    
    async def geocode_location(self, location_entity: LocationEntity) -> LocationEntity:
        """Geocode a location entity to get coordinates"""
        
        location_text = location_entity.text.lower().strip()
        
        # Check cache first
        if location_text in self.geocoding_cache:
            cached_result = self.geocoding_cache[location_text]
            location_entity.coordinates = cached_result.get('coordinates')
            location_entity.address = cached_result.get('address')
            location_entity.place_type = cached_result.get('place_type')
            location_entity.country = cached_result.get('country')
            location_entity.state = cached_result.get('state')
            return location_entity
        
        # Try fuzzy matching with Indian location database first
        similar_locations = self.location_db.find_similar_locations(location_text)
        
        if similar_locations:
            best_match = similar_locations[0]
            match_name, score, location_data = best_match
            
            if score > 80:  # High similarity
                location_entity.coordinates = (location_data['lat'], location_data['lon'])
                location_entity.state = location_data['state']
                location_entity.place_type = location_data['type']
                location_entity.country = 'India'
                location_entity.address = f"{match_name.title()}, {location_data['state']}, India"
                
                # Cache the result
                self.geocoding_cache[location_text] = {
                    'coordinates': location_entity.coordinates,
                    'address': location_entity.address,
                    'place_type': location_entity.place_type,
                    'country': location_entity.country,
                    'state': location_entity.state
                }
                
                return location_entity
        
        # Fallback to Nominatim geocoding
        try:
            # Add "India" to the query for better results
            query = f"{location_entity.text}, India"
            
            location = await asyncio.get_event_loop().run_in_executor(
                None, self.nominatim.geocode, query
            )
            
            if location:
                location_entity.coordinates = (location.latitude, location.longitude)
                location_entity.address = location.address
                
                # Extract additional info from address
                address_parts = location.address.split(', ')
                if len(address_parts) > 1:
                    location_entity.state = address_parts[-2] if 'India' in address_parts[-1] else None
                location_entity.country = 'India' if 'India' in location.address else None
                
                # Cache the result
                self.geocoding_cache[location_text] = {
                    'coordinates': location_entity.coordinates,
                    'address': location_entity.address,
                    'place_type': location_entity.place_type,
                    'country': location_entity.country,
                    'state': location_entity.state
                }
            
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Geocoding error for '{location_entity.text}': {e}")
        
        return location_entity

class GeolocationExtractor:
    """Main geolocation extraction engine"""
    
    def __init__(self):
        self.ner = NamedEntityRecognizer()
        self.geocoding_service = GeocodingService()
        
        self.is_loaded = False
        
        self.stats = {
            'total_processed': 0,
            'entities_extracted': 0,
            'successfully_geocoded': 0,
            'geocoding_errors': 0
        }
    
    async def load_models(self):
        """Load geolocation models"""
        print("Loading geolocation models...")
        await self.ner.load_models()
        self.is_loaded = True
        print("Geolocation models loaded successfully")
    
    async def extract_locations(
        self, 
        processed_report: ProcessedReport,
        hazard_prediction: HazardPrediction
    ) -> GeolocationResult:
        """Extract and geocode locations from a processed report"""
        
        if not self.is_loaded:
            await self.load_models()
        
        try:
            # Extract location entities from text
            entities = await self.ner.extract_entities(
                processed_report.normalized_content,
                processed_report.detected_language
            )
            
            # Geocode entities to get coordinates
            geocoded_entities = []
            for entity in entities:
                geocoded_entity = await self.geocoding_service.geocode_location(entity)
                geocoded_entities.append(geocoded_entity)
                
                if geocoded_entity.coordinates:
                    self.stats['successfully_geocoded'] += 1
                else:
                    self.stats['geocoding_errors'] += 1
            
            # Determine primary location
            primary_location = self._determine_primary_location(
                geocoded_entities, 
                processed_report
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence(geocoded_entities)
            
            # Update hazard prediction with location entities
            hazard_prediction.entities = [
                {
                    'text': entity.text,
                    'label': entity.label,
                    'coordinates': entity.coordinates,
                    'confidence': entity.confidence,
                    'address': entity.address
                }
                for entity in geocoded_entities
            ]
            
            result = GeolocationResult(
                report_id=processed_report.id,
                extracted_locations=geocoded_entities,
                primary_location=primary_location,
                confidence_score=confidence_score,
                processing_metadata={
                    'processed_at': datetime.now(),
                    'entities_count': len(geocoded_entities),
                    'geocoded_count': len([e for e in geocoded_entities if e.coordinates]),
                    'processing_version': '1.0'
                }
            )
            
            # Update statistics
            self.stats['total_processed'] += 1
            self.stats['entities_extracted'] += len(geocoded_entities)
            
            return result
            
        except Exception as e:
            print(f"Error extracting locations for report {processed_report.id}: {e}")
            
            # Return empty result on error
            return GeolocationResult(
                report_id=processed_report.id,
                extracted_locations=[],
                primary_location=None,
                confidence_score=0.0,
                processing_metadata={
                    'processed_at': datetime.now(),
                    'error': str(e),
                    'processing_version': '1.0'
                }
            )
    
    def _determine_primary_location(
        self, 
        entities: List[LocationEntity], 
        processed_report: ProcessedReport
    ) -> Optional[LocationEntity]:
        """Determine the primary location from extracted entities"""
        
        if not entities:
            return None
        
        # Prefer entities with coordinates
        geocoded_entities = [e for e in entities if e.coordinates]
        
        if geocoded_entities:
            # Sort by confidence and prefer coastal locations
            scored_entities = []
            
            for entity in geocoded_entities:
                score = entity.confidence
                
                # Boost score for coastal/water-related locations
                if any(keyword in entity.text.lower() for keyword in 
                       ['beach', 'port', 'harbor', 'lighthouse', 'coast', 'shore']):
                    score += 0.2
                
                # Boost score for known Indian coastal cities
                if entity.text.lower() in self.geocoding_service.location_db.coastal_locations:
                    score += 0.3
                
                scored_entities.append((entity, score))
            
            # Return entity with highest score
            return max(scored_entities, key=lambda x: x[1])[0]
        
        # Fallback to highest confidence entity without coordinates
        return max(entities, key=lambda x: x.confidence)
    
    def _calculate_confidence(self, entities: List[LocationEntity]) -> float:
        """Calculate overall confidence score for location extraction"""
        
        if not entities:
            return 0.0
        
        # Base confidence on entity confidences and geocoding success
        entity_confidences = [e.confidence for e in entities]
        geocoded_count = len([e for e in entities if e.coordinates])
        
        avg_confidence = sum(entity_confidences) / len(entity_confidences)
        geocoding_rate = geocoded_count / len(entities) if entities else 0
        
        # Combined confidence score
        confidence = (avg_confidence * 0.6) + (geocoding_rate * 0.4)
        
        return min(confidence, 1.0)
    
    async def extract_batch(
        self,
        processed_reports: List[ProcessedReport],
        hazard_predictions: List[HazardPrediction]
    ) -> List[GeolocationResult]:
        """Extract locations from a batch of reports"""
        
        print(f"Extracting locations from {len(processed_reports)} reports...")
        
        tasks = [
            self.extract_locations(report, prediction)
            for report, prediction in zip(processed_reports, hazard_predictions)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, GeolocationResult):
                valid_results.append(result)
            else:
                print(f"Geolocation extraction error: {result}")
        
        print(f"Successfully extracted locations from {len(valid_results)} reports")
        return valid_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get geolocation extraction statistics"""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset geolocation statistics"""
        self.stats = {
            'total_processed': 0,
            'entities_extracted': 0,
            'successfully_geocoded': 0,
            'geocoding_errors': 0
        }