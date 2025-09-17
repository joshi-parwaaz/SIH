"""
Main pipeline that orchestrates scrapers, ML models, and generates structured output.
Combines data from multiple sources and processes them through the ML pipeline.
"""

import time
import logging
from typing import List, Dict, Any
from datetime import datetime

# Import scrapers
from scrapers.incois_scraper import fetch_incois_alerts
from scrapers.twitter_scraper import fetch_twitter_alerts, SNSCRAPE_AVAILABLE
from scrapers.youtube_scraper import fetch_youtube_alerts

# Import models
from models.relevance_classifier import is_relevant
from models.extractor import extract_info

# Import schema
from app.schema import HazardReport, PipelineResponse, SourceStatus

logger = logging.getLogger(__name__)


class HazardAnalysisPipeline:
    """Main pipeline for hazard analysis."""
    
    def __init__(self):
        self.total_reports_processed = 0
        self.start_time = time.time()
        self.source_configs = {
            "INCOIS": {
                "enabled": True,
                "fetch_function": fetch_incois_alerts,
                "max_items": 20
            },
            "Twitter": {
                "enabled": SNSCRAPE_AVAILABLE,
                "fetch_function": fetch_twitter_alerts,
                "max_items": 15
            },
            "YouTube": {
                "enabled": True,  # Will check API key internally
                "fetch_function": fetch_youtube_alerts,
                "max_items": 10
            }
        }
    
    def fetch_from_source(self, source_name: str, config: Dict[str, Any]) -> SourceStatus:
        """
        Fetch data from a single source.
        
        Args:
            source_name (str): Name of the source
            config (dict): Source configuration
            
        Returns:
            SourceStatus: Status of the fetch operation
        """
        if not config["enabled"]:
            return SourceStatus(
                source_name=source_name,
                status="disabled",
                items_fetched=0,
                error_message="Source is disabled"
            )
        
        try:
            fetch_function = config["fetch_function"]
            max_items = config.get("max_items", 10)
            
            logger.info(f"Fetching from {source_name}...")
            
            if source_name == "Twitter":
                items = fetch_function(max_results=max_items)
            elif source_name == "YouTube":
                items = fetch_function(max_results=max_items)
            else:
                items = fetch_function()
            
            # Limit items if needed
            if len(items) > max_items:
                items = items[:max_items]
            
            return SourceStatus(
                source_name=source_name,
                status="success",
                items_fetched=len(items),
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Error fetching from {source_name}: {e}")
            return SourceStatus(
                source_name=source_name,
                status="error",
                items_fetched=0,
                error_message=str(e)
            )
    
    def process_text(self, text: str, source: str) -> HazardReport:
        """
        Process a single text through the ML pipeline.
        
        Args:
            text (str): Text to process
            source (str): Source of the text
            
        Returns:
            HazardReport: Processed report
        """
        try:
            # Extract information using ML models
            extracted_info = extract_info(text)
            
            # Calculate confidence based on various factors
            confidence = self.calculate_confidence(text, extracted_info)
            
            report = HazardReport(
                source=source,
                text=text,
                hazard_type=extracted_info["hazard_type"],
                severity=extracted_info["severity"],
                urgency=extracted_info["urgency"],
                sentiment=extracted_info["sentiment"],
                misinformation=extracted_info["misinformation"],
                location=extracted_info["location"],
                confidence=confidence,
                timestamp=datetime.utcnow()
            )
            
            self.total_reports_processed += 1
            return report
            
        except Exception as e:
            logger.error(f"Error processing text from {source}: {e}")
            # Return a minimal report on error
            return HazardReport(
                source=source,
                text=text,
                hazard_type="unknown",
                severity="medium",
                urgency="medium",
                sentiment="neutral",
                misinformation=False,
                location="unknown",
                confidence=0.1,
                timestamp=datetime.utcnow()
            )
    
    def calculate_confidence(self, text: str, extracted_info: Dict[str, Any]) -> float:
        """
        Calculate confidence score for a report.
        
        Args:
            text (str): Original text
            extracted_info (dict): Extracted information
            
        Returns:
            float: Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence
        
        # Boost confidence for specific sources
        if "INCOIS" in text or "official" in text.lower():
            confidence += 0.3
        
        # Boost for specific hazard types
        if extracted_info["hazard_type"] in ["tsunami", "cyclone", "flood"]:
            confidence += 0.2
        
        # Reduce for unknown hazard type
        if extracted_info["hazard_type"] == "unknown":
            confidence -= 0.2
        
        # Reduce for detected misinformation
        if extracted_info["misinformation"]:
            confidence -= 0.3
        
        # Boost for specific locations
        if extracted_info["location"] != "unknown":
            confidence += 0.1
        
        # Boost for higher urgency
        if extracted_info["urgency"] in ["high", "immediate"]:
            confidence += 0.1
        
        # Text length factor
        if len(text) > 50:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def run_pipeline(self) -> PipelineResponse:
        """
        Run the complete analysis pipeline.
        
        Returns:
            PipelineResponse: Complete analysis results
        """
        start_time = time.time()
        logger.info("Starting hazard analysis pipeline...")
        
        all_reports = []
        source_statuses = []
        total_sources_checked = 0
        
        # Process each source
        for source_name, config in self.source_configs.items():
            total_sources_checked += 1
            
            # Fetch data from source
            status = self.fetch_from_source(source_name, config)
            source_statuses.append(status)
            
            if status.status != "success":
                logger.warning(f"Skipping {source_name}: {status.error_message}")
                continue
            
            # Get the actual data
            try:
                fetch_function = config["fetch_function"]
                max_items = config.get("max_items", 10)
                
                if source_name == "Twitter":
                    texts = fetch_function(max_results=max_items)
                elif source_name == "YouTube":
                    texts = fetch_function(max_results=max_items)
                else:
                    texts = fetch_function()
                
                if texts and len(texts) > max_items:
                    texts = texts[:max_items]
                
                # Process each text
                for text in texts or []:
                    if text and isinstance(text, str) and len(text.strip()) > 10:
                        # Check relevance first
                        if is_relevant(text):
                            report = self.process_text(text, source_name)
                            all_reports.append(report)
                        else:
                            logger.debug(f"Filtered out irrelevant text: {text[:50]}...")
                
            except Exception as e:
                logger.error(f"Error processing {source_name} data: {e}")
        
        processing_time = time.time() - start_time
        
        # Sort reports by confidence (highest first)
        all_reports.sort(key=lambda x: x.confidence, reverse=True)
        
        response = PipelineResponse(
            reports=all_reports,
            total_sources_checked=total_sources_checked,
            relevant_reports=len(all_reports),
            processing_time_seconds=round(processing_time, 2),
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Pipeline completed: {len(all_reports)} reports in {processing_time:.2f}s")
        return response
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            dict: System status information
        """
        uptime = time.time() - self.start_time
        
        # Check source statuses
        source_statuses = []
        for source_name, config in self.source_configs.items():
            status = self.fetch_from_source(source_name, config)
            source_statuses.append(status)
        
        return {
            "status": "operational",
            "sources": source_statuses,
            "total_reports_processed": self.total_reports_processed,
            "uptime_seconds": round(uptime, 2),
            "version": "1.5",
            "timestamp": datetime.utcnow()
        }


# Global pipeline instance
_pipeline = None


def get_pipeline() -> HazardAnalysisPipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = HazardAnalysisPipeline()
    return _pipeline


def run_pipeline() -> Dict[str, Any]:
    """
    Main function to run the pipeline (for backward compatibility).
    
    Returns:
        dict: Pipeline results as dictionary
    """
    pipeline = get_pipeline()
    response = pipeline.run_pipeline()
    return response.dict()


if __name__ == "__main__":
    # Test the pipeline
    pipeline = HazardAnalysisPipeline()
    result = pipeline.run_pipeline()
    
    print(f"Found {result.relevant_reports} relevant reports:")
    for report in result.reports[:3]:  # Show first 3
        print(f"- [{report.source}] {report.hazard_type} ({report.confidence:.2f}): {report.text[:80]}...")