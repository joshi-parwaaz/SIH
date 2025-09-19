"""
Production-Ready Disaster Management Pipeline
Clean, optimized pipeline with enhanced ML components for SIH deployment.
"""

import time
import logging
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HazardReport:
    """Simple data class for hazard reports."""
    source: str
    text: str
    hazard_type: str
    severity: str
    urgency: str
    sentiment: str
    misinformation: bool
    location: str
    confidence: float
    timestamp: datetime


@dataclass  
class PipelineResult:
    """Simple data class for pipeline results."""
    reports: List[HazardReport]
    total_sources_checked: int
    relevant_reports: int
    processing_time_seconds: float
    timestamp: datetime

# Import all scrapers
from scrapers.incois_scraper import fetch_incois_alerts
from scrapers.custom_twitter_scraper import fetch_twitter_alerts
from scrapers.youtube_scraper import fetch_youtube_alerts
from scrapers.government_regional_sources import fetch_all_government_sources
from scrapers.google_news_scraper import fetch_google_news_alerts

# Import enhanced ML models
from models.enhanced_nlp_classifier import batch_classify_fast
from models.enhanced_anomaly_detector import EnhancedAnomalyDetector
from models.enhanced_reverse_image_checker import EnhancedReverseImageChecker
from models.extractor import extract_info

# Import schema
# Using simple dataclasses defined above

class ProductionPipeline:
    """
    Production-ready disaster management pipeline.
    Combines 5 data sources with 3 enhanced ML components.
    Supports dynamic region-based filtering.
    """
    
    def __init__(self, target_region=None):
        self.anomaly_detector = EnhancedAnomalyDetector()
        self.misinformation_checker = EnhancedReverseImageChecker()
        self.target_region = target_region  # Dynamic region filtering
        
        # Source configuration
        self.sources = {
            "INCOIS": fetch_incois_alerts,
            "Twitter": self._fetch_twitter_custom,
            "YouTube": fetch_youtube_alerts,
            "Government_Sources": fetch_all_government_sources,
            "Google_News": fetch_google_news_alerts
        }
        
        logger.info(f"Production pipeline initialized with 5 sources and 3 ML components")
        if target_region:
            logger.info(f"🎯 Region filter: {target_region}")
    
    def _get_region_keywords(self, region):
        """Get keywords for a specific region including major cities and districts."""
        region_keywords = {
            'odisha': ['odisha', 'orissa', 'bhubaneswar', 'cuttack', 'puri', 'berhampur', 'rourkela', 'sambalpur', 'balasore', 'kendrapara', 'jagatsinghpur'],
            'west bengal': ['west bengal', 'kolkata', 'howrah', 'durgapur', 'asansol', 'siliguri', 'malda', 'kharagpur', 'haldia'],
            'tamil nadu': ['tamil nadu', 'chennai', 'madurai', 'coimbatore', 'tiruchirappalli', 'salem', 'tirunelveli', 'erode', 'vellore'],
            'karnataka': ['karnataka', 'bangalore', 'bengaluru', 'mysore', 'hubli', 'mangalore', 'belgaum', 'davangere', 'bellary'],
            'kerala': ['kerala', 'thiruvananthapuram', 'kochi', 'kozhikode', 'thrissur', 'kollam', 'palakkad', 'alappuzha', 'kannur'],
            'gujarat': ['gujarat', 'ahmedabad', 'surat', 'vadodara', 'rajkot', 'bhavnagar', 'jamnagar', 'gandhinagar', 'anand'],
            'maharashtra': ['maharashtra', 'mumbai', 'pune', 'nagpur', 'thane', 'nashik', 'aurangabad', 'solapur', 'amravati'],
            'andhra pradesh': ['andhra pradesh', 'hyderabad', 'visakhapatnam', 'vijayawada', 'guntur', 'nellore', 'kurnool', 'rajahmundry', 'tirupati'],
            'telangana': ['telangana', 'hyderabad', 'warangal', 'nizamabad', 'karimnagar', 'khammam', 'mahbubnagar', 'nalgonda'],
            'goa': ['goa', 'panaji', 'margao', 'vasco', 'mapusa', 'ponda', 'bicholim'],
            'punjab': ['punjab', 'chandigarh', 'ludhiana', 'amritsar', 'jalandhar', 'patiala', 'bathinda', 'mohali'],
            'haryana': ['haryana', 'gurgaon', 'faridabad', 'panipat', 'ambala', 'yamunanagar', 'rohtak', 'hisar'],
            'rajasthan': ['rajasthan', 'jaipur', 'jodhpur', 'udaipur', 'kota', 'bikaner', 'ajmer', 'bhilwara'],
            'uttar pradesh': ['uttar pradesh', 'lucknow', 'kanpur', 'ghaziabad', 'agra', 'varanasi', 'meerut', 'allahabad'],
            'bihar': ['bihar', 'patna', 'gaya', 'bhagalpur', 'muzaffarpur', 'darbhanga', 'bihar sharif', 'arrah'],
            'jharkhand': ['jharkhand', 'ranchi', 'jamshedpur', 'dhanbad', 'bokaro', 'deoghar', 'phusro', 'hazaribagh'],
            'chhattisgarh': ['chhattisgarh', 'raipur', 'bhilai', 'bilaspur', 'korba', 'durg', 'rajnandgaon'],
            'uttarakhand': ['uttarakhand', 'dehradun', 'haridwar', 'roorkee', 'rudrapur', 'kashipur', 'haldwani'],
            'himachal pradesh': ['himachal pradesh', 'shimla', 'dharamshala', 'solan', 'mandi', 'kullu', 'hamirpur'],
            'jammu kashmir': ['jammu kashmir', 'srinagar', 'jammu', 'anantnag', 'baramulla', 'kupwara', 'udhampur'],
            'assam': ['assam', 'guwahati', 'dibrugarh', 'silchar', 'nagaon', 'tinsukia', 'jorhat', 'bongaigaon'],
            'manipur': ['manipur', 'imphal', 'thoubal', 'bishnupur', 'churachandpur', 'ukhrul'],
            'meghalaya': ['meghalaya', 'shillong', 'tura', 'jowai', 'nongstoin'],
            'mizoram': ['mizoram', 'aizawl', 'lunglei', 'saiha', 'champhai'],
            'nagaland': ['nagaland', 'kohima', 'dimapur', 'mokokchung', 'tuensang'],
            'tripura': ['tripura', 'agartala', 'dharmanagar', 'udaipur', 'kailashahar'],
            'sikkim': ['sikkim', 'gangtok', 'namchi', 'gyalshing', 'mangan'],
            'arunachal pradesh': ['arunachal pradesh', 'itanagar', 'naharlagun', 'pasighat', 'aalo']
        }
        
        region_lower = region.lower() if region else None
        return region_keywords.get(region_lower, [region_lower] if region_lower else [])
    
    def _fetch_twitter_custom(self, query="flood OR tsunami OR cyclone", max_results=10):
        """Fetch Twitter data with optional region filtering."""
        try:
            # Add region-specific terms to the query if target region is specified
            if self.target_region:
                region_keywords = self._get_region_keywords(self.target_region)
                region_query = " OR ".join(region_keywords[:5])  # Use top 5 keywords to avoid too long query
                final_query = f"({query}) AND ({region_query})"
            else:
                final_query = query
                
            # Use the fetch_twitter_alerts function instead
            tweets = fetch_twitter_alerts()
            
            # Extract text content (already in string format)
            tweet_texts = []
            for tweet in tweets:
                if isinstance(tweet, str):
                    tweet_texts.append(tweet)
            
            return tweet_texts
        except Exception as e:
            logger.error(f"Custom Twitter scraper failed: {e}")
            return []
    
    def _is_relevant_to_region(self, content, region=None):
        """Check if content is relevant to the specified region."""
        if not region:
            return True  # No region filter, include all content
            
        content_lower = content.lower()
        region_keywords = self._get_region_keywords(region)
        
        # Check if any region keyword appears in the content
        for keyword in region_keywords:
            if keyword in content_lower:
                return True
        return False

    def run_pipeline(self, region=None) -> PipelineResult:
        """
        Execute the complete production pipeline with optional region filtering.
        
        Args:
            region (str): Target region for filtering (e.g., 'Odisha', 'Tamil Nadu', 'Kerala')
        
        Returns:
            PipelineResult: Complete analysis results
        """
        start_time = time.time()
        all_reports = []
        source_stats = {}
        
        # Update target region for this run
        if region:
            self.target_region = region
            logger.info(f"🎯 Filtering results for region: {region}")
        
        logger.info("🚀 Starting production disaster management pipeline")
        
        # Step 1: Collect data from all sources
        for source_name, fetch_function in self.sources.items():
            source_start = time.time()
            
            try:
                raw_data = fetch_function()
                
                # Apply region filtering if specified
                if region and raw_data:
                    filtered_data = []
                    for item in raw_data:
                        if self._is_relevant_to_region(str(item), region):
                            filtered_data.append(item)
                    raw_data = filtered_data
                    logger.info(f"✅ {source_name}: Retrieved {len(raw_data)} items (region-filtered)")
                else:
                    logger.info(f"✅ {source_name}: Retrieved {len(raw_data)} items")
                
                # Convert to HazardReport objects with enhanced extraction
                for item in raw_data:
                    report = self._create_hazard_report(item, source_name)
                    if report:
                        all_reports.append(report)
                
                source_time = time.time() - source_start
                source_stats[source_name] = {
                    "count": len(raw_data),
                    "time": source_time,
                    "status": "success"
                }
                
            except Exception as e:
                logger.error(f"❌ {source_name} failed: {e}")
                source_stats[source_name] = {
                    "count": 0,
                    "time": 0,
                    "status": "failed",
                    "error": str(e)
                }
        
        # Step 2: Enhanced NLP classification (batch processing)
        if all_reports:
            logger.info(f"🧠 Running enhanced NLP classification on {len(all_reports)} reports")
            
            # Batch classify for efficiency
            texts = [report.text for report in all_reports]
            classifications = batch_classify_fast(texts)
            
            # Update reports with enhanced classifications
            relevant_reports = []
            for report, (is_relevant, confidence) in zip(all_reports, classifications):
                if is_relevant:
                    report.confidence = max(report.confidence, confidence)
                    relevant_reports.append(report)
            
            logger.info(f"✅ NLP: {len(relevant_reports)} relevant reports identified")
        else:
            relevant_reports = []
            logger.warning("No reports to classify")
        
        # Step 3: Enhanced anomaly detection
        anomaly_results = {}
        if relevant_reports:
            logger.info("🔍 Running enhanced anomaly detection")
            
            # Convert to dict format for anomaly detection  
            report_dicts = [
                {
                    'source': r.source,
                    'text': r.text, 
                    'hazard_type': r.hazard_type,
                    'location': r.location,
                    'confidence': r.confidence,
                    'timestamp': r.timestamp.isoformat()
                } for r in relevant_reports
            ]
            anomaly_results = self.anomaly_detector.analyze_all_anomalies(report_dicts)
            
            logger.info(f"✅ Anomaly detection: {anomaly_results.get('total_anomalies', 0)} anomalies detected")
        
        # Step 4: Enhanced misinformation detection (sample check)
        misinformation_results = {}
        if relevant_reports:
            logger.info("🛡️ Running enhanced misinformation detection")
            
            # Check top 10 reports for misinformation
            sample_reports = relevant_reports[:10]
            content_list = [(report.text, report.source) for report in sample_reports]
            
            misinformation_batch = self.misinformation_checker.batch_verify_content(content_list)
            
            misinformation_count = sum(1 for result in misinformation_batch 
                                     if result.get("is_likely_misinformation", False))
            avg_authenticity = (sum(result.get("authenticity_score", 0) for result in misinformation_batch) 
                              / len(misinformation_batch) if misinformation_batch else 0)
            
            misinformation_results = {
                "checked_reports": len(content_list),
                "suspicious_count": misinformation_count,
                "average_authenticity": avg_authenticity,
                "details": misinformation_batch
            }
            
            logger.info(f"✅ Misinformation check: {misinformation_count}/{len(content_list)} suspicious reports")
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Create comprehensive result
        result = PipelineResult(
            reports=relevant_reports,
            total_sources_checked=len(self.sources),
            relevant_reports=len(relevant_reports),
            processing_time_seconds=total_time,
            timestamp=datetime.now()
        )
        
        # Log final summary
        logger.info(f"🎯 Pipeline completed in {total_time:.2f}s")
        logger.info(f"📊 Found {len(relevant_reports)} relevant reports from {len(self.sources)} sources")
        logger.info(f"🔍 Detected {anomaly_results.get('total_anomalies', 0)} anomalies")
        logger.info(f"🛡️ Average authenticity score: {misinformation_results.get('average_authenticity', 0):.3f}")
        
        return result
    
    def _create_hazard_report(self, raw_text: str, source: str) -> HazardReport:
        """
        Create a HazardReport from raw text using enhanced extraction.
        
        Args:
            raw_text: Raw text from scraper
            source: Source name
            
        Returns:
            HazardReport object or None if invalid
        """
        try:
            if not raw_text or len(raw_text.strip()) < 10:
                return None
            
            # Enhanced extraction
            extracted_info = extract_info(raw_text)
            
            # Create report with enhanced confidence scoring
            report = HazardReport(
                text=raw_text.strip(),
                source=source,
                timestamp=datetime.now(),
                hazard_type=extracted_info.get("hazard_type", "unknown"),
                location=extracted_info.get("location", "unknown"),
                confidence=extracted_info.get("confidence", 0.5),
                severity=extracted_info.get("severity", "medium"),
                urgency="medium",  # Default urgency
                sentiment="neutral",  # Default sentiment
                misinformation=False  # Default to not misinformation
            )
            
            return report
            
        except Exception as e:
            logger.warning(f"Error creating hazard report: {e}")
            return None


def run_production_pipeline(region=None) -> PipelineResult:
    """
    Execute the production pipeline with optional region filtering.
    
    Args:
        region (str): Optional region name for filtering results
    
    Returns:
        PipelineResult: Complete analysis results
    """
    pipeline = ProductionPipeline(target_region=region)
    return pipeline.run_pipeline(region=region)


if __name__ == "__main__":
    # Test with different regions
    test_regions = ["Odisha", "Tamil Nadu", None]  # None = all regions
    
    for region in test_regions:
        print(f"\n{'='*60}")
        print(f"🎯 Testing region: {region or 'All regions'}")
        print(f"{'='*60}")
        
        # Run pipeline with region filter
        result = run_production_pipeline(region=region)
        
        print(f"\n🎉 Production pipeline completed successfully!")
        print(f"📊 {result.relevant_reports} relevant reports processed")
        print(f"⏱️ Total time: {result.processing_time_seconds:.2f}s")
        print(f"🎯 Region filter: {region or 'None (all regions)'}")
        
        # Show sample reports
        if result.reports:
            print(f"\n📋 Sample reports:")
            for i, report in enumerate(result.reports[:3]):  # Show first 3
                print(f"   {i+1}. [{report.source}] {report.text[:80]}...")
        else:
            print("\n📋 No reports found for this region")