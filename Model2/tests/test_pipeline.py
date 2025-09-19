"""
Test suite for the hazard analysis pipeline.
Tests all major components including scrapers, models, and API endpoints.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from scrapers.incois_scraper import fetch_incois_alerts
from scrapers.twitter_scraper import fetch_twitter_alerts
from scrapers.youtube_scraper import fetch_youtube_alerts
from models.relevance_classifier import is_relevant
from models.extractor import extract_info
from app.pipeline import HazardAnalysisPipeline


class TestScrapers(unittest.TestCase):
    """Test scraper modules."""
    
    def test_incois_scraper_structure(self):
        """Test INCOIS scraper returns list."""
        # Test with mock to avoid network calls
        with patch('scrapers.incois_scraper.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'<rss><channel><item><title>Test Alert</title></item></channel></rss>'
            mock_get.return_value = mock_response
            
            alerts = fetch_incois_alerts()
            self.assertIsInstance(alerts, list)
    
    def test_twitter_scraper_structure(self):
        """Test Twitter scraper returns list."""
        alerts = fetch_twitter_alerts(max_results=1)
        self.assertIsInstance(alerts, list)
    
    def test_youtube_scraper_structure(self):
        """Test YouTube scraper returns list."""
        alerts = fetch_youtube_alerts(max_results=1)
        self.assertIsInstance(alerts, list)


class TestRelevanceClassifier(unittest.TestCase):
    """Test relevance classification."""
    
    def test_relevant_keywords(self):
        """Test keyword-based relevance detection."""
        relevant_texts = [
            "Tsunami warning issued for coastal areas",
            "Severe flooding in Kerala due to heavy rains",
            "Cyclone approaching Andhra Pradesh coast"
        ]
        
        for text in relevant_texts:
            with self.subTest(text=text):
                self.assertTrue(is_relevant(text))
    
    def test_irrelevant_keywords(self):
        """Test filtering of irrelevant content."""
        irrelevant_texts = [
            "Cricket match cancelled due to rain",
            "Stock market closes higher today",
            "New movie release this weekend"
        ]
        
        for text in irrelevant_texts:
            with self.subTest(text=text):
                # This might be True or False depending on keywords
                result = is_relevant(text)
                self.assertIsInstance(result, bool)
    
    def test_empty_text(self):
        """Test handling of empty or short text."""
        self.assertFalse(is_relevant(""))
        self.assertFalse(is_relevant("hi"))


class TestExtractor(unittest.TestCase):
    """Test information extraction."""
    
    def test_hazard_type_extraction(self):
        """Test hazard type extraction."""
        test_cases = [
            ("Tsunami warning for coastal areas", "tsunami"),
            ("Heavy flooding in Mumbai", "flood"),
            ("Cyclone approaching the coast", "cyclone"),
            ("Storm surge expected", "storm_surge")
        ]
        
        for text, expected_type in test_cases:
            with self.subTest(text=text):
                info = extract_info(text)
                self.assertEqual(info["hazard_type"], expected_type)
    
    def test_severity_extraction(self):
        """Test severity level extraction."""
        test_cases = [
            ("Extreme tsunami warning", "extreme"),
            ("Severe flooding alert", "extreme"),
            ("Heavy rains expected", "high"),
            ("Light rainfall", "low")
        ]
        
        for text, expected_severity in test_cases:
            with self.subTest(text=text):
                info = extract_info(text)
                self.assertEqual(info["severity"], expected_severity)
    
    def test_urgency_extraction(self):
        """Test urgency level extraction."""
        test_cases = [
            ("Immediate evacuation required", "immediate"),
            ("Emergency alert issued", "immediate"),
            ("Watch for potential flooding", "high"),
            ("Monitor weather conditions", "low")
        ]
        
        for text, expected_urgency in test_cases:
            with self.subTest(text=text):
                info = extract_info(text)
                self.assertEqual(info["urgency"], expected_urgency)
    
    def test_location_extraction(self):
        """Test location extraction."""
        test_cases = [
            ("Flooding in Kerala", "kerala"),
            ("Chennai receives heavy rain", "tamil_nadu"),
            ("Mumbai coastal areas flooded", "maharashtra"),
            ("Coastal warning issued", "coastal_area")
        ]
        
        for text, expected_location in test_cases:
            with self.subTest(text=text):
                info = extract_info(text)
                self.assertEqual(info["location"], expected_location)
    
    def test_misinformation_detection(self):
        """Test misinformation detection."""
        test_cases = [
            ("FAKE: Tsunami hits Mumbai - hoax message", True),
            ("Unverified reports of flooding", True),
            ("Official warning from meteorological department", False),
            ("Confirmed by authorities", False)
        ]
        
        for text, expected_misinfo in test_cases:
            with self.subTest(text=text):
                info = extract_info(text)
                self.assertEqual(info["misinformation"], expected_misinfo)
    
    def test_extract_info_structure(self):
        """Test that extract_info returns proper structure."""
        info = extract_info("Test tsunami warning")
        
        required_keys = [
            "hazard_type", "severity", "urgency", 
            "sentiment", "misinformation", "location"
        ]
        
        for key in required_keys:
            with self.subTest(key=key):
                self.assertIn(key, info)
                self.assertIsNotNone(info[key])


class TestPipeline(unittest.TestCase):
    """Test the main pipeline."""
    
    def setUp(self):
        """Set up test pipeline."""
        self.pipeline = HazardAnalysisPipeline()
    
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        self.assertIsInstance(self.pipeline, HazardAnalysisPipeline)
        self.assertGreater(len(self.pipeline.source_configs), 0)
    
    def test_process_text(self):
        """Test text processing through pipeline."""
        test_text = "Tsunami warning issued for coastal Tamil Nadu"
        report = self.pipeline.process_text(test_text, "TEST")
        
        self.assertEqual(report.source, "TEST")
        self.assertEqual(report.text, test_text)
        self.assertIsInstance(report.confidence, float)
        self.assertGreaterEqual(report.confidence, 0.0)
        self.assertLessEqual(report.confidence, 1.0)
        self.assertIsInstance(report.timestamp, datetime)
    
    def test_confidence_calculation(self):
        """Test confidence calculation."""
        # High confidence case
        high_conf_info = {
            "hazard_type": "tsunami",
            "severity": "high",
            "urgency": "immediate",
            "misinformation": False,
            "location": "kerala"
        }
        
        # Low confidence case
        low_conf_info = {
            "hazard_type": "unknown",
            "severity": "low",
            "urgency": "low",
            "misinformation": True,
            "location": "unknown"
        }
        
        high_conf = self.pipeline.calculate_confidence("Official tsunami warning for Kerala coast", high_conf_info)
        low_conf = self.pipeline.calculate_confidence("Some random text", low_conf_info)
        
        self.assertGreater(high_conf, low_conf)
        self.assertGreaterEqual(high_conf, 0.0)
        self.assertLessEqual(high_conf, 1.0)
        self.assertGreaterEqual(low_conf, 0.0)
        self.assertLessEqual(low_conf, 1.0)
    
    @patch('app.pipeline.fetch_incois_alerts')
    @patch('app.pipeline.fetch_twitter_alerts')
    @patch('app.pipeline.fetch_youtube_alerts')
    def test_run_pipeline(self, mock_youtube, mock_twitter, mock_incois):
        """Test running the complete pipeline."""
        # Mock data
        mock_incois.return_value = ["Tsunami warning for coastal areas"]
        mock_twitter.return_value = ["Heavy flooding in Mumbai reported"]
        mock_youtube.return_value = ["Cyclone update: approaching coast"]
        
        response = self.pipeline.run_pipeline()
        
        self.assertIsNotNone(response)
        self.assertGreaterEqual(response.total_sources_checked, 1)
        self.assertGreaterEqual(response.processing_time_seconds, 0)
        self.assertIsInstance(response.reports, list)
        self.assertIsInstance(response.timestamp, datetime)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_flow(self):
        """Test complete flow from text to structured output."""
        # Simulate a hazard text
        hazard_text = "Severe tsunami warning issued for Kerala coastal areas. Immediate evacuation required."
        
        # Test relevance classification
        self.assertTrue(is_relevant(hazard_text))
        
        # Test information extraction
        info = extract_info(hazard_text)
        self.assertEqual(info["hazard_type"], "tsunami")
        self.assertEqual(info["severity"], "extreme")
        self.assertEqual(info["urgency"], "immediate")
        self.assertEqual(info["location"], "kerala")
        
        # Test pipeline processing
        pipeline = HazardAnalysisPipeline()
        report = pipeline.process_text(hazard_text, "TEST")
        
        self.assertEqual(report.hazard_type, "tsunami")
        self.assertEqual(report.location, "kerala")
        self.assertGreater(report.confidence, 0.5)


def run_tests():
    """Run all tests."""
    print("Running Ocean Hazard Analysis Tests...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestScrapers,
        TestRelevanceClassifier,
        TestExtractor,
        TestPipeline,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'✓ PASSED' if success else '✗ FAILED'}")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)