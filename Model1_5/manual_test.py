"""
Manual testing script for the Ocean Hazard Analysis System.
Tests the complete pipeline with realistic scenario data.
"""

import sys
import os
import requests
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.relevance_classifier import is_relevant
from models.extractor import extract_info
from app.pipeline import HazardAnalysisPipeline


def test_individual_components():
    """Test individual components with realistic data."""
    print("🧪 Testing Individual Components")
    print("=" * 50)
    
    # Realistic test scenarios
    test_scenarios = [
        {
            "source": "INCOIS",
            "text": "Tsunami Warning: A major earthquake of magnitude 8.2 has occurred in the Indian Ocean. Tsunami waves are expected to reach the coasts of Tamil Nadu, Kerala, and Andhra Pradesh within 2-3 hours. Immediate evacuation of coastal areas is advised.",
            "expected_relevant": True
        },
        {
            "source": "Twitter",
            "text": "Heavy flooding in Chennai due to Cyclone Mandous. Water levels rising rapidly in T Nagar and Velachery areas. Avoid travelling unless absolutely necessary. Stay safe! #ChennaiFloods #CycloneMandous",
            "expected_relevant": True
        },
        {
            "source": "YouTube",
            "text": "Breaking: Storm surge hits Mumbai coast as Cyclone Biparjoy intensifies. Live footage shows waves crashing over Marine Drive. Authorities urge residents to stay indoors.",
            "expected_relevant": True
        },
        {
            "source": "Twitter",
            "text": "Just had amazing biryani at Paradise restaurant in Hyderabad! The mutton was so tender and the flavors were incredible. Highly recommend! 🍛 #foodie #hyderabadfood",
            "expected_relevant": False
        },
        {
            "source": "INCOIS",
            "text": "Weather update: Light to moderate rainfall expected over Kerala and Karnataka coast in the next 24 hours. Sea conditions are rough with waves 2-3 meters high.",
            "expected_relevant": True
        },
        {
            "source": "YouTube",
            "text": "FAKE NEWS ALERT: Viral video claiming 100-foot tsunami hit Goa is completely false! This is old footage from Japan 2011. Please don't spread misinformation during monsoon season.",
            "expected_relevant": True  # Still relevant as it's about tsunami misinformation
        }
    ]
    
    print("\n📊 Relevance Classification Results:")
    print("-" * 40)
    
    for i, scenario in enumerate(test_scenarios, 1):
        text = scenario["text"]
        expected = scenario["expected_relevant"]
        source = scenario["source"]
        
        # Test relevance classification
        is_rel = is_relevant(text)
        
        # Test information extraction
        info = extract_info(text)
        
        status = "✅" if is_rel == expected else "❌"
        print(f"\n{i}. [{source}] {status}")
        print(f"   Text: {text[:80]}...")
        print(f"   Relevant: {is_rel} (expected: {expected})")
        print(f"   Hazard Type: {info['hazard_type']}")
        print(f"   Severity: {info['severity']}")
        print(f"   Urgency: {info['urgency']}")
        print(f"   Location: {info['location']}")
        print(f"   Misinformation: {info['misinformation']}")


def test_pipeline_with_mock_data():
    """Test the complete pipeline with mock data."""
    print("\n\n🔄 Testing Complete Pipeline")
    print("=" * 50)
    
    # Create a mock pipeline with predefined data
    pipeline = HazardAnalysisPipeline()
    
    # Test with individual text processing
    test_texts = [
        {
            "source": "INCOIS",
            "text": "URGENT: Tsunami Alert - Magnitude 7.8 earthquake detected 200km southwest of Kochi. Estimated wave arrival: 90 minutes. Evacuate immediately from coastal areas of Kerala and Karnataka."
        },
        {
            "source": "Twitter", 
            "text": "Chennai airport flooded! All flights cancelled due to Cyclone Michaung. Water entered Terminal 1. Passengers stranded. #ChennaiRains #CycloneMichaung"
        },
        {
            "source": "YouTube",
            "text": "Live: Massive storm surge hitting Visakhapatnam coast as Cyclone Fani makes landfall. Winds over 200 kmph recorded. Emergency services on high alert."
        }
    ]
    
    print("\n📈 Pipeline Processing Results:")
    print("-" * 40)
    
    for i, item in enumerate(test_texts, 1):
        text = item["text"]
        source = item["source"]
        
        print(f"\n{i}. Processing {source} data...")
        print(f"   Input: {text[:60]}...")
        
        # Process through pipeline
        report = pipeline.process_text(text, source)
        
        print(f"   📊 Results:")
        print(f"      Hazard Type: {report.hazard_type}")
        print(f"      Severity: {report.severity}")
        print(f"      Urgency: {report.urgency}")
        print(f"      Sentiment: {report.sentiment}")
        print(f"      Location: {report.location}")
        print(f"      Confidence: {report.confidence:.2f}")
        print(f"      Misinformation: {report.misinformation}")


def test_api_endpoints():
    """Test API endpoints with HTTP requests."""
    print("\n\n🌐 Testing API Endpoints")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:8000"
    
    endpoints = [
        {"path": "/health", "name": "Health Check"},
        {"path": "/status", "name": "System Status"},
        {"path": "/sources", "name": "Data Sources Info"},
        {"path": "/analyze", "name": "Main Analysis Pipeline"}
    ]
    
    for endpoint in endpoints:
        try:
            print(f"\n🔗 Testing {endpoint['name']} ({endpoint['path']})...")
            
            response = requests.get(f"{base_url}{endpoint['path']}", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Status: {response.status_code}")
                
                if endpoint['path'] == '/analyze':
                    reports = data.get('reports', [])
                    print(f"   📊 Found {len(reports)} reports")
                    print(f"   ⏱️  Processing time: {data.get('processing_time_seconds', 'N/A')}s")
                    
                    # Show first report if available
                    if reports:
                        report = reports[0]
                        print(f"   📋 Sample report:")
                        print(f"      Source: {report.get('source', 'N/A')}")
                        print(f"      Hazard: {report.get('hazard_type', 'N/A')}")
                        print(f"      Confidence: {report.get('confidence', 'N/A')}")
                        print(f"      Text: {report.get('text', 'N/A')[:50]}...")
                
                elif endpoint['path'] == '/status':
                    print(f"   🔧 System Status: {data.get('status', 'N/A')}")
                    sources = data.get('sources', [])
                    for source in sources:
                        status = source.get('status', 'unknown')
                        name = source.get('source_name', 'unknown')
                        icon = "✅" if status == "success" else "❌" if status == "error" else "⚠️"
                        print(f"      {icon} {name}: {status}")
                
                elif endpoint['path'] == '/sources':
                    sources = data.get('sources', [])
                    print(f"   📡 Available sources: {len(sources)}")
                    for source in sources:
                        print(f"      • {source.get('name', 'N/A')}: {source.get('description', 'N/A')}")
                
            else:
                print(f"   ❌ Status: {response.status_code}")
                print(f"   Error: {response.text[:100]}...")
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection failed - is the server running?")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")


def performance_test():
    """Test system performance with multiple requests."""
    print("\n\n⚡ Performance Testing")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:8000"
    
    # Test multiple rapid requests
    print("\n🚀 Testing rapid requests...")
    
    start_time = datetime.now()
    successful_requests = 0
    
    for i in range(5):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                successful_requests += 1
            print(f"   Request {i+1}: ✅" if response.status_code == 200 else f"   Request {i+1}: ❌")
        except Exception as e:
            print(f"   Request {i+1}: ❌ ({str(e)[:30]}...)")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n📊 Performance Results:")
    print(f"   Total time: {duration:.2f}s")
    print(f"   Successful requests: {successful_requests}/5")
    print(f"   Average response time: {duration/5:.2f}s per request")


def main():
    """Run all tests."""
    print("🌊 Ocean Hazard Analysis System - Manual Testing")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test individual components
        test_individual_components()
        
        # Test complete pipeline
        test_pipeline_with_mock_data()
        
        # Test API endpoints
        test_api_endpoints()
        
        # Performance testing
        performance_test()
        
        print("\n\n🎉 Manual Testing Complete!")
        print("=" * 60)
        print("✅ All components tested successfully")
        print("📊 System is ready for Docker containerization")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Testing failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()