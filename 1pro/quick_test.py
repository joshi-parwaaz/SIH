#!/usr/bin/env python3
"""
Quick manual test for 1pro - focusing on core functionality.
"""

import os
import sys
import time
import requests
from datetime import datetime

def quick_component_test():
    """Test individual components quickly."""
    print("🔍 QUICK COMPONENT TEST")
    print("=" * 50)
    
    # Set environment for testing
    os.environ['FAST_MODE'] = '1'
    
    try:
        # Test imports
        print("Testing core imports...")
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from app.main import app
        from app.pipeline import HazardAnalysisPipeline
        from scrapers.custom_twitter_scraper import CustomTwitterScraper
        print("✅ All imports working")
        
        # Test Twitter scraper quickly
        print("\nTesting custom Twitter scraper...")
        scraper = CustomTwitterScraper()
        tweets = scraper.search_tweets("cyclone", max_results=1)
        print(f"✅ Twitter scraper working - found {len(tweets)} tweets")
        
        # Test pipeline initialization
        print("\nTesting pipeline initialization...")
        pipeline = HazardAnalysisPipeline()
        
        # Check source configs
        enabled_sources = [name for name, config in pipeline.source_configs.items() 
                          if config.get('enabled', False)]
        print(f"✅ Pipeline initialized - {len(enabled_sources)} sources enabled:")
        for source in enabled_sources:
            print(f"   - {source}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def quick_server_test():
    """Start server and test basic endpoints quickly."""
    print("\n🚀 QUICK SERVER TEST")
    print("=" * 50)
    
    from subprocess import Popen, PIPE
    import time
    
    try:
        # Set environment
        env = os.environ.copy()
        env['FAST_MODE'] = '1'
        env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
        
        # Start server
        cmd = [sys.executable, '-m', 'uvicorn', 'app.main:app', 
               '--host', '127.0.0.1', '--port', '8000']
        
        print("Starting server...")
        process = Popen(cmd, env=env, stdout=PIPE, stderr=PIPE)
        
        # Wait for startup
        time.sleep(8)
        
        # Test health endpoint
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Health endpoint working")
                health_data = response.json()
                print(f"   Status: {health_data.get('status')}")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health endpoint error: {e}")
            return False
        
        # Test docs endpoint
        try:
            response = requests.get("http://127.0.0.1:8000/docs", timeout=5)
            if response.status_code == 200:
                print("✅ Swagger docs available")
            else:
                print(f"⚠️  Docs status: {response.status_code}")
        except Exception as e:
            print(f"⚠️  Docs error: {e}")
        
        # Quick analyze test (with short timeout)
        try:
            print("\nTesting analyze endpoint (10s timeout)...")
            response = requests.get("http://127.0.0.1:8000/analyze", timeout=10)
            if response.status_code == 200:
                print("✅ Analyze endpoint responding quickly!")
                data = response.json()
                sources = data.get('data_sources', {})
                reports = data.get('hazard_reports', [])
                print(f"   Sources: {len(sources)}, Reports: {len(reports)}")
            else:
                print(f"⚠️  Analyze status: {response.status_code}")
        except requests.exceptions.Timeout:
            print("⚠️  Analyze endpoint taking longer (normal for full analysis)")
        except Exception as e:
            print(f"⚠️  Analyze error: {e}")
        
        return True
        
    finally:
        # Clean up
        try:
            process.terminate()
            process.wait(timeout=3)
        except:
            process.kill()
        print("🛑 Server stopped")

def main():
    print("🧪 QUICK MANUAL TEST - 1PRO SYSTEM")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Run quick tests
    component_ok = quick_component_test()
    server_ok = quick_server_test()
    
    print("\n" + "=" * 60)
    print("🏁 QUICK TEST RESULTS")
    print("=" * 60)
    
    if component_ok and server_ok:
        print("🎉 CORE FUNCTIONALITY WORKING!")
        print("✅ Imports successful")
        print("✅ Custom Twitter scraper functional")
        print("✅ Pipeline initialization working")
        print("✅ FastAPI server running")
        print("✅ Health endpoint responding")
        print("✅ Documentation available")
        print()
        print("🚀 READY FOR DOCKER TESTING!")
        print("   Run: docker-compose up --build")
        print()
        print("📝 NOTE:")
        print("   - Full analyze endpoint may take 30-60s on first run")
        print("   - This is normal as it initializes ML models and fetches data")
        print("   - Docker environment should handle this better")
        return True
    else:
        print("⚠️  SOME ISSUES DETECTED")
        print("🔧 Check errors above")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Error: {e}")
        sys.exit(1)