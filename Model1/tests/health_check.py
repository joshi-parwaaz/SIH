#!/usr/bin/env python3
"""
Quick health check script for Model1.
Run this to verify system is running and basic functionality works.
"""

import requests
import json
import sys
from datetime import datetime


def check_health(base_url: str = "http://localhost:8000") -> bool:
    """Check if Model1 is healthy and responsive."""
    try:
        print("🔍 Checking Model1 health...")
        response = requests.get(f"{base_url}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ System is healthy!")
            print(f"   Version: {health_data.get('version', 'Unknown')}")
            print(f"   Status: {health_data.get('status', 'Unknown')}")
            print(f"   Timestamp: {health_data.get('timestamp', 'Unknown')}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Model1. Is it running?")
        print("   Try: docker-compose up -d")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_basic_functionality(base_url: str = "http://localhost:8000") -> bool:
    """Test basic functionality with a simple request."""
    try:
        print("\n🧪 Testing basic functionality...")
        
        test_data = {
            "text": "Tsunami warning for Chennai coast. Evacuate immediately.",
            "source": "health_check",
            "timestamp": datetime.now().isoformat()
        }
        
        response = requests.post(
            f"{base_url}/api/v1/process/single",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Basic processing works!")
            print(f"   Hazard detected: {result.get('is_hazard', 'Unknown')}")
            print(f"   Hazard type: {result.get('hazard_type', 'Unknown')}")
            print(f"   Processing time: {result.get('processing_time', 0):.3f}s")
            return True
        else:
            print(f"❌ Processing failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Run health check."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model1 Health Check")
    parser.add_argument("--url", default="http://localhost:8000",
                       help="Base URL for the API")
    
    args = parser.parse_args()
    
    print("🔍 Model1 Health Check")
    print("=" * 30)
    
    # Check health
    health_ok = check_health(args.url)
    
    if health_ok:
        # Test basic functionality
        basic_ok = test_basic_functionality(args.url)
        
        if basic_ok:
            print("\n✅ Model1 is ready for testing!")
            sys.exit(0)
        else:
            print("\n⚠️  Model1 is running but has functionality issues.")
            sys.exit(1)
    else:
        print("\n❌ Model1 is not healthy.")
        print("\n🛠️  Troubleshooting steps:")
        print("   1. Check if containers are running: docker-compose ps")
        print("   2. Check logs: docker-compose logs model-a-ml")
        print("   3. Restart services: docker-compose restart")
        print("   4. Check port availability: netstat -an | grep 8000")
        sys.exit(1)


if __name__ == "__main__":
    main()