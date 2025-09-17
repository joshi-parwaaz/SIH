"""
Automated test runner for Model1 system tests.
Run comprehensive tests to validate system readiness.
"""

import asyncio
import json
import time
import requests
import subprocess
import sys
import os
from datetime import datetime
from typing import Dict, List, Any
import threading
import concurrent.futures

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from test_data_samples import (
    MULTILINGUAL_SAMPLES, EDGE_CASES, PERFORMANCE_TEST_DATA,
    GOVERNMENT_ALERT_SAMPLES, SOCIAL_MEDIA_SAMPLES, ERROR_HANDLING_TESTS
)


class SystemTestRunner:
    """Comprehensive system test runner for Model1."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": [],
            "performance_metrics": {},
            "system_status": "unknown"
        }
    
    def log_test_result(self, test_name: str, passed: bool, details: Dict = None):
        """Log test result."""
        self.test_results["total_tests"] += 1
        if passed:
            self.test_results["passed_tests"] += 1
            status = "PASS"
        else:
            self.test_results["failed_tests"] += 1
            status = "FAIL"
        
        result = {
            "test_name": test_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.test_results["test_details"].append(result)
        
        # Print immediate feedback
        print(f"[{status}] {test_name}")
        if details and not passed:
            print(f"    Error: {details.get('error', 'Unknown error')}")
    
    def check_system_health(self) -> bool:
        """Check if the system is running and healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                self.log_test_result("System Health Check", True, health_data)
                return True
            else:
                self.log_test_result("System Health Check", False, 
                                   {"error": f"Status code: {response.status_code}"})
                return False
        except Exception as e:
            self.log_test_result("System Health Check", False, {"error": str(e)})
            return False
    
    def test_multilingual_processing(self) -> bool:
        """Test multilingual text processing."""
        print("\n=== Testing Multilingual Processing ===")
        all_passed = True
        
        for language, samples in MULTILINGUAL_SAMPLES.items():
            print(f"\nTesting {language.upper()} language:")
            
            for i, sample in enumerate(samples):
                test_name = f"Multilingual_{language}_{i+1}"
                
                try:
                    payload = {
                        "text": sample["text"],
                        "source": "test",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    response = requests.post(
                        f"{self.base_url}/api/v1/process/single",
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Validate expected results
                        passed = True
                        details = {"language": language, "sample_index": i}
                        
                        if result.get("is_hazard") != sample["expected_hazard"]:
                            passed = False
                            details["error"] = f"Hazard detection mismatch. Expected: {sample['expected_hazard']}, Got: {result.get('is_hazard')}"
                        
                        if passed and sample["expected_hazard"] and result.get("hazard_type") != sample["expected_type"]:
                            passed = False
                            details["error"] = f"Hazard type mismatch. Expected: {sample['expected_type']}, Got: {result.get('hazard_type')}"
                        
                        self.log_test_result(test_name, passed, details)
                        if not passed:
                            all_passed = False
                    else:
                        self.log_test_result(test_name, False, 
                                           {"error": f"HTTP {response.status_code}: {response.text}"})
                        all_passed = False
                        
                except Exception as e:
                    self.log_test_result(test_name, False, {"error": str(e)})
                    all_passed = False
        
        return all_passed
    
    def test_edge_cases(self) -> bool:
        """Test edge cases and error handling."""
        print("\n=== Testing Edge Cases ===")
        all_passed = True
        
        for i, case in enumerate(EDGE_CASES):
            test_name = f"EdgeCase_{case['category']}_{i+1}"
            
            try:
                payload = {
                    "text": case["text"],
                    "source": "edge_case_test",
                    "timestamp": datetime.now().isoformat()
                }
                
                response = requests.post(
                    f"{self.base_url}/api/v1/process/single",
                    json=payload,
                    timeout=15
                )
                
                # For edge cases, we mainly check that the system doesn't crash
                if response.status_code in [200, 422]:  # OK or validation error
                    self.log_test_result(test_name, True, 
                                       {"category": case["category"], "status_code": response.status_code})
                else:
                    self.log_test_result(test_name, False,
                                       {"error": f"Unexpected status code: {response.status_code}"})
                    all_passed = False
                    
            except Exception as e:
                self.log_test_result(test_name, False, {"error": str(e)})
                all_passed = False
        
        return all_passed
    
    def test_batch_processing(self) -> bool:
        """Test batch processing functionality."""
        print("\n=== Testing Batch Processing ===")
        
        try:
            # Use sample data for batch test
            batch_data = {
                "reports": [
                    {
                        "text": "Tsunami warning for Chennai coast",
                        "source": "test_batch",
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "text": "High waves at Mumbai beach",
                        "source": "test_batch",
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "text": "Beautiful weather in Goa",
                        "source": "test_batch",
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/v1/process/batch",
                json=batch_data,
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                processing_time = end_time - start_time
                
                passed = True
                details = {
                    "processing_time": processing_time,
                    "reports_processed": len(batch_data["reports"]),
                    "success_count": result.get("processed_count", 0)
                }
                
                if result.get("processed_count") != len(batch_data["reports"]):
                    passed = False
                    details["error"] = "Not all reports were processed"
                
                self.log_test_result("Batch_Processing", passed, details)
                return passed
            else:
                self.log_test_result("Batch_Processing", False,
                                   {"error": f"HTTP {response.status_code}: {response.text}"})
                return False
                
        except Exception as e:
            self.log_test_result("Batch_Processing", False, {"error": str(e)})
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks."""
        print("\n=== Testing Performance Benchmarks ===")
        
        # Test single request performance
        single_request_times = []
        for i in range(10):
            try:
                payload = {
                    "text": f"Test performance message {i}",
                    "source": "performance_test",
                    "timestamp": datetime.now().isoformat()
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/v1/process/single",
                    json=payload,
                    timeout=10
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    single_request_times.append(end_time - start_time)
                    
            except Exception as e:
                print(f"Performance test {i} failed: {e}")
        
        if single_request_times:
            avg_time = sum(single_request_times) / len(single_request_times)
            max_time = max(single_request_times)
            
            # Performance criteria: average < 2s, max < 5s
            passed = avg_time < 2.0 and max_time < 5.0
            
            details = {
                "average_response_time": avg_time,
                "max_response_time": max_time,
                "samples": len(single_request_times),
                "target_avg": 2.0,
                "target_max": 5.0
            }
            
            self.test_results["performance_metrics"]["single_request"] = details
            self.log_test_result("Performance_SingleRequest", passed, details)
            return passed
        else:
            self.log_test_result("Performance_SingleRequest", False, 
                               {"error": "No successful requests for performance testing"})
            return False
    
    def test_concurrent_load(self) -> bool:
        """Test concurrent request handling."""
        print("\n=== Testing Concurrent Load ===")
        
        def make_request(request_id):
            try:
                payload = {
                    "text": f"Concurrent test request {request_id}",
                    "source": "load_test",
                    "timestamp": datetime.now().isoformat()
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/v1/process/single",
                    json=payload,
                    timeout=15
                )
                end_time = time.time()
                
                return {
                    "success": response.status_code == 200,
                    "response_time": end_time - start_time,
                    "request_id": request_id
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "request_id": request_id
                }
        
        # Test with 10 concurrent requests
        concurrent_requests = 10
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(concurrent_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        successful_requests = [r for r in results if r["success"]]
        success_rate = len(successful_requests) / len(results)
        
        if successful_requests:
            avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
        else:
            avg_response_time = float('inf')
        
        # Criteria: >90% success rate, avg response time <3s
        passed = success_rate >= 0.9 and avg_response_time < 3.0
        
        details = {
            "concurrent_requests": concurrent_requests,
            "success_rate": success_rate,
            "successful_requests": len(successful_requests),
            "avg_response_time": avg_response_time,
            "target_success_rate": 0.9,
            "target_avg_time": 3.0
        }
        
        self.test_results["performance_metrics"]["concurrent_load"] = details
        self.log_test_result("Performance_ConcurrentLoad", passed, details)
        return passed
    
    def test_data_export(self) -> bool:
        """Test data export functionality."""
        print("\n=== Testing Data Export ===")
        
        endpoints_to_test = [
            ("/api/v1/export/geojson", "GeoJSON_Export"),
            ("/api/v1/statistics", "Statistics_Export")
        ]
        
        all_passed = True
        
        for endpoint, test_name in endpoints_to_test:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Basic validation
                    passed = True
                    details = {"endpoint": endpoint}
                    
                    if "geojson" in endpoint and data.get("type") != "FeatureCollection":
                        passed = False
                        details["error"] = "Invalid GeoJSON format"
                    elif "statistics" in endpoint and "total_reports" not in data:
                        passed = False
                        details["error"] = "Missing statistics data"
                    
                    self.log_test_result(test_name, passed, details)
                    if not passed:
                        all_passed = False
                else:
                    self.log_test_result(test_name, False,
                                       {"error": f"HTTP {response.status_code}: {response.text}"})
                    all_passed = False
                    
            except Exception as e:
                self.log_test_result(test_name, False, {"error": str(e)})
                all_passed = False
        
        return all_passed
    
    def run_all_tests(self) -> Dict:
        """Run all system tests."""
        print("🧪 Starting Model1 System Tests")
        print("=" * 50)
        
        start_time = time.time()
        
        # 1. Check system health
        if not self.check_system_health():
            print("❌ System health check failed. Cannot continue with tests.")
            self.test_results["system_status"] = "unhealthy"
            return self.test_results
        
        # 2. Run all test suites
        test_suites = [
            ("Multilingual Processing", self.test_multilingual_processing),
            ("Edge Cases", self.test_edge_cases),
            ("Batch Processing", self.test_batch_processing),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Concurrent Load", self.test_concurrent_load),
            ("Data Export", self.test_data_export)
        ]
        
        for suite_name, test_function in test_suites:
            print(f"\n🔄 Running {suite_name} tests...")
            try:
                test_function()
            except Exception as e:
                print(f"❌ Test suite {suite_name} failed with error: {e}")
                self.log_test_result(f"{suite_name}_Suite", False, {"error": str(e)})
        
        # Calculate final results
        end_time = time.time()
        total_time = end_time - start_time
        
        self.test_results["total_time"] = total_time
        self.test_results["success_rate"] = (
            self.test_results["passed_tests"] / max(self.test_results["total_tests"], 1) * 100
        )
        
        if self.test_results["success_rate"] >= 90:
            self.test_results["system_status"] = "ready_for_integration"
        elif self.test_results["success_rate"] >= 75:
            self.test_results["system_status"] = "needs_minor_fixes"
        else:
            self.test_results["system_status"] = "needs_major_fixes"
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    def print_test_summary(self):
        """Print test summary."""
        print("\n" + "=" * 50)
        print("🧪 MODEL1 SYSTEM TEST RESULTS")
        print("=" * 50)
        
        results = self.test_results
        
        print(f"📊 Overall Statistics:")
        print(f"   Total Tests: {results['total_tests']}")
        print(f"   Passed: {results['passed_tests']}")
        print(f"   Failed: {results['failed_tests']}")
        print(f"   Success Rate: {results['success_rate']:.1f}%")
        print(f"   Total Time: {results['total_time']:.2f}s")
        
        print(f"\n🎯 System Status: {results['system_status'].upper()}")
        
        if results['failed_tests'] > 0:
            print(f"\n❌ Failed Tests:")
            for test in results['test_details']:
                if test['status'] == 'FAIL':
                    print(f"   - {test['test_name']}: {test['details'].get('error', 'Unknown error')}")
        
        if 'performance_metrics' in results:
            print(f"\n⚡ Performance Metrics:")
            for metric, data in results['performance_metrics'].items():
                if 'average_response_time' in data:
                    print(f"   {metric}: {data['average_response_time']:.3f}s avg")
        
        # Integration readiness
        if results['success_rate'] >= 90:
            print(f"\n✅ Model1 is READY for team integration!")
        elif results['success_rate'] >= 75:
            print(f"\n⚠️  Model1 needs minor fixes before integration.")
        else:
            print(f"\n❌ Model1 needs significant fixes before integration.")
    
    def save_results(self, filename: str = None):
        """Save test results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model1_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n💾 Test results saved to: {filename}")


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model1 System Test Runner")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL for the API (default: http://localhost:8000)")
    parser.add_argument("--save", action="store_true", 
                       help="Save test results to JSON file")
    
    args = parser.parse_args()
    
    # Run tests
    runner = SystemTestRunner(base_url=args.url)
    results = runner.run_all_tests()
    
    # Save results if requested
    if args.save:
        runner.save_results()
    
    # Exit with appropriate code
    if results["success_rate"] >= 90:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()