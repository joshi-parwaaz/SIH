"""Test runner script for Ocean Hazard Prediction System."""

import pytest
import sys
import os
import coverage
from pathlib import Path

def run_tests(test_type="all", coverage_report=True, verbose=True):
    """
    Run the test suite for Ocean Hazard Prediction System.
    
    Args:
        test_type (str): Type of tests to run ("unit", "integration", "all")
        coverage_report (bool): Generate coverage report
        verbose (bool): Verbose output
    """
    
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Test configuration
    pytest_args = []
    
    if verbose:
        pytest_args.append("-v")
    
    # Select test directories based on test_type
    if test_type == "unit":
        pytest_args.append("tests/unit/")
    elif test_type == "integration":
        pytest_args.append("tests/integration/")
    elif test_type == "all":
        pytest_args.extend(["tests/unit/", "tests/integration/"])
    else:
        print(f"Invalid test type: {test_type}")
        return 1
    
    # Coverage configuration
    if coverage_report:
        pytest_args.extend([
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ])
    
    # Add other useful options
    pytest_args.extend([
        "--tb=short",  # Shorter traceback format
        "-x",          # Stop on first failure
        "--strict-markers",  # Strict marker checking
    ])
    
    print(f"Running {test_type} tests...")
    print(f"Command: pytest {' '.join(pytest_args)}")
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("✅ All tests passed!")
        if coverage_report:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("❌ Some tests failed!")
    
    return exit_code

def run_specific_test(test_path, coverage_report=False):
    """
    Run a specific test file or test function.
    
    Args:
        test_path (str): Path to test file or specific test
        coverage_report (bool): Generate coverage report
    """
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    pytest_args = ["-v", test_path]
    
    if coverage_report:
        pytest_args.extend([
            "--cov=src",
            "--cov-report=term-missing"
        ])
    
    print(f"Running specific test: {test_path}")
    exit_code = pytest.main(pytest_args)
    
    return exit_code

def run_performance_tests():
    """Run performance-specific tests."""
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    pytest_args = [
        "-v",
        "-m", "performance",  # Run only tests marked with @pytest.mark.performance
        "tests/"
    ]
    
    print("Running performance tests...")
    exit_code = pytest.main(pytest_args)
    
    return exit_code

def run_smoke_tests():
    """Run basic smoke tests to verify system functionality."""
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    pytest_args = [
        "-v",
        "-m", "smoke",  # Run only tests marked with @pytest.mark.smoke
        "tests/"
    ]
    
    print("Running smoke tests...")
    exit_code = pytest.main(pytest_args)
    
    return exit_code

def generate_test_report():
    """Generate comprehensive test report."""
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    pytest_args = [
        "tests/",
        "--html=reports/test_report.html",
        "--self-contained-html",
        "--cov=src",
        "--cov-report=html:reports/coverage",
        "--cov-report=xml:reports/coverage.xml",
        "--junit-xml=reports/junit.xml"
    ]
    
    # Create reports directory
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    print("Generating comprehensive test report...")
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("📋 Test report generated in reports/test_report.html")
        print("📊 Coverage report generated in reports/coverage/index.html")
    
    return exit_code

def validate_system():
    """Run system validation tests."""
    
    print("🔍 Running system validation...")
    
    # Check if all required modules can be imported
    try:
        from src.data_aggregation import HistoricalEvents, SensorData, GeospatialData, SocialSignals
        from src.feature_engineering import FeaturePipeline
        from src.predictive_modeling import ModelTrainer
        from src.output_generation import RiskScorer, HotspotMapper, AlertGenerator
        print("✅ All modules import successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1
    
    # Run basic functionality tests
    try:
        # Test basic instantiation
        historical = HistoricalEvents()
        sensor = SensorData()
        pipeline = FeaturePipeline()
        scorer = RiskScorer({})
        print("✅ All components instantiate successfully")
    except Exception as e:
        print(f"❌ Instantiation error: {e}")
        return 1
    
    # Run smoke tests
    smoke_exit_code = run_smoke_tests()
    
    if smoke_exit_code == 0:
        print("✅ System validation passed")
    else:
        print("❌ System validation failed")
    
    return smoke_exit_code

def main():
    """Main test runner with command line interface."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Ocean Hazard Prediction System Test Runner")
    parser.add_argument("--type", choices=["unit", "integration", "all", "performance", "smoke"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage report")
    parser.add_argument("--quiet", action="store_true", help="Quiet output")
    parser.add_argument("--specific", help="Run specific test file or function")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive test report")
    parser.add_argument("--validate", action="store_true", help="Run system validation")
    
    args = parser.parse_args()
    
    if args.validate:
        return validate_system()
    
    if args.report:
        return generate_test_report()
    
    if args.specific:
        return run_specific_test(args.specific, not args.no_coverage)
    
    if args.type == "performance":
        return run_performance_tests()
    
    if args.type == "smoke":
        return run_smoke_tests()
    
    return run_tests(
        test_type=args.type,
        coverage_report=not args.no_coverage,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)