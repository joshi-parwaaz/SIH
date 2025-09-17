#!/bin/bash
# Model1 Testing Script - Run all tests before integration

set -e  # Exit on any error

echo "🧪 Model1 Pre-Integration Testing Suite"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    case $1 in
        "INFO") echo -e "${BLUE}[INFO]${NC} $2" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $2" ;;
        "WARNING") echo -e "${YELLOW}[WARNING]${NC} $2" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $2" ;;
    esac
}

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_status "ERROR" "docker-compose is not installed or not in PATH"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_status "ERROR" "Python 3 is not installed or not in PATH"
    exit 1
fi

# Navigate to Model1 directory
cd "$(dirname "$0")/.."
print_status "INFO" "Working directory: $(pwd)"

# Phase 1: Environment Setup
print_status "INFO" "Phase 1: Setting up test environment..."

# Install test dependencies
print_status "INFO" "Installing test dependencies..."
pip3 install pytest pytest-asyncio httpx requests || {
    print_status "ERROR" "Failed to install test dependencies"
    exit 1
}

# Check if .env file exists
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        print_status "INFO" "Creating .env from .env.example"
        cp .env.example .env
    else
        print_status "WARNING" ".env file not found. Some tests may fail."
    fi
fi

# Phase 2: Start Services
print_status "INFO" "Phase 2: Starting Docker services..."

# Start the services
docker-compose up -d || {
    print_status "ERROR" "Failed to start Docker services"
    exit 1
}

print_status "INFO" "Waiting for services to be ready (60 seconds)..."
sleep 60

# Phase 3: Health Check
print_status "INFO" "Phase 3: Running health checks..."

python3 tests/health_check.py || {
    print_status "ERROR" "Health check failed"
    print_status "INFO" "Checking service logs..."
    docker-compose logs --tail=20 model-a-ml
    exit 1
}

print_status "SUCCESS" "Health check passed!"

# Phase 4: Unit Tests
print_status "INFO" "Phase 4: Running unit tests..."

if [ -f tests/test_ml_components.py ]; then
    pytest tests/test_ml_components.py -v --tb=short || {
        print_status "WARNING" "Some unit tests failed"
    }
else
    print_status "WARNING" "Unit tests not found"
fi

# Phase 5: API Tests
print_status "INFO" "Phase 5: Running API tests..."

if [ -f tests/test_api.py ]; then
    pytest tests/test_api.py -v --tb=short || {
        print_status "WARNING" "Some API tests failed"
    }
else
    print_status "WARNING" "API tests not found"
fi

# Phase 6: System Integration Tests
print_status "INFO" "Phase 6: Running system integration tests..."

python3 tests/run_system_tests.py --save || {
    print_status "ERROR" "System tests failed"
    exit 1
}

print_status "SUCCESS" "All tests completed!"

# Phase 7: Generate Report
print_status "INFO" "Phase 7: Generating test report..."

echo ""
echo "📊 Test Summary Report"
echo "====================="

# Check if test results file was created
LATEST_RESULTS=$(ls -t model1_test_results_*.json 2>/dev/null | head -n1)
if [ -n "$LATEST_RESULTS" ]; then
    python3 -c "
import json
with open('$LATEST_RESULTS', 'r') as f:
    data = json.load(f)
print(f'📊 Total Tests: {data[\"total_tests\"]}')
print(f'✅ Passed: {data[\"passed_tests\"]}')
print(f'❌ Failed: {data[\"failed_tests\"]}')
print(f'🎯 Success Rate: {data[\"success_rate\"]:.1f}%')
print(f'⏱️  Total Time: {data[\"total_time\"]:.2f}s')
print(f'🚀 Status: {data[\"system_status\"]}')
"
else
    print_status "WARNING" "Test results file not found"
fi

echo ""
echo "🔍 Next Steps:"
echo "=============="
echo "1. Review any failed tests above"
echo "2. Check docker-compose logs for detailed error information"
echo "3. If success rate > 90%, Model1 is ready for integration"
echo "4. If success rate < 90%, address the issues before integration"

# Check final status
if [ -n "$LATEST_RESULTS" ]; then
    SUCCESS_RATE=$(python3 -c "import json; data=json.load(open('$LATEST_RESULTS')); print(data['success_rate'])")
    if (( $(echo "$SUCCESS_RATE >= 90" | bc -l) )); then
        print_status "SUCCESS" "Model1 is READY for team integration! 🎉"
        exit 0
    else
        print_status "WARNING" "Model1 needs fixes before integration"
        exit 1
    fi
else
    print_status "WARNING" "Could not determine final test status"
    exit 1
fi