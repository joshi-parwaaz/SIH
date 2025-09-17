@echo off
REM Model1 Testing Script for Windows - Run all tests before integration

echo 🧪 Model1 Pre-Integration Testing Suite
echo ========================================

REM Navigate to Model1 directory
cd /d "%~dp0\.."
echo Working directory: %CD%

REM Phase 1: Environment Setup
echo.
echo [INFO] Phase 1: Setting up test environment...

REM Install test dependencies
echo [INFO] Installing test dependencies...
pip install pytest pytest-asyncio httpx requests
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install test dependencies
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    if exist .env.example (
        echo [INFO] Creating .env from .env.example
        copy .env.example .env
    ) else (
        echo [WARNING] .env file not found. Some tests may fail.
    )
)

REM Phase 2: Start Services
echo.
echo [INFO] Phase 2: Starting Docker services...

REM Start the services
docker-compose up -d
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to start Docker services
    exit /b 1
)

echo [INFO] Waiting for services to be ready (60 seconds)...
timeout /t 60 /nobreak >nul

REM Phase 3: Health Check
echo.
echo [INFO] Phase 3: Running health checks...

python tests\health_check.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Health check failed
    echo [INFO] Checking service logs...
    docker-compose logs --tail=20 model-a-ml
    exit /b 1
)

echo [SUCCESS] Health check passed!

REM Phase 4: Unit Tests
echo.
echo [INFO] Phase 4: Running unit tests...

if exist tests\test_ml_components.py (
    pytest tests\test_ml_components.py -v --tb=short
    if %ERRORLEVEL% neq 0 (
        echo [WARNING] Some unit tests failed
    )
) else (
    echo [WARNING] Unit tests not found
)

REM Phase 5: API Tests
echo.
echo [INFO] Phase 5: Running API tests...

if exist tests\test_api.py (
    pytest tests\test_api.py -v --tb=short
    if %ERRORLEVEL% neq 0 (
        echo [WARNING] Some API tests failed
    )
) else (
    echo [WARNING] API tests not found
)

REM Phase 6: System Integration Tests
echo.
echo [INFO] Phase 6: Running system integration tests...

python tests\run_system_tests.py --save
if %ERRORLEVEL% neq 0 (
    echo [ERROR] System tests failed
    exit /b 1
)

echo [SUCCESS] All tests completed!

REM Phase 7: Generate Report
echo.
echo [INFO] Phase 7: Generating test report...

echo.
echo 📊 Test Summary Report
echo =====================

REM Find latest results file
for /f %%i in ('dir /b /od model1_test_results_*.json 2^>nul') do set LATEST_RESULTS=%%i

if defined LATEST_RESULTS (
    python -c "import json; data=json.load(open('%LATEST_RESULTS%')); print(f'📊 Total Tests: {data[\"total_tests\"]}'); print(f'✅ Passed: {data[\"passed_tests\"]}'); print(f'❌ Failed: {data[\"failed_tests\"]}'); print(f'🎯 Success Rate: {data[\"success_rate\"]:.1f}%%'); print(f'⏱️  Total Time: {data[\"total_time\"]:.2f}s'); print(f'🚀 Status: {data[\"system_status\"]}')"
) else (
    echo [WARNING] Test results file not found
)

echo.
echo 🔍 Next Steps:
echo ==============
echo 1. Review any failed tests above
echo 2. Check docker-compose logs for detailed error information
echo 3. If success rate ^> 90%%, Model1 is ready for integration
echo 4. If success rate ^< 90%%, address the issues before integration

REM Check final status
if defined LATEST_RESULTS (
    for /f %%i in ('python -c "import json; data=json.load(open('%LATEST_RESULTS%')); print(data['success_rate'])"') do set SUCCESS_RATE=%%i
    python -c "import sys; sys.exit(0 if float('%SUCCESS_RATE%') >= 90 else 1)"
    if %ERRORLEVEL% equ 0 (
        echo [SUCCESS] Model1 is READY for team integration! 🎉
        exit /b 0
    ) else (
        echo [WARNING] Model1 needs fixes before integration
        exit /b 1
    )
) else (
    echo [WARNING] Could not determine final test status
    exit /b 1
)