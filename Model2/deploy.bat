@echo off
REM Enhanced Disaster Management API - Quick Deployment Script (Windows)
REM This script allows easy deployment of the new enhanced ML pipeline
REM while maintaining compatibility with existing backend integration.

echo 🚀 Enhanced Disaster Management API Deployment
echo =============================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

echo ✅ Docker and Docker Compose are available

REM Stop any existing containers
echo 🛑 Stopping any existing containers...
docker-compose down >nul 2>&1

REM Build and start the new enhanced API
echo 🔨 Building and starting enhanced disaster management API...
docker-compose up --build -d

REM Wait for service to be ready
echo ⏳ Waiting for API to be ready...
timeout /t 10 /nobreak >nul

REM Health check
echo 🔍 Checking API health...
set max_retries=30
set retry_count=0

:health_check_loop
curl -f -s http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ API is healthy and ready!
    goto api_ready
)

set /a retry_count+=1
echo ⏳ Waiting... (%retry_count%/%max_retries%)
timeout /t 2 /nobreak >nul

if %retry_count% LSS %max_retries% goto health_check_loop

echo ❌ API failed to start properly. Check logs with: docker-compose logs
pause
exit /b 1

:api_ready
REM Show API status
echo.
echo 🎉 Enhanced Disaster Management API is now running!
echo =============================================
echo 📍 API Base URL: http://localhost:8000
echo 🏥 Health Check: http://localhost:8000/health
echo 📊 Main Analysis: http://localhost:8000/analyze
echo 📈 System Status: http://localhost:8000/status
echo 📖 API Documentation: http://localhost:8000/docs
echo.
echo 🔄 Backward Compatible Endpoints:
echo    - POST/GET /api/analyze
echo    - POST/GET /api/hazard-analysis
echo.
echo 📊 Quick Test:
curl -s http://localhost:8000/health
echo.
echo.
echo 🛠️ Management Commands:
echo    - View logs: docker-compose logs -f
echo    - Stop API: docker-compose down
echo    - Restart: docker-compose restart
echo.
echo ✅ Your existing backend should work without any changes!
echo    Just point it to the same endpoints at http://localhost:8000
echo.
pause