@echo off
REM Ocean Hazard Platform - Model A Startup Script for Windows

echo Starting Ocean Hazard Platform - Model A...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo Please edit .env file with your configuration before running again.
    exit /b 1
)

REM Create necessary directories
echo Creating directories...
if not exist data mkdir data
if not exist logs mkdir logs
if not exist models mkdir models
if not exist config mkdir config

REM Build and start services
echo Building and starting services...
docker-compose up -d --build

REM Wait for services to be ready
echo Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Health check
echo Checking service health...
curl -s http://localhost:8000/health > temp_health.json 2>nul
if errorlevel 1 (
    echo Health check failed. Checking logs...
    docker-compose logs --tail=50 model-a-ml
    del temp_health.json 2>nul
    exit /b 1
)

findstr "healthy" temp_health.json >nul
if errorlevel 1 (
    echo Service health check failed. Checking logs...
    docker-compose logs --tail=50 model-a-ml
    del temp_health.json 2>nul
    exit /b 1
)

del temp_health.json 2>nul

echo ✅ Model A service is healthy and ready!
echo 🌐 API available at: http://localhost:8000
echo 📚 API documentation: http://localhost:8000/docs
echo 💾 MongoDB available at: mongodb://localhost:27017

echo 🚀 Ocean Hazard Platform - Model A is now running!
echo.
echo Quick commands:
echo   View logs:        docker-compose logs -f model-a-ml
echo   Stop services:    docker-compose down
echo   Restart:          docker-compose restart
echo   Health check:     curl http://localhost:8000/health