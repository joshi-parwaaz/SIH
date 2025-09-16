@echo off
REM Ocean Hazard Prediction System - Start Script for Windows
echo Starting Ocean Hazard Prediction System...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker Compose is not installed. Please install Docker Compose.
    pause
    exit /b 1
)

REM Create necessary directories
echo Creating necessary directories...
mkdir logs 2>nul
mkdir data\models 2>nul
mkdir data\exports 2>nul
mkdir data\maps 2>nul
mkdir notebooks 2>nul
mkdir nginx\logs 2>nul
mkdir monitoring 2>nul

REM Check for environment file
if not exist .env (
    echo Creating default .env file...
    (
        echo # Ocean Hazard Prediction System Environment Variables
        echo ENVIRONMENT=production
        echo LOG_LEVEL=INFO
        echo API_HOST=0.0.0.0
        echo API_PORT=8000
        echo.
        echo # Database Configuration
        echo MONGODB_URI=mongodb://admin:ocean_hazard_2024@mongo:27017/ocean_hazard_db?authSource=admin
        echo REDIS_URL=redis://:ocean_hazard_2024@redis:6379/0
        echo.
        echo # API Keys (replace with actual keys^)
        echo WEATHER_API_KEY=your_weather_api_key_here
        echo EARTHQUAKE_API_KEY=your_earthquake_api_key_here
        echo SOCIAL_MEDIA_API_KEY=your_social_media_api_key_here
        echo.
        echo # Email Configuration
        echo SMTP_HOST=smtp.gmail.com
        echo SMTP_PORT=587
        echo SMTP_USERNAME=your_email@example.com
        echo SMTP_PASSWORD=your_app_password
        echo.
        echo # Webhook Configuration
        echo WEBHOOK_SECRET=your_webhook_secret_here
        echo.
        echo # Security
        echo JWT_SECRET_KEY=your_jwt_secret_key_here
        echo ENCRYPTION_KEY=your_encryption_key_here
    ) > .env
    echo Please edit .env file with your actual API keys and configuration
)

REM Pull latest images
echo Pulling latest Docker images...
docker-compose pull

REM Build and start services
echo Building and starting services...
docker-compose up -d --build

REM Wait for services to be ready
echo Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Check service health
echo Checking service health...
docker-compose ps ocean-hazard-api | find "Up" >nul
if errorlevel 1 (
    echo Error: ocean-hazard-api failed to start
    docker-compose logs ocean-hazard-api
) else (
    echo √ ocean-hazard-api is running
)

docker-compose ps ocean-hazard-mongo | find "Up" >nul
if errorlevel 1 (
    echo Error: ocean-hazard-mongo failed to start
    docker-compose logs ocean-hazard-mongo
) else (
    echo √ ocean-hazard-mongo is running
)

docker-compose ps ocean-hazard-redis | find "Up" >nul
if errorlevel 1 (
    echo Error: ocean-hazard-redis failed to start
    docker-compose logs ocean-hazard-redis
) else (
    echo √ ocean-hazard-redis is running
)

REM Show service URLs
echo.
echo === Ocean Hazard Prediction System Started ===
echo API Documentation: http://localhost:8000/docs
echo API Health Check: http://localhost:8000/health
echo Grafana Dashboard: http://localhost:3000 (admin/ocean_hazard_2024)
echo Jupyter Notebook: http://localhost:8888 (token: ocean_hazard_2024)
echo Prometheus: http://localhost:9090
echo.
echo To view logs: docker-compose logs -f
echo To stop system: scripts\stop.bat
echo ================================================
pause