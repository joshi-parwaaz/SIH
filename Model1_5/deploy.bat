@echo off
REM Docker deployment script for Ocean Hazard Analysis System (Windows)

echo 🌊 Ocean Hazard Analysis System - Docker Deployment
echo ==================================================

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Parse command line arguments
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=start

set PROFILE=dev
if "%2"=="--prod" set PROFILE=prod
if "%2"=="--dev" set PROFILE=dev

if "%PROFILE%"=="prod" (
    set COMPOSE_PROFILES=--profile production --profile cache --profile database
    echo 🚀 Using production profile ^(nginx + redis + postgres^)
) else (
    set COMPOSE_PROFILES=
    echo 🔧 Using development profile ^(api only^)
)

REM Execute commands
if "%COMMAND%"=="start" goto start
if "%COMMAND%"=="stop" goto stop
if "%COMMAND%"=="restart" goto restart
if "%COMMAND%"=="logs" goto logs
if "%COMMAND%"=="status" goto status
if "%COMMAND%"=="clean" goto clean
if "%COMMAND%"=="build" goto build
if "%COMMAND%"=="help" goto help
goto unknown

:start
echo 📦 Starting Ocean Hazard Analysis System...
docker-compose up -d %COMPOSE_PROFILES%
echo ✅ System started successfully!
echo.
echo 🌐 Available endpoints:
if "%PROFILE%"=="prod" (
    echo   Main API: http://localhost
    echo   Direct API: http://localhost:8000
) else (
    echo   API: http://localhost:8000
)
echo   API Docs: http://localhost:8000/docs
echo   Health Check: http://localhost:8000/health
echo.
echo 📊 Check status: deploy.bat status
echo 📋 View logs: deploy.bat logs
goto end

:stop
echo 🛑 Stopping Ocean Hazard Analysis System...
docker-compose down
echo ✅ System stopped successfully!
goto end

:restart
echo 🔄 Restarting Ocean Hazard Analysis System...
docker-compose down
docker-compose up -d %COMPOSE_PROFILES%
echo ✅ System restarted successfully!
goto end

:logs
set SERVICE=%2
if "%SERVICE%"=="" set SERVICE=hazard-api
echo 📋 Showing logs for %SERVICE%...
docker-compose logs -f %SERVICE%
goto end

:status
echo 📊 Container Status:
docker-compose ps
echo.
echo 🔍 Health Check:
docker-compose exec hazard-api curl -s http://localhost:8000/health
goto end

:clean
echo 🧹 Cleaning up Docker containers and volumes...
docker-compose down -v --remove-orphans
docker system prune -f
echo ✅ Cleanup completed!
goto end

:build
echo 🔨 Building Docker image...
docker-compose build --no-cache
echo ✅ Build completed!
goto end

:help
echo Usage: deploy.bat [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   start     Start the application ^(default^)
echo   stop      Stop the application
echo   restart   Restart the application
echo   logs      Show application logs
echo   status    Show container status
echo   clean     Clean up containers and volumes
echo   build     Build the Docker image
echo   help      Show this help message
echo.
echo Options:
echo   --prod    Use production profile ^(includes nginx, redis, postgres^)
echo   --dev     Use development profile ^(api only^)
echo.
echo Examples:
echo   deploy.bat start --dev          Start in development mode
echo   deploy.bat start --prod         Start with all services
echo   deploy.bat logs hazard-api      Show API logs
echo   deploy.bat clean                Clean up everything
goto end

:unknown
echo ❌ Unknown command: %COMMAND%
echo Use 'deploy.bat help' for usage information.
goto end

:end
pause