@echo off
REM Ocean Hazard Prediction System - Stop Script for Windows
echo Stopping Ocean Hazard Prediction System...

REM Stop all services
echo Stopping all services...
docker-compose down

REM Optional: Remove volumes (uncomment if you want to clear all data)
REM echo Removing volumes...
REM docker-compose down -v

REM Optional: Remove images (uncomment if you want to remove built images)
REM echo Removing images...
REM docker-compose down --rmi all

echo System stopped successfully.

REM Show status
echo Checking remaining containers...
docker ps | find "ocean-hazard" >nul
if errorlevel 1 (
    echo No ocean-hazard containers running.
) else (
    docker ps | find "ocean-hazard"
)

pause