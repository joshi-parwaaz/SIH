@echo off
REM Ocean Hazard Platform - Model A Stop Script for Windows

echo Stopping Ocean Hazard Platform - Model A...

REM Stop all services
docker-compose down

echo ✅ All services stopped successfully.
echo.
echo To start again, run: scripts\start.bat