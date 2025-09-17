"""
FastAPI server for the Ocean Hazard Analysis API.
Provides endpoints for hazard analysis and system monitoring.
"""

import logging
import sys
import os
from datetime import datetime

# Add the parent directory to Python path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    print("FastAPI not available. Please install with: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False
    # Create dummy app for imports
    class FastAPI:
        def __init__(self, *args, **kwargs): pass
        def get(self, *args, **kwargs): return lambda f: f
        def add_middleware(self, *args, **kwargs): pass

from app.pipeline import get_pipeline, run_pipeline
from app.schema import PipelineResponse, SystemStatus, HealthCheck, ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Ocean Hazard Analysis API",
    description="Real-time ocean hazard detection and analysis using multiple data sources",
    version="1.5",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for tracking
startup_time = datetime.utcnow()
request_count = 0


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Ocean Hazard Analysis API v1.5...")
    
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available!")
        return
    
    # Initialize the pipeline
    try:
        pipeline = get_pipeline()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
    
    logger.info("API server started successfully")


@app.get("/", response_model=HealthCheck)
async def root():
    """Root endpoint with basic health check."""
    global request_count
    request_count += 1
    
    return HealthCheck(
        status="healthy",
        version="1.5",
        timestamp=datetime.utcnow()
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    global request_count
    request_count += 1
    
    return HealthCheck(
        status="healthy",
        version="1.5",
        timestamp=datetime.utcnow()
    )


@app.get("/analyze", response_model=PipelineResponse)
async def analyze():
    """
    Run the complete hazard analysis pipeline.
    
    This endpoint:
    1. Fetches data from all configured sources (INCOIS, Twitter, YouTube)
    2. Filters for relevant content using ML classification
    3. Extracts hazard information (type, severity, urgency, etc.)
    4. Returns structured JSON with all findings
    
    Returns:
        PipelineResponse: Complete analysis results
    """
    global request_count
    request_count += 1
    
    try:
        logger.info("Starting hazard analysis...")
        
        pipeline = get_pipeline()
        result = pipeline.run_pipeline()
        
        logger.info(f"Analysis completed: {result.relevant_reports} reports found")
        return result
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline failed: {str(e)}"
        )


@app.get("/analyze/json")
async def analyze_json():
    """
    Alternative endpoint that returns raw JSON (for backward compatibility).
    
    Returns:
        dict: Pipeline results as raw JSON
    """
    global request_count
    request_count += 1
    
    try:
        logger.info("Starting hazard analysis (JSON mode)...")
        
        result = run_pipeline()
        
        logger.info(f"Analysis completed: {len(result.get('reports', []))} reports found")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "analysis_failed", "message": str(e)}
        )


@app.get("/status", response_model=SystemStatus)
async def system_status():
    """
    Get detailed system status including source health.
    
    Returns:
        SystemStatus: Detailed system status information
    """
    global request_count
    request_count += 1
    
    try:
        pipeline = get_pipeline()
        status_info = pipeline.get_system_status()
        
        # Add request count and uptime
        uptime = (datetime.utcnow() - startup_time).total_seconds()
        status_info["uptime_seconds"] = round(uptime, 2)
        status_info["total_requests"] = request_count
        
        return SystemStatus(**status_info)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )


@app.get("/sources")
async def list_sources():
    """
    Get information about available data sources.
    
    Returns:
        dict: Information about all configured sources
    """
    global request_count
    request_count += 1
    
    try:
        pipeline = get_pipeline()
        
        sources_info = {
            "sources": [
                {
                    "name": "INCOIS",
                    "description": "Indian National Centre for Ocean Information Services",
                    "type": "official",
                    "url": "https://incois.gov.in",
                    "data_type": "XML feeds",
                    "coverage": "Indian Ocean region"
                },
                {
                    "name": "Twitter",
                    "description": "Social media monitoring for hazard-related content",
                    "type": "social",
                    "data_type": "Tweets",
                    "coverage": "India (geographic filter)"
                },
                {
                    "name": "YouTube",
                    "description": "Video content analysis for hazard information",
                    "type": "media",
                    "data_type": "Video titles and descriptions",
                    "coverage": "India region"
                }
            ],
            "total_sources": 3,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return JSONResponse(content=sources_info)
        
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list sources: {str(e)}"
        )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred",
            timestamp=datetime.utcnow()
        ).dict()
    )


if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available. Please install with:")
        print("pip install fastapi uvicorn")
        sys.exit(1)
    
    import uvicorn
    
    print("Starting Ocean Hazard Analysis API...")
    print("API Documentation: http://127.0.0.1:8000/docs")
    print("Analysis endpoint: http://127.0.0.1:8000/analyze")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )