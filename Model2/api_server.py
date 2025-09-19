"""
FastAPI server that wraps the production pipeline for backend compatibility.
Maintains the same API endpoints as the previous version.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import time
import asyncio

# Import our production pipeline
from production_pipeline import run_production_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Enhanced Disaster Management API",
    description="Production-ready disaster management system with enhanced ML capabilities",
    version="2.0.0"
)

# Add CORS middleware for backend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
startup_time = time.time()
total_requests = 0

class AnalysisResponse(BaseModel):
    """API response format compatible with existing backend."""
    reports: List[Dict[str, Any]]
    total_sources_checked: int
    relevant_reports: int
    processing_time_seconds: float
    timestamp: str
    status: str = "success"
    version: str = "2.0.0"
    region_filter: Optional[str] = None  # Added region filter info

class AnalysisRequest(BaseModel):
    """Request model for analysis with optional region parameter."""
    query: Optional[str] = "disaster alert"
    region: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    uptime_seconds: float
    version: str
    total_requests: int

class ErrorResponse(BaseModel):
    """Error response format."""
    status: str = "error"
    error: str
    timestamp: str


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint - API info."""
    return {
        "service": "Enhanced Disaster Management API",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - same as before for backend compatibility."""
    global total_requests
    total_requests += 1
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=time.time() - startup_time,
        version="2.0.0",
        total_requests=total_requests
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_hazards_post(request: AnalysisRequest):
    """
    Main analysis endpoint (POST) - enhanced version with region filtering.
    
    Accepts JSON payload with optional region parameter for targeted analysis.
    """
    return await _run_analysis(request.region)

@app.get("/analyze", response_model=AnalysisResponse)
async def analyze_hazards_get(region: Optional[str] = None):
    """
    Main analysis endpoint (GET) - enhanced version with region filtering.
    
    Accepts optional region query parameter for targeted analysis.
    """
    return await _run_analysis(region)

async def _run_analysis(region: Optional[str] = None):
    """
    Internal method to run analysis with optional region filtering.
    
    Args:
        region: Optional region name for filtering (e.g., 'Odisha', 'Tamil Nadu')
    """
    global total_requests
    total_requests += 1
    
    try:
        if region:
            logger.info(f"🚀 Starting enhanced analysis via API for region: {region}")
        else:
            logger.info("🚀 Starting enhanced analysis via API (all regions)")
        
        # Run the production pipeline with region parameter
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: run_production_pipeline(region=region)
        )
        
        # Convert to API-compatible format
        api_reports = []
        for report in result.reports:
            api_reports.append({
                "source": report.source,
                "text": report.text,
                "hazard_type": report.hazard_type,
                "severity": report.severity,
                "urgency": report.urgency,
                "sentiment": report.sentiment,
                "misinformation": report.misinformation,
                "location": report.location,
                "confidence": report.confidence,
                "timestamp": report.timestamp.isoformat()
            })
        
        logger.info(f"✅ API analysis completed: {len(api_reports)} reports in {result.processing_time_seconds:.2f}s")
        
        return AnalysisResponse(
            reports=api_reports,
            total_sources_checked=result.total_sources_checked,
            relevant_reports=result.relevant_reports,
            processing_time_seconds=result.processing_time_seconds,
            timestamp=result.timestamp.isoformat(),
            status="success",
            version="2.0.0",
            region_filter=region
        )
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=str(e),
                timestamp=datetime.now().isoformat()
            ).dict()
        )


@app.get("/status", response_model=Dict[str, Any])
async def system_status():
    """System status endpoint - backward compatibility."""
    return {
        "status": "operational",
        "version": "2.0.0",
        "uptime_seconds": time.time() - startup_time,
        "total_requests": total_requests,
        "sources": [
            "INCOIS", "Twitter", "YouTube", "Government_Sources", "Google_News"
        ],
        "ml_components": [
            "Enhanced NLP Classifier",
            "Enhanced Anomaly Detection", 
            "Enhanced Misinformation Detection"
        ],
        "capabilities": {
            "real_time_analysis": True,
            "multi_source_aggregation": True,
            "ml_enhanced_classification": True,
            "anomaly_detection": True,
            "misinformation_detection": True
        }
    }


@app.get("/api/analyze", response_model=AnalysisResponse)
async def api_analyze(region: Optional[str] = None):
    """Alternative endpoint path for backward compatibility."""
    return await _run_analysis(region)


# Legacy endpoints for backward compatibility
@app.post("/api/hazard-analysis", response_model=AnalysisResponse)
async def legacy_analyze_post(request: AnalysisRequest):
    """Legacy endpoint name for backward compatibility."""
    return await _run_analysis(request.region)

@app.get("/api/hazard-analysis", response_model=AnalysisResponse)
async def legacy_analyze_get(region: Optional[str] = None):
    """Legacy endpoint name for backward compatibility."""
    return await _run_analysis(region)


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "error": "Endpoint not found",
            "available_endpoints": ["/", "/health", "/analyze", "/status"],
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error", 
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Enhanced Disaster Management API Server")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )