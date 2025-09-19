"""
Simplified Pydantic models for production pipeline.
Only includes essential data structures.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any


class HazardReport(BaseModel):
    """Individual hazard report from a single source."""
    
    source: str = Field(..., description="Data source")
    text: str = Field(..., description="Original text content")
    hazard_type: str = Field(..., description="Type of hazard detected")
    severity: str = Field(..., description="Severity level")
    urgency: str = Field(..., description="Urgency level")
    sentiment: str = Field(..., description="Emotional sentiment")
    misinformation: bool = Field(..., description="Whether misinformation is detected")
    location: str = Field(..., description="Detected location")
    confidence: float = Field(default=0.5, description="Confidence score (0-1)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When processed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PipelineResponse(BaseModel):
    """Response from the production pipeline."""
    
    reports: List[HazardReport] = Field(..., description="List of hazard reports")
    total_sources_checked: int = Field(..., description="Number of sources checked")
    relevant_reports: int = Field(..., description="Number of relevant reports found")
    processing_time_seconds: float = Field(..., description="Time taken to process")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When analysis was performed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the error occurred")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheck(BaseModel):
    """Health check response."""
    
    status: str = Field(default="healthy", description="Service health status")
    version: str = Field(default="1.5", description="Model version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }