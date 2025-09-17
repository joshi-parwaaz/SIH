"""
Pydantic models for clean JSON schema definition.
Defines the structure for hazard reports and API responses.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional


class HazardReport(BaseModel):
    """Individual hazard report from a single source."""
    
    source: str = Field(..., description="Data source (INCOIS, Twitter, YouTube)")
    text: str = Field(..., description="Original text content")
    hazard_type: str = Field(..., description="Type of hazard detected")
    severity: str = Field(..., description="Severity level (low, medium, high, extreme)")
    urgency: str = Field(..., description="Urgency level (low, medium, high, immediate)")
    sentiment: str = Field(..., description="Emotional sentiment (panic, concern, calm, neutral)")
    misinformation: bool = Field(..., description="Whether misinformation is detected")
    location: str = Field(..., description="Detected location")
    confidence: float = Field(default=0.5, description="Confidence score (0-1)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the report was processed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PipelineResponse(BaseModel):
    """Response from the analysis pipeline."""
    
    reports: List[HazardReport] = Field(..., description="List of hazard reports")
    total_sources_checked: int = Field(..., description="Number of sources checked")
    relevant_reports: int = Field(..., description="Number of relevant reports found")
    processing_time_seconds: float = Field(..., description="Time taken to process")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the analysis was performed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SourceStatus(BaseModel):
    """Status of individual data sources."""
    
    source_name: str = Field(..., description="Name of the data source")
    status: str = Field(..., description="Status (success, error, disabled)")
    items_fetched: int = Field(default=0, description="Number of items fetched")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemStatus(BaseModel):
    """Overall system status information."""
    
    status: str = Field(..., description="Overall system status")
    sources: List[SourceStatus] = Field(..., description="Status of individual sources")
    total_reports_processed: int = Field(..., description="Total reports processed in session")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    version: str = Field(default="1.5", description="Model version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Status check time")
    
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