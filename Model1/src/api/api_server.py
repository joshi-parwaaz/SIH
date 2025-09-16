import asyncio
import uvicorn
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import our ML pipeline components
from ..data_ingestion import DataIngestionManager, RawReport
from ..preprocessing import PreprocessingPipeline, ProcessedReport
from ..nlp_analysis import NLPAnalysisEngine, HazardPrediction
from ..geolocation import GeolocationExtractor, GeolocationResult
from ..anomaly_detection import AnomalyDetectionEngine, AnomalyAlert
from ..feedback import FeedbackIntegrationSystem, FeedbackType, FeedbackAction
from ...config import config

# Global instances
ml_pipeline = None
data_manager = None
preprocessing_pipeline = None
nlp_engine = None
geolocation_extractor = None
anomaly_detector = None
feedback_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage ML model lifecycle"""
    global ml_pipeline, data_manager, preprocessing_pipeline, nlp_engine, geolocation_extractor, anomaly_detector, feedback_system
    
    # Startup
    print("Loading ML models...")
    
    # Initialize all components
    data_manager = DataIngestionManager()
    preprocessing_pipeline = PreprocessingPipeline()
    nlp_engine = NLPAnalysisEngine()
    geolocation_extractor = GeolocationExtractor()
    anomaly_detector = AnomalyDetectionEngine()
    feedback_system = FeedbackIntegrationSystem()
    
    # Load models
    await nlp_engine.load_models()
    await geolocation_extractor.load_models()
    
    print("All ML models loaded successfully!")
    
    yield
    
    # Shutdown
    print("Shutting down ML services...")

# Create FastAPI app
app = FastAPI(
    title="Ocean Hazard Platform - Model A API",
    description="Multilingual Hazard Detection & Extraction System",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class ReportInput(BaseModel):
    """Input model for hazard report"""
    content: str = Field(..., description="Text content of the report")
    source: str = Field(..., description="Source of the report (twitter, facebook, user_report, etc.)")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    location: Optional[Dict[str, Any]] = Field(None, description="Location metadata if available")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    media_urls: Optional[List[str]] = Field(default_factory=list)

class BatchReportInput(BaseModel):
    """Input model for batch processing"""
    reports: List[ReportInput] = Field(..., description="List of reports to process")
    process_anomalies: bool = Field(True, description="Whether to run anomaly detection")

class HazardAnalysisResponse(BaseModel):
    """Response model for hazard analysis"""
    report_id: str
    is_hazard: bool
    hazard_type: str
    confidence: float
    severity: str
    urgency: str
    sentiment: str
    sentiment_score: float
    misinformation_probability: float
    entities: List[Dict[str, Any]]
    processing_metadata: Dict[str, Any]

class GeolocationResponse(BaseModel):
    """Response model for geolocation extraction"""
    report_id: str
    extracted_locations: List[Dict[str, Any]]
    primary_location: Optional[Dict[str, Any]]
    confidence_score: float
    processing_metadata: Dict[str, Any]

class AnomalyResponse(BaseModel):
    """Response model for anomaly detection"""
    alert_id: str
    alert_type: str
    severity: str
    location: Optional[str]
    coordinates: Optional[List[float]]
    detection_time: datetime
    message: str
    metrics: Dict[str, Any]
    affected_reports: List[str]
    confidence_score: float

class FeedbackInput(BaseModel):
    """Input model for operator feedback"""
    report_id: str
    operator_id: str
    feedback_type: str
    action: str
    original_prediction: Dict[str, Any]
    corrected_prediction: Optional[Dict[str, Any]] = None
    confidence_score: float = Field(1.0, ge=0.0, le=1.0)
    notes: Optional[str] = None

class ProcessingResult(BaseModel):
    """Complete processing result"""
    report_id: str
    processed_report: Dict[str, Any]
    hazard_analysis: HazardAnalysisResponse
    geolocation: GeolocationResponse
    processing_time_ms: float
    status: str

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "models_loaded": {
            "nlp_engine": nlp_engine.is_loaded if nlp_engine else False,
            "geolocation_extractor": geolocation_extractor.is_loaded if geolocation_extractor else False
        }
    }

# Single report processing
@app.post("/analyze/single", response_model=ProcessingResult)
async def analyze_single_report(report_input: ReportInput):
    """Analyze a single hazard report"""
    start_time = datetime.now()
    
    try:
        # Create RawReport
        raw_report = RawReport(
            id=f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(report_input.content) % 10000}",
            source=report_input.source,
            content=report_input.content,
            timestamp=report_input.timestamp,
            location=report_input.location,
            metadata=report_input.metadata,
            media_urls=report_input.media_urls,
            ingested_at=datetime.now()
        )
        
        # Preprocessing
        processed_report = await preprocessing_pipeline.process_report(raw_report)
        
        # NLP Analysis
        hazard_prediction = await nlp_engine.analyze_report(processed_report)
        
        # Geolocation Extraction
        geolocation_result = await geolocation_extractor.extract_locations(
            processed_report, hazard_prediction
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ProcessingResult(
            report_id=raw_report.id,
            processed_report={
                "id": processed_report.id,
                "original_content": processed_report.original_content,
                "normalized_content": processed_report.normalized_content,
                "detected_language": processed_report.detected_language,
                "confidence_score": processed_report.confidence_score
            },
            hazard_analysis=HazardAnalysisResponse(
                report_id=hazard_prediction.report_id,
                is_hazard=hazard_prediction.is_hazard,
                hazard_type=hazard_prediction.hazard_type,
                confidence=hazard_prediction.confidence,
                severity=hazard_prediction.severity,
                urgency=hazard_prediction.urgency,
                sentiment=hazard_prediction.sentiment,
                sentiment_score=hazard_prediction.sentiment_score,
                misinformation_probability=hazard_prediction.misinformation_probability,
                entities=hazard_prediction.entities,
                processing_metadata=hazard_prediction.processing_metadata
            ),
            geolocation=GeolocationResponse(
                report_id=geolocation_result.report_id,
                extracted_locations=[{
                    "text": loc.text,
                    "label": loc.label,
                    "coordinates": loc.coordinates,
                    "confidence": loc.confidence,
                    "address": loc.address
                } for loc in geolocation_result.extracted_locations],
                primary_location={
                    "text": geolocation_result.primary_location.text,
                    "coordinates": geolocation_result.primary_location.coordinates,
                    "address": geolocation_result.primary_location.address
                } if geolocation_result.primary_location else None,
                confidence_score=geolocation_result.confidence_score,
                processing_metadata=geolocation_result.processing_metadata
            ),
            processing_time_ms=processing_time,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Batch processing
@app.post("/analyze/batch")
async def analyze_batch_reports(batch_input: BatchReportInput):
    """Analyze a batch of hazard reports"""
    start_time = datetime.now()
    
    try:
        # Convert to RawReports
        raw_reports = []
        for i, report_input in enumerate(batch_input.reports):
            raw_report = RawReport(
                id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                source=report_input.source,
                content=report_input.content,
                timestamp=report_input.timestamp,
                location=report_input.location,
                metadata=report_input.metadata,
                media_urls=report_input.media_urls,
                ingested_at=datetime.now()
            )
            raw_reports.append(raw_report)
        
        # Batch preprocessing
        processed_reports = await preprocessing_pipeline.process_batch(raw_reports)
        
        # Batch NLP analysis
        hazard_predictions = await nlp_engine.analyze_batch(processed_reports)
        
        # Batch geolocation extraction
        geolocation_results = await geolocation_extractor.extract_batch(
            processed_reports, hazard_predictions
        )
        
        # Anomaly detection if requested
        anomaly_alerts = []
        if batch_input.process_anomalies:
            anomaly_alerts = await anomaly_detector.detect_anomalies(
                processed_reports, hazard_predictions, geolocation_results
            )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "total_reports": len(raw_reports),
            "processed_reports": len(processed_reports),
            "hazard_predictions": len(hazard_predictions),
            "geolocation_results": len(geolocation_results),
            "anomaly_alerts": len(anomaly_alerts),
            "processing_time_ms": processing_time,
            "results": {
                "hazard_analyses": [
                    HazardAnalysisResponse(
                        report_id=pred.report_id,
                        is_hazard=pred.is_hazard,
                        hazard_type=pred.hazard_type,
                        confidence=pred.confidence,
                        severity=pred.severity,
                        urgency=pred.urgency,
                        sentiment=pred.sentiment,
                        sentiment_score=pred.sentiment_score,
                        misinformation_probability=pred.misinformation_probability,
                        entities=pred.entities,
                        processing_metadata=pred.processing_metadata
                    ) for pred in hazard_predictions
                ],
                "geolocation_results": [
                    GeolocationResponse(
                        report_id=geo.report_id,
                        extracted_locations=[{
                            "text": loc.text,
                            "label": loc.label,
                            "coordinates": loc.coordinates,
                            "confidence": loc.confidence,
                            "address": loc.address
                        } for loc in geo.extracted_locations],
                        primary_location={
                            "text": geo.primary_location.text,
                            "coordinates": geo.primary_location.coordinates,
                            "address": geo.primary_location.address
                        } if geo.primary_location else None,
                        confidence_score=geo.confidence_score,
                        processing_metadata=geo.processing_metadata
                    ) for geo in geolocation_results
                ],
                "anomaly_alerts": [
                    AnomalyResponse(
                        alert_id=alert.id,
                        alert_type=alert.alert_type,
                        severity=alert.severity,
                        location=alert.location,
                        coordinates=list(alert.coordinates) if alert.coordinates else None,
                        detection_time=alert.detection_time,
                        message=alert.message,
                        metrics=alert.metrics,
                        affected_reports=alert.affected_reports,
                        confidence_score=alert.confidence_score
                    ) for alert in anomaly_alerts
                ]
            },
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

# Anomaly detection endpoint
@app.get("/anomalies/recent")
async def get_recent_anomalies(hours: int = 24):
    """Get recent anomaly alerts"""
    try:
        recent_alerts = anomaly_detector.get_recent_alerts(hours)
        
        return {
            "time_window_hours": hours,
            "total_alerts": len(recent_alerts),
            "alerts": [
                AnomalyResponse(
                    alert_id=alert.id,
                    alert_type=alert.alert_type,
                    severity=alert.severity,
                    location=alert.location,
                    coordinates=list(alert.coordinates) if alert.coordinates else None,
                    detection_time=alert.detection_time,
                    message=alert.message,
                    metrics=alert.metrics,
                    affected_reports=alert.affected_reports,
                    confidence_score=alert.confidence_score
                ) for alert in recent_alerts
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving anomalies: {str(e)}")

# Feedback submission
@app.post("/feedback/submit")
async def submit_feedback(feedback_input: FeedbackInput):
    """Submit operator feedback"""
    try:
        feedback_id = await feedback_system.submit_feedback(
            report_id=feedback_input.report_id,
            operator_id=feedback_input.operator_id,
            feedback_type=FeedbackType(feedback_input.feedback_type),
            action=FeedbackAction(feedback_input.action),
            original_prediction=feedback_input.original_prediction,
            corrected_prediction=feedback_input.corrected_prediction,
            confidence_score=feedback_input.confidence_score,
            notes=feedback_input.notes
        )
        
        return {
            "feedback_id": feedback_id,
            "status": "submitted",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

# Statistics endpoints
@app.get("/stats/pipeline")
async def get_pipeline_statistics():
    """Get ML pipeline statistics"""
    try:
        return {
            "preprocessing": preprocessing_pipeline.get_statistics(),
            "nlp_analysis": nlp_engine.get_statistics(),
            "geolocation": geolocation_extractor.get_statistics(),
            "anomaly_detection": anomaly_detector.get_statistics(),
            "feedback": feedback_system.get_statistics(),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")

# Performance metrics
@app.get("/feedback/metrics")
async def get_performance_metrics(days: int = 30):
    """Get model performance metrics from feedback"""
    try:
        metrics = await feedback_system.get_performance_metrics(days)
        recommendations = await feedback_system.get_retraining_recommendations()
        
        return {
            "metrics_period_days": days,
            "performance_metrics": {
                model: {
                    "accuracy": metric.accuracy,
                    "precision": metric.precision,
                    "recall": metric.recall,
                    "f1_score": metric.f1_score,
                    "total_feedback": metric.total_feedback_count
                } for model, metric in metrics.items()
            },
            "retraining_recommendations": [
                {
                    "model_component": rec.model_component,
                    "priority": rec.priority,
                    "reason": rec.reason,
                    "recommended_action": rec.recommended_action,
                    "estimated_improvement": rec.estimated_improvement
                } for rec in recommendations
            ],
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")

# Data ingestion status
@app.get("/ingestion/status")
async def get_ingestion_status():
    """Get data ingestion status"""
    try:
        stats = data_manager.get_statistics()
        
        return {
            "ingestion_statistics": stats,
            "active_sources": {
                "twitter": "enabled",
                "government_alerts": "enabled", 
                "user_reports": "enabled"
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving ingestion status: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=config.get('API.HOST', '0.0.0.0'),
        port=config.get('API.PORT', 8000),
        reload=config.get('API.DEBUG', False)
    )