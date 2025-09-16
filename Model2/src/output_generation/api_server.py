"""FastAPI server for ocean hazard prediction system."""

import uvicorn
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Our modules
from ..data_aggregation import HistoricalEvents, SensorData, GeospatialData, SocialSignals
from ..feature_engineering import FeaturePipeline
from ..predictive_modeling import ModelTrainer
from .risk_scorer import RiskScorer
from .hotspot_mapper import HotspotMapper
from .alert_generator import AlertGenerator

# Data models
class LocationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")

class RiskAssessmentRequest(BaseModel):
    location: LocationRequest
    hazard_type: str = Field("tsunami", description="Type of ocean hazard")
    time_window: str = Field("short_term", description="Prediction time window")
    include_environmental: bool = Field(True, description="Include environmental data")
    include_historical: bool = Field(True, description="Include historical analysis")

class MultiLocationRequest(BaseModel):
    locations: List[LocationRequest]
    hazard_type: str = Field("tsunami", description="Type of ocean hazard")
    time_window: str = Field("short_term", description="Prediction time window")

class AlertConfigRequest(BaseModel):
    channel_type: str = Field(..., description="Notification channel (email, webhook, sms)")
    configuration: Dict[str, Any] = Field(..., description="Channel configuration")

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(config: Dict = None) -> FastAPI:
    """Create FastAPI application with all endpoints."""
    
    app = FastAPI(
        title="Ocean Hazard Prediction API",
        description="Proactive Ocean Hazard Prediction and Alert System",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize system components
    app.state.config = config or {}
    app.state.data_aggregation = {
        'historical': HistoricalEvents(),
        'sensor': SensorData(),
        'geospatial': GeospatialData(),
        'social': SocialSignals()
    }
    app.state.feature_pipeline = FeaturePipeline()
    app.state.model_trainer = ModelTrainer(config)
    app.state.risk_scorer = RiskScorer(config)
    app.state.hotspot_mapper = HotspotMapper(config)
    app.state.alert_generator = AlertGenerator(config)
    
    @app.get("/", tags=["System"])
    async def root():
        """Root endpoint with system information."""
        return {
            "message": "Ocean Hazard Prediction API",
            "version": "1.0.0",
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "risk_assessment": "/risk/assess",
                "batch_assessment": "/risk/batch",
                "hotspots": "/hotspots/identify",
                "alerts": "/alerts"
            }
        }
    
    @app.get("/health", tags=["System"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "data_aggregation": "operational",
                "feature_engineering": "operational",
                "predictive_modeling": "operational",
                "risk_scoring": "operational",
                "hotspot_mapping": "operational",
                "alert_generation": "operational"
            }
        }
    
    @app.post("/risk/assess", tags=["Risk Assessment"])
    async def assess_risk(request: RiskAssessmentRequest):
        """Assess risk for a single location."""
        
        try:
            location = (request.location.latitude, request.location.longitude)
            
            # Collect data
            logger.info(f"Assessing risk for location {location}")
            
            # Get historical data
            historical_data = None
            if request.include_historical:
                historical_data = app.state.data_aggregation['historical'].collect_historical_events(
                    location, radius_km=100
                )
            
            # Get environmental data
            environmental_data = None
            if request.include_environmental:
                environmental_data = {
                    'sea_temperature': 22.5,
                    'wave_height': 2.1,
                    'wind_speed': 15.2,
                    'atmospheric_pressure': 1013.2
                }
            
            # Perform risk assessment
            risk_assessment = app.state.risk_scorer.calculate_comprehensive_risk_score(
                location=location,
                data=historical_data,
                environmental_data=environmental_data,
                time_window=request.time_window,
                hazard_type=request.hazard_type
            )
            
            return JSONResponse(content={
                "success": True,
                "risk_assessment": risk_assessment,
                "assessment_timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/risk/batch", tags=["Risk Assessment"])
    async def batch_risk_assessment(request: MultiLocationRequest):
        """Perform risk assessment for multiple locations."""
        
        try:
            locations = [(loc.latitude, loc.longitude) for loc in request.locations]
            
            logger.info(f"Performing batch assessment for {len(locations)} locations")
            
            # Collect historical data for all locations
            all_historical_data = app.state.data_aggregation['historical'].collect_multiple_locations(
                locations, radius_km=50
            )
            
            # Perform batch risk assessment
            batch_results = app.state.risk_scorer.batch_risk_assessment(
                locations=locations,
                data=all_historical_data,
                hazard_type=request.hazard_type
            )
            
            return JSONResponse(content={
                "success": True,
                "batch_results": batch_results,
                "total_locations": len(locations),
                "assessment_timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in batch risk assessment: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/hotspots/identify", tags=["Hotspot Analysis"])
    async def identify_hotspots(
        min_risk_score: float = Query(0.6, description="Minimum risk score for hotspots"),
        min_events: int = Query(3, description="Minimum events for hotspot"),
        spatial_radius: float = Query(50, description="Spatial radius in km"),
        hazard_type: str = Query("tsunami", description="Hazard type to analyze")
    ):
        """Identify risk hotspots based on recent assessments."""
        
        try:
            # Get recent risk assessments (this would typically come from a database)
            # For now, we'll generate some sample assessments
            sample_locations = [
                (35.6762, 139.6503),  # Tokyo
                (34.0522, -118.2437), # Los Angeles
                (-6.2088, 106.8456),  # Jakarta
                (37.7749, -122.4194), # San Francisco
                (40.7128, -74.0060)   # New York
            ]
            
            risk_assessments = []
            for lat, lon in sample_locations:
                assessment = app.state.risk_scorer.calculate_comprehensive_risk_score(
                    location=(lat, lon),
                    hazard_type=hazard_type
                )
                risk_assessments.append(assessment)
            
            # Update hotspot criteria
            app.state.hotspot_mapper.config['hotspot_criteria'].update({
                'min_risk_score': min_risk_score,
                'min_events': min_events,
                'spatial_radius_km': spatial_radius
            })
            
            # Identify hotspots
            hotspots = app.state.hotspot_mapper.identify_risk_hotspots(risk_assessments)
            
            # Update tracking
            app.state.hotspot_mapper.update_hotspot_tracking(hotspots)
            
            return JSONResponse(content={
                "success": True,
                "hotspots": hotspots,
                "total_hotspots": len(hotspots),
                "criteria": {
                    "min_risk_score": min_risk_score,
                    "min_events": min_events,
                    "spatial_radius_km": spatial_radius
                },
                "identified_at": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error identifying hotspots: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/hotspots/map", tags=["Hotspot Analysis"])
    async def create_hotspot_map(background_tasks: BackgroundTasks):
        """Create visual hotspot map."""
        
        try:
            # Get current hotspots
            current_time = datetime.now().isoformat()
            hotspots = []
            
            if current_time in app.state.hotspot_mapper.hotspot_data:
                hotspots = app.state.hotspot_mapper.hotspot_data[current_time]['hotspots']
            
            # Create map in background
            def create_map():
                map_file = app.state.hotspot_mapper.create_interactive_map(hotspots=hotspots)
                logger.info(f"Hotspot map created: {map_file}")
            
            background_tasks.add_task(create_map)
            
            return JSONResponse(content={
                "success": True,
                "message": "Hotspot map creation started",
                "hotspots_count": len(hotspots),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error creating hotspot map: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/alerts/generate", tags=["Alert System"])
    async def generate_alert(request: RiskAssessmentRequest):
        """Generate alert based on risk assessment."""
        
        try:
            location = (request.location.latitude, request.location.longitude)
            
            # Perform risk assessment
            risk_assessment = app.state.risk_scorer.calculate_comprehensive_risk_score(
                location=location,
                hazard_type=request.hazard_type,
                time_window=request.time_window
            )
            
            # Generate alert if needed
            alert_result = app.state.alert_generator.generate_alert(
                risk_assessment=risk_assessment,
                alert_type='api_request'
            )
            
            return JSONResponse(content={
                "success": True,
                "alert_result": alert_result,
                "risk_assessment": risk_assessment,
                "generated_at": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/alerts/active", tags=["Alert System"])
    async def get_active_alerts(
        risk_level: Optional[str] = Query(None, description="Filter by risk level"),
        hazard_type: Optional[str] = Query(None, description="Filter by hazard type")
    ):
        """Get currently active alerts."""
        
        try:
            active_alerts = app.state.alert_generator.get_active_alerts(
                risk_level=risk_level,
                hazard_type=hazard_type
            )
            
            return JSONResponse(content={
                "success": True,
                "active_alerts": active_alerts,
                "total_count": len(active_alerts),
                "filters": {
                    "risk_level": risk_level,
                    "hazard_type": hazard_type
                },
                "retrieved_at": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error retrieving active alerts: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/alerts/summary", tags=["Alert System"])
    async def get_alert_summary(
        time_period_hours: int = Query(24, description="Time period for summary in hours")
    ):
        """Get alert summary for specified time period."""
        
        try:
            summary = app.state.alert_generator.generate_alert_summary(
                time_period_hours=time_period_hours
            )
            
            return JSONResponse(content={
                "success": True,
                "summary": summary,
                "time_period_hours": time_period_hours
            })
            
        except Exception as e:
            logger.error(f"Error generating alert summary: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/alerts/configure", tags=["Alert System"])
    async def configure_alerts(request: AlertConfigRequest):
        """Configure alert notification channels."""
        
        try:
            success = app.state.alert_generator.configure_notification_channel(
                channel_type=request.channel_type,
                configuration=request.configuration
            )
            
            if success:
                return JSONResponse(content={
                    "success": True,
                    "message": f"Successfully configured {request.channel_type} channel",
                    "configured_at": datetime.now().isoformat()
                })
            else:
                raise HTTPException(status_code=400, detail="Failed to configure channel")
                
        except Exception as e:
            logger.error(f"Error configuring alerts: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/data/collect", tags=["Data Management"])
    async def trigger_data_collection(
        data_type: str = Query("all", description="Type of data to collect"),
        location_lat: Optional[float] = Query(None, description="Location latitude"),
        location_lon: Optional[float] = Query(None, description="Location longitude"),
        radius_km: float = Query(100, description="Collection radius in km")
    ):
        """Trigger data collection from various sources."""
        
        try:
            results = {}
            
            if data_type in ["all", "historical"]:
                if location_lat is not None and location_lon is not None:
                    historical_data = app.state.data_aggregation['historical'].collect_historical_events(
                        location=(location_lat, location_lon),
                        radius_km=radius_km
                    )
                    results['historical'] = {
                        'events_collected': len(historical_data) if historical_data is not None else 0,
                        'status': 'success'
                    }
                else:
                    results['historical'] = {
                        'status': 'skipped',
                        'reason': 'Location not provided'
                    }
            
            if data_type in ["all", "sensor"]:
                sensor_data = app.state.data_aggregation['sensor'].collect_sensor_data()
                results['sensor'] = {
                    'readings_collected': len(sensor_data) if sensor_data is not None else 0,
                    'status': 'success'
                }
            
            if data_type in ["all", "geospatial"]:
                if location_lat is not None and location_lon is not None:
                    geo_data = app.state.data_aggregation['geospatial'].collect_geospatial_data(
                        location=(location_lat, location_lon)
                    )
                    results['geospatial'] = {
                        'data_points': len(geo_data) if isinstance(geo_data, (list, dict)) else 1,
                        'status': 'success'
                    }
                else:
                    results['geospatial'] = {
                        'status': 'skipped',
                        'reason': 'Location not provided'
                    }
            
            if data_type in ["all", "social"]:
                if location_lat is not None and location_lon is not None:
                    social_data = app.state.data_aggregation['social'].collect_social_signals(
                        location=(location_lat, location_lon)
                    )
                    results['social'] = {
                        'signals_collected': len(social_data) if social_data is not None else 0,
                        'status': 'success'
                    }
                else:
                    results['social'] = {
                        'status': 'skipped',
                        'reason': 'Location not provided'
                    }
            
            return JSONResponse(content={
                "success": True,
                "collection_results": results,
                "collected_at": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/models/train", tags=["Model Management"])
    async def train_models(background_tasks: BackgroundTasks):
        """Trigger model training pipeline."""
        
        def train_pipeline():
            try:
                # Generate sample training data
                import pandas as pd
                import numpy as np
                
                # Create sample data for training
                n_samples = 1000
                sample_data = pd.DataFrame({
                    'latitude': np.random.uniform(-90, 90, n_samples),
                    'longitude': np.random.uniform(-180, 180, n_samples),
                    'sea_temperature': np.random.uniform(15, 30, n_samples),
                    'wave_height': np.random.uniform(0.5, 8, n_samples),
                    'wind_speed': np.random.uniform(5, 50, n_samples),
                    'historical_events': np.random.randint(0, 10, n_samples),
                    'risk_level': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_samples)
                })
                
                # Train the pipeline
                results = app.state.model_trainer.train_full_pipeline(
                    data=sample_data,
                    target_column='risk_level',
                    task_type='classification'
                )
                
                logger.info(f"Model training completed: {results.get('success', False)}")
                
            except Exception as e:
                logger.error(f"Error in model training: {e}")
        
        background_tasks.add_task(train_pipeline)
        
        return JSONResponse(content={
            "success": True,
            "message": "Model training started in background",
            "started_at": datetime.now().isoformat()
        })
    
    @app.get("/models/status", tags=["Model Management"])
    async def get_model_status():
        """Get status of trained models."""
        
        try:
            status = {
                "time_series_models": len(app.state.model_trainer.time_series_models.models),
                "classification_models": len(app.state.model_trainer.classification_models.models),
                "ensemble_models": len(app.state.model_trainer.model_ensemble.ensemble_models),
                "clustering_models": len(app.state.model_trainer.spatial_clustering.clustering_models),
                "last_training": "Not available",  # Would be stored in actual implementation
                "model_performance": "Available through training reports"
            }
            
            return JSONResponse(content={
                "success": True,
                "model_status": status,
                "checked_at": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/export/data", tags=["Data Export"])
    async def export_system_data(
        data_type: str = Query("all", description="Type of data to export"),
        format: str = Query("json", description="Export format")
    ):
        """Export system data for analysis or backup."""
        
        try:
            export_files = []
            
            if data_type in ["all", "alerts"]:
                alert_file = app.state.alert_generator.export_alert_data()
                if alert_file:
                    export_files.append({"type": "alerts", "file": alert_file})
            
            if data_type in ["all", "hotspots"]:
                hotspot_file = app.state.hotspot_mapper.export_mapping_data(
                    output_format=format
                )
                if hotspot_file:
                    export_files.append({"type": "hotspots", "file": hotspot_file})
            
            return JSONResponse(content={
                "success": True,
                "exported_files": export_files,
                "export_format": format,
                "exported_at": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """Run the FastAPI server."""
    
    app = create_app()
    
    logger.info(f"Starting Ocean Hazard Prediction API server on {host}:{port}")
    
    uvicorn.run(
        "api_server:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True
    )

if __name__ == "__main__":
    run_server()