import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json

# Machine learning
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from ..data_ingestion import RawReport
from ..preprocessing import ProcessedReport
from ..nlp_analysis import HazardPrediction
from ..geolocation import GeolocationResult
from ..anomaly_detection import AnomalyAlert
from config import config

class FeedbackType(Enum):
    """Types of feedback"""
    HAZARD_CLASSIFICATION = "hazard_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    MISINFORMATION_DETECTION = "misinformation_detection"
    LOCATION_EXTRACTION = "location_extraction"
    ANOMALY_DETECTION = "anomaly_detection"
    OVERALL_ACCURACY = "overall_accuracy"

class FeedbackAction(Enum):
    """Feedback actions"""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"
    MISSING = "missing"
    FALSE_POSITIVE = "false_positive"
    UPDATE_LABEL = "update_label"

@dataclass
class OperatorFeedback:
    """Operator feedback on model predictions"""
    id: str
    report_id: str
    operator_id: str
    feedback_type: FeedbackType
    action: FeedbackAction
    
    # Original prediction
    original_prediction: Dict[str, Any]
    
    # Corrected values
    corrected_prediction: Optional[Dict[str, Any]]
    
    # Feedback details
    confidence_score: float  # Operator confidence in feedback
    notes: Optional[str]
    timestamp: datetime
    
    # Model performance impact
    impact_score: Optional[float] = None
    validation_status: Optional[str] = None

@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics from feedback"""
    model_component: str
    total_feedback_count: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Breakdown by action
    correct_predictions: int
    incorrect_predictions: int
    false_positives: int
    missing_predictions: int
    
    # Time-based metrics
    metrics_period: str
    last_updated: datetime

@dataclass
class RetrainingRecommendation:
    """Recommendation for model retraining"""
    model_component: str
    priority: str  # 'low', 'medium', 'high', 'critical'
    reason: str
    feedback_count: int
    performance_degradation: float
    recommended_action: str
    dataset_requirements: Dict[str, Any]
    estimated_improvement: float

class HazardClassificationFeedbackHandler:
    """Handle feedback for hazard classification model"""
    
    def __init__(self):
        self.feedback_data = []
        self.performance_metrics = {}
        
    async def process_feedback(self, feedback: OperatorFeedback) -> Dict[str, Any]:
        """Process hazard classification feedback"""
        
        if feedback.feedback_type != FeedbackType.HAZARD_CLASSIFICATION:
            return {'error': 'Invalid feedback type'}
        
        self.feedback_data.append(feedback)
        
        # Extract original and corrected predictions
        original = feedback.original_prediction
        corrected = feedback.corrected_prediction
        
        result = {
            'feedback_id': feedback.id,
            'processed': True,
            'impact_assessment': {}
        }
        
        # Analyze the feedback
        if feedback.action == FeedbackAction.CORRECT:
            result['impact_assessment']['accuracy_impact'] = 0.0
            result['impact_assessment']['confidence_boost'] = 0.1
            
        elif feedback.action == FeedbackAction.INCORRECT:
            # Calculate severity of error
            if original and 'predictions' in original:
                top_prediction = max(original['predictions'], key=lambda x: x['confidence'])
                error_severity = top_prediction['confidence']
                
                result['impact_assessment']['accuracy_impact'] = -error_severity
                result['impact_assessment']['retraining_urgency'] = 'medium' if error_severity > 0.8 else 'low'
        
        elif feedback.action == FeedbackAction.UPDATE_LABEL and corrected:
            # Calculate difference between original and corrected
            if original and corrected:
                original_hazard = max(original['predictions'], key=lambda x: x['confidence'])['hazard_type']
                corrected_hazard = corrected.get('hazard_type', original_hazard)
                
                if original_hazard != corrected_hazard:
                    result['impact_assessment']['label_correction'] = {
                        'from': original_hazard,
                        'to': corrected_hazard,
                        'importance': 'high'
                    }
        
        # Update feedback impact score
        feedback.impact_score = self._calculate_impact_score(feedback)
        
        return result
    
    def _calculate_impact_score(self, feedback: OperatorFeedback) -> float:
        """Calculate impact score for feedback"""
        
        base_score = 1.0
        
        # Weight by operator confidence
        confidence_weight = feedback.confidence_score
        
        # Weight by action type
        action_weights = {
            FeedbackAction.CORRECT: 0.1,
            FeedbackAction.INCORRECT: 0.8,
            FeedbackAction.UPDATE_LABEL: 0.9,
            FeedbackAction.FALSE_POSITIVE: 0.7,
            FeedbackAction.MISSING: 0.6
        }
        
        action_weight = action_weights.get(feedback.action, 0.5)
        
        # Calculate final impact
        impact = base_score * confidence_weight * action_weight
        
        return min(impact, 1.0)
    
    async def get_performance_metrics(self, days: int = 30) -> ModelPerformanceMetrics:
        """Calculate performance metrics from feedback"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_feedback = [f for f in self.feedback_data if f.timestamp >= cutoff_date]
        
        if not recent_feedback:
            return ModelPerformanceMetrics(
                model_component='hazard_classification',
                total_feedback_count=0,
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                correct_predictions=0, incorrect_predictions=0,
                false_positives=0, missing_predictions=0,
                metrics_period=f"{days}d",
                last_updated=datetime.now()
            )
        
        # Count feedback by action
        action_counts = {}
        for action in FeedbackAction:
            action_counts[action] = len([f for f in recent_feedback if f.action == action])
        
        total_count = len(recent_feedback)
        correct_count = action_counts[FeedbackAction.CORRECT]
        incorrect_count = action_counts[FeedbackAction.INCORRECT]
        
        # Calculate basic metrics
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        # For precision/recall, we need more sophisticated calculation
        # This is a simplified version
        precision = correct_count / (correct_count + action_counts[FeedbackAction.FALSE_POSITIVE]) if (correct_count + action_counts[FeedbackAction.FALSE_POSITIVE]) > 0 else 0.0
        recall = correct_count / (correct_count + action_counts[FeedbackAction.MISSING]) if (correct_count + action_counts[FeedbackAction.MISSING]) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return ModelPerformanceMetrics(
            model_component='hazard_classification',
            total_feedback_count=total_count,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            correct_predictions=correct_count,
            incorrect_predictions=incorrect_count,
            false_positives=action_counts[FeedbackAction.FALSE_POSITIVE],
            missing_predictions=action_counts[FeedbackAction.MISSING],
            metrics_period=f"{days}d",
            last_updated=datetime.now()
        )

class LocationExtractionFeedbackHandler:
    """Handle feedback for location extraction model"""
    
    def __init__(self):
        self.feedback_data = []
        
    async def process_feedback(self, feedback: OperatorFeedback) -> Dict[str, Any]:
        """Process location extraction feedback"""
        
        if feedback.feedback_type != FeedbackType.LOCATION_EXTRACTION:
            return {'error': 'Invalid feedback type'}
        
        self.feedback_data.append(feedback)
        
        result = {
            'feedback_id': feedback.id,
            'processed': True,
            'impact_assessment': {}
        }
        
        # Analyze location extraction feedback
        if feedback.action == FeedbackAction.INCORRECT:
            original = feedback.original_prediction
            if original and 'extracted_locations' in original:
                location_count = len(original['extracted_locations'])
                
                result['impact_assessment']['location_extraction_accuracy'] = -0.5
                result['impact_assessment']['geocoding_reliability'] = -0.3
                
                if location_count == 0:
                    result['impact_assessment']['missing_detection'] = True
                else:
                    result['impact_assessment']['false_detection'] = True
        
        elif feedback.action == FeedbackAction.UPDATE_LABEL and feedback.corrected_prediction:
            # Handle location corrections
            corrected = feedback.corrected_prediction
            
            result['impact_assessment']['location_correction'] = {
                'corrected_coordinates': corrected.get('coordinates'),
                'corrected_address': corrected.get('address'),
                'importance': 'high'
            }
        
        feedback.impact_score = self._calculate_location_impact(feedback)
        
        return result
    
    def _calculate_location_impact(self, feedback: OperatorFeedback) -> float:
        """Calculate impact score for location feedback"""
        
        base_impact = 0.5
        
        # Location accuracy is critical for hazard response
        if feedback.action in [FeedbackAction.INCORRECT, FeedbackAction.UPDATE_LABEL]:
            base_impact = 0.8
        
        return base_impact * feedback.confidence_score

class AnomalyDetectionFeedbackHandler:
    """Handle feedback for anomaly detection"""
    
    def __init__(self):
        self.feedback_data = []
        
    async def process_feedback(self, feedback: OperatorFeedback) -> Dict[str, Any]:
        """Process anomaly detection feedback"""
        
        if feedback.feedback_type != FeedbackType.ANOMALY_DETECTION:
            return {'error': 'Invalid feedback type'}
        
        self.feedback_data.append(feedback)
        
        result = {
            'feedback_id': feedback.id,
            'processed': True,
            'impact_assessment': {}
        }
        
        # Analyze anomaly detection feedback
        if feedback.action == FeedbackAction.FALSE_POSITIVE:
            # Anomaly was flagged but shouldn't have been
            original = feedback.original_prediction
            if original and 'alert_type' in original:
                alert_type = original['alert_type']
                
                result['impact_assessment']['false_positive_type'] = alert_type
                result['impact_assessment']['threshold_adjustment'] = 'increase'
                result['impact_assessment']['impact'] = -0.6
        
        elif feedback.action == FeedbackAction.MISSING:
            # Anomaly was missed
            result['impact_assessment']['missed_anomaly'] = True
            result['impact_assessment']['threshold_adjustment'] = 'decrease'
            result['impact_assessment']['impact'] = -0.8
        
        elif feedback.action == FeedbackAction.CORRECT:
            # Anomaly was correctly detected
            result['impact_assessment']['impact'] = 0.1
        
        feedback.impact_score = self._calculate_anomaly_impact(feedback)
        
        return result
    
    def _calculate_anomaly_impact(self, feedback: OperatorFeedback) -> float:
        """Calculate impact score for anomaly feedback"""
        
        # Anomaly detection feedback is very important
        impact_weights = {
            FeedbackAction.FALSE_POSITIVE: 0.8,
            FeedbackAction.MISSING: 0.9,
            FeedbackAction.CORRECT: 0.1
        }
        
        base_impact = impact_weights.get(feedback.action, 0.5)
        return base_impact * feedback.confidence_score

class RetrainingManager:
    """Manage model retraining based on feedback"""
    
    def __init__(self):
        self.retraining_thresholds = {
            'accuracy_threshold': config.get('FEEDBACK.RETRAINING.ACCURACY_THRESHOLD', 0.85),
            'feedback_count_threshold': config.get('FEEDBACK.RETRAINING.FEEDBACK_COUNT_THRESHOLD', 100),
            'days_since_training': config.get('FEEDBACK.RETRAINING.DAYS_SINCE_TRAINING', 30)
        }
        
        self.last_training_dates = {}
        
    async def assess_retraining_needs(
        self, 
        performance_metrics: List[ModelPerformanceMetrics]
    ) -> List[RetrainingRecommendation]:
        """Assess which models need retraining"""
        
        recommendations = []
        
        for metrics in performance_metrics:
            recommendation = await self._assess_single_model(metrics)
            if recommendation:
                recommendations.append(recommendation)
        
        return recommendations
    
    async def _assess_single_model(
        self, 
        metrics: ModelPerformanceMetrics
    ) -> Optional[RetrainingRecommendation]:
        """Assess retraining needs for a single model"""
        
        model_name = metrics.model_component
        
        # Check if model needs retraining
        needs_retraining = False
        priority = 'low'
        reasons = []
        
        # Accuracy threshold
        if metrics.accuracy < self.retraining_thresholds['accuracy_threshold']:
            needs_retraining = True
            priority = 'high'
            reasons.append(f"Accuracy below threshold: {metrics.accuracy:.3f} < {self.retraining_thresholds['accuracy_threshold']}")
        
        # Feedback count threshold
        if metrics.total_feedback_count >= self.retraining_thresholds['feedback_count_threshold']:
            needs_retraining = True
            if priority == 'low':
                priority = 'medium'
            reasons.append(f"Sufficient feedback data available: {metrics.total_feedback_count} samples")
        
        # Time since last training
        last_training = self.last_training_dates.get(model_name)
        if last_training:
            days_since_training = (datetime.now() - last_training).days
            if days_since_training >= self.retraining_thresholds['days_since_training']:
                needs_retraining = True
                if priority == 'low':
                    priority = 'medium'
                reasons.append(f"Long time since last training: {days_since_training} days")
        
        # High error rate
        error_rate = (metrics.incorrect_predictions + metrics.false_positives) / max(metrics.total_feedback_count, 1)
        if error_rate > 0.3:  # 30% error rate
            needs_retraining = True
            priority = 'critical'
            reasons.append(f"High error rate: {error_rate:.3f}")
        
        if not needs_retraining:
            return None
        
        # Calculate expected improvement
        current_performance = metrics.accuracy
        expected_improvement = min(0.95 - current_performance, 0.2)  # Cap at 20% improvement
        
        # Determine dataset requirements
        dataset_requirements = {
            'min_samples': max(1000, metrics.total_feedback_count * 2),
            'include_feedback_corrections': True,
            'balance_classes': True,
            'validation_split': 0.2
        }
        
        # Recommended action
        if priority == 'critical':
            recommended_action = 'immediate_retraining'
        elif priority == 'high':
            recommended_action = 'schedule_retraining_this_week'
        else:
            recommended_action = 'schedule_retraining_next_cycle'
        
        return RetrainingRecommendation(
            model_component=model_name,
            priority=priority,
            reason='; '.join(reasons),
            feedback_count=metrics.total_feedback_count,
            performance_degradation=1.0 - metrics.accuracy,
            recommended_action=recommended_action,
            dataset_requirements=dataset_requirements,
            estimated_improvement=expected_improvement
        )

class FeedbackIntegrationSystem:
    """Main feedback integration system"""
    
    def __init__(self):
        self.hazard_handler = HazardClassificationFeedbackHandler()
        self.location_handler = LocationExtractionFeedbackHandler()
        self.anomaly_handler = AnomalyDetectionFeedbackHandler()
        self.retraining_manager = RetrainingManager()
        
        # Feedback storage
        self.all_feedback = []
        
        # Statistics
        self.stats = {
            'total_feedback_received': 0,
            'feedback_by_type': {ft.value: 0 for ft in FeedbackType},
            'feedback_by_action': {fa.value: 0 for fa in FeedbackAction},
            'average_operator_confidence': 0.0,
            'last_feedback_time': None
        }
    
    async def submit_feedback(
        self,
        report_id: str,
        operator_id: str,
        feedback_type: FeedbackType,
        action: FeedbackAction,
        original_prediction: Dict[str, Any],
        corrected_prediction: Optional[Dict[str, Any]] = None,
        confidence_score: float = 1.0,
        notes: Optional[str] = None
    ) -> str:
        """Submit operator feedback"""
        
        feedback_id = str(uuid.uuid4())
        
        feedback = OperatorFeedback(
            id=feedback_id,
            report_id=report_id,
            operator_id=operator_id,
            feedback_type=feedback_type,
            action=action,
            original_prediction=original_prediction,
            corrected_prediction=corrected_prediction,
            confidence_score=confidence_score,
            notes=notes,
            timestamp=datetime.now()
        )
        
        # Store feedback
        self.all_feedback.append(feedback)
        
        # Route to appropriate handler
        if feedback_type == FeedbackType.HAZARD_CLASSIFICATION:
            result = await self.hazard_handler.process_feedback(feedback)
        elif feedback_type == FeedbackType.LOCATION_EXTRACTION:
            result = await self.location_handler.process_feedback(feedback)
        elif feedback_type == FeedbackType.ANOMALY_DETECTION:
            result = await self.anomaly_handler.process_feedback(feedback)
        else:
            result = {'feedback_id': feedback_id, 'processed': True}
        
        # Update statistics
        self.stats['total_feedback_received'] += 1
        self.stats['feedback_by_type'][feedback_type.value] += 1
        self.stats['feedback_by_action'][action.value] += 1
        self.stats['last_feedback_time'] = datetime.now()
        
        # Update average confidence
        total_confidence = sum(f.confidence_score for f in self.all_feedback)
        self.stats['average_operator_confidence'] = total_confidence / len(self.all_feedback)
        
        print(f"Feedback submitted: {feedback_id} for report {report_id}")
        
        return feedback_id
    
    async def get_performance_metrics(self, days: int = 30) -> Dict[str, ModelPerformanceMetrics]:
        """Get performance metrics for all models"""
        
        metrics = {}
        
        # Hazard classification metrics
        hazard_metrics = await self.hazard_handler.get_performance_metrics(days)
        metrics['hazard_classification'] = hazard_metrics
        
        # Add other model metrics as they become available
        
        return metrics
    
    async def get_retraining_recommendations(self) -> List[RetrainingRecommendation]:
        """Get retraining recommendations based on feedback"""
        
        performance_metrics = await self.get_performance_metrics()
        metrics_list = list(performance_metrics.values())
        
        return await self.retraining_manager.assess_retraining_needs(metrics_list)
    
    def get_feedback_by_report(self, report_id: str) -> List[OperatorFeedback]:
        """Get all feedback for a specific report"""
        return [f for f in self.all_feedback if f.report_id == report_id]
    
    def get_feedback_by_operator(self, operator_id: str, days: int = 30) -> List[OperatorFeedback]:
        """Get feedback submitted by specific operator"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [f for f in self.all_feedback 
                if f.operator_id == operator_id and f.timestamp >= cutoff_date]
    
    def get_high_impact_feedback(self, threshold: float = 0.7) -> List[OperatorFeedback]:
        """Get feedback with high impact scores"""
        return [f for f in self.all_feedback 
                if f.impact_score and f.impact_score >= threshold]
    
    async def export_feedback_for_retraining(
        self, 
        model_component: str,
        days: int = 90
    ) -> Dict[str, Any]:
        """Export feedback data for model retraining"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter feedback by model component and date
        relevant_feedback = []
        for feedback in self.all_feedback:
            if (feedback.timestamp >= cutoff_date and 
                feedback.feedback_type.value == model_component):
                relevant_feedback.append(feedback)
        
        # Prepare training data
        training_data = {
            'feedback_records': [asdict(f) for f in relevant_feedback],
            'export_timestamp': datetime.now(),
            'model_component': model_component,
            'data_period_days': days,
            'total_records': len(relevant_feedback),
            'metadata': {
                'accuracy_corrections': len([f for f in relevant_feedback if f.action == FeedbackAction.UPDATE_LABEL]),
                'false_positives': len([f for f in relevant_feedback if f.action == FeedbackAction.FALSE_POSITIVE]),
                'missing_detections': len([f for f in relevant_feedback if f.action == FeedbackAction.MISSING])
            }
        }
        
        return training_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback system statistics"""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset feedback statistics"""
        self.stats = {
            'total_feedback_received': 0,
            'feedback_by_type': {ft.value: 0 for ft in FeedbackType},
            'feedback_by_action': {fa.value: 0 for fa in FeedbackAction},
            'average_operator_confidence': 0.0,
            'last_feedback_time': None
        }