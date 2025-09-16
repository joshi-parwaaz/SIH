"""Model ensemble for combining multiple predictions."""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import json

# ML models
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import joblib

logger = logging.getLogger(__name__)

class ModelEnsemble:
    """Ensemble system for combining multiple model predictions."""
    
    def __init__(self):
        self.base_models = {}
        self.ensemble_models = {}
        self.model_weights = {}
        self.performance_history = {}
        
    def add_base_model(self, model_name: str, model, model_type: str = 'classification'):
        """Add a base model to the ensemble."""
        
        self.base_models[model_name] = {
            'model': model,
            'type': model_type,
            'added_at': datetime.now().isoformat()
        }
        
        logger.info(f"Added base model: {model_name} ({model_type})")
    
    def create_voting_ensemble(self, model_names: List[str], 
                             ensemble_name: str,
                             voting: str = 'soft',
                             weights: Optional[List[float]] = None) -> bool:
        """Create a voting ensemble from selected base models."""
        
        try:
            # Validate model names
            for name in model_names:
                if name not in self.base_models:
                    raise ValueError(f"Model {name} not found in base models")
            
            # Check model types are consistent
            model_types = [self.base_models[name]['type'] for name in model_names]
            if len(set(model_types)) > 1:
                raise ValueError("All models must be of the same type for voting ensemble")
            
            model_type = model_types[0]
            
            # Create estimator list
            estimators = [(name, self.base_models[name]['model']) for name in model_names]
            
            # Create ensemble
            if model_type == 'classification':
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting=voting,
                    weights=weights
                )
            else:  # regression
                ensemble = VotingRegressor(
                    estimators=estimators,
                    weights=weights
                )
            
            self.ensemble_models[ensemble_name] = {
                'model': ensemble,
                'type': 'voting',
                'base_models': model_names,
                'model_type': model_type,
                'voting': voting,
                'weights': weights,
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"Created voting ensemble: {ensemble_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating voting ensemble {ensemble_name}: {e}")
            return False
    
    def create_stacking_ensemble(self, model_names: List[str],
                               ensemble_name: str,
                               meta_learner=None,
                               cv_folds: int = 5) -> bool:
        """Create a stacking ensemble with meta-learner."""
        
        try:
            # Validate model names
            for name in model_names:
                if name not in self.base_models:
                    raise ValueError(f"Model {name} not found in base models")
            
            # Check model types are consistent
            model_types = [self.base_models[name]['type'] for name in model_names]
            if len(set(model_types)) > 1:
                raise ValueError("All models must be of the same type for stacking ensemble")
            
            model_type = model_types[0]
            
            # Default meta-learner
            if meta_learner is None:
                if model_type == 'classification':
                    meta_learner = LogisticRegression(random_state=42)
                else:
                    from sklearn.linear_model import LinearRegression
                    meta_learner = LinearRegression()
            
            # Create estimator list
            estimators = [(name, self.base_models[name]['model']) for name in model_names]
            
            # Create stacking ensemble
            if model_type == 'classification':
                from sklearn.ensemble import StackingClassifier
                ensemble = StackingClassifier(
                    estimators=estimators,
                    final_estimator=meta_learner,
                    cv=cv_folds,
                    stack_method='predict_proba' if hasattr(meta_learner, 'predict_proba') else 'predict'
                )
            else:  # regression
                from sklearn.ensemble import StackingRegressor
                ensemble = StackingRegressor(
                    estimators=estimators,
                    final_estimator=meta_learner,
                    cv=cv_folds
                )
            
            self.ensemble_models[ensemble_name] = {
                'model': ensemble,
                'type': 'stacking',
                'base_models': model_names,
                'model_type': model_type,
                'meta_learner': type(meta_learner).__name__,
                'cv_folds': cv_folds,
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"Created stacking ensemble: {ensemble_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating stacking ensemble {ensemble_name}: {e}")
            return False
    
    def create_weighted_average_ensemble(self, model_names: List[str],
                                       ensemble_name: str,
                                       weights: Optional[List[float]] = None) -> bool:
        """Create a weighted average ensemble."""
        
        try:
            # Validate model names
            for name in model_names:
                if name not in self.base_models:
                    raise ValueError(f"Model {name} not found in base models")
            
            # Default equal weights
            if weights is None:
                weights = [1.0 / len(model_names)] * len(model_names)
            elif len(weights) != len(model_names):
                raise ValueError("Number of weights must match number of models")
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            self.ensemble_models[ensemble_name] = {
                'model': None,  # Custom implementation
                'type': 'weighted_average',
                'base_models': model_names,
                'weights': weights.tolist(),
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"Created weighted average ensemble: {ensemble_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating weighted average ensemble {ensemble_name}: {e}")
            return False
    
    def train_ensemble(self, ensemble_name: str, X_train: np.ndarray, 
                      y_train: np.ndarray) -> bool:
        """Train an ensemble model."""
        
        try:
            if ensemble_name not in self.ensemble_models:
                raise ValueError(f"Ensemble {ensemble_name} not found")
            
            ensemble_info = self.ensemble_models[ensemble_name]
            
            if ensemble_info['type'] in ['voting', 'stacking']:
                # Train sklearn ensemble
                ensemble_model = ensemble_info['model']
                ensemble_model.fit(X_train, y_train)
                
                logger.info(f"Trained ensemble: {ensemble_name}")
                return True
                
            elif ensemble_info['type'] == 'weighted_average':
                # For weighted average, we need to ensure base models are trained
                for model_name in ensemble_info['base_models']:
                    base_model = self.base_models[model_name]['model']
                    if not hasattr(base_model, 'predict'):
                        logger.warning(f"Base model {model_name} may not be trained")
                
                logger.info(f"Weighted average ensemble ready: {ensemble_name}")
                return True
            
        except Exception as e:
            logger.error(f"Error training ensemble {ensemble_name}: {e}")
            return False
    
    def predict_ensemble(self, ensemble_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using an ensemble model."""
        
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble {ensemble_name} not found")
        
        ensemble_info = self.ensemble_models[ensemble_name]
        
        if ensemble_info['type'] in ['voting', 'stacking']:
            # Use sklearn ensemble prediction
            ensemble_model = ensemble_info['model']
            return ensemble_model.predict(X)
            
        elif ensemble_info['type'] == 'weighted_average':
            # Implement weighted average prediction
            predictions = []
            weights = np.array(ensemble_info['weights'])
            
            for model_name in ensemble_info['base_models']:
                base_model = self.base_models[model_name]['model']
                pred = base_model.predict(X)
                predictions.append(pred)
            
            # Weighted average
            predictions = np.array(predictions)
            weighted_pred = np.average(predictions, axis=0, weights=weights)
            
            return weighted_pred
        
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_info['type']}")
    
    def predict_proba_ensemble(self, ensemble_name: str, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities from ensemble (classification only)."""
        
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble {ensemble_name} not found")
        
        ensemble_info = self.ensemble_models[ensemble_name]
        
        if ensemble_info['model_type'] != 'classification':
            raise ValueError("predict_proba only available for classification ensembles")
        
        if ensemble_info['type'] in ['voting', 'stacking']:
            # Use sklearn ensemble prediction
            ensemble_model = ensemble_info['model']
            if hasattr(ensemble_model, 'predict_proba'):
                return ensemble_model.predict_proba(X)
            else:
                raise ValueError("Ensemble model does not support predict_proba")
                
        elif ensemble_info['type'] == 'weighted_average':
            # Implement weighted average for probabilities
            probabilities = []
            weights = np.array(ensemble_info['weights'])
            
            for model_name in ensemble_info['base_models']:
                base_model = self.base_models[model_name]['model']
                if hasattr(base_model, 'predict_proba'):
                    proba = base_model.predict_proba(X)
                    probabilities.append(proba)
                else:
                    raise ValueError(f"Base model {model_name} does not support predict_proba")
            
            # Weighted average of probabilities
            probabilities = np.array(probabilities)
            weighted_proba = np.average(probabilities, axis=0, weights=weights)
            
            return weighted_proba
    
    def evaluate_ensemble(self, ensemble_name: str, X_test: np.ndarray, 
                         y_test: np.ndarray) -> Dict:
        """Evaluate ensemble performance."""
        
        try:
            ensemble_info = self.ensemble_models[ensemble_name]
            model_type = ensemble_info.get('model_type', 'classification')
            
            # Get predictions
            y_pred = self.predict_ensemble(ensemble_name, X_test)
            
            # Calculate metrics
            metrics = {}
            
            if model_type == 'classification':
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                
                # Get probabilities if available
                try:
                    y_proba = self.predict_proba_ensemble(ensemble_name, X_test)
                    from sklearn.metrics import log_loss, roc_auc_score
                    
                    metrics['log_loss'] = log_loss(y_test, y_proba)
                    
                    # ROC AUC for binary classification
                    if len(np.unique(y_test)) == 2:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
                        
                except Exception as e:
                    logger.warning(f"Could not calculate probability-based metrics: {e}")
                
            else:  # regression
                metrics['mse'] = mean_squared_error(y_test, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y_test, y_pred)
                
                # R-squared
                from sklearn.metrics import r2_score
                metrics['r2'] = r2_score(y_test, y_pred)
            
            # Store performance history
            if ensemble_name not in self.performance_history:
                self.performance_history[ensemble_name] = []
            
            performance_record = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'test_size': len(y_test)
            }
            self.performance_history[ensemble_name].append(performance_record)
            
            logger.info(f"Evaluated ensemble {ensemble_name}: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble {ensemble_name}: {e}")
            return {}
    
    def compare_ensembles(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Compare performance of all ensembles."""
        
        results = []
        
        for ensemble_name in self.ensemble_models:
            try:
                metrics = self.evaluate_ensemble(ensemble_name, X_test, y_test)
                
                result = {
                    'ensemble_name': ensemble_name,
                    'ensemble_type': self.ensemble_models[ensemble_name]['type'],
                    'model_type': self.ensemble_models[ensemble_name].get('model_type', 'unknown'),
                    **metrics
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error comparing ensemble {ensemble_name}: {e}")
        
        return pd.DataFrame(results)
    
    def get_feature_importance(self, ensemble_name: str) -> Dict:
        """Get feature importance from ensemble if available."""
        
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble {ensemble_name} not found")
        
        ensemble_info = self.ensemble_models[ensemble_name]
        
        try:
            if ensemble_info['type'] in ['voting', 'stacking']:
                ensemble_model = ensemble_info['model']
                
                if hasattr(ensemble_model, 'feature_importances_'):
                    return {'feature_importances': ensemble_model.feature_importances_.tolist()}
                
                # For voting ensemble, get importance from base models
                if ensemble_info['type'] == 'voting':
                    importances = {}
                    for name, estimator in ensemble_model.named_estimators_.items():
                        if hasattr(estimator, 'feature_importances_'):
                            importances[name] = estimator.feature_importances_.tolist()
                        elif hasattr(estimator, 'coef_'):
                            importances[name] = np.abs(estimator.coef_).flatten().tolist()
                    
                    return {'base_model_importances': importances}
                    
            elif ensemble_info['type'] == 'weighted_average':
                # Get importance from base models
                importances = {}
                weights = ensemble_info['weights']
                
                for i, model_name in enumerate(ensemble_info['base_models']):
                    base_model = self.base_models[model_name]['model']
                    weight = weights[i]
                    
                    if hasattr(base_model, 'feature_importances_'):
                        importances[model_name] = {
                            'importance': base_model.feature_importances_.tolist(),
                            'weight': weight
                        }
                    elif hasattr(base_model, 'coef_'):
                        importances[model_name] = {
                            'importance': np.abs(base_model.coef_).flatten().tolist(),
                            'weight': weight
                        }
                
                return {'weighted_base_importances': importances}
            
        except Exception as e:
            logger.error(f"Error getting feature importance for {ensemble_name}: {e}")
        
        return {}
    
    def optimize_ensemble_weights(self, ensemble_name: str, 
                                X_val: np.ndarray, y_val: np.ndarray,
                                method: str = 'grid_search') -> bool:
        """Optimize ensemble weights using validation data."""
        
        try:
            if ensemble_name not in self.ensemble_models:
                raise ValueError(f"Ensemble {ensemble_name} not found")
            
            ensemble_info = self.ensemble_models[ensemble_name]
            
            if ensemble_info['type'] != 'weighted_average':
                logger.warning(f"Weight optimization only supported for weighted_average ensembles")
                return False
            
            model_names = ensemble_info['base_models']
            n_models = len(model_names)
            
            # Get base model predictions
            base_predictions = []
            for model_name in model_names:
                base_model = self.base_models[model_name]['model']
                pred = base_model.predict(X_val)
                base_predictions.append(pred)
            
            base_predictions = np.array(base_predictions).T  # Shape: (n_samples, n_models)
            
            # Grid search for optimal weights
            best_weights = None
            best_score = float('inf') if ensemble_info.get('model_type') == 'regression' else 0.0
            
            if method == 'grid_search':
                # Generate weight combinations
                from itertools import product
                
                weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                
                for weights in product(weight_options, repeat=n_models):
                    weights = np.array(weights)
                    
                    # Skip if all weights are zero
                    if np.sum(weights) == 0:
                        continue
                    
                    # Normalize weights
                    weights = weights / np.sum(weights)
                    
                    # Calculate ensemble prediction
                    ensemble_pred = np.average(base_predictions, axis=1, weights=weights)
                    
                    # Calculate score
                    if ensemble_info.get('model_type') == 'regression':
                        score = mean_squared_error(y_val, ensemble_pred)
                        if score < best_score:
                            best_score = score
                            best_weights = weights
                    else:  # classification
                        score = accuracy_score(y_val, ensemble_pred)
                        if score > best_score:
                            best_score = score
                            best_weights = weights
            
            # Update ensemble weights
            if best_weights is not None:
                self.ensemble_models[ensemble_name]['weights'] = best_weights.tolist()
                logger.info(f"Optimized weights for {ensemble_name}: {best_weights} (score: {best_score:.4f})")
                return True
            else:
                logger.warning(f"Could not find better weights for {ensemble_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error optimizing ensemble weights {ensemble_name}: {e}")
            return False
    
    def get_ensemble_info(self) -> Dict:
        """Get information about all ensembles."""
        
        info = {
            'base_models': {
                name: {
                    'type': model_info['type'],
                    'added_at': model_info['added_at']
                }
                for name, model_info in self.base_models.items()
            },
            'ensembles': {
                name: {
                    'type': ensemble_info['type'],
                    'base_models': ensemble_info['base_models'],
                    'created_at': ensemble_info['created_at']
                }
                for name, ensemble_info in self.ensemble_models.items()
            },
            'performance_history': self.performance_history
        }
        
        return info
    
    def save_ensemble(self, ensemble_name: str, filepath: str):
        """Save ensemble model to disk."""
        
        try:
            if ensemble_name not in self.ensemble_models:
                raise ValueError(f"Ensemble {ensemble_name} not found")
            
            ensemble_info = self.ensemble_models[ensemble_name].copy()
            
            # For sklearn ensembles, save the trained model
            if ensemble_info['type'] in ['voting', 'stacking']:
                model_data = {
                    'ensemble_info': ensemble_info,
                    'model': ensemble_info['model']
                }
            else:
                # For custom ensembles, save configuration
                model_data = {
                    'ensemble_info': ensemble_info,
                    'base_models': {
                        name: self.base_models[name] 
                        for name in ensemble_info['base_models']
                    }
                }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Saved ensemble {ensemble_name} to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving ensemble {ensemble_name}: {e}")
    
    def load_ensemble(self, ensemble_name: str, filepath: str):
        """Load ensemble model from disk."""
        
        try:
            model_data = joblib.load(filepath)
            
            # Restore ensemble
            self.ensemble_models[ensemble_name] = model_data['ensemble_info']
            
            # Restore base models if included
            if 'base_models' in model_data:
                for name, model_info in model_data['base_models'].items():
                    self.base_models[name] = model_info
            
            logger.info(f"Loaded ensemble {ensemble_name} from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading ensemble {ensemble_name}: {e}")
    
    def remove_ensemble(self, ensemble_name: str):
        """Remove an ensemble model."""
        
        if ensemble_name in self.ensemble_models:
            del self.ensemble_models[ensemble_name]
            
            if ensemble_name in self.performance_history:
                del self.performance_history[ensemble_name]
            
            logger.info(f"Removed ensemble: {ensemble_name}")
        else:
            logger.warning(f"Ensemble {ensemble_name} not found")
    
    def clear_ensembles(self):
        """Clear all ensemble models."""
        
        self.ensemble_models.clear()
        self.performance_history.clear()
        logger.info("Cleared all ensemble models")