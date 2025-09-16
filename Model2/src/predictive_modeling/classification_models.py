"""Classification models for hazard risk assessment."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging

# Sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, accuracy_score
)
from sklearn.preprocessing import LabelEncoder

# XGBoost
try:
    import xgboost as xgb
except ImportError:
    xgb = None

import joblib
from config import config

logger = logging.getLogger(__name__)

class ClassificationModels:
    """Classification models for hazard risk assessment."""
    
    def __init__(self):
        self.models = {}
        self.model_configs = config.models
        self.label_encoders = {}
        self.trained_models = {}
        
    def prepare_classification_data(self, features: pd.DataFrame, 
                                  target: pd.Series,
                                  test_size: float = 0.2) -> Tuple:
        """Prepare data for classification models."""
        
        # Remove any infinite or NaN values
        features_clean = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Align features and target
        features_aligned, target_aligned = features_clean.align(target, axis=0, join='inner')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_aligned, target_aligned, 
            test_size=test_size, 
            random_state=42,
            stratify=target_aligned if len(target_aligned.unique()) > 1 else None
        )
        
        return X_train, X_test, y_train, y_test
    
    def create_risk_labels(self, risk_scores: pd.Series, 
                          method: str = 'quantile') -> pd.Series:
        """Create risk labels from continuous scores."""
        
        if method == 'quantile':
            # Create labels based on quantiles
            labels = pd.cut(
                risk_scores, 
                bins=[0, 0.33, 0.66, 1.0],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
        elif method == 'threshold':
            # Create labels based on fixed thresholds
            thresholds = config.get('prediction.risk_threshold', {})
            low_thresh = thresholds.get('low', 0.3)
            medium_thresh = thresholds.get('medium', 0.6)
            high_thresh = thresholds.get('high', 0.8)
            
            labels = pd.cut(
                risk_scores,
                bins=[0, low_thresh, medium_thresh, high_thresh, 1.0],
                labels=['Very Low', 'Low', 'Medium', 'High'],
                include_lowest=True
            )
        else:
            raise ValueError(f"Unknown labeling method: {method}")
        
        return labels
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Train Random Forest classifier."""
        
        logger.info("Training Random Forest classifier")
        
        # Get configuration
        rf_config = self.model_configs.get('random_forest', {})
        
        # Initialize model
        model = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 100),
            max_depth=rf_config.get('max_depth', 10),
            random_state=rf_config.get('random_state', 42),
            n_jobs=-1
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_pred_proba = model.predict_proba(X_train)
        test_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store model
        model_key = 'random_forest'
        self.trained_models[model_key] = {
            'model': model,
            'feature_names': X_train.columns.tolist(),
            'classes': model.classes_.tolist()
        }
        
        results = {
            'model_type': 'Random Forest',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'feature_importance': feature_importance,
            'model_key': model_key,
            'classification_report': classification_report(y_test, test_pred),
            'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
        }
        
        # Add AUC if binary classification
        if len(model.classes_) == 2:
            train_auc = roc_auc_score(y_train, train_pred_proba[:, 1])
            test_auc = roc_auc_score(y_test, test_pred_proba[:, 1])
            results.update({
                'train_auc': train_auc,
                'test_auc': test_auc
            })
        
        logger.info(f"Random Forest trained. Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
        
        return results
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Train XGBoost classifier."""
        
        if xgb is None:
            logger.error("XGBoost not available. Install with: pip install xgboost")
            return {'error': 'XGBoost not installed'}
        
        logger.info("Training XGBoost classifier")
        
        # Encode labels if they are strings
        if y_train.dtype == 'object':
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            self.label_encoders['xgboost'] = le
        else:
            y_train_encoded = y_train
            y_test_encoded = y_test
        
        # Get configuration
        xgb_config = self.model_configs.get('xgboost', {})
        
        # Initialize model
        model = xgb.XGBClassifier(
            n_estimators=xgb_config.get('n_estimators', 100),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            max_depth=xgb_config.get('max_depth', 6),
            random_state=42,
            eval_metric='logloss'
        )
        
        # Train model
        model.fit(
            X_train, y_train_encoded,
            eval_set=[(X_test, y_test_encoded)],
            verbose=False
        )
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_pred_proba = model.predict_proba(X_train)
        test_pred_proba = model.predict_proba(X_test)
        
        # Decode predictions if needed
        if 'xgboost' in self.label_encoders:
            le = self.label_encoders['xgboost']
            train_pred = le.inverse_transform(train_pred)
            test_pred = le.inverse_transform(test_pred)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store model
        model_key = 'xgboost'
        self.trained_models[model_key] = {
            'model': model,
            'feature_names': X_train.columns.tolist(),
            'label_encoder': self.label_encoders.get('xgboost')
        }
        
        results = {
            'model_type': 'XGBoost',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'feature_importance': feature_importance,
            'model_key': model_key,
            'classification_report': classification_report(y_test, test_pred),
            'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
        }
        
        # Add AUC if binary classification
        if len(np.unique(y_train_encoded)) == 2:
            train_auc = roc_auc_score(y_train_encoded, train_pred_proba[:, 1])
            test_auc = roc_auc_score(y_test_encoded, test_pred_proba[:, 1])
            results.update({
                'train_auc': train_auc,
                'test_auc': test_auc
            })
        
        logger.info(f"XGBoost trained. Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
        
        return results
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Train Logistic Regression classifier."""
        
        logger.info("Training Logistic Regression classifier")
        
        # Initialize model
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='ovr'
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_pred_proba = model.predict_proba(X_train)
        test_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        # Store model
        model_key = 'logistic_regression'
        self.trained_models[model_key] = {
            'model': model,
            'feature_names': X_train.columns.tolist(),
            'classes': model.classes_.tolist()
        }
        
        results = {
            'model_type': 'Logistic Regression',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'model_key': model_key,
            'classification_report': classification_report(y_test, test_pred),
            'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
        }
        
        # Add AUC if binary classification
        if len(model.classes_) == 2:
            train_auc = roc_auc_score(y_train, train_pred_proba[:, 1])
            test_auc = roc_auc_score(y_test, test_pred_proba[:, 1])
            results.update({
                'train_auc': train_auc,
                'test_auc': test_auc
            })
        
        logger.info(f"Logistic Regression trained. Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
        
        return results
    
    def train_neural_network(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Train Neural Network classifier."""
        
        logger.info("Training Neural Network classifier")
        
        # Initialize model
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_pred_proba = model.predict_proba(X_train)
        test_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        # Store model
        model_key = 'neural_network'
        self.trained_models[model_key] = {
            'model': model,
            'feature_names': X_train.columns.tolist(),
            'classes': model.classes_.tolist()
        }
        
        results = {
            'model_type': 'Neural Network',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'model_key': model_key,
            'classification_report': classification_report(y_test, test_pred),
            'confusion_matrix': confusion_matrix(y_test, test_pred).tolist(),
            'n_iterations': model.n_iter_
        }
        
        # Add AUC if binary classification
        if len(model.classes_) == 2:
            train_auc = roc_auc_score(y_train, train_pred_proba[:, 1])
            test_auc = roc_auc_score(y_test, test_pred_proba[:, 1])
            results.update({
                'train_auc': train_auc,
                'test_auc': test_auc
            })
        
        logger.info(f"Neural Network trained. Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
        
        return results
    
    def predict_risk_class(self, model_key: str, features: pd.DataFrame) -> Dict:
        """Predict risk class for new data."""
        
        if model_key not in self.trained_models:
            raise ValueError(f"Model {model_key} not found. Train the model first.")
        
        model_info = self.trained_models[model_key]
        model = model_info['model']
        
        # Ensure features match training features
        training_features = model_info['feature_names']
        features_aligned = features.reindex(columns=training_features, fill_value=0)
        
        try:
            # Make predictions
            predictions = model.predict(features_aligned)
            probabilities = model.predict_proba(features_aligned)
            
            # Decode predictions if needed
            if model_key in self.label_encoders:
                le = self.label_encoders[model_key]
                predictions = le.inverse_transform(predictions)
            
            return {
                'model_key': model_key,
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'classes': model_info.get('classes', model.classes_.tolist())
            }
            
        except Exception as e:
            logger.error(f"Error making predictions with {model_key}: {e}")
            return {'error': str(e)}
    
    def train_all_models(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Train all available classification models."""
        
        logger.info("Training all classification models")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_classification_data(features, target)
        
        results = {}
        
        # Train Random Forest
        try:
            rf_results = self.train_random_forest(X_train, y_train, X_test, y_test)
            results['random_forest'] = rf_results
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            results['random_forest'] = {'error': str(e)}
        
        # Train XGBoost
        try:
            xgb_results = self.train_xgboost(X_train, y_train, X_test, y_test)
            results['xgboost'] = xgb_results
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
            results['xgboost'] = {'error': str(e)}
        
        # Train Logistic Regression
        try:
            lr_results = self.train_logistic_regression(X_train, y_train, X_test, y_test)
            results['logistic_regression'] = lr_results
        except Exception as e:
            logger.error(f"Error training Logistic Regression: {e}")
            results['logistic_regression'] = {'error': str(e)}
        
        # Train Neural Network
        try:
            nn_results = self.train_neural_network(X_train, y_train, X_test, y_test)
            results['neural_network'] = nn_results
        except Exception as e:
            logger.error(f"Error training Neural Network: {e}")
            results['neural_network'] = {'error': str(e)}
        
        # Find best model
        best_model = None
        best_score = 0
        
        for model_name, model_results in results.items():
            if 'error' not in model_results:
                test_f1 = model_results.get('test_f1', 0)
                if test_f1 > best_score:
                    best_score = test_f1
                    best_model = model_name
        
        results['best_model'] = best_model
        results['best_score'] = best_score
        
        logger.info(f"All models trained. Best model: {best_model} (F1: {best_score:.4f})")
        
        return results
    
    def perform_hyperparameter_tuning(self, model_type: str, 
                                    X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Perform hyperparameter tuning for specified model."""
        
        logger.info(f"Performing hyperparameter tuning for {model_type}")
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'xgboost' and xgb is not None:
            model = xgb.XGBClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:
            return {'error': f'Hyperparameter tuning not implemented for {model_type}'}
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=5, 
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save_models(self, filepath_prefix: str):
        """Save all trained models."""
        
        for model_key, model_info in self.trained_models.items():
            filepath = f"{filepath_prefix}_{model_key}.pkl"
            
            try:
                joblib.dump(model_info, filepath)
                logger.info(f"Saved model {model_key} to {filepath}")
                
            except Exception as e:
                logger.error(f"Error saving model {model_key}: {e}")
    
    def load_models(self, filepath_prefix: str):
        """Load trained models from disk."""
        
        import os
        import glob
        
        # Find all model files
        pattern = f"{filepath_prefix}_*.pkl"
        model_files = glob.glob(pattern)
        
        for filepath in model_files:
            try:
                model_key = os.path.basename(filepath).replace(f"{os.path.basename(filepath_prefix)}_", "").replace(".pkl", "")
                model_info = joblib.load(filepath)
                self.trained_models[model_key] = model_info
                logger.info(f"Loaded model {model_key} from {filepath}")
                
            except Exception as e:
                logger.error(f"Error loading model from {filepath}: {e}")