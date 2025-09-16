"""Model training orchestration and management."""

import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import joblib
from pathlib import Path

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, regression_report

# Import our custom modules
from .time_series_models import TimeSeriesModels
from .classification_models import ClassificationModels
from .spatial_clustering import SpatialClustering
from .model_ensemble import ModelEnsemble

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Orchestrates training of all predictive models."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._load_default_config()
        self.models = {}
        self.training_history = {}
        self.data_processors = {}
        
        # Initialize model components
        self.time_series_models = TimeSeriesModels()
        self.classification_models = ClassificationModels()
        self.spatial_clustering = SpatialClustering()
        self.model_ensemble = ModelEnsemble()
        
        # Create output directories
        self.model_dir = self.config.get('model_directory', 'models')
        self.results_dir = self.config.get('results_directory', 'results')
        self._create_directories()
        
    def _load_default_config(self) -> Dict:
        """Load default configuration."""
        
        return {
            'model_directory': 'models',
            'results_directory': 'results',
            'train_test_split': 0.2,
            'validation_split': 0.2,
            'random_state': 42,
            'cross_validation_folds': 5,
            'hyperparameter_tuning': True,
            'ensemble_methods': ['voting', 'stacking', 'weighted_average'],
            'model_types': {
                'time_series': ['lstm', 'arima', 'prophet'],
                'classification': ['random_forest', 'xgboost', 'neural_network'],
                'clustering': ['dbscan', 'kmeans']
            }
        }
    
    def _create_directories(self):
        """Create necessary directories."""
        
        for directory in [self.model_dir, self.results_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_path: str):
        """Load configuration from file."""
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def prepare_data(self, data: pd.DataFrame, target_column: str,
                    task_type: str = 'classification') -> Dict:
        """Prepare data for training."""
        
        logger.info(f"Preparing data for {task_type} task")
        
        # Basic data validation
        if target_column not in data.columns:
            raise ValueError(f"Target column {target_column} not found in data")
        
        # Handle missing values
        data_clean = data.dropna()
        logger.info(f"Removed {len(data) - len(data_clean)} rows with missing values")
        
        # Separate features and target
        X = data_clean.drop(columns=[target_column])
        y = data_clean[target_column]
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            logger.info(f"Encoding categorical columns: {list(categorical_columns)}")
            
            # Simple label encoding for now
            label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
            
            self.data_processors['label_encoders'] = label_encoders
        
        # Handle datetime columns
        datetime_columns = X.select_dtypes(include=['datetime64']).columns
        if len(datetime_columns) > 0:
            logger.info(f"Processing datetime columns: {list(datetime_columns)}")
            
            for col in datetime_columns:
                # Extract temporal features
                X[f'{col}_year'] = X[col].dt.year
                X[f'{col}_month'] = X[col].dt.month
                X[f'{col}_day'] = X[col].dt.day
                X[f'{col}_hour'] = X[col].dt.hour
                X[f'{col}_dayofweek'] = X[col].dt.dayofweek
                
                # Drop original datetime column
                X = X.drop(columns=[col])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        self.data_processors['scaler'] = scaler
        
        # Handle target variable for classification
        if task_type == 'classification':
            if y.dtype == 'object':
                target_encoder = LabelEncoder()
                y_encoded = target_encoder.fit_transform(y)
                self.data_processors['target_encoder'] = target_encoder
                y = pd.Series(y_encoded, index=y.index)
        
        # Train-test split
        test_size = self.config.get('train_test_split', 0.2)
        random_state = self.config.get('random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state,
            stratify=y if task_type == 'classification' else None
        )
        
        # Validation split
        val_size = self.config.get('validation_split', 0.2)
        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=random_state,
                stratify=y_train if task_type == 'classification' else None
            )
        else:
            X_val, y_val = None, None
        
        data_splits = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_val': X_val,
            'y_val': y_val,
            'feature_names': X.columns.tolist(),
            'n_features': X_scaled.shape[1],
            'n_samples': len(data_clean),
            'task_type': task_type
        }
        
        logger.info(f"Data preparation complete. Training samples: {len(X_train)}, "
                   f"Test samples: {len(X_test)}, Features: {X_scaled.shape[1]}")
        
        return data_splits
    
    def train_time_series_models(self, data: pd.DataFrame, 
                                target_column: str,
                                time_column: str = 'timestamp') -> Dict:
        """Train time series prediction models."""
        
        logger.info("Training time series models")
        
        results = {}
        model_types = self.config.get('model_types', {}).get('time_series', ['lstm', 'arima'])
        
        # Prepare time series data
        if time_column not in data.columns:
            logger.warning(f"Time column {time_column} not found. Using index.")
            time_series_data = data[target_column].values
        else:
            data_sorted = data.sort_values(time_column)
            time_series_data = data_sorted[target_column].values
        
        # Train each model type
        for model_type in model_types:
            try:
                logger.info(f"Training {model_type} model")
                
                if model_type == 'lstm':
                    result = self.time_series_models.train_lstm_model(
                        time_series_data,
                        sequence_length=self.config.get('lstm_sequence_length', 10),
                        epochs=self.config.get('lstm_epochs', 50)
                    )
                    
                elif model_type == 'arima':
                    result = self.time_series_models.train_arima_model(time_series_data)
                    
                elif model_type == 'sarima':
                    result = self.time_series_models.train_sarima_model(time_series_data)
                    
                elif model_type == 'prophet':
                    # Prepare Prophet data format
                    if time_column in data.columns:
                        prophet_data = data[[time_column, target_column]].copy()
                        prophet_data.columns = ['ds', 'y']
                    else:
                        prophet_data = pd.DataFrame({
                            'ds': pd.date_range(start='2020-01-01', periods=len(data), freq='D'),
                            'y': data[target_column].values
                        })
                    
                    result = self.time_series_models.train_prophet_model(prophet_data)
                
                results[model_type] = result
                
                # Add to ensemble
                if result.get('success', False):
                    model_name = f"time_series_{model_type}"
                    self.model_ensemble.add_base_model(
                        model_name, 
                        result['model'], 
                        'regression'
                    )
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
                results[model_type] = {'success': False, 'error': str(e)}
        
        return results
    
    def train_classification_models(self, data_splits: Dict) -> Dict:
        """Train classification models."""
        
        logger.info("Training classification models")
        
        results = {}
        model_types = self.config.get('model_types', {}).get('classification', 
                                                             ['random_forest', 'xgboost'])
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        # Train each model type
        for model_type in model_types:
            try:
                logger.info(f"Training {model_type} classification model")
                
                if model_type == 'random_forest':
                    result = self.classification_models.train_random_forest(
                        X_train, y_train, X_test, y_test
                    )
                    
                elif model_type == 'xgboost':
                    result = self.classification_models.train_xgboost(
                        X_train, y_train, X_test, y_test
                    )
                    
                elif model_type == 'logistic_regression':
                    result = self.classification_models.train_logistic_regression(
                        X_train, y_train, X_test, y_test
                    )
                    
                elif model_type == 'neural_network':
                    result = self.classification_models.train_neural_network(
                        X_train, y_train, X_test, y_test,
                        epochs=self.config.get('nn_epochs', 100)
                    )
                
                results[model_type] = result
                
                # Add to ensemble
                if result.get('success', False):
                    model_name = f"classification_{model_type}"
                    self.model_ensemble.add_base_model(
                        model_name, 
                        result['model'], 
                        'classification'
                    )
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
                results[model_type] = {'success': False, 'error': str(e)}
        
        return results
    
    def train_clustering_models(self, data: pd.DataFrame) -> Dict:
        """Train spatial clustering models."""
        
        logger.info("Training clustering models")
        
        results = {}
        
        # Ensure spatial columns exist
        if not all(col in data.columns for col in ['latitude', 'longitude']):
            logger.warning("Latitude/longitude columns not found. Skipping clustering.")
            return {}
        
        # Prepare spatial data
        spatial_data = self.spatial_clustering.prepare_spatial_data(data)
        
        model_types = self.config.get('model_types', {}).get('clustering', ['dbscan', 'kmeans'])
        
        for model_type in model_types:
            try:
                logger.info(f"Training {model_type} clustering")
                
                if model_type == 'dbscan':
                    # Optimize parameters if requested
                    if self.config.get('hyperparameter_tuning', True):
                        opt_result = self.spatial_clustering.optimize_clustering_parameters(
                            spatial_data, 'dbscan'
                        )
                        params = opt_result['best_params']
                    else:
                        params = {'eps': 0.5, 'min_samples': 5}
                    
                    result = self.spatial_clustering.perform_dbscan_clustering(
                        spatial_data, **params
                    )
                    
                elif model_type == 'kmeans':
                    # Optimize parameters if requested
                    if self.config.get('hyperparameter_tuning', True):
                        opt_result = self.spatial_clustering.optimize_clustering_parameters(
                            spatial_data, 'kmeans'
                        )
                        params = opt_result['best_params']
                    else:
                        params = {'n_clusters': 5}
                    
                    result = self.spatial_clustering.perform_kmeans_clustering(
                        spatial_data, **params
                    )
                    
                elif model_type == 'hierarchical':
                    result = self.spatial_clustering.perform_hierarchical_clustering(
                        spatial_data, n_clusters=5
                    )
                
                results[model_type] = result
                
                # Identify hotspots
                if result.get('cluster_statistics'):
                    hotspots = self.spatial_clustering.identify_hotspots(result)
                    results[f'{model_type}_hotspots'] = hotspots
                
            except Exception as e:
                logger.error(f"Error training {model_type} clustering: {e}")
                results[model_type] = {'success': False, 'error': str(e)}
        
        return results
    
    def create_model_ensembles(self, data_splits: Dict) -> Dict:
        """Create and train ensemble models."""
        
        logger.info("Creating model ensembles")
        
        results = {}
        ensemble_methods = self.config.get('ensemble_methods', ['voting', 'stacking'])
        
        # Get available classification models for ensemble
        classification_models = [
            name for name in self.model_ensemble.base_models.keys()
            if 'classification' in name
        ]
        
        if len(classification_models) < 2:
            logger.warning("Need at least 2 models for ensemble. Skipping ensemble creation.")
            return {}
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        for method in ensemble_methods:
            try:
                ensemble_name = f"ensemble_{method}"
                
                if method == 'voting':
                    success = self.model_ensemble.create_voting_ensemble(
                        classification_models, 
                        ensemble_name,
                        voting='soft'
                    )
                    
                elif method == 'stacking':
                    success = self.model_ensemble.create_stacking_ensemble(
                        classification_models,
                        ensemble_name
                    )
                    
                elif method == 'weighted_average':
                    success = self.model_ensemble.create_weighted_average_ensemble(
                        classification_models,
                        ensemble_name
                    )
                
                if success:
                    # Train ensemble
                    trained = self.model_ensemble.train_ensemble(
                        ensemble_name, X_train.values, y_train.values
                    )
                    
                    if trained:
                        # Evaluate ensemble
                        metrics = self.model_ensemble.evaluate_ensemble(
                            ensemble_name, X_test.values, y_test.values
                        )
                        
                        results[ensemble_name] = {
                            'success': True,
                            'method': method,
                            'models': classification_models,
                            'metrics': metrics
                        }
                        
                        # Optimize weights for weighted average
                        if method == 'weighted_average' and data_splits['X_val'] is not None:
                            self.model_ensemble.optimize_ensemble_weights(
                                ensemble_name,
                                data_splits['X_val'].values,
                                data_splits['y_val'].values
                            )
                
            except Exception as e:
                logger.error(f"Error creating {method} ensemble: {e}")
                results[f"ensemble_{method}"] = {'success': False, 'error': str(e)}
        
        return results
    
    def cross_validate_models(self, data_splits: Dict) -> Dict:
        """Perform cross-validation on trained models."""
        
        logger.info("Performing cross-validation")
        
        cv_results = {}
        cv_folds = self.config.get('cross_validation_folds', 5)
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        
        # Cross-validate classification models
        for model_name, model_info in self.model_ensemble.base_models.items():
            if model_info['type'] == 'classification':
                try:
                    model = model_info['model']
                    
                    # Perform cross-validation
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=cv_folds, 
                        scoring='accuracy'
                    )
                    
                    cv_results[model_name] = {
                        'cv_scores': cv_scores.tolist(),
                        'mean_score': cv_scores.mean(),
                        'std_score': cv_scores.std(),
                        'cv_folds': cv_folds
                    }
                    
                    logger.info(f"{model_name} CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                    
                except Exception as e:
                    logger.error(f"Error in cross-validation for {model_name}: {e}")
                    cv_results[model_name] = {'error': str(e)}
        
        return cv_results
    
    def hyperparameter_tuning(self, data_splits: Dict) -> Dict:
        """Perform hyperparameter tuning for select models."""
        
        if not self.config.get('hyperparameter_tuning', True):
            logger.info("Hyperparameter tuning disabled")
            return {}
        
        logger.info("Performing hyperparameter tuning")
        
        tuning_results = {}
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        for model_name, model_info in self.model_ensemble.base_models.items():
            if model_info['type'] == 'classification':
                try:
                    # Extract model type from name
                    model_type = model_name.split('_')[-1]
                    
                    if model_type in param_grids:
                        logger.info(f"Tuning hyperparameters for {model_name}")
                        
                        model = model_info['model']
                        param_grid = param_grids[model_type]
                        
                        # Perform grid search
                        grid_search = GridSearchCV(
                            model, param_grid,
                            cv=3,  # Reduced for faster tuning
                            scoring='accuracy',
                            n_jobs=-1
                        )
                        
                        grid_search.fit(X_train, y_train)
                        
                        tuning_results[model_name] = {
                            'best_params': grid_search.best_params_,
                            'best_score': grid_search.best_score_,
                            'cv_results': {
                                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                                'params': grid_search.cv_results_['params']
                            }
                        }
                        
                        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                        
                except Exception as e:
                    logger.error(f"Error tuning {model_name}: {e}")
                    tuning_results[model_name] = {'error': str(e)}
        
        return tuning_results
    
    def train_full_pipeline(self, data: pd.DataFrame, 
                           target_column: str,
                           time_column: str = None,
                           task_type: str = 'classification') -> Dict:
        """Train complete ML pipeline."""
        
        logger.info("Starting full pipeline training")
        
        pipeline_results = {
            'timestamp': datetime.now().isoformat(),
            'task_type': task_type,
            'data_info': {
                'n_samples': len(data),
                'n_features': len(data.columns) - 1,
                'target_column': target_column
            }
        }
        
        try:
            # 1. Data preparation
            logger.info("Step 1: Data preparation")
            data_splits = self.prepare_data(data, target_column, task_type)
            pipeline_results['data_preparation'] = {
                'success': True,
                'train_samples': len(data_splits['X_train']),
                'test_samples': len(data_splits['X_test']),
                'n_features': data_splits['n_features']
            }
            
            # 2. Time series models (if applicable)
            if time_column and task_type == 'regression':
                logger.info("Step 2: Time series modeling")
                ts_results = self.train_time_series_models(data, target_column, time_column)
                pipeline_results['time_series_models'] = ts_results
            
            # 3. Classification/Regression models
            logger.info("Step 3: Classification/Regression modeling")
            if task_type == 'classification':
                model_results = self.train_classification_models(data_splits)
            else:
                # For regression, we would implement similar logic
                model_results = self.train_classification_models(data_splits)  # Placeholder
            
            pipeline_results['predictive_models'] = model_results
            
            # 4. Spatial clustering
            logger.info("Step 4: Spatial clustering")
            clustering_results = self.train_clustering_models(data)
            pipeline_results['clustering_models'] = clustering_results
            
            # 5. Model ensembles
            logger.info("Step 5: Model ensembles")
            ensemble_results = self.create_model_ensembles(data_splits)
            pipeline_results['ensemble_models'] = ensemble_results
            
            # 6. Cross-validation
            logger.info("Step 6: Cross-validation")
            cv_results = self.cross_validate_models(data_splits)
            pipeline_results['cross_validation'] = cv_results
            
            # 7. Hyperparameter tuning
            logger.info("Step 7: Hyperparameter tuning")
            tuning_results = self.hyperparameter_tuning(data_splits)
            pipeline_results['hyperparameter_tuning'] = tuning_results
            
            # 8. Save models
            logger.info("Step 8: Saving models")
            self.save_all_models()
            pipeline_results['model_saving'] = {'success': True}
            
            # 9. Generate report
            logger.info("Step 9: Generating report")
            self.generate_training_report(pipeline_results)
            
            pipeline_results['success'] = True
            pipeline_results['completion_time'] = datetime.now().isoformat()
            
            logger.info("Full pipeline training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in pipeline training: {e}")
            pipeline_results['success'] = False
            pipeline_results['error'] = str(e)
        
        # Store training history
        self.training_history[datetime.now().isoformat()] = pipeline_results
        
        return pipeline_results
    
    def save_all_models(self):
        """Save all trained models."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual models
        model_dir = Path(self.model_dir) / timestamp
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save time series models
        self.time_series_models.save_models(str(model_dir / "time_series"))
        
        # Save classification models
        self.classification_models.save_models(str(model_dir / "classification"))
        
        # Save clustering models
        self.spatial_clustering.save_models(str(model_dir / "clustering"))
        
        # Save ensemble models
        for ensemble_name in self.model_ensemble.ensemble_models:
            filepath = model_dir / f"ensemble_{ensemble_name}.pkl"
            self.model_ensemble.save_ensemble(ensemble_name, str(filepath))
        
        # Save data processors
        processors_path = model_dir / "data_processors.pkl"
        joblib.dump(self.data_processors, processors_path)
        
        # Save configuration
        config_path = model_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"All models saved to {model_dir}")
    
    def load_models(self, model_dir: str):
        """Load previously trained models."""
        
        model_path = Path(model_dir)
        
        if not model_path.exists():
            raise ValueError(f"Model directory {model_dir} does not exist")
        
        # Load configuration
        config_path = model_path / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Load data processors
        processors_path = model_path / "data_processors.pkl"
        if processors_path.exists():
            self.data_processors = joblib.load(processors_path)
        
        # Load individual models
        self.time_series_models.load_models(str(model_path / "time_series"))
        self.classification_models.load_models(str(model_path / "classification"))
        self.spatial_clustering.load_models(str(model_path / "clustering"))
        
        # Load ensemble models
        ensemble_files = list(model_path.glob("ensemble_*.pkl"))
        for filepath in ensemble_files:
            ensemble_name = filepath.stem.replace("ensemble_", "")
            self.model_ensemble.load_ensemble(ensemble_name, str(filepath))
        
        logger.info(f"Models loaded from {model_dir}")
    
    def generate_training_report(self, results: Dict):
        """Generate comprehensive training report."""
        
        report_path = Path(self.results_dir) / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Create summary
        summary = {
            'training_timestamp': results.get('timestamp'),
            'task_type': results.get('task_type'),
            'data_info': results.get('data_info'),
            'success': results.get('success', False)
        }
        
        # Model performance summary
        if 'predictive_models' in results:
            model_performances = {}
            for model_name, model_result in results['predictive_models'].items():
                if model_result.get('success') and 'metrics' in model_result:
                    model_performances[model_name] = model_result['metrics']
            summary['model_performances'] = model_performances
        
        # Ensemble performance summary
        if 'ensemble_models' in results:
            ensemble_performances = {}
            for ensemble_name, ensemble_result in results['ensemble_models'].items():
                if ensemble_result.get('success') and 'metrics' in ensemble_result:
                    ensemble_performances[ensemble_name] = ensemble_result['metrics']
            summary['ensemble_performances'] = ensemble_performances
        
        # Best model identification
        if 'cross_validation' in results:
            best_model = None
            best_score = 0
            
            for model_name, cv_result in results['cross_validation'].items():
                if 'mean_score' in cv_result and cv_result['mean_score'] > best_score:
                    best_score = cv_result['mean_score']
                    best_model = model_name
            
            summary['best_model'] = {
                'name': best_model,
                'cv_score': best_score
            }
        
        # Complete results
        full_report = {
            'summary': summary,
            'detailed_results': results,
            'model_info': self.model_ensemble.get_ensemble_info()
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(full_report, f, indent=2, default=str)
        
        logger.info(f"Training report saved to {report_path}")
        
        return full_report
    
    def get_model_recommendations(self, results: Dict) -> Dict:
        """Provide model recommendations based on training results."""
        
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'recommendations': []
        }
        
        # Analyze cross-validation results
        if 'cross_validation' in results:
            cv_scores = {}
            for model_name, cv_result in results['cross_validation'].items():
                if 'mean_score' in cv_result:
                    cv_scores[model_name] = cv_result['mean_score']
            
            if cv_scores:
                best_model = max(cv_scores, key=cv_scores.get)
                recommendations['recommendations'].append({
                    'type': 'best_individual_model',
                    'model': best_model,
                    'score': cv_scores[best_model],
                    'reason': 'Highest cross-validation score'
                })
        
        # Analyze ensemble performance
        if 'ensemble_models' in results:
            ensemble_scores = {}
            for ensemble_name, ensemble_result in results['ensemble_models'].items():
                if ensemble_result.get('success') and 'metrics' in ensemble_result:
                    # Use accuracy for classification, R2 for regression
                    score = ensemble_result['metrics'].get('accuracy', 
                           ensemble_result['metrics'].get('r2', 0))
                    ensemble_scores[ensemble_name] = score
            
            if ensemble_scores:
                best_ensemble = max(ensemble_scores, key=ensemble_scores.get)
                recommendations['recommendations'].append({
                    'type': 'best_ensemble_model',
                    'model': best_ensemble,
                    'score': ensemble_scores[best_ensemble],
                    'reason': 'Highest ensemble performance'
                })
        
        # Model complexity recommendations
        simple_models = ['logistic_regression', 'random_forest']
        complex_models = ['neural_network', 'xgboost']
        
        recommendations['recommendations'].append({
            'type': 'deployment_recommendation',
            'recommendation': 'Use ensemble for highest accuracy, individual models for faster inference',
            'details': {
                'production': 'Consider ensemble models for batch predictions',
                'real_time': 'Use simpler models for low-latency requirements'
            }
        })
        
        return recommendations