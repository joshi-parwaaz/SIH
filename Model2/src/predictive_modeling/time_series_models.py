"""Time series models for hazard prediction."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

# Sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib

from config import config

logger = logging.getLogger(__name__)

class TimeSeriesModels:
    """Time series models for hazard prediction."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_configs = config.models
        self.trained_models = {}
        
    def prepare_lstm_data(self, data: pd.DataFrame, target_column: str,
                         sequence_length: int = 24, test_size: float = 0.2) -> Tuple:
        """Prepare data for LSTM model."""
        
        # Ensure data is sorted by time
        data = data.sort_index()
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[[target_column]])
        self.scalers[target_column] = scaler
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split into train/test
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def build_lstm_model(self, sequence_length: int = 24, features: int = 1) -> Sequential:
        """Build LSTM model architecture."""
        
        lstm_config = self.model_configs.get('lstm', {})
        
        model = Sequential([
            LSTM(lstm_config.get('hidden_units', 128), 
                 return_sequences=True, 
                 input_shape=(sequence_length, features)),
            Dropout(0.2),
            LSTM(lstm_config.get('hidden_units', 128) // 2, 
                 return_sequences=False),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_lstm_model(self, data: pd.DataFrame, target_column: str,
                        sequence_length: int = None) -> Dict:
        """Train LSTM model for time series prediction."""
        
        logger.info(f"Training LSTM model for {target_column}")
        
        if sequence_length is None:
            sequence_length = self.model_configs.get('lstm', {}).get('sequence_length', 24)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_lstm_data(
            data, target_column, sequence_length
        )
        
        # Build model
        model = self.build_lstm_model(sequence_length)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
        ]
        
        # Train model
        lstm_config = self.model_configs.get('lstm', {})
        history = model.fit(
            X_train, y_train,
            epochs=lstm_config.get('epochs', 100),
            batch_size=lstm_config.get('batch_size', 32),
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate model
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Inverse transform predictions
        scaler = self.scalers[target_column]
        train_pred_inv = scaler.inverse_transform(train_pred)
        test_pred_inv = scaler.inverse_transform(test_pred)
        y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_pred_inv))
        test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred_inv))
        train_mae = mean_absolute_error(y_train_inv, train_pred_inv)
        test_mae = mean_absolute_error(y_test_inv, test_pred_inv)
        
        # Store model
        model_key = f'lstm_{target_column}'
        self.trained_models[model_key] = {
            'model': model,
            'scaler': scaler,
            'sequence_length': sequence_length,
            'target_column': target_column
        }
        
        results = {
            'model_type': 'LSTM',
            'target_column': target_column,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'model_key': model_key,
            'history': history.history
        }
        
        logger.info(f"LSTM model trained. Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")
        
        return results
    
    def build_cnn_lstm_model(self, sequence_length: int = 24, features: int = 1) -> Model:
        """Build CNN-LSTM hybrid model."""
        
        inputs = Input(shape=(sequence_length, features))
        
        # CNN layers for feature extraction
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # LSTM layers for temporal modeling
        x = LSTM(50, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(25)(x)
        x = Dropout(0.2)(x)
        
        # Dense layers
        x = Dense(25, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_arima_model(self, data: pd.Series, order: Tuple[int, int, int] = (2, 1, 2)) -> Dict:
        """Train ARIMA model for time series prediction."""
        
        logger.info(f"Training ARIMA model with order {order}")
        
        try:
            # Prepare data
            data_clean = data.dropna()
            
            # Split data
            split_idx = int(len(data_clean) * 0.8)
            train_data = data_clean[:split_idx]
            test_data = data_clean[split_idx:]
            
            # Fit ARIMA model
            model = ARIMA(train_data, order=order)
            fitted_model = model.fit()
            
            # Make predictions
            train_pred = fitted_model.fittedvalues
            test_pred = fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(train_data[1:], train_pred[1:]))
            test_rmse = np.sqrt(mean_squared_error(test_data, test_pred))
            train_mae = mean_absolute_error(train_data[1:], train_pred[1:])
            test_mae = mean_absolute_error(test_data, test_pred)
            
            # Store model
            model_key = f'arima_{data.name}'
            self.trained_models[model_key] = {
                'model': fitted_model,
                'order': order,
                'target_column': data.name
            }
            
            results = {
                'model_type': 'ARIMA',
                'target_column': data.name,
                'order': order,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'model_key': model_key,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
            logger.info(f"ARIMA model trained. Test RMSE: {test_rmse:.4f}, AIC: {fitted_model.aic:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            return {'error': str(e)}
    
    def train_sarima_model(self, data: pd.Series, 
                          order: Tuple[int, int, int] = (1, 1, 1),
                          seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)) -> Dict:
        """Train SARIMA model for seasonal time series."""
        
        logger.info(f"Training SARIMA model with order {order}, seasonal {seasonal_order}")
        
        try:
            data_clean = data.dropna()
            
            # Split data
            split_idx = int(len(data_clean) * 0.8)
            train_data = data_clean[:split_idx]
            test_data = data_clean[split_idx:]
            
            # Fit SARIMA model
            model = SARIMAX(
                train_data, 
                order=order, 
                seasonal_order=seasonal_order
            )
            fitted_model = model.fit(disp=False)
            
            # Make predictions
            train_pred = fitted_model.fittedvalues
            test_pred = fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(train_data, train_pred))
            test_rmse = np.sqrt(mean_squared_error(test_data, test_pred))
            train_mae = mean_absolute_error(train_data, train_pred)
            test_mae = mean_absolute_error(test_data, test_pred)
            
            # Store model
            model_key = f'sarima_{data.name}'
            self.trained_models[model_key] = {
                'model': fitted_model,
                'order': order,
                'seasonal_order': seasonal_order,
                'target_column': data.name
            }
            
            results = {
                'model_type': 'SARIMA',
                'target_column': data.name,
                'order': order,
                'seasonal_order': seasonal_order,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'model_key': model_key,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
            logger.info(f"SARIMA model trained. Test RMSE: {test_rmse:.4f}, AIC: {fitted_model.aic:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training SARIMA model: {e}")
            return {'error': str(e)}
    
    def train_prophet_model(self, data: pd.DataFrame, date_column: str, 
                           value_column: str) -> Dict:
        """Train Prophet model for time series forecasting."""
        
        if Prophet is None:
            logger.error("Prophet not available. Install with: pip install prophet")
            return {'error': 'Prophet not installed'}
        
        logger.info(f"Training Prophet model for {value_column}")
        
        try:
            # Prepare data for Prophet
            prophet_data = data[[date_column, value_column]].copy()
            prophet_data.columns = ['ds', 'y']
            prophet_data = prophet_data.dropna()
            
            # Split data
            split_idx = int(len(prophet_data) * 0.8)
            train_data = prophet_data[:split_idx]
            test_data = prophet_data[split_idx:]
            
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_mode='multiplicative'
            )
            
            # Add custom seasonalities for ocean data
            model.add_seasonality(
                name='tidal', 
                period=0.5,  # ~12 hours (tidal cycle)
                fourier_order=3
            )
            
            model.fit(train_data)
            
            # Make predictions
            future = model.make_future_dataframe(periods=len(test_data), freq='H')
            forecast = model.predict(future)
            
            # Extract predictions
            train_pred = forecast['yhat'][:len(train_data)]
            test_pred = forecast['yhat'][len(train_data):]
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(train_data['y'], train_pred))
            test_rmse = np.sqrt(mean_squared_error(test_data['y'], test_pred))
            train_mae = mean_absolute_error(train_data['y'], train_pred)
            test_mae = mean_absolute_error(test_data['y'], test_pred)
            
            # Store model
            model_key = f'prophet_{value_column}'
            self.trained_models[model_key] = {
                'model': model,
                'target_column': value_column,
                'date_column': date_column
            }
            
            results = {
                'model_type': 'Prophet',
                'target_column': value_column,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'model_key': model_key,
                'forecast': forecast
            }
            
            logger.info(f"Prophet model trained. Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            return {'error': str(e)}
    
    def predict_future(self, model_key: str, steps: int = 24) -> Dict:
        """Make future predictions using trained model."""
        
        if model_key not in self.trained_models:
            raise ValueError(f"Model {model_key} not found. Train the model first.")
        
        model_info = self.trained_models[model_key]
        model_type = model_key.split('_')[0]
        
        try:
            if model_type == 'lstm':
                # LSTM prediction logic would go here
                # This is a simplified version
                predictions = np.random.normal(0.5, 0.1, steps)  # Placeholder
                
            elif model_type in ['arima', 'sarima']:
                model = model_info['model']
                predictions = model.forecast(steps=steps)
                
            elif model_type == 'prophet':
                model = model_info['model']
                future = model.make_future_dataframe(periods=steps, freq='H')
                forecast = model.predict(future)
                predictions = forecast['yhat'][-steps:].values
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            return {
                'model_key': model_key,
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                'steps': steps,
                'model_type': model_type
            }
            
        except Exception as e:
            logger.error(f"Error making predictions with {model_key}: {e}")
            return {'error': str(e)}
    
    def evaluate_model(self, model_key: str, test_data: pd.Series) -> Dict:
        """Evaluate trained model on test data."""
        
        if model_key not in self.trained_models:
            raise ValueError(f"Model {model_key} not found")
        
        # Make predictions
        predictions = self.predict_future(model_key, len(test_data))
        
        if 'error' in predictions:
            return predictions
        
        pred_values = np.array(predictions['predictions'])
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test_data, pred_values))
        mae = mean_absolute_error(test_data, pred_values)
        r2 = r2_score(test_data, pred_values)
        
        # Calculate directional accuracy
        direction_actual = np.diff(test_data) > 0
        direction_pred = np.diff(pred_values) > 0
        directional_accuracy = np.mean(direction_actual == direction_pred)
        
        return {
            'model_key': model_key,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'directional_accuracy': directional_accuracy,
            'predictions': pred_values.tolist()
        }
    
    def save_models(self, filepath_prefix: str):
        """Save all trained models."""
        
        for model_key, model_info in self.trained_models.items():
            filepath = f"{filepath_prefix}_{model_key}.pkl"
            
            try:
                if 'model' in model_info and hasattr(model_info['model'], 'save'):
                    # TensorFlow/Keras models
                    model_info['model'].save(filepath.replace('.pkl', '.h5'))
                    # Save additional info
                    model_data = {k: v for k, v in model_info.items() if k != 'model'}
                    joblib.dump(model_data, filepath)
                else:
                    # Other models
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
                
                # Load model info
                model_info = joblib.load(filepath)
                
                # Check for TensorFlow model
                h5_filepath = filepath.replace('.pkl', '.h5')
                if os.path.exists(h5_filepath):
                    model_info['model'] = tf.keras.models.load_model(h5_filepath)
                
                self.trained_models[model_key] = model_info
                logger.info(f"Loaded model {model_key} from {filepath}")
                
            except Exception as e:
                logger.error(f"Error loading model from {filepath}: {e}")