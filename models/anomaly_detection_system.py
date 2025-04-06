import numpy as np
import pandas as pd
from models.lstm_autoencoder import LSTMAutoencoder
from models.isolation_forest import IsolationForestDetector
import os
import joblib

class AnomalyDetectionSystem:
    """Combined anomaly detection system for transportation predictive maintenance."""
    
    def __init__(self):
        """Initialize the anomaly detection system with multiple models."""
        self.models = {
            'airplane': {
                'lstm': None,
                'isolation_forest': None
            },
            'truck': {
                'lstm': None,
                'isolation_forest': None
            },
            'railway': {
                'lstm': None,
                'isolation_forest': None
            }
        }
        
        self.feature_columns = {
            'airplane': [
                'engine_temperature', 'engine_vibration', 'fuel_pressure', 
                'oil_pressure', 'rotation_speed', 'altitude', 'airspeed', 
                'exhaust_gas_temp', 'hydraulic_pressure', 'cabin_pressure'
            ],
            'truck': [
                'engine_temperature', 'engine_vibration', 'fuel_pressure', 
                'oil_pressure', 'rotation_speed', 'tire_pressure', 
                'brake_temperature', 'transmission_temp', 'battery_voltage',
                'coolant_level'
            ],
            'railway': [
                'engine_temperature', 'engine_vibration', 'hydraulic_pressure', 
                'oil_pressure', 'rotation_speed', 'axle_temperature', 
                'catenary_voltage', 'pantograph_force', 'brake_cylinder_pressure',
                'traction_motor_temp'
            ]
        }
    
    def _preprocess_data(self, df, vehicle_type):
        """Preprocess data for model training or prediction.
        
        Args:
            df (pd.DataFrame): Data to preprocess
            vehicle_type (str): Type of vehicle
            
        Returns:
            np.array: Preprocessed data
        """
        # Filter relevant columns
        if vehicle_type not in self.feature_columns:
            raise ValueError(f"Unsupported vehicle type: {vehicle_type}")
            
        feature_cols = self.feature_columns[vehicle_type]
        
        # Ensure all necessary columns exist
        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Get only the feature columns
        X = df[feature_cols].values
        
        return X
    
    def fit_models(self, data, vehicle_types=None):
        """Fit all models on historical data.
        
        Args:
            data (dict): Dictionary with DataFrames for each vehicle type
            vehicle_types (list, optional): List of vehicle types to fit models for,
                                            or all if None
        
        Returns:
            self: The fitted model instance
        """
        types_to_fit = vehicle_types if vehicle_types else list(self.models.keys())
        
        for vehicle_type in types_to_fit:
            if vehicle_type not in data or data[vehicle_type].empty:
                print(f"No data available for {vehicle_type}. Skipping model fitting.")
                continue
                
            # Get data for this vehicle type
            df = data[vehicle_type]
            
            # Preprocess data
            X = self._preprocess_data(df, vehicle_type)
            
            # Initialize and fit LSTM Autoencoder
            lstm = LSTMAutoencoder(sequence_length=10)
            lstm.fit(X, epochs=20, batch_size=32)
            self.models[vehicle_type]['lstm'] = lstm
            
            # Initialize and fit Isolation Forest
            iso_forest = IsolationForestDetector()
            iso_forest.fit(X)
            self.models[vehicle_type]['isolation_forest'] = iso_forest
            
            print(f"Models for {vehicle_type} trained successfully.")
        
        return self
    
    def detect_anomalies(self, data, vehicle_types=None):
        """Detect anomalies in current data.
        
        Args:
            data (dict): Dictionary with DataFrames for each vehicle type
            vehicle_types (list, optional): List of vehicle types to detect anomalies for,
                                            or all if None
        
        Returns:
            dict: Dictionary with anomaly detection results for each vehicle type
        """
        results = {}
        
        types_to_check = vehicle_types if vehicle_types else list(self.models.keys())
        
        for vehicle_type in types_to_check:
            if vehicle_type not in data or data[vehicle_type].empty:
                results[vehicle_type] = {
                    'anomaly': False,
                    'model': None,
                    'score': 0.0,
                    'anomaly_metrics': []
                }
                continue
            
            # Get data for this vehicle type
            df = data[vehicle_type]
            
            # Check if models are trained
            if self.models[vehicle_type]['lstm'] is None or self.models[vehicle_type]['isolation_forest'] is None:
                results[vehicle_type] = {
                    'anomaly': False,
                    'model': 'None - Models not trained',
                    'score': 0.0,
                    'anomaly_metrics': []
                }
                continue
                
            # Preprocess data
            X = self._preprocess_data(df, vehicle_type)
            
            # Get predictions from both models
            lstm_result = self.models[vehicle_type]['lstm'].predict(X)
            iso_result = self.models[vehicle_type]['isolation_forest'].predict(X)
            
            # Check results from both models
            lstm_anomaly = lstm_result['anomaly'] if len(lstm_result['predictions']) > 0 else False
            iso_anomaly = iso_result['anomaly'] if len(iso_result['predictions']) > 0 else False
            
            # Combine results (conservative approach: anomaly if either model detects it)
            is_anomaly = lstm_anomaly or iso_anomaly
            
            # Determine which model detected the anomaly and score
            if is_anomaly:
                if lstm_anomaly and iso_anomaly:
                    model = 'Both LSTM and Isolation Forest'
                    # Average the scores, normalizing LSTM scores first
                    lstm_score = np.max(lstm_result['scores']) / lstm_result['threshold'] if len(lstm_result['scores']) > 0 else 0
                    iso_score = np.max(iso_result['scores']) if len(iso_result['scores']) > 0 else 0
                    score = (lstm_score + iso_score) / 2
                elif lstm_anomaly:
                    model = 'LSTM Autoencoder'
                    score = np.max(lstm_result['scores']) / lstm_result['threshold'] if len(lstm_result['scores']) > 0 else 0
                else:
                    model = 'Isolation Forest'
                    score = np.max(iso_result['scores']) if len(iso_result['scores']) > 0 else 0
                    
                # Get anomalous metrics
                anomaly_metrics = []
                
                # If the last entry has an anomaly, use the anomaly_metrics field if available
                if 'anomaly_metrics' in df.columns[-1] and len(df['anomaly_metrics']) > 0:
                    last_entry = df.iloc[-1]
                    if isinstance(last_entry['anomaly_metrics'], list) and len(last_entry['anomaly_metrics']) > 0:
                        anomaly_metrics = last_entry['anomaly_metrics']
            else:
                model = 'None'
                score = 0.0
                anomaly_metrics = []
            
            results[vehicle_type] = {
                'anomaly': is_anomaly,
                'model': model,
                'score': float(score),
                'anomaly_metrics': anomaly_metrics
            }
        
        return results
    
    def save_models(self, model_dir='saved_models'):
        """Save all trained models to disk.
        
        Args:
            model_dir (str): Directory to save models to
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        for vehicle_type, models in self.models.items():
            vehicle_dir = os.path.join(model_dir, vehicle_type)
            if not os.path.exists(vehicle_dir):
                os.makedirs(vehicle_dir)
                
            # Save LSTM
            if models['lstm'] is not None:
                lstm_dir = os.path.join(vehicle_dir, 'lstm')
                if not os.path.exists(lstm_dir):
                    os.makedirs(lstm_dir)
                models['lstm'].save(lstm_dir)
                
            # Save Isolation Forest
            if models['isolation_forest'] is not None:
                iso_dir = os.path.join(vehicle_dir, 'isolation_forest')
                if not os.path.exists(iso_dir):
                    os.makedirs(iso_dir)
                models['isolation_forest'].save(iso_dir)
                
        # Save feature columns configuration
        joblib.dump(self.feature_columns, os.path.join(model_dir, 'feature_columns.pkl'))
        
        print(f"All models saved to {model_dir}")
    
    @classmethod
    def load_models(cls, model_dir='saved_models'):
        """Load all models from disk.
        
        Args:
            model_dir (str): Directory to load models from
            
        Returns:
            AnomalyDetectionSystem: Loaded model instance
        """
        instance = cls()
        
        # Load feature columns configuration
        feature_cols_path = os.path.join(model_dir, 'feature_columns.pkl')
        if os.path.exists(feature_cols_path):
            instance.feature_columns = joblib.load(feature_cols_path)
        
        # Load models for each vehicle type
        for vehicle_type in instance.models.keys():
            vehicle_dir = os.path.join(model_dir, vehicle_type)
            if not os.path.exists(vehicle_dir):
                continue
                
            # Load LSTM
            lstm_dir = os.path.join(vehicle_dir, 'lstm')
            if os.path.exists(lstm_dir):
                try:
                    instance.models[vehicle_type]['lstm'] = LSTMAutoencoder.load(lstm_dir)
                    print(f"LSTM model for {vehicle_type} loaded successfully.")
                except Exception as e:
                    print(f"Error loading LSTM model for {vehicle_type}: {e}")
                    
            # Load Isolation Forest
            iso_dir = os.path.join(vehicle_dir, 'isolation_forest')
            if os.path.exists(iso_dir):
                try:
                    instance.models[vehicle_type]['isolation_forest'] = IsolationForestDetector.load(iso_dir)
                    print(f"Isolation Forest model for {vehicle_type} loaded successfully.")
                except Exception as e:
                    print(f"Error loading Isolation Forest model for {vehicle_type}: {e}")
        
        return instance