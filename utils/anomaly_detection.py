import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed

class IsolationForestDetector:
    """Anomaly detection using Isolation Forest algorithm."""
    
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _preprocess(self, df, fit=False):
        """Preprocess the data for anomaly detection."""
        # Select numerical columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['is_anomaly', 'timestamp']]
        
        X = df[numeric_cols].values
        
        # Scale the data
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, numeric_cols
    
    def fit(self, df):
        """Fit the model to the data."""
        X_scaled, _ = self._preprocess(df, fit=True)
        self.model.fit(X_scaled)
        self.is_fitted = True
    
    def predict(self, df):
        """Predict anomalies in the data."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X_scaled, _ = self._preprocess(df, fit=False)
        
        # Predict (1 for inliers, -1 for outliers)
        predictions = self.model.predict(X_scaled)
        
        # Convert to binary (0 for normal, 1 for anomaly)
        anomalies = np.where(predictions == -1, 1, 0)
        
        # Calculate anomaly scores (negative scores = more anomalous)
        scores = -self.model.score_samples(X_scaled)
        
        # Normalize scores to [0, 1] range
        if len(scores) > 0:
            min_score, max_score = np.min(scores), np.max(scores)
            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(scores)
        else:
            normalized_scores = np.array([])
        
        return anomalies, normalized_scores


class LSTMAutoencoder:
    """Anomaly detection using LSTM Autoencoder."""
    
    def __init__(self, timesteps=10, n_features=None, threshold_multiplier=2.0):
        self.timesteps = timesteps
        self.n_features = n_features
        self.threshold_multiplier = threshold_multiplier
        self.model = None
        self.scaler = StandardScaler()
        self.reconstruction_error_threshold = None
        self.is_fitted = False
    
    def _build_model(self, n_features):
        """Build the LSTM Autoencoder model."""
        # Build encoder
        inputs = Input(shape=(self.timesteps, n_features))
        encoded = LSTM(32, activation='relu', return_sequences=False)(inputs)
        
        # Build decoder
        decoded = RepeatVector(self.timesteps)(encoded)
        decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(n_features))(decoded)
        
        # Build autoencoder
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def _preprocess(self, df, fit=False):
        """Preprocess the data for the LSTM Autoencoder."""
        # Select numerical columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['is_anomaly', 'timestamp']]
        
        X = df[numeric_cols].values
        
        # Scale the data
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_sequences = []
        for i in range(len(X_scaled) - self.timesteps + 1):
            X_sequences.append(X_scaled[i:i+self.timesteps])
        
        if not X_sequences:
            return np.array([]), numeric_cols
        
        return np.array(X_sequences), numeric_cols
    
    def fit(self, df, epochs=10, batch_size=32, validation_split=0.1):
        """Fit the LSTM Autoencoder to the data."""
        X_sequences, numeric_cols = self._preprocess(df, fit=True)
        
        if len(X_sequences) == 0:
            raise ValueError("Not enough data points to create sequences.")
        
        self.n_features = X_sequences.shape[2]
        
        # Build the model
        self.model = self._build_model(self.n_features)
        
        # Train the model
        self.model.fit(
            X_sequences, X_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        # Calculate reconstruction error on training data
        reconstructions = self.model.predict(X_sequences)
        mse = np.mean(np.power(X_sequences - reconstructions, 2), axis=(1, 2))
        
        # Set threshold as mean + std * multiplier
        self.reconstruction_error_threshold = np.mean(mse) + np.std(mse) * self.threshold_multiplier
        
        self.is_fitted = True
    
    def predict(self, df):
        """Predict anomalies in the data."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X_sequences, _ = self._preprocess(df, fit=False)
        
        if len(X_sequences) == 0:
            return np.array([]), np.array([])
        
        # Calculate reconstruction error
        reconstructions = self.model.predict(X_sequences)
        mse = np.mean(np.power(X_sequences - reconstructions, 2), axis=(1, 2))
        
        # Detect anomalies
        anomalies = np.where(mse > self.reconstruction_error_threshold, 1, 0)
        
        # Calculate anomaly scores (0 to 1 range)
        scores = mse / (self.reconstruction_error_threshold * 2)
        scores = np.clip(scores, 0, 1)
        
        # Extend predictions to match original data length
        # (we lose timesteps-1 predictions at the beginning)
        padding = np.zeros(self.timesteps - 1)
        extended_anomalies = np.concatenate([padding, anomalies])
        extended_scores = np.concatenate([padding, scores])
        
        return extended_anomalies, extended_scores


class AnomalyDetectionSystem:
    """Complete anomaly detection system that combines multiple models."""
    
    def __init__(self):
        # Hardware metrics detectors
        self.hw_isolation_forest = IsolationForestDetector(contamination=0.05)
        self.hw_lstm_autoencoder = LSTMAutoencoder(timesteps=10)
        
        # Media metrics detectors
        self.media_isolation_forest = IsolationForestDetector(contamination=0.05)
        self.media_lstm_autoencoder = LSTMAutoencoder(timesteps=10)
        
        # Tracking if models are fitted
        self.hw_if_fitted = False
        self.hw_lstm_fitted = False
        self.media_if_fitted = False
        self.media_lstm_fitted = False
    
    def fit_models(self, hw_df, media_df):
        """Fit all models on historical data."""
        # Only fit if we have enough data
        min_data_points = 50  # Minimum number of data points to fit models
        
        if len(hw_df) >= min_data_points:
            # Fit hardware metrics models
            try:
                self.hw_isolation_forest.fit(hw_df)
                self.hw_if_fitted = True
            except Exception as e:
                print(f"Error fitting hardware Isolation Forest: {e}")
            
            try:
                if len(hw_df) >= min_data_points + 10:  # Need extra points for LSTM sequences
                    self.hw_lstm_autoencoder.fit(hw_df, epochs=5)
                    self.hw_lstm_fitted = True
            except Exception as e:
                print(f"Error fitting hardware LSTM Autoencoder: {e}")
        
        if len(media_df) >= min_data_points:
            # Fit media metrics models
            try:
                self.media_isolation_forest.fit(media_df)
                self.media_if_fitted = True
            except Exception as e:
                print(f"Error fitting media Isolation Forest: {e}")
            
            try:
                if len(media_df) >= min_data_points + 10:  # Need extra points for LSTM sequences
                    self.media_lstm_autoencoder.fit(media_df, epochs=5)
                    self.media_lstm_fitted = True
            except Exception as e:
                print(f"Error fitting media LSTM Autoencoder: {e}")
    
    def detect_anomalies(self, hw_df, media_df):
        """Detect anomalies in current data."""
        results = {
            'hardware': {'anomaly': False, 'score': 0.0, 'model': None},
            'media': {'anomaly': False, 'score': 0.0, 'model': None}
        }
        
        # Detect hardware anomalies
        if self.hw_if_fitted and len(hw_df) > 0:
            try:
                anomalies, scores = self.hw_isolation_forest.predict(hw_df)
                if len(anomalies) > 0 and anomalies[-1] == 1:
                    results['hardware'] = {
                        'anomaly': True,
                        'score': float(scores[-1]),
                        'model': 'IsolationForest'
                    }
            except Exception as e:
                print(f"Error detecting hardware anomalies with Isolation Forest: {e}")
        
        if not results['hardware']['anomaly'] and self.hw_lstm_fitted and len(hw_df) >= self.hw_lstm_autoencoder.timesteps:
            try:
                anomalies, scores = self.hw_lstm_autoencoder.predict(hw_df)
                if len(anomalies) > 0 and anomalies[-1] == 1:
                    results['hardware'] = {
                        'anomaly': True,
                        'score': float(scores[-1]),
                        'model': 'LSTMAutoencoder'
                    }
            except Exception as e:
                print(f"Error detecting hardware anomalies with LSTM Autoencoder: {e}")
        
        # Detect media anomalies
        if self.media_if_fitted and len(media_df) > 0:
            try:
                anomalies, scores = self.media_isolation_forest.predict(media_df)
                if len(anomalies) > 0 and anomalies[-1] == 1:
                    results['media'] = {
                        'anomaly': True,
                        'score': float(scores[-1]),
                        'model': 'IsolationForest'
                    }
            except Exception as e:
                print(f"Error detecting media anomalies with Isolation Forest: {e}")
        
        if not results['media']['anomaly'] and self.media_lstm_fitted and len(media_df) >= self.media_lstm_autoencoder.timesteps:
            try:
                anomalies, scores = self.media_lstm_autoencoder.predict(media_df)
                if len(anomalies) > 0 and anomalies[-1] == 1:
                    results['media'] = {
                        'anomaly': True,
                        'score': float(scores[-1]),
                        'model': 'LSTMAutoencoder'
                    }
            except Exception as e:
                print(f"Error detecting media anomalies with LSTM Autoencoder: {e}")
        
        return results
