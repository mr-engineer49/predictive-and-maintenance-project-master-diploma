import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


class PCADetector:
    """Anomaly detection using PCA reconstruction error."""
    
    def __init__(self, n_components=0.95, threshold_multiplier=2.0):
        self.n_components = n_components
        self.threshold_multiplier = threshold_multiplier
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.reconstruction_error_threshold = None
        self.is_fitted = False
    
    def _preprocess(self, df, fit=False):
        """Preprocess the data for PCA."""
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
        """Fit the PCA model to the data."""
        X_scaled, _ = self._preprocess(df, fit=True)
        
        if len(X_scaled) == 0:
            raise ValueError("Not enough data points.")
        
        # Fit PCA
        self.pca.fit(X_scaled)
        
        # Calculate reconstruction error on training data
        X_transformed = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        mse = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
        
        # Set threshold as mean + std * multiplier
        self.reconstruction_error_threshold = np.mean(mse) + np.std(mse) * self.threshold_multiplier
        
        self.is_fitted = True
    
    def predict(self, df):
        """Predict anomalies in the data."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X_scaled, _ = self._preprocess(df, fit=False)
        
        if len(X_scaled) == 0:
            return np.array([]), np.array([])
        
        # Calculate reconstruction error
        X_transformed = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        mse = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
        
        # Detect anomalies
        anomalies = np.where(mse > self.reconstruction_error_threshold, 1, 0)
        
        # Calculate anomaly scores (0 to 1 range)
        scores = mse / (self.reconstruction_error_threshold * 2)
        scores = np.clip(scores, 0, 1)
        
        return anomalies, scores


class AnomalyDetectionSystem:
    """Complete anomaly detection system that combines multiple models."""
    
    def __init__(self):
        # Hardware metrics detectors
        self.hw_isolation_forest = IsolationForestDetector(contamination=0.05)
        self.hw_pca_detector = PCADetector(n_components=0.95)
        
        # Media metrics detectors
        self.media_isolation_forest = IsolationForestDetector(contamination=0.05)
        self.media_pca_detector = PCADetector(n_components=0.95)
        
        # Tracking if models are fitted
        self.hw_if_fitted = False
        self.hw_pca_fitted = False
        self.media_if_fitted = False
        self.media_pca_fitted = False
    
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
                self.hw_pca_detector.fit(hw_df)
                self.hw_pca_fitted = True
            except Exception as e:
                print(f"Error fitting hardware PCA Detector: {e}")
        
        if len(media_df) >= min_data_points:
            # Fit media metrics models
            try:
                self.media_isolation_forest.fit(media_df)
                self.media_if_fitted = True
            except Exception as e:
                print(f"Error fitting media Isolation Forest: {e}")
            
            try:
                self.media_pca_detector.fit(media_df)
                self.media_pca_fitted = True
            except Exception as e:
                print(f"Error fitting media PCA Detector: {e}")
    
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
        
        if not results['hardware']['anomaly'] and self.hw_pca_fitted and len(hw_df) > 0:
            try:
                anomalies, scores = self.hw_pca_detector.predict(hw_df)
                if len(anomalies) > 0 and anomalies[-1] == 1:
                    results['hardware'] = {
                        'anomaly': True,
                        'score': float(scores[-1]),
                        'model': 'PCA'
                    }
            except Exception as e:
                print(f"Error detecting hardware anomalies with PCA: {e}")
        
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
        
        if not results['media']['anomaly'] and self.media_pca_fitted and len(media_df) > 0:
            try:
                anomalies, scores = self.media_pca_detector.predict(media_df)
                if len(anomalies) > 0 and anomalies[-1] == 1:
                    results['media'] = {
                        'anomaly': True,
                        'score': float(scores[-1]),
                        'model': 'PCA'
                    }
            except Exception as e:
                print(f"Error detecting media anomalies with PCA: {e}")
        
        return results
