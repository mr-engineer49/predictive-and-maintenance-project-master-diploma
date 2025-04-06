import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os

class IsolationForestDetector:
    """Isolation Forest for anomaly detection in transportation systems."""
    
    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        """Initialize the Isolation Forest model.
        
        Args:
            contamination (float): Expected proportion of anomalies
            n_estimators (int): Number of base estimators
            random_state (int): Random state for reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Create a preprocessing pipeline
        self.preprocessing = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', RobustScaler())  # Use robust scaler to handle outliers better
        ])
        
    def fit(self, data):
        """Fit the model to the data.
        
        Args:
            data (np.array): Data for training
            
        Returns:
            self: The fitted model instance
        """
        try:
            # Clean the data and handle extreme values
            clean_data = np.copy(data)
            
            # Replace infinities and NaNs with zeros
            clean_data = np.nan_to_num(clean_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Handle extreme values by clipping
            for col in range(clean_data.shape[1]):
                col_data = clean_data[:, col]
                if np.any(col_data != 0):
                    # Get valid range (excluding zeros which might be from nan replacement)
                    valid_data = col_data[col_data != 0]
                    if len(valid_data) > 0:
                        # Calculate percentiles for clipping
                        p01 = np.percentile(valid_data, 1)
                        p99 = np.percentile(valid_data, 99)
                        # Clip the data to this range
                        col_data = np.clip(col_data, p01, p99)
                        clean_data[:, col] = col_data
            
            # Apply preprocessing pipeline
            processed_data = self.preprocessing.fit_transform(clean_data)
            
            # Fit the model
            self.model.fit(processed_data)
            
            return self
            
        except Exception as e:
            print(f"Error fitting Isolation Forest: {str(e)}")
            # Create a dummy model that will still work
            dummy_data = np.zeros((10, data.shape[1]))
            self.preprocessing.fit(dummy_data)
            self.model.fit(dummy_data)
            return self
    
    def predict(self, data):
        """Predict anomalies in new data.
        
        Args:
            data (np.array): New data to predict anomalies in
            
        Returns:
            dict: Dictionary with anomaly scores and binary predictions
        """
        try:
            if len(data) == 0:
                return {
                    'anomaly': False,
                    'scores': np.array([]),
                    'predictions': np.array([])
                }
            
            # Clean data
            clean_data = np.copy(data)
            clean_data = np.nan_to_num(clean_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply preprocessing pipeline
            processed_data = self.preprocessing.transform(clean_data)
            
            # Get predictions (-1 for anomalies, 1 for normal)
            predictions = self.model.predict(processed_data)
            
            # Convert to binary (1 for anomalies, 0 for normal)
            binary_predictions = (predictions == -1).astype(int)
            
            # Get decision scores
            scores = self.model.decision_function(processed_data)
            # Convert to anomaly scores (higher means more anomalous)
            anomaly_scores = -scores
            
            # Determine if any point is an anomaly
            is_anomaly = bool(np.any(binary_predictions))
            
            return {
                'anomaly': is_anomaly,
                'scores': anomaly_scores,
                'predictions': binary_predictions
            }
            
        except Exception as e:
            print(f"Error in anomaly prediction: {str(e)}")
            # Return a default response
            return {
                'anomaly': False,
                'scores': np.zeros(len(data)),
                'predictions': np.zeros(len(data), dtype=bool),
                'error': str(e)
            }
    
    def save(self, model_dir='saved_models'):
        """Save the trained model to disk.
        
        Args:
            model_dir (str): Directory to save model to
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Save model
        joblib.dump(self.model, os.path.join(model_dir, 'isolation_forest.pkl'))
        
        # Save preprocessing pipeline
        joblib.dump(self.preprocessing, os.path.join(model_dir, 'preprocessing.pkl'))
    
    @classmethod
    def load(cls, model_dir='saved_models'):
        """Load a trained model from disk.
        
        Args:
            model_dir (str): Directory to load model from
            
        Returns:
            IsolationForestDetector: Loaded model instance
        """
        # Create instance
        instance = cls()
        
        # Load model
        instance.model = joblib.load(os.path.join(model_dir, 'isolation_forest.pkl'))
        
        # Load preprocessing pipeline
        instance.preprocessing = joblib.load(os.path.join(model_dir, 'preprocessing.pkl'))
        
        return instance