import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

class HardwareMetricsGenerator:
    """Generate simulated hardware metrics for a multimedia processing system."""
    
    def __init__(self, anomaly_probability=0.05):
        self.anomaly_probability = anomaly_probability
        # Normal operating ranges
        self.normal_ranges = {
            'cpu_usage': (20, 60),       # percentage
            'gpu_usage': (30, 70),       # percentage
            'memory_usage': (30, 70),    # percentage
            'disk_io': (5, 30),          # MB/s
            'network_io': (10, 50),      # MB/s
            'temperature': (40, 70)      # degrees Celsius
        }
        # Initialize trend values
        self.trend_values = {metric: np.mean(range_val) for metric, range_val in self.normal_ranges.items()}
        # Current state (normal or anomaly)
        self.current_state = "normal"
        # Counter for anomaly duration
        self.anomaly_counter = 0
    
    def _introduce_anomaly(self, metric, value):
        """Introduce an anomaly to the given metric value."""
        if metric == 'cpu_usage':
            return min(value * 2, 100)  # CPU spike
        elif metric == 'gpu_usage':
            return min(value * 1.8, 100)  # GPU spike
        elif metric == 'memory_usage':
            return min(value * 1.5, 100)  # Memory leak
        elif metric == 'disk_io':
            return value * 3  # Disk I/O surge
        elif metric == 'network_io':
            return value * 0.2  # Network throttling
        elif metric == 'temperature':
            return value * 1.3  # Overheating
        return value
    
    def _update_trend(self):
        """Update the trend values with small random variations."""
        for metric, range_val in self.normal_ranges.items():
            min_val, max_val = range_val
            current = self.trend_values[metric]
            # Random walk
            change = np.random.normal(0, (max_val - min_val) * 0.05)
            new_val = current + change
            # Ensure within range
            new_val = max(min_val, min(new_val, max_val))
            self.trend_values[metric] = new_val
    
    def generate_metrics(self):
        """Generate the current set of hardware metrics."""
        # Decide if we should transition to/from anomaly state
        if self.current_state == "normal" and np.random.random() < self.anomaly_probability:
            self.current_state = "anomaly"
            self.anomaly_counter = np.random.randint(5, 15)  # Anomaly lasts for 5-15 data points
        
        if self.current_state == "anomaly":
            self.anomaly_counter -= 1
            if self.anomaly_counter <= 0:
                self.current_state = "normal"
        
        # Update trend values
        self._update_trend()
        
        # Generate current metrics
        metrics = {}
        for metric, trend_value in self.trend_values.items():
            # Add noise
            noise = np.random.normal(0, (self.normal_ranges[metric][1] - self.normal_ranges[metric][0]) * 0.1)
            value = trend_value + noise
            
            # Apply anomaly if in anomaly state
            if self.current_state == "anomaly":
                value = self._introduce_anomaly(metric, value)
            
            metrics[metric] = max(0, value)  # Ensure non-negative
        
        metrics['timestamp'] = datetime.now()
        metrics['is_anomaly'] = 1 if self.current_state == "anomaly" else 0
        
        return metrics


class MediaMetricsGenerator:
    """Generate simulated media processing metrics."""
    
    def __init__(self, anomaly_probability=0.05):
        self.anomaly_probability = anomaly_probability
        # Normal operating ranges
        self.normal_ranges = {
            'frame_rate': (25, 30),         # FPS
            'bitrate': (5000, 10000),       # Kbps
            'encoding_time': (10, 30),      # ms per frame
            'frame_drops': (0, 2),          # count per minute
            'audio_sync_offset': (-5, 5),   # ms
            'compression_ratio': (0.6, 0.8) # ratio
        }
        # Initialize trend values
        self.trend_values = {metric: np.mean(range_val) for metric, range_val in self.normal_ranges.items()}
        # Current state (normal or anomaly)
        self.current_state = "normal"
        # Counter for anomaly duration
        self.anomaly_counter = 0
    
    def _introduce_anomaly(self, metric, value):
        """Introduce an anomaly to the given metric value."""
        if metric == 'frame_rate':
            return max(value * 0.5, 1)  # Frame rate drop
        elif metric == 'bitrate':
            return value * 0.4  # Bitrate drop
        elif metric == 'encoding_time':
            return value * 3  # Slow encoding
        elif metric == 'frame_drops':
            return value + 15  # Frame drops
        elif metric == 'audio_sync_offset':
            return value + 100 if value >= 0 else value - 100  # Audio sync issues
        elif metric == 'compression_ratio':
            return max(value * 1.5, 1)  # Compression issues
        return value
    
    def _update_trend(self):
        """Update the trend values with small random variations."""
        for metric, range_val in self.normal_ranges.items():
            min_val, max_val = range_val
            current = self.trend_values[metric]
            # Random walk
            change = np.random.normal(0, (max_val - min_val) * 0.05)
            new_val = current + change
            # Ensure within range
            new_val = max(min_val, min(new_val, max_val))
            self.trend_values[metric] = new_val
    
    def generate_metrics(self):
        """Generate the current set of media metrics."""
        # Decide if we should transition to/from anomaly state
        if self.current_state == "normal" and np.random.random() < self.anomaly_probability:
            self.current_state = "anomaly"
            self.anomaly_counter = np.random.randint(5, 15)  # Anomaly lasts for 5-15 data points
        
        if self.current_state == "anomaly":
            self.anomaly_counter -= 1
            if self.anomaly_counter <= 0:
                self.current_state = "normal"
        
        # Update trend values
        self._update_trend()
        
        # Generate current metrics
        metrics = {}
        for metric, trend_value in self.trend_values.items():
            # Add noise
            noise = np.random.normal(0, (self.normal_ranges[metric][1] - self.normal_ranges[metric][0]) * 0.1)
            value = trend_value + noise
            
            # Apply anomaly if in anomaly state
            if self.current_state == "anomaly":
                value = self._introduce_anomaly(metric, value)
            
            metrics[metric] = max(0, value) if metric != 'audio_sync_offset' else value
        
        metrics['timestamp'] = datetime.now()
        metrics['is_anomaly'] = 1 if self.current_state == "anomaly" else 0
        
        return metrics


def get_historical_data(days=7, interval_minutes=15):
    """Generate historical hardware and media metrics data."""
    hw_gen = HardwareMetricsGenerator(anomaly_probability=0.02)
    media_gen = MediaMetricsGenerator(anomaly_probability=0.02)
    
    # Calculate number of data points
    points = int((days * 24 * 60) / interval_minutes)
    
    # Create timestamp range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = [start_time + timedelta(minutes=i*interval_minutes) for i in range(points)]
    
    hw_data = []
    media_data = []
    
    for ts in timestamps:
        # Generate hardware metrics
        hw_metrics = hw_gen.generate_metrics()
        hw_metrics['timestamp'] = ts
        hw_data.append(hw_metrics)
        
        # Generate media metrics
        media_metrics = media_gen.generate_metrics()
        media_metrics['timestamp'] = ts
        media_data.append(media_metrics)
    
    hw_df = pd.DataFrame(hw_data)
    media_df = pd.DataFrame(media_data)
    
    return hw_df, media_df


class MetricsCollector:
    """Collects and stores metrics from generators in real-time."""
    
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.hw_gen = HardwareMetricsGenerator()
        self.media_gen = MediaMetricsGenerator()
        
        # Initialize empty dataframes for real-time data
        self.hw_data = pd.DataFrame()
        self.media_data = pd.DataFrame()
    
    def update(self):
        """Update metrics with new data points."""
        # Generate new data points
        hw_metrics = self.hw_gen.generate_metrics()
        media_metrics = self.media_gen.generate_metrics()
        
        # Convert to dataframes
        hw_df = pd.DataFrame([hw_metrics])
        media_df = pd.DataFrame([media_metrics])
        
        # Append to existing data
        self.hw_data = pd.concat([self.hw_data, hw_df])
        self.media_data = pd.concat([self.media_data, media_df])
        
        # Trim to max_points
        if len(self.hw_data) > self.max_points:
            self.hw_data = self.hw_data.iloc[-self.max_points:]
        if len(self.media_data) > self.max_points:
            self.media_data = self.media_data.iloc[-self.max_points:]
        
        return hw_df.iloc[0].to_dict(), media_df.iloc[0].to_dict()
    
    def get_data(self):
        """Get the current data."""
        return self.hw_data, self.media_data
