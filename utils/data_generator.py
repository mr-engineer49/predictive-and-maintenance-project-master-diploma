import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json

class TransportationDataGenerator:
    """Generate simulated sensor data for transportation vehicles (airplanes, trucks, railways)."""
    
    def __init__(self, vehicle_type='airplane', anomaly_probability=0.05, random_seed=42):
        """Initialize the data generator.
        
        Args:
            vehicle_type (str): Type of vehicle ('airplane', 'truck', or 'railway')
            anomaly_probability (float): Probability of introducing anomalies
            random_seed (int): Random seed for reproducibility
        """
        self.vehicle_type = vehicle_type.lower()
        self.anomaly_probability = anomaly_probability
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Define normal operating ranges for each vehicle type
        self.normal_ranges = self._initialize_normal_ranges()
        
        # Initialize trends with random values within normal range
        self.trends = self._initialize_trends()
        
        # Track which metrics have anomalies currently
        self.active_anomalies = {}
    
    def _initialize_normal_ranges(self):
        """Initialize normal operating ranges for each vehicle type."""
        if self.vehicle_type == 'airplane':
            return {
                'engine_temperature': (320, 450),      # Celsius
                'engine_vibration': (0.1, 0.8),        # g-force units
                'fuel_pressure': (30, 40),             # PSI
                'oil_pressure': (45, 60),              # PSI
                'rotation_speed': (2000, 3500),        # RPM
                'altitude': (0, 40000),                # feet
                'airspeed': (150, 500),                # knots
                'exhaust_gas_temp': (400, 600),        # Celsius
                'hydraulic_pressure': (2800, 3200),    # PSI
                'cabin_pressure': (11, 15)             # PSI
            }
        elif self.vehicle_type == 'truck':
            return {
                'engine_temperature': (80, 105),       # Celsius
                'engine_vibration': (0.05, 0.4),       # g-force units
                'fuel_pressure': (50, 65),             # PSI
                'oil_pressure': (30, 45),              # PSI
                'rotation_speed': (600, 2000),         # RPM
                'tire_pressure': (100, 120),           # PSI
                'brake_temperature': (50, 200),        # Celsius
                'transmission_temp': (70, 95),         # Celsius
                'battery_voltage': (12.5, 14.5),       # Volts
                'coolant_level': (80, 100)             # Percentage
            }
        elif self.vehicle_type == 'railway':
            return {
                'engine_temperature': (60, 95),        # Celsius
                'engine_vibration': (0.02, 0.3),       # g-force units
                'hydraulic_pressure': (140, 180),      # Bar
                'oil_pressure': (3, 6),                # Bar
                'rotation_speed': (300, 1200),         # RPM
                'axle_temperature': (20, 60),          # Celsius
                'catenary_voltage': (24000, 27000),    # Volts
                'pantograph_force': (60, 90),          # Newtons
                'brake_cylinder_pressure': (3, 5),     # Bar
                'traction_motor_temp': (50, 80)        # Celsius
            }
        else:
            raise ValueError(f"Unsupported vehicle type: {self.vehicle_type}")
    
    def _initialize_trends(self):
        """Initialize random starting points for each metric within normal range."""
        trends = {}
        for metric, (min_val, max_val) in self.normal_ranges.items():
            # Start at a random point within the normal range
            mid_point = (min_val + max_val) / 2
            range_width = (max_val - min_val) * 0.3  # Start near the middle
            trends[metric] = random.uniform(mid_point - range_width, mid_point + range_width)
        return trends
    
    def _update_trend(self, metric, current_value, drift_factor=0.02):
        """Update a metric value with small random variations.
        
        Args:
            metric (str): Name of the metric to update
            current_value (float): Current value of the metric
            drift_factor (float): Maximum percentage change per update
            
        Returns:
            float: Updated value with random drift
        """
        min_val, max_val = self.normal_ranges[metric]
        range_size = max_val - min_val
        
        # Calculate maximum allowed drift
        max_drift = range_size * drift_factor
        
        # Generate random drift
        drift = random.uniform(-max_drift, max_drift)
        
        # Update value with drift
        new_value = current_value + drift
        
        # Ensure value stays within a larger range (allowing for minor excursions)
        extended_min = min_val - range_size * 0.1
        extended_max = max_val + range_size * 0.1
        
        # Apply soft boundaries by adding resistance when approaching limits
        if new_value < min_val:
            # Pull back toward normal range when below minimum
            pull_factor = (min_val - new_value) / (min_val - extended_min) * 0.7
            new_value += (min_val - new_value) * pull_factor
        elif new_value > max_val:
            # Pull back toward normal range when above maximum
            pull_factor = (new_value - max_val) / (extended_max - max_val) * 0.7
            new_value -= (new_value - max_val) * pull_factor
        
        return new_value
    
    def _introduce_anomaly(self, metric, current_value):
        """Introduce an anomaly to the given metric value.
        
        Args:
            metric (str): Name of the metric
            current_value (float): Current value of the metric
            
        Returns:
            float: Value with anomaly introduced
        """
        min_val, max_val = self.normal_ranges[metric]
        range_size = max_val - min_val
        
        # Different anomaly patterns based on metric type
        if 'temperature' in metric or 'temp' in metric:
            # Temperature metrics typically spike upward during failures
            anomaly_value = current_value + range_size * random.uniform(0.15, 0.4)
        elif 'pressure' in metric:
            # Pressure can drop during leaks or spike during blockages
            if random.random() < 0.7:  # 70% chance of pressure drop
                anomaly_value = current_value - range_size * random.uniform(0.2, 0.5)
            else:  # 30% chance of pressure spike
                anomaly_value = current_value + range_size * random.uniform(0.15, 0.3)
        elif 'vibration' in metric:
            # Vibration typically increases with mechanical issues
            anomaly_value = current_value + range_size * random.uniform(0.3, 0.8)
        elif 'speed' in metric or 'rpm' in metric:
            # Rotation speed might fluctuate unusually
            deviation = range_size * random.uniform(0.1, 0.25)
            anomaly_value = current_value + deviation * (-1 if random.random() < 0.5 else 1)
        elif 'voltage' in metric:
            # Voltage drops are common electrical issues
            anomaly_value = current_value - range_size * random.uniform(0.15, 0.4)
        else:
            # Generic anomaly for other metrics
            deviation = range_size * random.uniform(0.15, 0.4)
            anomaly_value = current_value + deviation * (-1 if random.random() < 0.5 else 1)
        
        return anomaly_value
    
    def _should_start_anomaly(self):
        """Determine if a new anomaly should be started."""
        return random.random() < self.anomaly_probability
    
    def _should_continue_anomaly(self, duration_steps):
        """Determine if an existing anomaly should continue.
        
        Args:
            duration_steps (int): How many steps the anomaly has been active
            
        Returns:
            bool: Whether to continue the anomaly
        """
        # Most anomalies last 3-7 steps before resolving
        if duration_steps > random.randint(3, 7):
            return False
        return True
    
    def generate_metrics(self):
        """Generate a new set of metrics data.
        
        Returns:
            dict: Dictionary of metrics with values
        """
        metrics = {}
        anomaly_present = False
        anomaly_metrics = []
        
        # Update each metric
        for metric in self.normal_ranges.keys():
            current_value = self.trends[metric]
            
            # Check if this metric already has an active anomaly
            if metric in self.active_anomalies:
                # Increment anomaly duration
                self.active_anomalies[metric] += 1
                
                # Decide whether to continue the anomaly
                if self._should_continue_anomaly(self.active_anomalies[metric]):
                    # Continue anomaly
                    new_value = self._introduce_anomaly(metric, current_value)
                    anomaly_present = True
                    anomaly_metrics.append(metric)
                else:
                    # End anomaly
                    new_value = self._update_trend(metric, current_value)
                    del self.active_anomalies[metric]
            else:
                # No active anomaly for this metric
                if self._should_start_anomaly():
                    # Start new anomaly
                    new_value = self._introduce_anomaly(metric, current_value)
                    self.active_anomalies[metric] = 1
                    anomaly_present = True
                    anomaly_metrics.append(metric)
                else:
                    # Normal update
                    new_value = self._update_trend(metric, current_value)
            
            # Update trend and add to metrics
            self.trends[metric] = new_value
            metrics[metric] = round(new_value, 2)
        
        # Add metadata
        metrics['timestamp'] = datetime.now()
        metrics['vehicle_type'] = self.vehicle_type
        metrics['vehicle_id'] = f"{self.vehicle_type}_01"  # Add ID for tracking multiple vehicles
        metrics['is_anomaly'] = int(anomaly_present)
        metrics['anomaly_metrics'] = anomaly_metrics if anomaly_present else []
        
        return metrics
    
    def generate_kafka_message(self):
        """Generate a message for Kafka in JSON format.
        
        Returns:
            str: JSON string with metrics data
        """
        metrics = self.generate_metrics()
        
        # Convert timestamp to string for JSON serialization
        metrics['timestamp'] = metrics['timestamp'].isoformat()
        
        return json.dumps(metrics)
    
    def generate_batch(self, n_samples, time_delta=None):
        """Generate a batch of consecutive samples.
        
        Args:
            n_samples (int): Number of samples to generate
            time_delta (timedelta, optional): Time between samples
            
        Returns:
            pd.DataFrame: DataFrame with generated samples
        """
        samples = []
        
        # Reset active anomalies for a clean slate
        self.active_anomalies = {}
        
        for i in range(n_samples):
            metrics = self.generate_metrics()
            samples.append(metrics)
            
            # Adjust timestamp for sequential data if time_delta provided
            if time_delta and i < n_samples - 1:
                # Use fixed timestamps for historical data
                next_time = metrics['timestamp'] + time_delta
                # Temporarily modify current time for the next sample
                metrics['timestamp'] = next_time
        
        return pd.DataFrame(samples)


def get_historical_data(days=7, interval_minutes=15, vehicle_type='airplane'):
    """Generate historical data for training and testing.
    
    Args:
        days (int): Number of days of historical data
        interval_minutes (int): Interval between data points in minutes
        vehicle_type (str): Type of vehicle to generate data for
        
    Returns:
        pd.DataFrame: DataFrame with historical data
    """
    # Calculate number of samples
    n_samples = int((days * 24 * 60) / interval_minutes)
    
    # Create data generator with low anomaly probability for historical data
    generator = TransportationDataGenerator(
        vehicle_type=vehicle_type,
        anomaly_probability=0.02
    )
    
    # Generate batch with time intervals
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Create a range of timestamps
    timestamps = [start_time + timedelta(minutes=i*interval_minutes) for i in range(n_samples)]
    
    # Generate data
    data = generator.generate_batch(n_samples, timedelta(minutes=interval_minutes))
    
    # Ensure timestamps are in sequence
    data['timestamp'] = timestamps
    
    return data


class MetricsCollector:
    """Collects and stores metrics from generators in real-time."""
    
    def __init__(self, vehicle_types=['airplane', 'truck', 'railway'], max_points=1000):
        """Initialize metrics collectors for different vehicle types.
        
        Args:
            vehicle_types (list): List of vehicle types to collect metrics for
            max_points (int): Maximum number of data points to store per vehicle type
        """
        self.max_points = max_points
        self.generators = {}
        self.data = {}
        
        # Initialize generators and data storage for each vehicle type
        for vehicle_type in vehicle_types:
            self.generators[vehicle_type] = TransportationDataGenerator(vehicle_type=vehicle_type)
            self.data[vehicle_type] = pd.DataFrame()
    
    def update(self, vehicle_type=None):
        """Update metrics with new data points.
        
        Args:
            vehicle_type (str, optional): Specific vehicle type to update, or all if None
            
        Returns:
            dict: Dictionary with the latest metrics for each vehicle type
        """
        latest_metrics = {}
        
        # Update specified vehicle type or all vehicle types
        types_to_update = [vehicle_type] if vehicle_type else self.generators.keys()
        
        for vtype in types_to_update:
            if vtype not in self.generators:
                continue
                
            # Generate new metrics
            metrics = self.generators[vtype].generate_metrics()
            latest_metrics[vtype] = metrics
            
            # Convert to DataFrame row
            metrics_df = pd.DataFrame([metrics])
            
            # Append to stored data
            if self.data[vtype].empty:
                self.data[vtype] = metrics_df
            else:
                self.data[vtype] = pd.concat([self.data[vtype], metrics_df], ignore_index=True)
            
            # Maintain maximum size
            if len(self.data[vtype]) > self.max_points:
                self.data[vtype] = self.data[vtype].iloc[-self.max_points:]
        
        return latest_metrics
    
    def get_data(self, vehicle_type=None):
        """Get the current data for the specified vehicle type(s).
        
        Args:
            vehicle_type (str, optional): Vehicle type to get data for, or all if None
            
        Returns:
            dict: Dictionary with DataFrames for each vehicle type
        """
        if vehicle_type:
            if vehicle_type in self.data:
                return {vehicle_type: self.data[vehicle_type]}
            return {}
        
        return self.data
        
    def get_data_for_vehicle(self, vehicle_type):
        """Get data for a specific vehicle type.
        
        Args:
            vehicle_type (str): Vehicle type to get data for
            
        Returns:
            pd.DataFrame: DataFrame with data for the vehicle type
        """
        if vehicle_type in self.data:
            return self.data[vehicle_type]
        return pd.DataFrame()
        
    def add_external_data(self, vehicle_type, df):
        """Add external data to the existing data.
        
        Args:
            vehicle_type (str): Vehicle type to add data for
            df (pd.DataFrame): DataFrame with data to add
            
        Returns:
            None
        """
        if vehicle_type not in self.data:
            self.data[vehicle_type] = df
            return
            
        # Make sure timestamps are datetime objects
        if 'timestamp' in df.columns and 'timestamp' in self.data[vehicle_type].columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except:
                    pass
                    
            if not pd.api.types.is_datetime64_any_dtype(self.data[vehicle_type]['timestamp']):
                try:
                    self.data[vehicle_type]['timestamp'] = pd.to_datetime(self.data[vehicle_type]['timestamp'])
                except:
                    pass
        
        # Add the new data
        self.data[vehicle_type] = pd.concat([self.data[vehicle_type], df], ignore_index=True)
        
        # Remove duplicates based on timestamp if it exists
        if 'timestamp' in self.data[vehicle_type].columns:
            self.data[vehicle_type] = self.data[vehicle_type].drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Maintain maximum size
        if len(self.data[vehicle_type]) > self.max_points:
            self.data[vehicle_type] = self.data[vehicle_type].iloc[-self.max_points:]