import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import threading
from datetime import datetime, timedelta

# Import custom modules
from utils.data_generator import TransportationDataGenerator, MetricsCollector, get_historical_data
from models.anomaly_detection_system import AnomalyDetectionSystem
from utils.dashboard_components import create_dashboard
from api.flask_api import app as flask_app, alerts, vehicle_data, anomaly_results

# Set page configuration
st.set_page_config(
    page_title="Transportation Predictive Maintenance",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Flask API in a separate thread
def start_flask_api():
    from api.flask_api import start_api_server
    start_api_server(port=5001)

# Initialize Kafka producer simulator in a separate thread (for demo)
def start_kafka_producer():
    from api.kafka_consumer import KafkaLogProducer
    producer = KafkaLogProducer(
        bootstrap_servers=['localhost:9092'],
        topic='vehicle_metrics',
        simulation_interval=3
    )
    producer.start()

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.dark_mode = False
    
    # Initialize data collectors and models
    st.session_state.metrics_collector = MetricsCollector(
        vehicle_types=['airplane', 'truck', 'railway'],
        max_points=1000
    )
    st.session_state.anomaly_detector = AnomalyDetectionSystem()
    
    # Initialize alerts list
    st.session_state.alerts = []
    
    # Generate some historical data for each vehicle type
    st.session_state.historical_data = {
        'airplane': get_historical_data(days=7, vehicle_type='airplane'),
        'truck': get_historical_data(days=7, vehicle_type='truck'),
        'railway': get_historical_data(days=7, vehicle_type='railway')
    }
    
    # Initialize the anomaly detector with historical data
    st.session_state.anomaly_detector.fit_models(st.session_state.historical_data)
    
    # Start Flask API in a separate thread
    flask_thread = threading.Thread(target=start_flask_api)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Start Kafka producer simulator (optional, for demo only)
    # In a real deployment, this would be replaced with a real Kafka consumer
    kafka_thread = threading.Thread(target=start_kafka_producer)
    kafka_thread.daemon = True
    kafka_thread.start()

# Apply dark mode if enabled
if st.session_state.get('dark_mode', False):
    # Dark mode
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #F0F2F6;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    # Light mode (default Streamlit)
    st.markdown("""
    <style>
    .stApp {
        background-color: #FAFBFC;
        color: #172B4D;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("Control Panel")
    
    # Update frequency
    update_frequency = st.slider(
        "Update Frequency (seconds)",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        key="update_frequency_slider"
    )
    
    # Vehicle type filter
    st.subheader("Vehicle Types")
    vehicle_filter = {}
    for vehicle_type in ['airplane', 'truck', 'railway']:
        vehicle_filter[vehicle_type] = st.checkbox(
            vehicle_type.capitalize(),
            value=True,
            key=f"vehicle_filter_{vehicle_type}"
        )
    
    # Alert thresholds
    st.subheader("Alert Thresholds")
    
    anomaly_score_threshold = st.slider(
        "Anomaly Score Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.1,
        key="anomaly_threshold_slider"
    )
    
    # System controls
    st.subheader("System Controls")
    
    if st.button("Reset Alerts", key="reset_alerts_button"):
        st.session_state.alerts = []
        # Also clear the API alerts
        alerts.clear()
        st.success("Alerts cleared!")
    
    if st.button("Retrain Anomaly Models", key="retrain_models_button"):
        with st.spinner("Retraining models..."):
            # Get current data
            vehicle_data_for_training = st.session_state.metrics_collector.get_data()
            
            # Make sure we have enough data
            has_enough_data = all(len(df) > 50 for df in vehicle_data_for_training.values())
            
            if has_enough_data:
                st.session_state.anomaly_detector.fit_models(vehicle_data_for_training)
                st.success("Models retrained successfully!")
            else:
                st.error("Not enough data to retrain models. Continue collecting more data.")

# Main loop for real-time updates
def update_data():
    """Update data and detect anomalies."""
    # Update metrics for filtered vehicle types
    filtered_vehicle_types = [vtype for vtype, enabled in vehicle_filter.items() if enabled]
    latest_metrics = {}
    
    for vehicle_type in filtered_vehicle_types:
        metrics = st.session_state.metrics_collector.update(vehicle_type)
        latest_metrics.update(metrics)
    
    # Get full datasets
    vehicle_data_local = st.session_state.metrics_collector.get_data()
    
    # Filter data based on selected vehicle types
    filtered_data = {vtype: data for vtype, data in vehicle_data_local.items() if vtype in filtered_vehicle_types}
    
    # Detect anomalies
    anomaly_results_local = st.session_state.anomaly_detector.detect_anomalies(filtered_data)
    
    # Create alerts for anomalies if detected
    current_time = datetime.now().isoformat()
    
    for vehicle_type, result in anomaly_results_local.items():
        if result['anomaly']:
            # Get the metrics that are anomalous
            anomaly_metrics = result.get('anomaly_metrics', [])
            metrics_str = ', '.join(anomaly_metrics) if anomaly_metrics else "multiple metrics"
            
            # Create alert
            alert = {
                'title': f"Anomaly Detected in {vehicle_type.capitalize()}",
                'message': f"Unusual behavior detected in {metrics_str} with confidence score {result['score']:.2f}",
                'severity': "Critical" if result['score'] > anomaly_score_threshold else "Warning",
                'timestamp': current_time,
                'vehicle_type': vehicle_type,
                'vehicle_id': f"{vehicle_type}_01",
                'metrics': anomaly_metrics
            }
            
            # Add to session state
            st.session_state.alerts.insert(0, alert)
            
            # Limit alerts to 20 most recent
            st.session_state.alerts = st.session_state.alerts[:20]
            
            # Also update the API alerts
            alerts.insert(0, alert)
            alerts[:] = alerts[:100]  # Keep only 100 most recent
    
    # Update API data
    global vehicle_data, anomaly_results
    vehicle_data.update(filtered_data)
    anomaly_results.update(anomaly_results_local)
    
    return filtered_data, anomaly_results_local

# Create the dashboard with placeholders for updates
create_dashboard()

# Create a container for status
status_container = st.empty()

# Main update loop
try:
    while True:
        # Update data
        start_time = time.time()
        st.session_state.vehicle_data, st.session_state.anomaly_results = update_data()
        
        # Display status
        with status_container.container():
            end_time = time.time()
            st.text(f"Last update: {datetime.now().strftime('%H:%M:%S')} (took {(end_time - start_time):.2f}s)")
        
        # Wait before next update
        time.sleep(update_frequency)
        
        # Force rerun to update the dashboard
        st.rerun()
except Exception as e:
    st.error(f"An error occurred: {str(e)}")