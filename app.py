import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import threading
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import custom modules
from utils.data_generator import TransportationDataGenerator, MetricsCollector, get_historical_data
from utils.external_data_fetcher import TransportationDataFetcher, CustomDataFetcher
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
    
    # Initialize the real-time external data fetcher
    st.session_state.external_data_fetcher = TransportationDataFetcher(
        cache_limit=1000
    )
    
    # Initialize anomaly detection system
    st.session_state.anomaly_detector = AnomalyDetectionSystem()
    
    # Initialize alerts list
    st.session_state.alerts = []
    
    # Flag to toggle between simulated and real data sources
    st.session_state.use_real_data = False
    
    # Counter for mixed data approach (periodic real data fetching)
    st.session_state.update_counter = 0
    
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
    
    # Start external data fetcher in background
    st.session_state.external_data_fetcher.start()

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
    
    # Data source selection
    st.subheader("Data Source Settings")
    
    # Make sure use_real_data is initialized in session state
    if 'use_real_data' not in st.session_state:
        st.session_state.use_real_data = False
        
    # Toggle for real-time data fetching
    use_real_data = st.checkbox(
        "Use Real-time Transportation Data",
        value=st.session_state.use_real_data,
        key="use_real_data_checkbox",
        help="When checked, the system will attempt to fetch real-time transportation data from public sources. When unchecked, it uses simulated data."
    )
    
    # Update the session state if the checkbox changed
    if use_real_data != st.session_state.use_real_data:
        st.session_state.use_real_data = use_real_data
        st.info("Data source changed. Next update will use the selected source.")
    
    # Option for mixed data approach
    mixed_data_frequency = st.slider(
        "Real Data Fetch Frequency",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        key="mixed_data_frequency_slider",
        help="How often to fetch real data (every N updates). Higher values increase performance but reduce the amount of real data."
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
            
            # If using real data, also include real data for training
            if st.session_state.use_real_data:
                real_data = st.session_state.external_data_fetcher.get_data()
                # Merge real data with simulated data
                for vehicle_type, df in real_data.items():
                    if not df.empty and vehicle_type in vehicle_data_for_training:
                        vehicle_data_for_training[vehicle_type] = pd.concat([
                            vehicle_data_for_training[vehicle_type], 
                            df
                        ]).drop_duplicates().reset_index(drop=True)
            
            # Make sure we have enough data
            has_enough_data = all(len(df) > 50 for df in vehicle_data_for_training.values() if not df.empty)
            
            if has_enough_data:
                st.session_state.anomaly_detector.fit_models(vehicle_data_for_training)
                st.success("Models retrained successfully!")
            else:
                st.error("Not enough data to retrain models. Continue collecting more data.")

# Main loop for real-time updates
def update_data():
    """Update data and detect anomalies."""
    # Initialize the update counter if not present
    if 'update_counter' not in st.session_state:
        st.session_state.update_counter = 0
    
    # Increment the update counter
    st.session_state.update_counter += 1
    
    # Get the list of filtered vehicle types
    filtered_vehicle_types = [vtype for vtype, enabled in vehicle_filter.items() if enabled]
    latest_metrics = {}
    
    # Determine if we should use real data in this update
    use_real_data = st.session_state.use_real_data
    mixed_data_frequency = st.session_state.get('mixed_data_frequency_slider', 5)
    
    # If the counter has reached the frequency, use real data
    use_mixed_approach = use_real_data and st.session_state.update_counter % mixed_data_frequency == 0
    
    # Get simulated data
    if not use_real_data:
        # Update metrics using simulated data
        for vehicle_type in filtered_vehicle_types:
            metrics = st.session_state.metrics_collector.update(vehicle_type)
            latest_metrics.update(metrics)
        
        # Get full datasets
        vehicle_data_local = st.session_state.metrics_collector.get_data()
    
    # Use real data
    else:
        try:
            logger.info("Fetching real-time transportation data...")
            
            # Get real data
            real_data = st.session_state.external_data_fetcher.get_data()
            
            # If we're doing a mixed approach, also get simulated data
            if use_mixed_approach:
                # Update simulated data as well
                for vehicle_type in filtered_vehicle_types:
                    metrics = st.session_state.metrics_collector.update(vehicle_type)
                    latest_metrics.update(metrics)
                
                # Get both datasets
                simulated_data = st.session_state.metrics_collector.get_data()
                
                # Merge the two datasets
                vehicle_data_local = {}
                for vehicle_type in filtered_vehicle_types:
                    if vehicle_type in real_data and not real_data[vehicle_type].empty:
                        # If we have real data for this vehicle type, use it
                        real_df = real_data[vehicle_type].copy()
                        
                        # Make sure the real data has all the required columns
                        if 'is_anomaly' not in real_df.columns:
                            real_df['is_anomaly'] = 0
                        if 'anomaly_metrics' not in real_df.columns:
                            real_df['anomaly_metrics'] = [[] for _ in range(len(real_df))]
                            
                        # Combine with simulated data
                        if vehicle_type in simulated_data:
                            vehicle_data_local[vehicle_type] = pd.concat([
                                simulated_data[vehicle_type],
                                real_df
                            ]).reset_index(drop=True)
                        else:
                            vehicle_data_local[vehicle_type] = real_df
                    elif vehicle_type in simulated_data:
                        # Fall back to simulated data if no real data
                        vehicle_data_local[vehicle_type] = simulated_data[vehicle_type]
            else:
                # Only use real data
                vehicle_data_local = {}
                for vehicle_type in filtered_vehicle_types:
                    if vehicle_type in real_data and not real_data[vehicle_type].empty:
                        # Process real data
                        real_df = real_data[vehicle_type].copy()
                        
                        # Make sure the real data has all the required columns
                        if 'is_anomaly' not in real_df.columns:
                            real_df['is_anomaly'] = 0
                        if 'anomaly_metrics' not in real_df.columns:
                            real_df['anomaly_metrics'] = [[] for _ in range(len(real_df))]
                            
                        vehicle_data_local[vehicle_type] = real_df
                    else:
                        # If no real data for this vehicle type, get some simulated data
                        metrics = st.session_state.metrics_collector.update(vehicle_type)
                        latest_metrics.update(metrics)
                        vehicle_data_local[vehicle_type] = st.session_state.metrics_collector.get_data_for_vehicle(vehicle_type)
                        
            # Add the real data to the metrics collector for future reference and training
            for vehicle_type, df in real_data.items():
                if not df.empty and vehicle_type in filtered_vehicle_types:
                    st.session_state.metrics_collector.add_external_data(vehicle_type, df)
                    
        except Exception as e:
            logger.error(f"Error fetching real data: {str(e)}")
            # Fall back to simulated data
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
    
    # Add data source indicator to the status
    data_source = "Real-time API" if use_real_data else "Simulated"
    if use_mixed_approach:
        data_source = "Mixed (Real + Simulated)"
    
    return filtered_data, anomaly_results_local, data_source

# Create the dashboard with placeholders for updates
create_dashboard()

# Create a container for status
status_container = st.empty()

# Main update loop
try:
    while True:
        # Update data
        start_time = time.time()
        st.session_state.vehicle_data, st.session_state.anomaly_results, data_source = update_data()
        
        # Display status
        with status_container.container():
            end_time = time.time()
            st.text(f"Last update: {datetime.now().strftime('%H:%M:%S')} (took {(end_time - start_time):.2f}s) - Data Source: {data_source}")
        
        # Wait before next update
        time.sleep(update_frequency)
        
        # Force rerun to update the dashboard
        st.rerun()
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    logger.error(f"Error in main loop: {str(e)}", exc_info=True)