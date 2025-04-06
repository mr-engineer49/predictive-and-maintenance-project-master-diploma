import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Import utilities
from utils.data_generator import (
    HardwareMetricsGenerator, 
    MediaMetricsGenerator,
    MetricsCollector,
    get_historical_data
)
from utils.anomaly_detection import AnomalyDetectionSystem
from utils.dashboard_components import (
    header_section,
    system_health_indicators,
    metrics_cards,
    real_time_charts,
    anomaly_visualization,
    alert_system,
    maintenance_recommendations,
    historical_analysis
)

# Set page configuration
st.set_page_config(
    page_title="Media Processing Predictive Maintenance",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.dark_mode = False
    st.session_state.metrics_collector = MetricsCollector(max_points=1000)
    st.session_state.anomaly_detector = AnomalyDetectionSystem()
    st.session_state.alerts = []
    st.session_state.historical_hw_df, st.session_state.historical_media_df = get_historical_data(days=7)
    # Initialize the anomaly detector with historical data
    st.session_state.anomaly_detector.fit_models(
        st.session_state.historical_hw_df,
        st.session_state.historical_media_df
    )

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

# Display header
header_section()

# Sidebar configuration
with st.sidebar:
    st.header("Control Panel")
    
    # Update frequency
    update_frequency = st.slider(
        "Update Frequency (seconds)",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )
    
    # Alert thresholds
    st.subheader("Alert Thresholds")
    
    cpu_threshold = st.slider("CPU Usage (%)", 70, 95, 80)
    gpu_threshold = st.slider("GPU Usage (%)", 70, 95, 85)
    memory_threshold = st.slider("Memory Usage (%)", 70, 95, 80)
    
    frame_rate_threshold = st.slider("Min Frame Rate (FPS)", 10, 24, 20)
    frame_drops_threshold = st.slider("Max Frame Drops (/min)", 2, 15, 5)
    
    # System controls
    st.subheader("System Controls")
    
    if st.button("Reset Alerts"):
        st.session_state.alerts = []
        st.success("Alerts cleared!")
    
    if st.button("Retrain Anomaly Models"):
        with st.spinner("Retraining models..."):
            # Get current data
            hw_df, media_df = st.session_state.metrics_collector.get_data()
            
            # Make sure we have enough data
            if len(hw_df) > 50 and len(media_df) > 50:
                st.session_state.anomaly_detector.fit_models(hw_df, media_df)
                st.success("Models retrained successfully!")
            else:
                st.error("Not enough data to retrain models. Continue collecting more data.")

# Main dashboard layout
# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Real-time Monitoring", "Anomaly Detection", "Historical Analysis"])

# Update function to refresh data
def update_data():
    # Generate new data point
    hw_metrics, media_metrics = st.session_state.metrics_collector.update()
    
    # Get full datasets
    hw_df, media_df = st.session_state.metrics_collector.get_data()
    
    # Check for anomalies
    anomaly_results = st.session_state.anomaly_detector.detect_anomalies(hw_df, media_df)
    
    # Check for threshold breaches and create alerts
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Check hardware thresholds
    if hw_metrics['cpu_usage'] > cpu_threshold:
        st.session_state.alerts.insert(0, {
            'title': "High CPU Usage Alert",
            'message': f"CPU usage is at {hw_metrics['cpu_usage']:.1f}%, which exceeds the threshold of {cpu_threshold}%",
            'severity': "Warning" if hw_metrics['cpu_usage'] < 90 else "Critical",
            'time': current_time
        })
    
    if hw_metrics['gpu_usage'] > gpu_threshold:
        st.session_state.alerts.insert(0, {
            'title': "High GPU Usage Alert",
            'message': f"GPU usage is at {hw_metrics['gpu_usage']:.1f}%, which exceeds the threshold of {gpu_threshold}%",
            'severity': "Warning" if hw_metrics['gpu_usage'] < 90 else "Critical",
            'time': current_time
        })
    
    if hw_metrics['memory_usage'] > memory_threshold:
        st.session_state.alerts.insert(0, {
            'title': "High Memory Usage Alert",
            'message': f"Memory usage is at {hw_metrics['memory_usage']:.1f}%, which exceeds the threshold of {memory_threshold}%",
            'severity': "Warning" if hw_metrics['memory_usage'] < 90 else "Critical",
            'time': current_time
        })
    
    # Check media thresholds
    if media_metrics['frame_rate'] < frame_rate_threshold:
        st.session_state.alerts.insert(0, {
            'title': "Low Frame Rate Alert",
            'message': f"Frame rate is at {media_metrics['frame_rate']:.1f} FPS, which is below the threshold of {frame_rate_threshold} FPS",
            'severity': "Warning" if media_metrics['frame_rate'] > frame_rate_threshold * 0.7 else "Critical",
            'time': current_time
        })
    
    if media_metrics['frame_drops'] > frame_drops_threshold:
        st.session_state.alerts.insert(0, {
            'title': "High Frame Drop Alert",
            'message': f"Frame drops are at {media_metrics['frame_drops']:.1f} per minute, which exceeds the threshold of {frame_drops_threshold}",
            'severity': "Warning" if media_metrics['frame_drops'] < frame_drops_threshold * 1.5 else "Critical",
            'time': current_time
        })
    
    # Create alerts for anomalies if detected
    if anomaly_results['hardware']['anomaly']:
        st.session_state.alerts.insert(0, {
            'title': "Hardware Anomaly Detected",
            'message': f"Unusual hardware behavior detected by {anomaly_results['hardware']['model']} model with confidence score {anomaly_results['hardware']['score']:.2f}",
            'severity': "Warning" if anomaly_results['hardware']['score'] < 0.7 else "Critical",
            'time': current_time
        })
    
    if anomaly_results['media']['anomaly']:
        st.session_state.alerts.insert(0, {
            'title': "Media Processing Anomaly Detected",
            'message': f"Unusual media processing behavior detected by {anomaly_results['media']['model']} model with confidence score {anomaly_results['media']['score']:.2f}",
            'severity': "Warning" if anomaly_results['media']['score'] < 0.7 else "Critical",
            'time': current_time
        })
    
    # Limit alerts to 20 most recent
    st.session_state.alerts = st.session_state.alerts[:20]
    
    return hw_metrics, media_metrics, hw_df, media_df, anomaly_results

# Real-time Monitoring tab
with tab1:
    # Create placeholders for real-time updates
    system_health_placeholder = st.empty()
    metrics_cards_placeholder = st.empty()
    charts_placeholder = st.empty()
    alerts_placeholder = st.empty()

# Anomaly Detection tab
with tab2:
    # Create placeholders for anomaly detection
    anomaly_viz_placeholder = st.empty()
    recommendations_placeholder = st.empty()

# Historical Analysis tab
with tab3:
    # Create placeholder for historical analysis
    historical_placeholder = st.empty()

# Main loop for real-time updates
while True:
    # Update data
    hw_metrics, media_metrics, hw_df, media_df, anomaly_results = update_data()
    
    # Update Real-time Monitoring tab
    with tab1:
        with system_health_placeholder.container():
            system_health_indicators(
                anomaly_results['hardware'], 
                anomaly_results['media']
            )
        
        with metrics_cards_placeholder.container():
            metrics_cards(hw_metrics, media_metrics)
        
        with charts_placeholder.container():
            real_time_charts(hw_df, media_df)
        
        with alerts_placeholder.container():
            alert_system(st.session_state.alerts)
    
    # Update Anomaly Detection tab
    with tab2:
        with anomaly_viz_placeholder.container():
            anomaly_visualization(hw_df, media_df)
        
        with recommendations_placeholder.container():
            maintenance_recommendations(
                anomaly_results['hardware'], 
                anomaly_results['media']
            )
    
    # Update Historical Analysis tab
    with tab3:
        with historical_placeholder.container():
            historical_analysis(hw_df, media_df)
    
    # Wait before next update
    time.sleep(update_frequency)
