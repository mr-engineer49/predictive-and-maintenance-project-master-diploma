import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import base64
import io
from datetime import datetime, timedelta

def system_optimization(hw_df, media_df, key_prefix=""):
    """Display system optimization recommendations and tools."""
    st.subheader("System Optimization")
    
    if len(hw_df) < 10 or len(media_df) < 10:
        st.info("Not enough data for optimization recommendations.")
        return
    
    # Calculate resource utilization statistics
    avg_cpu = hw_df['cpu_usage'].mean()
    avg_gpu = hw_df['gpu_usage'].mean()
    avg_memory = hw_df['memory_usage'].mean()
    avg_frame_rate = media_df['frame_rate'].mean()
    avg_bitrate = media_df['bitrate'].mean()
    avg_frame_drops = media_df['frame_drops'].mean()
    
    # Calculate efficiency scores (fictional metrics for demonstration)
    cpu_efficiency = 100 - (abs(avg_cpu - 50) * 2)  # Optimal is 50%
    gpu_efficiency = 100 - (abs(avg_gpu - 60) * 2)  # Optimal is 60%
    memory_efficiency = 100 - (abs(avg_memory - 60) * 2)  # Optimal is 60%
    
    media_quality = min(100, max(0, (avg_frame_rate / 30) * 100)) - (avg_frame_drops * 5)
    encoding_efficiency = min(100, max(0, (avg_bitrate / 8000) * 100))
    
    overall_efficiency = (cpu_efficiency + gpu_efficiency + memory_efficiency + media_quality + encoding_efficiency) / 5
    
    # Display efficiency gauge
    st.markdown("### Overall System Efficiency")
    
    efficiency_color = "#36B37E" if overall_efficiency > 80 else "#FFAB00" if overall_efficiency > 60 else "#FF5630"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_efficiency,
        title={'text': "System Efficiency Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': efficiency_color},
            'steps': [
                {'range': [0, 60], 'color': "#FF563030"},
                {'range': [60, 80], 'color': "#FFAB0030"},
                {'range': [80, 100], 'color': "#36B37E30"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Resource allocation recommendations
    st.markdown("### Resource Allocation Recommendations")
    
    cols = st.columns(3)
    
    with cols[0]:
        cpu_rec = "Optimal" if cpu_efficiency > 80 else "Underutilized" if avg_cpu < 40 else "Overutilized"
        cpu_advice = "Resource allocation is optimal." if cpu_efficiency > 80 else \
                    "Consider allocating more tasks to utilize available CPU capacity." if avg_cpu < 40 else \
                    "Consider reducing CPU load or upgrading hardware."
        
        st.markdown(f"""
        <div style='background-color: #F4F5F7; padding: 15px; border-radius: 5px;'>
            <h4>CPU Resources</h4>
            <p><strong>Status:</strong> {cpu_rec}</p>
            <p>{cpu_advice}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        gpu_rec = "Optimal" if gpu_efficiency > 80 else "Underutilized" if avg_gpu < 50 else "Overutilized"
        gpu_advice = "Resource allocation is optimal." if gpu_efficiency > 80 else \
                    "GPU resources are underutilized. Consider offloading more processing to GPU." if avg_gpu < 50 else \
                    "Consider reducing GPU load, upgrading hardware, or optimizing GPU-intensive tasks."
        
        st.markdown(f"""
        <div style='background-color: #F4F5F7; padding: 15px; border-radius: 5px;'>
            <h4>GPU Resources</h4>
            <p><strong>Status:</strong> {gpu_rec}</p>
            <p>{gpu_advice}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        memory_rec = "Optimal" if memory_efficiency > 80 else "Underutilized" if avg_memory < 50 else "Overutilized"
        memory_advice = "Resource allocation is optimal." if memory_efficiency > 80 else \
                      "Memory resources are underutilized." if avg_memory < 50 else \
                      "Consider increasing available memory or optimizing memory usage."
        
        st.markdown(f"""
        <div style='background-color: #F4F5F7; padding: 15px; border-radius: 5px;'>
            <h4>Memory Resources</h4>
            <p><strong>Status:</strong> {memory_rec}</p>
            <p>{memory_advice}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Media processing optimization
    st.markdown("### Media Processing Optimization")
    
    cols = st.columns(2)
    
    with cols[0]:
        quality_status = "Excellent" if media_quality > 80 else "Good" if media_quality > 60 else "Poor"
        quality_advice = "Video quality is excellent." if media_quality > 80 else \
                        "Consider optimizing encoding settings to improve quality." if media_quality > 60 else \
                        "Review encoding pipeline for potential issues."
        
        st.markdown(f"""
        <div style='background-color: #F4F5F7; padding: 15px; border-radius: 5px;'>
            <h4>Video Quality</h4>
            <p><strong>Status:</strong> {quality_status}</p>
            <p>{quality_advice}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        encoding_status = "Efficient" if encoding_efficiency > 80 else "Acceptable" if encoding_efficiency > 60 else "Inefficient"
        encoding_advice = "Encoding efficiency is optimal." if encoding_efficiency > 80 else \
                         "Encoding settings could be improved for better efficiency." if encoding_efficiency > 60 else \
                         "Consider revising encoding parameters for better efficiency."
        
        st.markdown(f"""
        <div style='background-color: #F4F5F7; padding: 15px; border-radius: 5px;'>
            <h4>Encoding Efficiency</h4>
            <p><strong>Status:</strong> {encoding_status}</p>
            <p>{encoding_advice}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Maintenance schedule section
    st.markdown("### Maintenance Schedule")
    
    # Function to predict next maintenance date based on anomaly frequency
    def predict_maintenance_date():
        if 'is_anomaly' not in hw_df.columns or 'is_anomaly' not in media_df.columns:
            return datetime.now() + timedelta(days=30)
        
        # Count anomalies and calculate frequency
        hw_anomalies = hw_df['is_anomaly'].sum()
        media_anomalies = media_df['is_anomaly'].sum()
        total_anomalies = hw_anomalies + media_anomalies
        
        total_records = len(hw_df) + len(media_df)
        anomaly_rate = total_anomalies / total_records if total_records > 0 else 0
        
        # Higher anomaly rate means earlier maintenance
        if anomaly_rate > 0.1:
            return datetime.now() + timedelta(days=7)
        elif anomaly_rate > 0.05:
            return datetime.now() + timedelta(days=14)
        else:
            return datetime.now() + timedelta(days=30)
    
    next_maintenance = predict_maintenance_date()
    days_until = (next_maintenance - datetime.now()).days
    
    st.markdown(f"""
    <div style='background-color: #EAE6FF; padding: 20px; border-radius: 5px; border-left: 5px solid #6554C0;'>
        <h4 style='margin-top: 0;'>Next Recommended Maintenance</h4>
        <p><strong>Date:</strong> {next_maintenance.strftime('%Y-%m-%d')}</p>
        <p><strong>In:</strong> {days_until} days</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate maintenance checklist
    st.markdown("### Maintenance Checklist")
    
    checklist_items = [
        "Check cooling systems and clean dust from hardware",
        "Verify storage capacity and free up space if necessary",
        "Update system and media processing software",
        "Run hardware diagnostics",
        "Check network connections and bandwidth",
        "Verify media encoding settings and profiles",
        "Backup configuration and important data"
    ]
    
    for i, item in enumerate(checklist_items):
        st.checkbox(item, key=f"{key_prefix}checklist_{i}")

def export_report(hw_df, media_df, anomaly_results, key_prefix=""):
    """Provide export functionality for reports."""
    st.subheader("Export Report")
    
    report_type = st.selectbox(
        "Select Report Type",
        ["System Health Summary", "Anomaly Detection Report", "Performance Metrics", "Full System Report"],
        key=f"{key_prefix}report_type"
    )
    
    time_period = st.selectbox(
        "Select Time Period",
        ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Data"],
        key=f"{key_prefix}export_time_period"
    )
    
    include_charts = st.checkbox("Include Charts", value=True, key=f"{key_prefix}include_charts")
    include_recommendations = st.checkbox("Include Recommendations", value=True, key=f"{key_prefix}include_recommendations")
    
    if st.button("Generate Report", key=f"{key_prefix}generate_report"):
        # Create in-memory CSV for export
        if report_type == "System Health Summary" or report_type == "Full System Report":
            hw_csv = hw_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Hardware Metrics CSV",
                data=hw_csv,
                file_name=f"hardware_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key=f"{key_prefix}download_hw"
            )
            
            media_csv = media_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Media Metrics CSV",
                data=media_csv,
                file_name=f"media_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key=f"{key_prefix}download_media"
            )
        
        # Create an example PDF report (simulated)
        st.markdown("### Report Preview")
        
        st.markdown(f"""
        <div style='background-color: #F4F5F7; padding: 20px; border-radius: 5px;'>
            <h2>System Health Report</h2>
            <p><strong>Report Type:</strong> {report_type}</p>
            <p><strong>Time Period:</strong> {time_period}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <hr>
            <h3>Summary</h3>
            <p>Hardware Status: {'Critical' if anomaly_results['hardware']['anomaly'] else 'Normal'}</p>
            <p>Media Processing Status: {'Critical' if anomaly_results['media']['anomaly'] else 'Normal'}</p>
            <p>CPU Average: {hw_df['cpu_usage'].mean():.1f}%</p>
            <p>GPU Average: {hw_df['gpu_usage'].mean():.1f}%</p>
            <p>Frame Rate Average: {media_df['frame_rate'].mean():.1f} FPS</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulate PDF download
        st.download_button(
            label="Download PDF Report",
            data="This would be a PDF report in a real application",
            file_name=f"system_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            key=f"{key_prefix}download_pdf"
        )

def resource_efficiency_metrics(hw_df, media_df, key_prefix=""):
    """Display resource efficiency metrics."""
    st.subheader("Resource Efficiency Metrics")
    
    if len(hw_df) < 10 or len(media_df) < 10:
        st.info("Not enough data for efficiency metrics calculation.")
        return
    
    # Calculate efficiency metrics
    processing_efficiency = min(100, max(0, 100 * (media_df['frame_rate'].mean() / (hw_df['gpu_usage'].mean() + 0.1))))
    memory_throughput = min(100, max(0, 100 * (media_df['bitrate'].mean() / (hw_df['memory_usage'].mean() + 0.1))))
    power_efficiency = min(100, max(0, 100 - (hw_df['temperature'].mean() / (hw_df['cpu_usage'].mean() + 0.1) * 5)))
    storage_efficiency = min(100, max(0, 100 - hw_df['disk_io'].mean() / 5))
    network_efficiency = min(100, max(0, 100 - 100 * (hw_df['network_io'].mean() / (media_df['bitrate'].mean() + 0.1))))
    
    # Create radar chart for efficiency metrics
    categories = ['Processing', 'Memory', 'Power', 'Storage', 'Network']
    values = [processing_efficiency, memory_throughput, power_efficiency, storage_efficiency, network_efficiency]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Efficiency',
        line_color='#0747A6'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=400,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display individual metrics with interpretations
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("Processing Efficiency", f"{processing_efficiency:.1f}%")
        st.markdown("**Interpretation:** Ratio of frame rate to GPU usage. Higher is better.")
    
    with cols[1]:
        st.metric("Memory Efficiency", f"{memory_throughput:.1f}%")
        st.markdown("**Interpretation:** Ratio of bitrate to memory usage. Higher is better.")
    
    with cols[2]:
        st.metric("Power Efficiency", f"{power_efficiency:.1f}%")
        st.markdown("**Interpretation:** Inversely related to temperature vs CPU usage. Higher is better.")
    
    cols = st.columns(2)
    
    with cols[0]:
        st.metric("Storage Efficiency", f"{storage_efficiency:.1f}%")
        st.markdown("**Interpretation:** Inversely related to disk I/O. Higher is better.")
    
    with cols[1]:
        st.metric("Network Efficiency", f"{network_efficiency:.1f}%")
        st.markdown("**Interpretation:** Ratio of network usage to bitrate. Higher is better.")