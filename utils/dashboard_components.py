import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import base64
import io
from datetime import datetime, timedelta

def header_section():
    """Create the header section of the dashboard."""
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.title("Media Processing Predictive Maintenance")
        st.markdown("Real-time monitoring and anomaly detection for multimedia processing systems")
    
    with col2:
        st.markdown("<div style='text-align: right; padding-top: 20px;'>", unsafe_allow_html=True)
        if st.session_state.get('dark_mode', False):
            if st.button("☀️ Light Mode"):
                st.session_state['dark_mode'] = False
                st.rerun()
        else:
            if st.button("🌙 Dark Mode"):
                st.session_state['dark_mode'] = True
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def system_health_indicators(hw_anomaly, media_anomaly):
    """Display system health indicators."""
    st.subheader("System Health")
    
    cols = st.columns(2)
    
    with cols[0]:
        hw_status = "Critical" if hw_anomaly['anomaly'] and hw_anomaly['score'] > 0.7 else \
                  "Warning" if hw_anomaly['anomaly'] else "Normal"
        
        hw_color = "#FF5630" if hw_status == "Critical" else \
                 "#FFAB00" if hw_status == "Warning" else "#36B37E"
        
        st.markdown(
            f"""
            <div style='background-color: {hw_color}30; padding: 10px; border-radius: 5px; border-left: 5px solid {hw_color};'>
                <h4 style='margin: 0; color: {hw_color};'>Hardware Status: {hw_status}</h4>
                <p style='margin: 5px 0 0 0;'>
                    {"Potential issue detected" if hw_anomaly['anomaly'] else "Operating normally"}
                    {f" (Model: {hw_anomaly['model']})" if hw_anomaly['anomaly'] else ""}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with cols[1]:
        media_status = "Critical" if media_anomaly['anomaly'] and media_anomaly['score'] > 0.7 else \
                    "Warning" if media_anomaly['anomaly'] else "Normal"
        
        media_color = "#FF5630" if media_status == "Critical" else \
                   "#FFAB00" if media_status == "Warning" else "#36B37E"
        
        st.markdown(
            f"""
            <div style='background-color: {media_color}30; padding: 10px; border-radius: 5px; border-left: 5px solid {media_color};'>
                <h4 style='margin: 0; color: {media_color};'>Media Processing Status: {media_status}</h4>
                <p style='margin: 5px 0 0 0;'>
                    {"Potential issue detected" if media_anomaly['anomaly'] else "Operating normally"}
                    {f" (Model: {media_anomaly['model']})" if media_anomaly['anomaly'] else ""}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

def metrics_cards(hw_metrics, media_metrics):
    """Display key metrics in card format."""
    hw_cols = st.columns(3)
    
    with hw_cols[0]:
        st.metric(
            "CPU Usage", 
            f"{hw_metrics['cpu_usage']:.1f}%",
            f"{hw_metrics['cpu_usage'] - 50:.1f}%" if hw_metrics['cpu_usage'] > 50 else None,
            delta_color="inverse"
        )
    
    with hw_cols[1]:
        st.metric(
            "GPU Usage", 
            f"{hw_metrics['gpu_usage']:.1f}%",
            f"{hw_metrics['gpu_usage'] - 60:.1f}%" if hw_metrics['gpu_usage'] > 60 else None,
            delta_color="inverse"
        )
    
    with hw_cols[2]:
        st.metric(
            "Memory Usage", 
            f"{hw_metrics['memory_usage']:.1f}%",
            f"{hw_metrics['memory_usage'] - 60:.1f}%" if hw_metrics['memory_usage'] > 60 else None,
            delta_color="inverse"
        )
    
    media_cols = st.columns(3)
    
    with media_cols[0]:
        st.metric(
            "Frame Rate", 
            f"{media_metrics['frame_rate']:.1f} FPS",
            f"{media_metrics['frame_rate'] - 24:.1f}" if media_metrics['frame_rate'] < 24 else None
        )
    
    with media_cols[1]:
        st.metric(
            "Bitrate", 
            f"{media_metrics['bitrate']:.0f} Kbps",
            f"{media_metrics['bitrate'] - 5000:.0f}" if media_metrics['bitrate'] < 5000 else None
        )
    
    with media_cols[2]:
        st.metric(
            "Frame Drops", 
            f"{media_metrics['frame_drops']:.1f}/min",
            f"{media_metrics['frame_drops'] - 2:.1f}" if media_metrics['frame_drops'] > 2 else None,
            delta_color="inverse"
        )

def real_time_charts(hw_df, media_df):
    """Create real-time charts for hardware and media metrics."""
    # Limit to last 100 data points
    if len(hw_df) > 100:
        hw_df = hw_df.tail(100)
    if len(media_df) > 100:
        media_df = media_df.tail(100)
    
    # Hardware metrics plots
    st.subheader("Hardware Metrics")
    
    # CPU, GPU, Memory usage
    fig_usage = go.Figure()
    fig_usage.add_trace(go.Scatter(
        x=hw_df['timestamp'], 
        y=hw_df['cpu_usage'],
        mode='lines',
        name='CPU Usage (%)',
        line=dict(color='#0747A6', width=2)
    ))
    fig_usage.add_trace(go.Scatter(
        x=hw_df['timestamp'], 
        y=hw_df['gpu_usage'],
        mode='lines',
        name='GPU Usage (%)',
        line=dict(color='#00B8D9', width=2)
    ))
    fig_usage.add_trace(go.Scatter(
        x=hw_df['timestamp'], 
        y=hw_df['memory_usage'],
        mode='lines',
        name='Memory Usage (%)',
        line=dict(color='#6554C0', width=2)
    ))
    
    fig_usage.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0),
        xaxis_title=None,
        yaxis_title='Percentage (%)',
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig_usage, use_container_width=True)
    
    # Temperature and I/O metrics
    hw_cols = st.columns(2)
    
    with hw_cols[0]:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=hw_df['timestamp'], 
            y=hw_df['temperature'],
            mode='lines',
            name='Temperature (°C)',
            line=dict(color='#FF5630', width=2)
        ))
        
        # Add danger zone
        fig_temp.add_shape(
            type="rect",
            x0=hw_df['timestamp'].min(),
            x1=hw_df['timestamp'].max(),
            y0=80,
            y1=100,
            fillcolor="rgba(255, 86, 48, 0.2)",
            line=dict(width=0),
            layer="below"
        )
        
        fig_temp.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=None,
            yaxis_title='Temperature (°C)'
        )
        
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with hw_cols[1]:
        fig_io = go.Figure()
        fig_io.add_trace(go.Scatter(
            x=hw_df['timestamp'], 
            y=hw_df['disk_io'],
            mode='lines',
            name='Disk I/O (MB/s)',
            line=dict(color='#36B37E', width=2)
        ))
        fig_io.add_trace(go.Scatter(
            x=hw_df['timestamp'], 
            y=hw_df['network_io'],
            mode='lines',
            name='Network I/O (MB/s)',
            line=dict(color='#00B8D9', width=2)
        ))
        
        fig_io.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0),
            xaxis_title=None,
            yaxis_title='MB/s'
        )
        
        st.plotly_chart(fig_io, use_container_width=True)
    
    # Media metrics plots
    st.subheader("Media Processing Metrics")
    
    media_cols = st.columns(2)
    
    with media_cols[0]:
        fig_media1 = go.Figure()
        fig_media1.add_trace(go.Scatter(
            x=media_df['timestamp'], 
            y=media_df['frame_rate'],
            mode='lines',
            name='Frame Rate (FPS)',
            line=dict(color='#0747A6', width=2)
        ))
        
        # Add warning zone for frame rate
        fig_media1.add_shape(
            type="rect",
            x0=media_df['timestamp'].min(),
            x1=media_df['timestamp'].max(),
            y0=0,
            y1=24,
            fillcolor="rgba(255, 171, 0, 0.2)",
            line=dict(width=0),
            layer="below"
        )
        
        fig_media1.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=None,
            yaxis_title='FPS'
        )
        
        st.plotly_chart(fig_media1, use_container_width=True)
    
    with media_cols[1]:
        fig_media2 = go.Figure()
        fig_media2.add_trace(go.Scatter(
            x=media_df['timestamp'], 
            y=media_df['bitrate'],
            mode='lines',
            name='Bitrate (Kbps)',
            line=dict(color='#6554C0', width=2)
        ))
        
        fig_media2.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=None,
            yaxis_title='Kbps'
        )
        
        st.plotly_chart(fig_media2, use_container_width=True)
    
    media_cols2 = st.columns(2)
    
    with media_cols2[0]:
        fig_media3 = go.Figure()
        fig_media3.add_trace(go.Scatter(
            x=media_df['timestamp'], 
            y=media_df['frame_drops'],
            mode='lines',
            name='Frame Drops (per minute)',
            line=dict(color='#FF5630', width=2)
        ))
        
        fig_media3.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=None,
            yaxis_title='Drops/minute'
        )
        
        st.plotly_chart(fig_media3, use_container_width=True)
    
    with media_cols2[1]:
        fig_media4 = go.Figure()
        fig_media4.add_trace(go.Scatter(
            x=media_df['timestamp'], 
            y=media_df['audio_sync_offset'],
            mode='lines',
            name='Audio Sync Offset (ms)',
            line=dict(color='#00B8D9', width=2)
        ))
        
        # Add reference line at 0
        fig_media4.add_shape(
            type="line",
            x0=media_df['timestamp'].min(),
            x1=media_df['timestamp'].max(),
            y0=0,
            y1=0,
            line=dict(color="rgba(0, 0, 0, 0.3)", width=1, dash="dash")
        )
        
        fig_media4.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=None,
            yaxis_title='Offset (ms)'
        )
        
        st.plotly_chart(fig_media4, use_container_width=True)

def anomaly_visualization(hw_df, media_df):
    """Create visualizations for anomaly detection."""
    st.subheader("Anomaly Detection")
    
    # If dataframes contain anomaly information, visualize it
    if 'is_anomaly' in hw_df.columns and 'is_anomaly' in media_df.columns:
        # Create a combined chart for anomalies
        fig = go.Figure()
        
        # Hardware metrics with anomaly highlighting
        for metric in ['cpu_usage', 'gpu_usage', 'memory_usage']:
            if metric in hw_df.columns:
                fig.add_trace(go.Scatter(
                    x=hw_df['timestamp'],
                    y=hw_df[metric],
                    mode='lines',
                    name=f'{metric.replace("_", " ").title()}',
                    line=dict(width=1.5)
                ))
        
        # Add points for anomalies
        anomaly_df = hw_df[hw_df['is_anomaly'] == 1]
        if not anomaly_df.empty:
            for metric in ['cpu_usage', 'gpu_usage', 'memory_usage']:
                if metric in anomaly_df.columns:
                    fig.add_trace(go.Scatter(
                        x=anomaly_df['timestamp'],
                        y=anomaly_df[metric],
                        mode='markers',
                        name=f'{metric.replace("_", " ").title()} Anomaly',
                        marker=dict(
                            size=10,
                            color='red',
                            symbol='circle-open'
                        )
                    ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Hardware Metrics with Anomaly Detection",
            xaxis_title=None,
            yaxis_title='Value',
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a similar chart for media metrics
        fig2 = go.Figure()
        
        # Media metrics with anomaly highlighting
        for metric in ['frame_rate', 'bitrate', 'frame_drops']:
            if metric in media_df.columns:
                fig2.add_trace(go.Scatter(
                    x=media_df['timestamp'],
                    y=media_df[metric],
                    mode='lines',
                    name=f'{metric.replace("_", " ").title()}',
                    line=dict(width=1.5)
                ))
        
        # Add points for anomalies
        anomaly_df = media_df[media_df['is_anomaly'] == 1]
        if not anomaly_df.empty:
            for metric in ['frame_rate', 'bitrate', 'frame_drops']:
                if metric in anomaly_df.columns:
                    fig2.add_trace(go.Scatter(
                        x=anomaly_df['timestamp'],
                        y=anomaly_df[metric],
                        mode='markers',
                        name=f'{metric.replace("_", " ").title()} Anomaly',
                        marker=dict(
                            size=10,
                            color='red',
                            symbol='circle-open'
                        )
                    ))
        
        fig2.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Media Processing Metrics with Anomaly Detection",
            xaxis_title=None,
            yaxis_title='Value',
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig2, use_container_width=True)

def alert_system(alerts):
    """Display alerts and notifications."""
    st.subheader("Alerts & Notifications")
    
    if not alerts:
        st.info("No active alerts at this time.")
        return
    
    for i, alert in enumerate(alerts):
        severity_color = "#FF5630" if alert['severity'] == "Critical" else \
                        "#FFAB00" if alert['severity'] == "Warning" else "#36B37E"
        
        st.markdown(
            f"""
            <div style='background-color: {severity_color}20; 
                        padding: 10px; 
                        border-radius: 5px;
                        border-left: 5px solid {severity_color};
                        margin-bottom: 10px;'>
                <div style='display: flex; justify-content: space-between;'>
                    <h4 style='margin: 0; color: {severity_color};'>{alert['title']}</h4>
                    <span style='color: #666; font-size: 0.8em;'>{alert['time']}</span>
                </div>
                <p style='margin: 5px 0 0 0;'>{alert['message']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

def maintenance_recommendations(hw_anomaly, media_anomaly):
    """Generate maintenance recommendations based on detected anomalies."""
    st.subheader("Predictive Maintenance Recommendations")
    
    if not hw_anomaly['anomaly'] and not media_anomaly['anomaly']:
        st.success("All systems are operating normally. No maintenance actions required at this time.")
        return
    
    recommendations = []
    
    if hw_anomaly['anomaly']:
        recommendations.append({
            "title": "Hardware System Alert",
            "recommendations": [
                "Check system temperature and cooling",
                "Monitor resource usage patterns during processing",
                "Verify hardware component health status",
                "Review background processes and services"
            ]
        })
    
    if media_anomaly['anomaly']:
        recommendations.append({
            "title": "Media Processing Alert",
            "recommendations": [
                "Check encoding parameters and settings",
                "Verify input media source quality",
                "Monitor processing pipeline for bottlenecks",
                "Review codec compatibility and performance",
                "Check for storage I/O limitations"
            ]
        })
    
    for rec in recommendations:
        st.markdown(f"#### {rec['title']}")
        for item in rec['recommendations']:
            st.markdown(f"- {item}")

def historical_analysis(hw_df, media_df, key_prefix=""):
    """Display historical analysis of system performance."""
    st.subheader("Historical Performance Analysis")
    
    if len(hw_df) < 2 or len(media_df) < 2:
        st.info("Not enough historical data available for analysis.")
        return
    
    # Time period selector
    time_period = st.selectbox(
        "Select Time Period",
        ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
        index=0,
        key=f"{key_prefix}historical_time_period"
    )
    
    # Filter data based on selected time period
    now = datetime.now()
    if time_period == "Last 1 Hour":
        start_time = now - timedelta(hours=1)
    elif time_period == "Last 6 Hours":
        start_time = now - timedelta(hours=6)
    elif time_period == "Last 24 Hours":
        start_time = now - timedelta(hours=24)
    else:  # Last 7 Days
        start_time = now - timedelta(days=7)
    
    hw_filtered = hw_df[hw_df['timestamp'] >= start_time]
    media_filtered = media_df[media_df['timestamp'] >= start_time]
    
    if len(hw_filtered) < 2 or len(media_filtered) < 2:
        st.info("Not enough data available for the selected time period.")
        return
    
    # Create historical trend charts
    cols = st.columns(2)
    
    with cols[0]:
        # Hardware historical trends
        hw_metrics = ['cpu_usage', 'gpu_usage', 'memory_usage']
        
        fig = go.Figure()
        for metric in hw_metrics:
            if metric in hw_filtered.columns:
                fig.add_trace(go.Scatter(
                    x=hw_filtered['timestamp'],
                    y=hw_filtered[metric],
                    mode='lines',
                    name=f'{metric.replace("_", " ").title()}'
                ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Hardware Metrics Trends",
            xaxis_title=None,
            yaxis_title='Percentage (%)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with cols[1]:
        # Media historical trends
        media_metrics = ['frame_rate', 'frame_drops']
        
        fig = go.Figure()
        for metric in media_metrics:
            if metric in media_filtered.columns:
                fig.add_trace(go.Scatter(
                    x=media_filtered['timestamp'],
                    y=media_filtered[metric],
                    mode='lines',
                    name=f'{metric.replace("_", " ").title()}'
                ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Media Processing Metrics Trends",
            xaxis_title=None,
            yaxis_title='Value'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Calculate summary statistics
    st.markdown("### Performance Summary")
    
    hw_summary = {}
    for metric in ['cpu_usage', 'gpu_usage', 'memory_usage']:
        if metric in hw_filtered.columns:
            hw_summary[metric] = {
                'mean': hw_filtered[metric].mean(),
                'max': hw_filtered[metric].max(),
                'min': hw_filtered[metric].min(),
                'std': hw_filtered[metric].std()
            }
    
    media_summary = {}
    for metric in ['frame_rate', 'bitrate', 'frame_drops']:
        if metric in media_filtered.columns:
            media_summary[metric] = {
                'mean': media_filtered[metric].mean(),
                'max': media_filtered[metric].max(),
                'min': media_filtered[metric].min(),
                'std': media_filtered[metric].std()
            }
    
    # Display summary tables
    cols = st.columns(2)
    
    with cols[0]:
        st.markdown("#### Hardware Metrics")
        hw_summary_df = pd.DataFrame({
            'Metric': [k.replace('_', ' ').title() for k in hw_summary.keys()],
            'Average': [f"{v['mean']:.1f}" for v in hw_summary.values()],
            'Max': [f"{v['max']:.1f}" for v in hw_summary.values()],
            'Min': [f"{v['min']:.1f}" for v in hw_summary.values()]
        })
        st.dataframe(hw_summary_df, hide_index=True)
    
    with cols[1]:
        st.markdown("#### Media Processing Metrics")
        media_summary_df = pd.DataFrame({
            'Metric': [k.replace('_', ' ').title() for k in media_summary.keys()],
            'Average': [f"{v['mean']:.1f}" for v in media_summary.values()],
            'Max': [f"{v['max']:.1f}" for v in media_summary.values()],
            'Min': [f"{v['min']:.1f}" for v in media_summary.values()]
        })
        st.dataframe(media_summary_df, hide_index=True)
    
    # Anomaly distribution
    if 'is_anomaly' in hw_filtered.columns and 'is_anomaly' in media_filtered.columns:
        st.markdown("### Anomaly Distribution")
        
        cols = st.columns(2)
        
        with cols[0]:
            hw_anomaly_count = hw_filtered['is_anomaly'].sum()
            hw_total = len(hw_filtered)
            hw_anomaly_pct = (hw_anomaly_count / hw_total) * 100 if hw_total > 0 else 0
            
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=['Normal', 'Anomaly'],
                values=[hw_total - hw_anomaly_count, hw_anomaly_count],
                hole=0.5,
                marker_colors=['#36B37E', '#FF5630']
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=30, b=10),
                title="Hardware Anomalies",
                annotations=[dict(text=f"{hw_anomaly_pct:.1f}%", x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with cols[1]:
            media_anomaly_count = media_filtered['is_anomaly'].sum()
            media_total = len(media_filtered)
            media_anomaly_pct = (media_anomaly_count / media_total) * 100 if media_total > 0 else 0
            
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=['Normal', 'Anomaly'],
                values=[media_total - media_anomaly_count, media_anomaly_count],
                hole=0.5,
                marker_colors=['#36B37E', '#FF5630']
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=30, b=10),
                title="Media Processing Anomalies",
                annotations=[dict(text=f"{media_anomaly_pct:.1f}%", x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            st.plotly_chart(fig, use_container_width=True)
