import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import base64
import io

def header_section():
    """Create the header section of the dashboard."""
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.title("Transportation Predictive Maintenance")
        st.markdown("Real-time monitoring and anomaly detection for transportation systems")
    
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

def system_health_indicators(anomaly_results):
    """Display system health indicators for different vehicle types."""
    st.subheader("System Health")
    
    # Check if we have any results to display
    if not anomaly_results or len(anomaly_results) == 0:
        st.info("No vehicle data available yet. Collecting data...")
        return
    
    # Create a health indicator for each vehicle type - ensure at least 1 column
    num_columns = len(anomaly_results) if len(anomaly_results) > 0 else 1
    cols = st.columns(num_columns)
    
    # Display each vehicle's health status
    for i, (vehicle_type, result) in enumerate(anomaly_results.items()):
        with cols[i % num_columns]:  # Use modulo to handle any number of items safely
            # Default status if any fields are missing
            status = "Normal"
            if result.get('anomaly', False):
                if result.get('score', 0) > 0.7:
                    status = "Critical"
                else:
                    status = "Warning"
            
            status_color = "#FF5630" if status == "Critical" else \
                          "#FFAB00" if status == "Warning" else "#36B37E"
            
            vehicle_name = vehicle_type.capitalize()
            
            st.markdown(
                f"""
                <div style='background-color: {status_color}30; padding: 10px; border-radius: 5px; border-left: 5px solid {status_color};'>
                    <h4 style='margin: 0; color: {status_color};'>{vehicle_name} Status: {status}</h4>
                    <p style='margin: 5px 0 0 0;'>
                        {"Potential issue detected" if result.get('anomaly', False) else "Operating normally"}
                        {f" (Model: {result.get('model', 'Unknown')})" if result.get('anomaly', False) else ""}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

def metrics_cards(vehicle_data):
    """Display key metrics for each vehicle type in card format."""
    # Group metrics by vehicle type
    for vehicle_type, data in vehicle_data.items():
        if data.empty:
            continue
            
        # Get latest data point
        latest = data.iloc[-1]
        
        st.subheader(f"{vehicle_type.capitalize()} Metrics")
        
        # Determine which metrics to show based on vehicle type
        if vehicle_type == 'airplane':
            metrics = [
                ('Engine Temperature', 'engine_temperature', '°C'),
                ('Engine Vibration', 'engine_vibration', 'g'),
                ('Fuel Pressure', 'fuel_pressure', 'PSI'),
                ('Oil Pressure', 'oil_pressure', 'PSI'),
                ('Rotation Speed', 'rotation_speed', 'RPM')
            ]
        elif vehicle_type == 'truck':
            metrics = [
                ('Engine Temperature', 'engine_temperature', '°C'),
                ('Tire Pressure', 'tire_pressure', 'PSI'),
                ('Brake Temperature', 'brake_temperature', '°C'),
                ('Battery Voltage', 'battery_voltage', 'V'),
                ('Coolant Level', 'coolant_level', '%')
            ]
        elif vehicle_type == 'railway':
            metrics = [
                ('Engine Temperature', 'engine_temperature', '°C'),
                ('Axle Temperature', 'axle_temperature', '°C'),
                ('Hydraulic Pressure', 'hydraulic_pressure', 'Bar'),
                ('Catenary Voltage', 'catenary_voltage', 'V'),
                ('Traction Motor Temp', 'traction_motor_temp', '°C')
            ]
        else:
            metrics = []
            for col in data.columns:
                if col not in ['timestamp', 'vehicle_type', 'vehicle_id', 'is_anomaly', 'anomaly_metrics']:
                    metrics.append((col.replace('_', ' ').title(), col, ''))
        
        # Create columns for metrics
        cols = st.columns(len(metrics))
        
        # Display each metric
        for i, (label, col, unit) in enumerate(metrics):
            if col not in latest:
                continue
                
            with cols[i]:
                value = latest[col]
                
                # Display delta if we have at least 2 data points
                delta = None
                if len(data) > 1:
                    previous = data.iloc[-2][col]
                    delta = value - previous
                    
                # Determine whether higher is better for delta color
                delta_color = "normal"
                if 'temperature' in col or 'vibration' in col:
                    delta_color = "inverse"  # Lower is better
                
                st.metric(
                    f"{label}",
                    f"{value:.1f} {unit}",
                    f"{delta:.1f} {unit}" if delta is not None else None,
                    delta_color=delta_color
                )

def real_time_charts(vehicle_data):
    """Create real-time charts for each vehicle type."""
    for vehicle_type, data in vehicle_data.items():
        if data.empty or len(data) < 2:
            continue
            
        st.subheader(f"{vehicle_type.capitalize()} Monitoring")
        
        # Limit to last 100 data points
        df = data.tail(100).copy()
        
        # Determine which metrics to chart based on vehicle type
        if vehicle_type == 'airplane':
            primary_metrics = ['engine_temperature', 'engine_vibration', 'fuel_pressure']
            secondary_metrics = ['oil_pressure', 'rotation_speed', 'exhaust_gas_temp']
        elif vehicle_type == 'truck':
            primary_metrics = ['engine_temperature', 'tire_pressure', 'brake_temperature']
            secondary_metrics = ['battery_voltage', 'coolant_level', 'transmission_temp']
        elif vehicle_type == 'railway':
            primary_metrics = ['engine_temperature', 'axle_temperature', 'hydraulic_pressure']
            secondary_metrics = ['catenary_voltage', 'traction_motor_temp', 'pantograph_force']
        else:
            # Dynamic selection based on available columns
            all_metrics = [col for col in df.columns if col not in ['timestamp', 'vehicle_type', 'vehicle_id', 'is_anomaly', 'anomaly_metrics']]
            middle = len(all_metrics) // 2
            primary_metrics = all_metrics[:middle]
            secondary_metrics = all_metrics[middle:]
        
        # Create two charts side by side
        cols = st.columns(2)
        
        # Primary metrics chart
        with cols[0]:
            fig1 = go.Figure()
            for metric in primary_metrics:
                if metric in df.columns:
                    fig1.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[metric],
                        mode='lines',
                        name=metric.replace('_', ' ').title()
                    ))
            
            fig1.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                title="Primary Metrics",
                xaxis_title=None,
                yaxis_title="Value"
            )
            
            st.plotly_chart(fig1, use_container_width=True)
        
        # Secondary metrics chart
        with cols[1]:
            fig2 = go.Figure()
            for metric in secondary_metrics:
                if metric in df.columns:
                    fig2.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[metric],
                        mode='lines',
                        name=metric.replace('_', ' ').title()
                    ))
            
            fig2.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                title="Secondary Metrics",
                xaxis_title=None,
                yaxis_title="Value"
            )
            
            st.plotly_chart(fig2, use_container_width=True)

def anomaly_visualization(vehicle_data):
    """Create visualizations for anomaly detection."""
    for vehicle_type, data in vehicle_data.items():
        if data.empty or len(data) < 10:
            continue
            
        st.subheader(f"{vehicle_type.capitalize()} Anomaly Detection")
        
        # Determine which metrics to visualize based on vehicle type
        if vehicle_type == 'airplane':
            metrics_to_show = ['engine_temperature', 'engine_vibration', 'fuel_pressure']
        elif vehicle_type == 'truck':
            metrics_to_show = ['engine_temperature', 'tire_pressure', 'brake_temperature']
        elif vehicle_type == 'railway':
            metrics_to_show = ['engine_temperature', 'axle_temperature', 'hydraulic_pressure']
        else:
            # Dynamic selection of top 3 metrics
            all_metrics = [col for col in data.columns if col not in ['timestamp', 'vehicle_type', 'vehicle_id', 'is_anomaly', 'anomaly_metrics']]
            metrics_to_show = all_metrics[:3]
        
        # Get data with anomalies
        anomaly_df = data[data['is_anomaly'] == 1]
        
        # Create visualization
        fig = go.Figure()
        
        # Add lines for each metric
        for metric in metrics_to_show:
            if metric in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['timestamp'],
                    y=data[metric],
                    mode='lines',
                    name=metric.replace('_', ' ').title()
                ))
                
                # Add markers for anomalies
                if not anomaly_df.empty:
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
            title=f"{vehicle_type.capitalize()} Metrics with Anomaly Detection",
            xaxis_title=None,
            yaxis_title="Value"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def alert_system(alerts):
    """Display alerts and notifications."""
    st.subheader("Alerts & Notifications")
    
    if not alerts:
        st.info("No active alerts at this time.")
        return
    
    for alert in alerts:
        severity = alert.get('severity', 'Info')
        severity_color = "#FF5630" if severity == "Critical" else \
                         "#FFAB00" if severity == "Warning" else "#36B37E"
        
        st.markdown(
            f"""
            <div style='background-color: {severity_color}20; 
                        padding: 10px; 
                        border-radius: 5px;
                        border-left: 5px solid {severity_color};
                        margin-bottom: 10px;'>
                <div style='display: flex; justify-content: space-between;'>
                    <h4 style='margin: 0; color: {severity_color};'>{alert.get('title', 'Alert')}</h4>
                    <span style='color: #666; font-size: 0.8em;'>{alert.get('timestamp', datetime.now().isoformat())}</span>
                </div>
                <p style='margin: 5px 0 0 0;'>{alert.get('message', '')}</p>
                <p style='margin: 5px 0 0 0; font-size: 0.9em;'>
                    <strong>Vehicle:</strong> {alert.get('vehicle_type', 'Unknown').capitalize()} 
                    {alert.get('vehicle_id', '')}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

def maintenance_recommendations(anomaly_results):
    """Generate maintenance recommendations based on detected anomalies."""
    st.subheader("Predictive Maintenance Recommendations")
    
    # Check if any vehicle has anomalies
    any_anomalies = any(result['anomaly'] for result in anomaly_results.values())
    
    if not any_anomalies:
        st.success("All systems are operating normally. No maintenance actions required at this time.")
        return
    
    # Generate recommendations for each vehicle type with anomalies
    for vehicle_type, result in anomaly_results.items():
        if not result['anomaly']:
            continue
            
        # Get anomalous metrics
        anomaly_metrics = result.get('anomaly_metrics', [])
        
        # Determine recommendations based on vehicle type and anomalous metrics
        recommendations = []
        
        if vehicle_type == 'airplane':
            if 'engine_temperature' in anomaly_metrics:
                recommendations.append("Inspect engine cooling system and heat exchangers")
            if 'engine_vibration' in anomaly_metrics:
                recommendations.append("Perform engine vibration analysis and balance check")
            if 'fuel_pressure' in anomaly_metrics:
                recommendations.append("Check fuel pump and inspect fuel lines for leaks")
            if 'oil_pressure' in anomaly_metrics:
                recommendations.append("Check oil levels and inspect for oil leaks")
            if 'rotation_speed' in anomaly_metrics:
                recommendations.append("Inspect engine control unit and throttle connections")
            
            # Generic recommendations if specific metrics not identified
            if not recommendations:
                recommendations = [
                    "Schedule detailed engine inspection",
                    "Perform comprehensive systems check before next flight",
                    "Review recent flight data for patterns"
                ]
        
        elif vehicle_type == 'truck':
            if 'engine_temperature' in anomaly_metrics:
                recommendations.append("Check cooling system and thermostat operation")
            if 'tire_pressure' in anomaly_metrics:
                recommendations.append("Inspect tires for damage and adjust pressure")
            if 'brake_temperature' in anomaly_metrics:
                recommendations.append("Inspect brake pads and rotors for wear")
            if 'battery_voltage' in anomaly_metrics:
                recommendations.append("Test battery health and charging system")
            if 'transmission_temp' in anomaly_metrics:
                recommendations.append("Check transmission fluid level and condition")
            
            # Generic recommendations
            if not recommendations:
                recommendations = [
                    "Schedule detailed powertrain inspection",
                    "Perform general diagnostics scan",
                    "Check for fault codes in ECU"
                ]
        
        elif vehicle_type == 'railway':
            if 'engine_temperature' in anomaly_metrics:
                recommendations.append("Inspect locomotive cooling system")
            if 'axle_temperature' in anomaly_metrics:
                recommendations.append("Check wheel bearings and lubrication")
            if 'hydraulic_pressure' in anomaly_metrics:
                recommendations.append("Inspect hydraulic pumps and lines for leaks")
            if 'catenary_voltage' in anomaly_metrics:
                recommendations.append("Inspect pantograph and electrical collection system")
            if 'traction_motor_temp' in anomaly_metrics:
                recommendations.append("Inspect traction motors and cooling systems")
            
            # Generic recommendations
            if not recommendations:
                recommendations = [
                    "Schedule comprehensive locomotive inspection",
                    "Perform electrical systems check",
                    "Inspect braking systems"
                ]
        
        else:
            # Generic recommendations for unknown vehicle types
            recommendations = [
                "Schedule comprehensive vehicle inspection",
                "Check all systems showing anomalous behavior",
                "Review maintenance history for recurring issues"
            ]
        
        # Display recommendations for this vehicle type
        severity = "Critical" if result['score'] > 0.7 else "Moderate"
        severity_color = "#FF5630" if severity == "Critical" else "#FFAB00"
        
        st.markdown(
            f"""
            <div style='background-color: {severity_color}20; 
                        padding: 15px; 
                        border-radius: 5px;
                        border-left: 5px solid {severity_color};
                        margin-bottom: 20px;'>
                <h4 style='margin: 0; color: {severity_color};'>{vehicle_type.capitalize()} - {severity} Priority Maintenance</h4>
                <p style='margin: 5px 0 15px 0;'>Based on {result['model']} detection with confidence score {result['score']:.2f}</p>
                <ul style='margin: 0; padding-left: 20px;'>
                    {"".join([f"<li>{rec}</li>" for rec in recommendations])}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

def historical_analysis(vehicle_data, key_prefix=""):
    """Display historical analysis of system performance."""
    st.subheader("Historical Performance Analysis")
    
    # Select vehicle type to analyze
    vehicle_types = list(vehicle_data.keys())
    
    if not vehicle_types:
        st.info("No vehicle data available for analysis.")
        return
    
    selected_vehicle = st.selectbox(
        "Select Vehicle Type",
        vehicle_types,
        key=f"{key_prefix}selected_vehicle"
    )
    
    df = vehicle_data.get(selected_vehicle, pd.DataFrame())
    
    if df.empty or len(df) < 2:
        st.info(f"Not enough historical data available for {selected_vehicle}.")
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
    
    filtered = df[df['timestamp'] >= start_time]
    
    if len(filtered) < 2:
        st.info("Not enough data available for the selected time period.")
        return
    
    # Determine metrics to analyze based on vehicle type
    if selected_vehicle == 'airplane':
        metrics = ['engine_temperature', 'engine_vibration', 'fuel_pressure', 'oil_pressure', 'rotation_speed']
    elif selected_vehicle == 'truck':
        metrics = ['engine_temperature', 'tire_pressure', 'brake_temperature', 'battery_voltage', 'coolant_level']
    elif selected_vehicle == 'railway':
        metrics = ['engine_temperature', 'axle_temperature', 'hydraulic_pressure', 'catenary_voltage', 'traction_motor_temp']
    else:
        metrics = [col for col in df.columns if col not in ['timestamp', 'vehicle_type', 'vehicle_id', 'is_anomaly', 'anomaly_metrics']]
    
    # Create historical trend charts
    fig = go.Figure()
    for metric in metrics:
        if metric in filtered.columns:
            fig.add_trace(go.Scatter(
                x=filtered['timestamp'],
                y=filtered[metric],
                mode='lines',
                name=metric.replace('_', ' ').title()
            ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        title=f"{selected_vehicle.capitalize()} Historical Trends",
        xaxis_title=None,
        yaxis_title="Value"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate summary statistics
    st.markdown("### Performance Summary")
    
    summary = {}
    for metric in metrics:
        if metric in filtered.columns:
            summary[metric] = {
                'mean': filtered[metric].mean(),
                'max': filtered[metric].max(),
                'min': filtered[metric].min(),
                'std': filtered[metric].std()
            }
    
    # Display summary table
    summary_df = pd.DataFrame({
        'Metric': [k.replace('_', ' ').title() for k in summary.keys()],
        'Average': [f"{v['mean']:.1f}" for v in summary.values()],
        'Maximum': [f"{v['max']:.1f}" for v in summary.values()],
        'Minimum': [f"{v['min']:.1f}" for v in summary.values()],
        'Std Dev': [f"{v['std']:.1f}" for v in summary.values()]
    })
    
    st.dataframe(summary_df, hide_index=True)
    
    # Anomaly distribution if available
    if 'is_anomaly' in filtered.columns:
        st.markdown("### Anomaly Distribution")
        
        anomaly_count = filtered['is_anomaly'].sum()
        total_count = len(filtered)
        anomaly_pct = (anomaly_count / total_count) * 100 if total_count > 0 else 0
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=['Normal', 'Anomaly'],
            values=[total_count - anomaly_count, anomaly_count],
            hole=0.5,
            marker_colors=['#36B37E', '#FF5630']
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            title=f"{selected_vehicle.capitalize()} Anomalies",
            annotations=[dict(text=f"{anomaly_pct:.1f}%", x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)

def export_report(vehicle_data, anomaly_results, key_prefix=""):
    """Export data and analysis reports."""
    st.subheader("Export Report")
    
    # Select vehicle type to export
    vehicle_types = list(vehicle_data.keys())
    
    if not vehicle_types:
        st.info("No vehicle data available for export.")
        return
    
    selected_vehicle = st.selectbox(
        "Select Vehicle Type",
        vehicle_types,
        key=f"{key_prefix}export_vehicle"
    )
    
    # Report type selection
    report_type = st.selectbox(
        "Select Report Type",
        ["System Health Summary", "Anomaly Detection Report", "Performance Metrics", "Full System Report"],
        key=f"{key_prefix}export_report_type"
    )
    
    # Time period selection
    time_period = st.selectbox(
        "Select Time Period",
        ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Data"],
        key=f"{key_prefix}export_time_period"
    )
    
    if st.button("Generate Report", key=f"{key_prefix}generate_report"):
        df = vehicle_data.get(selected_vehicle, pd.DataFrame())
        
        if df.empty:
            st.error(f"No data available for {selected_vehicle}.")
            return
        
        # Filter data based on selected time period
        now = datetime.now()
        if time_period == "Last Hour":
            start_time = now - timedelta(hours=1)
        elif time_period == "Last 24 Hours":
            start_time = now - timedelta(hours=24)
        elif time_period == "Last 7 Days":
            start_time = now - timedelta(days=7)
        elif time_period == "Last 30 Days":
            start_time = now - timedelta(days=30)
        else:  # All Data
            start_time = datetime.min
        
        filtered_df = df[df['timestamp'] >= start_time]
        
        if len(filtered_df) == 0:
            st.error(f"No data available for the selected time period.")
            return
        
        # Create CSV for export
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV Data",
            data=csv,
            file_name=f"{selected_vehicle}_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key=f"{key_prefix}download_csv"
        )
        
        # Generate report preview
        st.markdown("### Report Preview")
        
        anomaly_status = "Normal"
        if selected_vehicle in anomaly_results and anomaly_results[selected_vehicle]['anomaly']:
            anomaly_score = anomaly_results[selected_vehicle]['score']
            anomaly_status = "Critical" if anomaly_score > 0.7 else "Warning"
        
        # Determine metrics to include in report
        if selected_vehicle == 'airplane':
            key_metrics = ['engine_temperature', 'engine_vibration', 'fuel_pressure']
        elif selected_vehicle == 'truck':
            key_metrics = ['engine_temperature', 'tire_pressure', 'brake_temperature']
        elif selected_vehicle == 'railway':
            key_metrics = ['engine_temperature', 'axle_temperature', 'hydraulic_pressure']
        else:
            key_metrics = [col for col in filtered_df.columns if col not in ['timestamp', 'vehicle_type', 'vehicle_id', 'is_anomaly', 'anomaly_metrics']][:3]
        
        # Calculate statistics for key metrics
        metric_stats = {}
        for metric in key_metrics:
            if metric in filtered_df.columns:
                metric_stats[metric] = {
                    'mean': filtered_df[metric].mean(),
                    'max': filtered_df[metric].max(),
                    'min': filtered_df[metric].min()
                }
        
        # Create report preview
        st.markdown(
            f"""
            <div style='background-color: #F4F5F7; padding: 20px; border-radius: 5px;'>
                <h2>{selected_vehicle.capitalize()} {report_type}</h2>
                <p><strong>Time Period:</strong> {time_period}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Data Points:</strong> {len(filtered_df)}</p>
                <hr>
                <h3>Summary</h3>
                <p><strong>System Status:</strong> {anomaly_status}</p>
                <p><strong>Anomalies Detected:</strong> {filtered_df['is_anomaly'].sum()}</p>
                <p><strong>Anomaly Rate:</strong> {(filtered_df['is_anomaly'].sum() / len(filtered_df) * 100):.1f}%</p>
                <h3>Key Metrics</h3>
                <table style='width: 100%;'>
                    <tr>
                        <th style='text-align: left; padding: 5px;'>Metric</th>
                        <th style='text-align: right; padding: 5px;'>Average</th>
                        <th style='text-align: right; padding: 5px;'>Maximum</th>
                        <th style='text-align: right; padding: 5px;'>Minimum</th>
                    </tr>
                    {"".join([f"<tr><td style='padding: 5px;'>{m.replace('_', ' ').title()}</td><td style='text-align: right; padding: 5px;'>{stats['mean']:.1f}</td><td style='text-align: right; padding: 5px;'>{stats['max']:.1f}</td><td style='text-align: right; padding: 5px;'>{stats['min']:.1f}</td></tr>" for m, stats in metric_stats.items()])}
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

def system_efficiency(vehicle_data, anomaly_results, key_prefix=""):
    """Display system efficiency metrics and optimization recommendations."""
    st.subheader("System Efficiency & Optimization")
    
    # Select vehicle type to analyze
    vehicle_types = list(vehicle_data.keys())
    
    if not vehicle_types:
        st.info("No vehicle data available for analysis.")
        return
    
    selected_vehicle = st.selectbox(
        "Select Vehicle Type",
        vehicle_types,
        key=f"{key_prefix}efficiency_vehicle"
    )
    
    df = vehicle_data.get(selected_vehicle, pd.DataFrame())
    
    if df.empty or len(df) < 10:
        st.info(f"Not enough data available for {selected_vehicle} efficiency analysis.")
        return
    
    # Calculate efficiency metrics based on vehicle type
    if selected_vehicle == 'airplane':
        # For airplanes, lower engine temp and vibration with stable fuel pressure is efficient
        temp_efficiency = max(0, min(100, 100 - (df['engine_temperature'].mean() - 350) / 2))
        vibration_efficiency = max(0, min(100, 100 - df['engine_vibration'].mean() * 100))
        fuel_efficiency = max(0, min(100, 100 - abs(df['fuel_pressure'].std() * 5)))
        
        # Overall efficiency score
        efficiency_score = (temp_efficiency + vibration_efficiency + fuel_efficiency) / 3
        
        # Recommendations based on efficiency
        if df['engine_temperature'].mean() > 400:
            temp_rec = "Engine running hot. Consider maintenance check on cooling systems."
        else:
            temp_rec = "Engine temperature within optimal range."
            
        if df['engine_vibration'].mean() > 0.6:
            vibration_rec = "High engine vibration detected. Inspect engine mounts and balance."
        else:
            vibration_rec = "Engine vibration within acceptable limits."
            
        if df['fuel_pressure'].std() > 2:
            fuel_rec = "Unstable fuel pressure. Check fuel pump and regulator."
        else:
            fuel_rec = "Fuel system operating efficiently."
    
    elif selected_vehicle == 'truck':
        # For trucks, focus on engine temp, tire pressure uniformity, and brake temperature
        temp_efficiency = max(0, min(100, 100 - abs(df['engine_temperature'].mean() - 90) * 2))
        tire_efficiency = max(0, min(100, 100 - df['tire_pressure'].std() * 10))
        brake_efficiency = max(0, min(100, 100 - (df['brake_temperature'].mean() - 100) / 2))
        
        # Overall efficiency score
        efficiency_score = (temp_efficiency + tire_efficiency + brake_efficiency) / 3
        
        # Recommendations
        if abs(df['engine_temperature'].mean() - 90) > 10:
            temp_rec = "Engine temperature outside optimal range. Check cooling system."
        else:
            temp_rec = "Engine temperature optimal for efficiency."
            
        if df['tire_pressure'].std() > 5:
            tire_rec = "Uneven tire pressure detected. Adjust all tires to recommended PSI."
        else:
            tire_rec = "Tire pressure balanced for optimal fuel efficiency."
            
        if df['brake_temperature'].mean() > 150:
            brake_rec = "Brake temperatures elevated. Check for dragging or excessive use."
        else:
            brake_rec = "Brake system operating efficiently."
    
    elif selected_vehicle == 'railway':
        # For railways, focus on traction efficiency, temperature, and electrical systems
        motor_efficiency = max(0, min(100, 100 - (df['traction_motor_temp'].mean() - 65) * 2))
        power_efficiency = max(0, min(100, 100 - abs(df['catenary_voltage'].std() / 100)))
        axle_efficiency = max(0, min(100, 100 - (df['axle_temperature'].mean() - 40) * 2))
        
        # Overall efficiency score
        efficiency_score = (motor_efficiency + power_efficiency + axle_efficiency) / 3
        
        # Recommendations
        if df['traction_motor_temp'].mean() > 70:
            temp_rec = "Traction motors running hot. Check cooling systems and load balancing."
        else:
            temp_rec = "Traction motor temperature within efficient range."
            
        if df['catenary_voltage'].std() > 500:
            power_rec = "Unstable power collection. Inspect pantograph and contact wire."
        else:
            power_rec = "Power collection system operating efficiently."
            
        if df['axle_temperature'].mean() > 50:
            axle_rec = "Elevated axle temperatures. Check bearings and lubrication."
        else:
            axle_rec = "Axle system operating within optimal temperature range."
    
    else:
        # Generic efficiency metrics for unknown vehicle types
        metrics = [col for col in df.columns if col not in ['timestamp', 'vehicle_type', 'vehicle_id', 'is_anomaly', 'anomaly_metrics']]
        
        if not metrics:
            st.error("No metrics available for efficiency analysis.")
            return
            
        # Use standard deviation as a proxy for stability (lower is better)
        stability_scores = []
        for metric in metrics[:3]:  # Use top 3 metrics
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            # Higher score for lower relative std dev
            stability_scores.append(100 - min(100, (std_val / (mean_val + 0.001)) * 100))
        
        efficiency_score = sum(stability_scores) / len(stability_scores)
    
    # Initialize default recommendations (to avoid unbound variable errors)
    temp_rec = "Regular maintenance recommended to maintain system efficiency."
    vibration_rec = "Monitor system for signs of increasing vibration or instability."
    fuel_rec = "Optimize operational parameters based on historical performance."
    
    # Display efficiency gauge
    efficiency_color = "#36B37E" if efficiency_score > 80 else "#FFAB00" if efficiency_score > 60 else "#FF5630"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=efficiency_score,
        title={'text': f"{selected_vehicle.capitalize()} Efficiency Score"},
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
    
    # Display recommendations
    st.markdown("### Optimization Recommendations")
    
    st.markdown(
        f"""
        <div style='background-color: #F4F5F7; padding: 15px; border-radius: 5px; margin-bottom: 15px;'>
            <h4>Temperature Management</h4>
            <p>{temp_rec}</p>
        </div>
        
        <div style='background-color: #F4F5F7; padding: 15px; border-radius: 5px; margin-bottom: 15px;'>
            <h4>{'Vibration Control' if selected_vehicle == 'airplane' else 'Pressure Management' if selected_vehicle == 'truck' else 'Power System'}</h4>
            <p>{vibration_rec}</p>
        </div>
        
        <div style='background-color: #F4F5F7; padding: 15px; border-radius: 5px;'>
            <h4>{'Fuel System' if selected_vehicle == 'airplane' else 'Brake System' if selected_vehicle == 'truck' else 'Axle System'}</h4>
            <p>{fuel_rec}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display predictive maintenance timeline if anomalies were detected
    if selected_vehicle in anomaly_results and anomaly_results[selected_vehicle]['anomaly']:
        st.markdown("### Predictive Maintenance Timeline")
        
        # Determine urgency based on anomaly score
        score = anomaly_results[selected_vehicle]['score']
        if score > 0.8:
            days_until = 1
            urgency = "Immediate attention required"
        elif score > 0.6:
            days_until = 3
            urgency = "Maintenance required within 3 days"
        elif score > 0.4:
            days_until = 7
            urgency = "Schedule maintenance within 1 week"
        else:
            days_until = 14
            urgency = "Routine maintenance recommended"
        
        maintenance_date = datetime.now() + timedelta(days=days_until)
        
        st.markdown(
            f"""
            <div style='background-color: #EAE6FF; padding: 20px; border-radius: 5px; border-left: 5px solid #6554C0;'>
                <h4 style='margin-top: 0;'>Next Recommended Maintenance</h4>
                <p><strong>Date:</strong> {maintenance_date.strftime('%Y-%m-%d')}</p>
                <p><strong>In:</strong> {days_until} days</p>
                <p><strong>Priority:</strong> {urgency}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Maintenance checklist
        st.markdown("### Maintenance Checklist")
        
        if selected_vehicle == 'airplane':
            checklist_items = [
                "Inspect engine cooling system",
                "Check engine mounts and vibration dampeners",
                "Test fuel pump and pressure regulators",
                "Verify oil pressure and levels",
                "Inspect all flight control surfaces"
            ]
        elif selected_vehicle == 'truck':
            checklist_items = [
                "Check engine cooling system and thermostat",
                "Inspect and adjust tire pressure on all wheels",
                "Examine brake pads and rotors for wear",
                "Test battery and charging system",
                "Check transmission fluid level and condition"
            ]
        elif selected_vehicle == 'railway':
            checklist_items = [
                "Inspect traction motors and cooling systems",
                "Check pantograph and contact wire condition",
                "Examine axle bearings and lubrication",
                "Test braking systems and pressure",
                "Inspect hydraulic systems for leaks"
            ]
        else:
            checklist_items = [
                "Perform comprehensive system inspection",
                "Check all components showing irregular patterns",
                "Test control systems and sensors",
                "Verify fluid levels and pressures",
                "Inspect for structural damage or wear"
            ]
        
        for i, item in enumerate(checklist_items):
            st.checkbox(item, key=f"{key_prefix}checklist_{i}")
            
def create_dashboard():
    """Main function to create the complete dashboard layout."""
    # Display header
    header_section()
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Real-time Monitoring", 
        "Anomaly Detection", 
        "Historical Analysis", 
        "System Optimization"
    ])
    
    # Get data from session state with fallbacks for missing data
    vehicle_data = st.session_state.get('vehicle_data', {})
    if not vehicle_data:
        st.warning("No vehicle data available yet. Initializing data collection...")
        
    # Initialize anomaly_results if it doesn't exist
    anomaly_results = st.session_state.get('anomaly_results', {})
    if not isinstance(anomaly_results, dict):
        st.warning("Invalid anomaly results format. Resetting...")
        anomaly_results = {}
        
    alerts = st.session_state.get('alerts', [])
    
    # Tab 1: Real-time Monitoring
    with tab1:
        system_health_indicators(anomaly_results)
        metrics_cards(vehicle_data)
        real_time_charts(vehicle_data)
        alert_system(alerts)
    
    # Tab 2: Anomaly Detection
    with tab2:
        anomaly_visualization(vehicle_data)
        maintenance_recommendations(anomaly_results)
    
    # Tab 3: Historical Analysis
    with tab3:
        historical_analysis(vehicle_data, key_prefix="tab3_")
    
    # Tab 4: System Optimization
    with tab4:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            system_efficiency(vehicle_data, anomaly_results, key_prefix="tab4_eff_")
        
        with col2:
            export_report(vehicle_data, anomaly_results, key_prefix="tab4_exp_")