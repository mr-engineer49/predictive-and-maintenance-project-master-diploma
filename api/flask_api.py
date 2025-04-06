from flask import Flask, jsonify, request
import pandas as pd
import json
from datetime import datetime

app = Flask(__name__)

# Global variables to store alerts and state
alerts = []
vehicle_data = {}
anomaly_results = {}

@app.route('/health', methods=['GET'])
def health_check():
    """API endpoint for health check."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/alerts', methods=['GET'])
def get_alerts():
    """API endpoint to get all current alerts."""
    global alerts
    return jsonify({'alerts': alerts})

@app.route('/alerts/latest', methods=['GET'])
def get_latest_alerts():
    """API endpoint to get latest alerts, with optional filtering."""
    global alerts
    
    # Get query parameters
    vehicle_type = request.args.get('vehicle_type')
    limit = request.args.get('limit', default=10, type=int)
    
    # Filter alerts if vehicle_type is provided
    filtered_alerts = alerts
    if vehicle_type:
        filtered_alerts = [alert for alert in alerts if alert.get('vehicle_type') == vehicle_type]
    
    # Limit number of alerts returned
    latest_alerts = filtered_alerts[:limit]
    
    return jsonify({
        'count': len(latest_alerts),
        'alerts': latest_alerts
    })

@app.route('/alerts', methods=['POST'])
def add_alert():
    """API endpoint to add a new alert (internal use)."""
    global alerts
    
    # Get alert data from request
    alert_data = request.json
    
    # Validate alert data
    if not alert_data or not isinstance(alert_data, dict):
        return jsonify({'error': 'Invalid alert data'}), 400
    
    # Add timestamp if not provided
    if 'timestamp' not in alert_data:
        alert_data['timestamp'] = datetime.now().isoformat()
    
    # Add alert to list
    alerts.insert(0, alert_data)
    
    # Keep only the latest 100 alerts
    alerts = alerts[:100]
    
    return jsonify({'status': 'success', 'alert': alert_data})

@app.route('/vehicle/status', methods=['GET'])
def get_vehicle_status():
    """API endpoint to get current status of vehicles."""
    global vehicle_data, anomaly_results
    
    # Get query parameters
    vehicle_type = request.args.get('vehicle_type')
    
    # Filter vehicle data if vehicle_type is provided
    if vehicle_type and vehicle_type in vehicle_data:
        status_data = {
            vehicle_type: {
                'latest_metrics': vehicle_data[vehicle_type].iloc[-1].to_dict() if not vehicle_data[vehicle_type].empty else {},
                'anomaly_status': anomaly_results.get(vehicle_type, {'anomaly': False, 'model': 'None', 'score': 0.0})
            }
        }
    else:
        # Return status for all vehicle types
        status_data = {}
        for vtype in vehicle_data.keys():
            status_data[vtype] = {
                'latest_metrics': vehicle_data[vtype].iloc[-1].to_dict() if not vehicle_data[vtype].empty else {},
                'anomaly_status': anomaly_results.get(vtype, {'anomaly': False, 'model': 'None', 'score': 0.0})
            }
    
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'vehicles': status_data
    })

@app.route('/metrics', methods=['POST'])
def update_metrics():
    """API endpoint for Kafka consumer to update metrics (internal use)."""
    global vehicle_data
    
    # Get metrics data from request
    metrics_data = request.json
    
    if not metrics_data:
        return jsonify({'error': 'No metrics data provided'}), 400
    
    # Process incoming metrics
    vehicle_type = metrics_data.get('vehicle_type')
    
    if not vehicle_type:
        return jsonify({'error': 'Vehicle type not specified'}), 400
    
    # Convert to DataFrame format for internal processing
    # (This would normally be handled by the Kafka consumer)
    if vehicle_type not in vehicle_data:
        vehicle_data[vehicle_type] = pd.DataFrame()
    
    # Add metrics to DataFrame
    new_row = pd.DataFrame([metrics_data])
    
    if vehicle_data[vehicle_type].empty:
        vehicle_data[vehicle_type] = new_row
    else:
        vehicle_data[vehicle_type] = pd.concat([vehicle_data[vehicle_type], new_row], ignore_index=True)
    
    # Maintain maximum size of 1000 data points per vehicle type
    if len(vehicle_data[vehicle_type]) > 1000:
        vehicle_data[vehicle_type] = vehicle_data[vehicle_type].iloc[-1000:]
    
    return jsonify({'status': 'success'})

@app.route('/anomaly/update', methods=['POST'])
def update_anomaly_results():
    """API endpoint to update anomaly detection results (internal use)."""
    global anomaly_results
    
    # Get anomaly results from request
    new_results = request.json
    
    if not new_results:
        return jsonify({'error': 'No anomaly results provided'}), 400
    
    # Update anomaly results
    anomaly_results.update(new_results)
    
    return jsonify({'status': 'success'})

def start_api_server(port=5001):
    """Start the Flask API server."""
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)