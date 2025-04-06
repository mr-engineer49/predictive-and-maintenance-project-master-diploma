import json
import threading
import time
from kafka import KafkaConsumer
import requests
from datetime import datetime

class KafkaLogConsumer:
    """Consumer for transportation system logs from Kafka."""
    
    def __init__(
        self,
        bootstrap_servers=['localhost:9092'],
        topic='vehicle_metrics',
        group_id='anomaly_detection_group',
        api_url='http://localhost:5001'
    ):
        """Initialize the Kafka consumer.
        
        Args:
            bootstrap_servers (list): List of Kafka broker addresses
            topic (str): Kafka topic to consume from
            group_id (str): Consumer group ID
            api_url (str): URL of the Flask API
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.api_url = api_url
        self.consumer = None
        self.running = False
        self.thread = None
    
    def connect(self):
        """Connect to Kafka and create consumer."""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset='latest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                enable_auto_commit=True
            )
            return True
        except Exception as e:
            print(f"Error connecting to Kafka: {e}")
            return False
    
    def process_message(self, message):
        """Process a message from Kafka.
        
        Args:
            message: Message from Kafka
            
        Returns:
            bool: Whether the message was processed successfully
        """
        try:
            # Extract message value
            metrics = message.value
            
            # Ensure timestamp is in ISO format
            if 'timestamp' in metrics and not isinstance(metrics['timestamp'], str):
                metrics['timestamp'] = datetime.now().isoformat()
            
            # Send to API for processing
            response = requests.post(
                f"{self.api_url}/metrics",
                json=metrics,
                headers={'Content-Type': 'application/json'}
            )
            
            # Check if successful
            if response.status_code != 200:
                print(f"Error sending metrics to API: {response.text}")
                return False
            
            # Create alert if anomaly detected
            if metrics.get('is_anomaly', 0) == 1:
                # Prepare alert data
                alert = {
                    'title': f"Anomaly detected in {metrics['vehicle_type']}",
                    'message': f"Anomalous behavior detected in {', '.join(metrics['anomaly_metrics'])}",
                    'severity': "Critical" if len(metrics['anomaly_metrics']) > 1 else "Warning",
                    'timestamp': metrics['timestamp'],
                    'vehicle_type': metrics['vehicle_type'],
                    'vehicle_id': metrics.get('vehicle_id', f"{metrics['vehicle_type']}_unknown"),
                    'metrics': metrics['anomaly_metrics']
                }
                
                # Send alert to API
                alert_response = requests.post(
                    f"{self.api_url}/alerts",
                    json=alert,
                    headers={'Content-Type': 'application/json'}
                )
                
                if alert_response.status_code != 200:
                    print(f"Error sending alert to API: {alert_response.text}")
            
            return True
        except Exception as e:
            print(f"Error processing message: {e}")
            return False
    
    def consume(self):
        """Consume messages from Kafka and process them."""
        if not self.consumer:
            if not self.connect():
                print("Failed to connect to Kafka. Cannot consume messages.")
                return
        
        print(f"Started consuming from topic: {self.topic}")
        
        while self.running:
            try:
                # Poll for messages with timeout
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        self.process_message(message)
                        
            except Exception as e:
                print(f"Error consuming messages: {e}")
                time.sleep(5)  # Wait before retrying
    
    def start(self):
        """Start consuming messages in a separate thread."""
        if self.running:
            print("Consumer is already running.")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.consume)
        self.thread.daemon = True
        self.thread.start()
        print("Kafka consumer started.")
    
    def stop(self):
        """Stop consuming messages."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.consumer:
            self.consumer.close()
        print("Kafka consumer stopped.")


class KafkaLogProducer:
    """Simulated producer for vehicle metrics to Kafka for testing."""
    
    def __init__(
        self,
        bootstrap_servers=['localhost:9092'],
        topic='vehicle_metrics',
        simulation_interval=2
    ):
        """Initialize the Kafka producer simulator.
        
        Args:
            bootstrap_servers (list): List of Kafka broker addresses
            topic (str): Kafka topic to produce to
            simulation_interval (int): Seconds between simulated messages
        """
        from kafka import KafkaProducer
        from utils.data_generator import TransportationDataGenerator
        
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.simulation_interval = simulation_interval
        self.producer = None
        self.generators = {
            'airplane': TransportationDataGenerator(vehicle_type='airplane'),
            'truck': TransportationDataGenerator(vehicle_type='truck'),
            'railway': TransportationDataGenerator(vehicle_type='railway')
        }
        self.running = False
        self.thread = None
    
    def connect(self):
        """Connect to Kafka and create producer."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            return True
        except Exception as e:
            print(f"Error connecting to Kafka: {e}")
            return False
    
    def simulate(self):
        """Generate and send simulated metrics to Kafka."""
        if not self.producer:
            if not self.connect():
                print("Failed to connect to Kafka. Cannot send messages.")
                return
        
        print(f"Started producing to topic: {self.topic}")
        
        while self.running:
            try:
                # Generate metrics for each vehicle type
                for vehicle_type, generator in self.generators.items():
                    # Generate metrics
                    metrics = generator.generate_metrics()
                    
                    # Convert timestamp to string for JSON serialization
                    metrics['timestamp'] = metrics['timestamp'].isoformat()
                    
                    # Send to Kafka
                    self.producer.send(self.topic, metrics)
                
                # Flush to ensure messages are sent
                self.producer.flush()
                
                # Wait for next iteration
                time.sleep(self.simulation_interval)
                
            except Exception as e:
                print(f"Error producing messages: {e}")
                time.sleep(5)  # Wait before retrying
    
    def start(self):
        """Start producing messages in a separate thread."""
        if self.running:
            print("Producer is already running.")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.simulate)
        self.thread.daemon = True
        self.thread.start()
        print("Kafka producer simulator started.")
    
    def stop(self):
        """Stop producing messages."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.producer:
            self.producer.close()
        print("Kafka producer simulator stopped.")