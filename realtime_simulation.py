import pandas as pd
import numpy as np
import time
import json
import random
import datetime
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# For real Kafka implementation (commented out by default)
# from kafka import KafkaProducer, KafkaConsumer

class ThreatDetectionSimulator:
    """
    Simulates real-time network traffic for threat detection.
    """
    
    def __init__(self, data_path, model, preprocessor=None, batch_size=32, delay=0.1, buffer_size=1000):
        """
        Initialize the simulator.
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset file
        model : object
            Trained model for prediction
        preprocessor : object, default=None
            Preprocessor to apply to the data (scaler, feature selection, etc.)
        batch_size : int, default=32
            Number of samples to process at once
        delay : float, default=0.1
            Delay between batches in seconds (simulates real-time)
        buffer_size : int, default=1000
            Size of the buffer for visualization
        """
        self.data_path = data_path
        self.model = model
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.delay = delay
        self.buffer_size = buffer_size
        
        # Statistics and buffer for visualization
        self.total_processed = 0
        self.total_alerts = 0
        self.prediction_times = []
        self.alert_buffer = []
        self.benign_buffer = []
        self.timestamp_buffer = []
        
        # For logging
        self.log_file = f"realtime_simulation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # For real-time visualization
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.line1 = None
        self.line2 = None
        
        # Is the simulation running?
        self.running = False
    
    def preprocess_batch(self, batch):
        """
        Preprocess a batch of data.
        
        Parameters:
        -----------
        batch : DataFrame
            Batch of data to preprocess
            
        Returns:
        --------
        array-like
            Preprocessed data
        """
        # Basic preprocessing (if no preprocessor provided)
        if self.preprocessor is None:
            # Remove non-feature columns
            if ' Label' in batch.columns:
                batch = batch.drop([' Label'], axis=1)
            
            # Handle missing values
            batch.replace([np.inf, -np.inf], np.nan, inplace=True)
            for col in batch.select_dtypes(include=np.number).columns:
                batch[col].fillna(batch[col].median(), inplace=True)
            
            return batch.values
        
        # Use provided preprocessor
        return self.preprocessor.transform(batch)
    
    def predict_batch(self, batch):
        """
        Make predictions on a batch of data.
        
        Parameters:
        -----------
        batch : array-like
            Preprocessed batch of data
            
        Returns:
        --------
        tuple
            (predictions, prediction_probabilities)
        """
        start_time = time.time()
        
        # Make predictions
        try:
            predictions = self.model.predict(batch)
            
            # Get prediction probabilities if available
            try:
                pred_proba = self.model.predict_proba(batch)
            except:
                # If predict_proba not available, use dummy values
                pred_proba = np.zeros((len(predictions), 2))
                pred_proba[:, 1] = predictions
        except Exception as e:
            print(f"Error making predictions: {e}")
            predictions = np.zeros(len(batch))
            pred_proba = np.zeros((len(predictions), 2))
        
        prediction_time = time.time() - start_time
        self.prediction_times.append(prediction_time)
        
        return predictions, pred_proba
    
    def log_predictions(self, batch, predictions, probabilities, true_labels=None):
        """
        Log prediction results.
        
        Parameters:
        -----------
        batch : DataFrame
            Original batch of data
        predictions : array-like
            Predicted labels
        probabilities : array-like
            Prediction probabilities
        true_labels : array-like, default=None
            True labels if available
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        for i in range(len(predictions)):
            # Determine if alert (non-zero label means attack)
            is_alert = predictions[i] != 0
            
            # Update alert count
            if is_alert:
                self.total_alerts += 1
            
            # Update buffers for visualization
            current_time = time.time()
            if is_alert:
                self.alert_buffer.append(current_time)
            else:
                self.benign_buffer.append(current_time)
            
            self.timestamp_buffer.append(current_time)
            
            # Only keep the most recent entries in the buffer
            if len(self.timestamp_buffer) > self.buffer_size:
                self.timestamp_buffer.pop(0)
                
            if len(self.alert_buffer) > self.buffer_size:
                self.alert_buffer.pop(0)
                
            if len(self.benign_buffer) > self.buffer_size:
                self.benign_buffer.pop(0)
            
            # Log the result
            log_entry = {
                'timestamp': timestamp,
                'prediction': int(predictions[i]),
                'is_alert': bool(is_alert),
                'probability': float(np.max(probabilities[i]))
            }
            
            # Add true label if available
            if true_labels is not None:
                log_entry['true_label'] = int(true_labels[i])
                log_entry['correct'] = bool(predictions[i] == true_labels[i])
            
            # Write to log file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def start_kafka_producer(self, topic='network_traffic', bootstrap_servers='localhost:9092'):
        """
        Start producing data to Kafka (for a real Kafka implementation).
        
        Parameters:
        -----------
        topic : str, default='network_traffic'
            Kafka topic to produce to
        bootstrap_servers : str, default='localhost:9092'
            Kafka bootstrap servers
        """
        print("Kafka producer implementation is commented out by default.")
        print("Uncomment the Kafka-related code to use this feature.")
        
        # # Initialize Kafka producer
        # producer = KafkaProducer(
        #     bootstrap_servers=bootstrap_servers,
        #     value_serializer=lambda v: json.dumps(v).encode('utf-8')
        # )
        # 
        # # Read data
        # df = pd.read_csv(self.data_path)
        # 
        # # Send data in batches
        # for i in range(0, len(df), self.batch_size):
        #     batch = df.iloc[i:i+self.batch_size]
        #     
        #     # Convert each row to a dictionary and send
        #     for _, row in batch.iterrows():
        #         producer.send(topic, dict(row))
        #     
        #     producer.flush()
        #     time.sleep(self.delay)
        # 
        # print(f"Finished sending {len(df)} records to Kafka topic '{topic}'")
    
    def start_kafka_consumer(self, topic='network_traffic', bootstrap_servers='localhost:9092'):
        """
        Start consuming data from Kafka and making predictions (for a real Kafka implementation).
        
        Parameters:
        -----------
        topic : str, default='network_traffic'
            Kafka topic to consume from
        bootstrap_servers : str, default='localhost:9092'
            Kafka bootstrap servers
        """
        print("Kafka consumer implementation is commented out by default.")
        print("Uncomment the Kafka-related code to use this feature.")
        
        # # Initialize Kafka consumer
        # consumer = KafkaConsumer(
        #     topic,
        #     bootstrap_servers=bootstrap_servers,
        #     value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        #     auto_offset_reset='earliest',
        #     group_id='threat_detection_group'
        # )
        # 
        # batch = []
        # 
        # # Process messages
        # for message in consumer:
        #     # Add to batch
        #     batch.append(message.value)
        #     
        #     # Process when batch is full
        #     if len(batch) >= self.batch_size:
        #         batch_df = pd.DataFrame(batch)
        #         X = self.preprocess_batch(batch_df)
        #         predictions, probabilities = self.predict_batch(X)
        #         
        #         # Extract true labels if available
        #         true_labels = None
        #         if ' Label' in batch_df.columns:
        #             true_labels = batch_df[' Label'].values
        #         
        #         self.log_predictions(batch_df, predictions, probabilities, true_labels)
        #         
        #         # Update statistics
        #         self.total_processed += len(batch)
        #         
        #         # Clear batch
        #         batch = []
        #         
        #         # Print status
        #         print(f"Processed: {self.total_processed}, Alerts: {self.total_alerts}")
    
    def simulate(self, duration=60, output_file=None):
        """
        Run the simulation directly (without Kafka).
        
        Parameters:
        -----------
        duration : int, default=60
            Duration of the simulation in seconds
        output_file : str, default=None
            Path to save the animation
        """
        # Initialize
        self.running = True
        self.total_processed = 0
        self.total_alerts = 0
        self.prediction_times = []
        self.alert_buffer = []
        self.benign_buffer = []
        self.timestamp_buffer = []
        
        # Read data
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # Set up visualization
        self._setup_visualization()
        
        # Start the simulation in a separate thread
        self.simulation_thread = threading.Thread(
            target=self._simulation_worker, args=(df, duration)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Start the animation
        if output_file:
            print(f"Saving animation to {output_file}...")
            ani = FuncAnimation(self.fig, self._update_plot, interval=200)
            ani.save(output_file, writer='pillow', fps=5)
        else:
            print("Real-time visualization is disabled. Use output_file parameter to save animation.")
        
        # Wait for the simulation to complete
        self.simulation_thread.join()
        
        # Print summary
        avg_prediction_time = np.mean(self.prediction_times) if self.prediction_times else 0
        print("\nSimulation Summary:")
        print(f"Total records processed: {self.total_processed}")
        print(f"Total alerts generated: {self.total_alerts}")
        print(f"Alert rate: {self.total_alerts / self.total_processed * 100:.2f}%")
        print(f"Average prediction time: {avg_prediction_time * 1000:.2f} ms")
        
        self.running = False
        
        # Return statistics
        return {
            'total_processed': self.total_processed,
            'total_alerts': self.total_alerts,
            'alert_rate': self.total_alerts / self.total_processed,
            'avg_prediction_time': avg_prediction_time
        }
    
    def _simulation_worker(self, df, duration):
        """
        Worker function to run the simulation.
        
        Parameters:
        -----------
        df : DataFrame
            Data to simulate
        duration : int
            Duration of the simulation in seconds
        """
        start_time = time.time()
        end_time = start_time + duration
        
        # Process data in batches
        i = 0
        with tqdm(total=duration, desc="Simulation Progress") as pbar:
            while time.time() < end_time and self.running:
                # Get batch (with wrapping if needed)
                batch_indices = [i % len(df) + j for j in range(self.batch_size)]
                batch = df.iloc[batch_indices].copy()
                
                # Extract true labels if available
                true_labels = None
                if ' Label' in batch.columns:
                    true_labels = batch[' Label'].values
                
                # Preprocess
                X = self.preprocess_batch(batch)
                
                # Predict
                predictions, probabilities = self.predict_batch(X)
                
                # Log
                self.log_predictions(batch, predictions, probabilities, true_labels)
                
                # Update statistics
                self.total_processed += len(batch)
                
                # Move to the next batch
                i += self.batch_size
                
                # Delay to simulate real-time
                time.sleep(self.delay)
                
                # Update progress bar
                elapsed = time.time() - start_time
                pbar.update(min(elapsed - pbar.n, duration - pbar.n))
    
    def _setup_visualization(self):
        """
        Set up real-time visualization of the simulation.
        """
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Initialize plots
        self.line1, = self.ax1.plot([], [], 'r-', label='Alert Rate')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Alert Rate')
        self.ax1.set_title('Real-time Alert Rate')
        self.ax1.legend()
        
        self.line2, = self.ax2.plot([], [], 'b-', label='Traffic Volume')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Records per Second')
        self.ax2.set_title('Traffic Volume')
        self.ax2.legend()
        
        plt.tight_layout()
    
    def _update_plot(self, frame):
        """
        Update the visualization plots.
        
        Parameters:
        -----------
        frame : int
            Animation frame number
        """
        # No data yet
        if not self.timestamp_buffer:
            return self.line1, self.line2
        
        # Get the earliest and latest timestamp
        min_time = min(self.timestamp_buffer)
        max_time = max(self.timestamp_buffer)
        
        # Avoid division by zero
        if max_time == min_time:
            return self.line1, self.line2
        
        # Calculate time range for x-axis
        time_range = np.linspace(min_time, max_time, 100)
        
        # Calculate alert rate over time
        alert_counts = np.zeros_like(time_range)
        total_counts = np.zeros_like(time_range)
        
        for i, t in enumerate(time_range):
            # Count records before time t
            alert_counts[i] = sum(1 for alert_time in self.alert_buffer if alert_time <= t)
            total_counts[i] = sum(1 for ts in self.timestamp_buffer if ts <= t)
        
        # Calculate alert rate and traffic volume
        alert_rate = np.zeros_like(time_range)
        for i in range(len(time_range)):
            if total_counts[i] > 0:
                alert_rate[i] = alert_counts[i] / total_counts[i]
        
        # Smooth traffic volume
        window_size = 10
        traffic_volume = np.zeros_like(time_range)
        for i in range(len(time_range)):
            window_start = max(0, i - window_size)
            if i > 0:
                volume = (total_counts[i] - total_counts[window_start]) / (time_range[i] - time_range[window_start])
                traffic_volume[i] = volume
        
        # Normalize time for display
        normalized_time = time_range - min_time
        
        # Update plots
        self.ax1.set_xlim(0, max(normalized_time))
        self.ax1.set_ylim(0, max(1.0, max(alert_rate) * 1.1))
        self.line1.set_data(normalized_time, alert_rate)
        
        self.ax2.set_xlim(0, max(normalized_time))
        self.ax2.set_ylim(0, max(1.0, max(traffic_volume) * 1.1))
        self.line2.set_data(normalized_time, traffic_volume)
        
        return self.line1, self.line2


def run_simulation_demo(model, data_path, duration=60, batch_size=32, delay=0.1, output_file='simulation.gif'):
    """
    Run a demonstration of the real-time threat detection simulation.
    
    Parameters:
    -----------
    model : object
        Trained model for prediction
    data_path : str
        Path to the dataset file
    duration : int, default=60
        Duration of the simulation in seconds
    batch_size : int, default=32
        Number of samples to process at once
    delay : float, default=0.1
        Delay between batches in seconds
    output_file : str, default='simulation.gif'
        Path to save the animation
        
    Returns:
    --------
    dict
        Simulation statistics
    """
    # Create the simulator
    simulator = ThreatDetectionSimulator(
        data_path=data_path,
        model=model,
        batch_size=batch_size,
        delay=delay
    )
    
    # Run the simulation
    print(f"Starting simulation for {duration} seconds...")
    stats = simulator.simulate(duration=duration, output_file=output_file)
    
    return stats


if __name__ == "__main__":
    print("This module provides simulation capabilities for real-time threat detection.")
    print("Import and use the ThreatDetectionSimulator class in your main script:") 