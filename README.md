# Real-Time Threat Detection Using CIC-IDS-2017

This project implements and compares various machine learning models for intrusion detection using the CIC-IDS-2017 dataset. It features a comprehensive approach to network security, handling class imbalance, feature selection, model explanation, and simulated real-time detection.

## Models Implemented

1. **TabNet** - A novel deep learning architecture for tabular data that uses sequential attention
2. **CatBoost** - A gradient boosting algorithm that handles categorical features effectively
3. **XGBoost** - A traditional gradient boosting implementation, known for its performance
4. **Random Forest** - A widely-used ensemble learning method
5. **TabNet-CatBoost Ensemble** - A novel ensemble approach combining TabNet and CatBoost

## Dataset

The CIC-IDS-2017 dataset contains network traffic data with various attack types:
- DDoS attacks
- Brute Force attacks
- XSS
- SQL Injection
- Infiltration
- Port Scanning
- Botnet

## Features

### Advanced Preprocessing
- **Class Imbalance Handling**: Implements SMOTE, ADASYN, and SMOTETomek
- **Feature Selection**: Uses filter-based and wrapper methods to select the most relevant features
- **Data Cleaning**: Handles missing values, infinite values, and standardizes the features

### Comprehensive Model Evaluation
- **Performance Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC, PR-AUC
- **Visualization**: ROC curves, Precision-Recall curves, confusion matrices
- **Statistical Testing**: Significance tests between model performances

### Model Explainability
- **SHAP Values**: For tree-based models (Random Forest, XGBoost, CatBoost)
- **TabNet Explanations**: Built-in feature importance and attention masks
- **Feature Importance**: Visualization of the most important features for each model

### Real-Time Simulation
- **Streaming Data**: Simulates real-time detection with streaming network traffic
- **Performance Monitoring**: Tracks detection rate, latency, and traffic volume
- **Visualization**: Real-time animation of alert rates and traffic patterns
- **Kafka Integration**: Optional Kafka producer/consumer for true streaming applications

## Setup and Installation

1. Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the analysis:
```
python intrusion_detection.py
```

4. For a quick test with a smaller dataset:
```
python quick_test.py
```

## Results

Results of the model comparison will be saved in the `results` directory:
- `model_comparison_metrics.png` - Comparison of all performance metrics
- `model_metrics.csv` - CSV file with all metrics
- `roc_curves.png` - ROC curves for all models
- `pr_curves.png` - Precision-Recall curves for all models
- `confusion_matrix_*.png` - Confusion matrices for each model
- `training_times.png` - Comparison of training times
- `statistical_significance.csv` - Results of statistical significance tests
- `shap_*.png` - SHAP feature importance visualizations
- `tabnet_importance.png` - TabNet feature importance
- `simulation_animation.gif` - Animation of real-time detection (if enabled)

## Usage

### Basic Analysis
```python
# Run the complete analysis pipeline
python intrusion_detection.py
```

### Customizing Preprocessing
You can modify the `comprehensive_preprocessing` function call in `intrusion_detection.py`:
```python
prep_results = comprehensive_preprocessing(
    df, 
    test_size=0.2, 
    balance_method='smote',  # Options: 'smote', 'adasyn', 'smotetomek'
    feature_select_method='filter',  # Options: 'filter', 'wrapper'
    n_features=30,  # Number of features to select
    random_state=RANDOM_STATE
)
```

### Enabling Real-Time Simulation
To run the real-time simulation, set the simulation flag to `True` in `intrusion_detection.py`:
```python
# Run real-time simulation demo
if True:  # Set to True to run simulation
    print("\nRunning real-time threat detection simulation...")
    # ... simulation code ...
```

### Enabling Ensemble Weight Optimization
To optimize the ensemble weights, set the optimization flag to `True` in `intrusion_detection.py`:
```python
# Optionally train an optimized ensemble
if True:  # Set to True to run optimization
    print("\nOptimizing ensemble weights...")
    # ... optimization code ...
```

## Advanced Features

### Implementing Kafka Streaming
The codebase includes commented-out Kafka integration. To enable:

1. Ensure Kafka is installed and running
2. Uncomment Kafka-related code in `realtime_simulation.py`
3. Run the Kafka producer and consumer functions:
```python
# Start the Kafka producer
simulator.start_kafka_producer()

# In another process/terminal
simulator.start_kafka_consumer()
```

## Notes

- By default, the script uses only one day of data (Monday) for faster processing. To use all data, uncomment the relevant line in the main function.
- For large datasets, consider using a subset of the data or increasing computational resources.
- The ensemble model can be further optimized by uncommenting the weight optimization code in the main function.

## References

- CIC-IDS-2017 Dataset: https://www.unb.ca/cic/datasets/ids-2017.html
- TabNet: https://arxiv.org/abs/1908.07442
- CatBoost: https://arxiv.org/abs/1706.09516
- SHAP: https://github.com/slundberg/shap
- SMOTE: https://arxiv.org/abs/1106.1813
