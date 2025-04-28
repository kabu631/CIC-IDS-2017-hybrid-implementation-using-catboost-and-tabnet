import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from ensemble_model import TabNetCatBoostEnsemble

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def main():
    print("Running quick test with a small subset of data...")
    
    # Load a small sample of data
    print("Loading data sample...")
    file_path = 'Monday-WorkingHours.pcap_ISCX.csv'
    
    # Read only a subset of rows for quick testing
    df = pd.read_csv(file_path, nrows=10000, low_memory=False)
    print(f"Sample size: {df.shape[0]} rows")
    
    # Basic preprocessing
    print("Preprocessing data...")
    
    # Handle missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df[' Label'])
    
    # Print class distribution
    print("Class distribution:")
    print(df['Label'].value_counts())
    
    # Prepare features and target
    X = df.drop([' Label', 'Label'], axis=1)
    y = df['Label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test Random Forest
    print("\nTesting Random Forest...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_time = time.time() - start_time
    
    # Test CatBoost
    print("\nTesting CatBoost...")
    start_time = time.time()
    cb = CatBoostClassifier(iterations=10, random_seed=RANDOM_STATE, verbose=0)
    cb.fit(X_train_scaled, y_train)
    cb_pred = cb.predict(X_test_scaled)
    cb_time = time.time() - start_time
    
    # Test Ensemble
    print("\nTesting TabNet-CatBoost Ensemble...")
    start_time = time.time()
    ensemble = TabNetCatBoostEnsemble(tabnet_weight=0.5, random_state=RANDOM_STATE)
    ensemble.fit(X_train_scaled, y_train)
    ensemble_pred = ensemble.predict(X_test_scaled)
    ensemble_time = time.time() - start_time
    
    # Evaluate and report results
    results = []
    
    for name, pred, train_time in [
        ('Random Forest', rf_pred, rf_time),
        ('CatBoost', cb_pred, cb_time),
        ('TabNet-CatBoost Ensemble', ensemble_pred, ensemble_time)
    ]:
        accuracy = accuracy_score(y_test, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average='weighted'
        )
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Training Time: {train_time:.2f} seconds")
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Training Time': train_time
        })
    
    # Create and show summary table
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Summary:")
    print(results_df)
    
    print("\nQuick test completed. Run the full intrusion_detection.py for complete analysis.")

if __name__ == "__main__":
    main() 