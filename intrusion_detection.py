import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import glob
import os
import torch
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

# Custom modules
from ensemble_model import TabNetCatBoostEnsemble, optimize_ensemble_weight
from preprocessing import comprehensive_preprocessing, handle_imbalance, select_features
from model_evaluation import (evaluate_classifier, plot_roc_curves, plot_precision_recall_curves,
                              plot_confusion_matrices, plot_model_comparison, explain_with_shap,
                              explain_tabnet, generate_comprehensive_report)
from realtime_simulation import ThreatDetectionSimulator, run_simulation_demo

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

def load_data(file_paths):
    """
    Load and concatenate multiple CSV files
    """
    print("Loading data...")
    dataframes = []
    
    for file_path in file_paths:
        print(f"Reading {file_path}...")
        df = pd.read_csv(file_path, low_memory=False)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Total samples: {combined_df.shape[0]}")
    return combined_df

def train_random_forest(X_train, y_train, random_state=RANDOM_STATE):
    """
    Train Random Forest model
    """
    print("\nTraining Random Forest...")
    start_time = time.time()
    
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    return rf_model, training_time

def train_xgboost(X_train, y_train, random_state=RANDOM_STATE):
    """
    Train XGBoost model
    """
    print("\nTraining XGBoost...")
    start_time = time.time()
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        random_state=random_state,
        use_label_encoder=False, 
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    return xgb_model, training_time

def train_catboost(X_train, y_train, random_state=RANDOM_STATE):
    """
    Train CatBoost model
    """
    print("\nTraining CatBoost...")
    start_time = time.time()
    
    catboost_model = CatBoostClassifier(
        iterations=100,
        random_seed=random_state,
        verbose=0
    )
    catboost_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    return catboost_model, training_time

def train_tabnet(X_train, y_train, X_test, y_test, num_classes, random_state=RANDOM_STATE):
    """
    Train TabNet model
    """
    print("\nTraining TabNet...")
    start_time = time.time()
    
    # Convert to proper format for TabNet
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    
    # Define TabNet model
    tabnet_model = TabNetClassifier(
        n_d=64, n_a=64, n_steps=5,
        gamma=1.5, n_independent=2, n_shared=2,
        cat_idxs=[], cat_dims=[], cat_emb_dim=[],
        lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"gamma": 0.95, "step_size": 20},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        epsilon=1e-15,
        device_name='auto',
        verbose=0
    )
    
    # Train the model
    tabnet_model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['accuracy'],
        max_epochs=100,
        patience=15,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
    )
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    return tabnet_model, training_time

def train_ensemble(X_train, y_train, tabnet_weight=0.5, random_state=RANDOM_STATE):
    """
    Train TabNet-CatBoost Ensemble model
    """
    print("\nTraining TabNet-CatBoost Ensemble...")
    start_time = time.time()
    
    # Use specified weight
    ensemble_model = TabNetCatBoostEnsemble(tabnet_weight=tabnet_weight, random_state=random_state)
    ensemble_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    return ensemble_model, training_time

def main():
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Get all CSV files in the directory
    file_paths = glob.glob("*.csv")
    
    # Option 1: Load all datasets
    # df = load_data(file_paths)
    
    # Option 2: Use a smaller subset for faster processing (e.g., one day of data)
    subset_file_paths = ['Monday-WorkingHours.pcap_ISCX.csv']
    df = load_data(subset_file_paths)
    
    # Comprehensive preprocessing (including class imbalance handling and feature selection)
    print("\nPerforming comprehensive preprocessing...")
    prep_results = comprehensive_preprocessing(
        df, 
        test_size=0.2, 
        balance_method='smote',  # Options: 'smote', 'adasyn', 'smotetomek'
        feature_select_method='filter',  # Options: 'filter', 'wrapper'
        n_features=30,  # Number of features to select
        random_state=RANDOM_STATE
    )
    
    # Extract preprocessed data
    X_train = prep_results['X_train']
    X_test = prep_results['X_test']
    y_train = prep_results['y_train']
    y_test = prep_results['y_test']
    label_encoder = prep_results['label_encoder']
    selected_features = prep_results['selected_features']
    
    # Number of classes
    num_classes = len(np.unique(y_train))
    print(f"Number of classes: {num_classes}")
    
    # Create a mapping of class indices to original label names
    class_names = {i: name for i, name in enumerate(label_encoder.classes_)}
    
    # Train all models
    models = {}
    training_times = {}
    
    # Train and save Random Forest
    rf_model, rf_time = train_random_forest(X_train, y_train)
    models['Random Forest'] = rf_model
    training_times['Random Forest'] = rf_time
    
    # Train and save XGBoost
    xgb_model, xgb_time = train_xgboost(X_train, y_train)
    models['XGBoost'] = xgb_model
    training_times['XGBoost'] = xgb_time
    
    # Train and save CatBoost
    catboost_model, catboost_time = train_catboost(X_train, y_train)
    models['CatBoost'] = catboost_model
    training_times['CatBoost'] = catboost_time
    
    # Train and save TabNet
    tabnet_model, tabnet_time = train_tabnet(X_train, y_train, X_test, y_test, num_classes)
    models['TabNet'] = tabnet_model
    training_times['TabNet'] = tabnet_time
    
    # Train and save TabNet-CatBoost Ensemble
    ensemble_model, ensemble_time = train_ensemble(X_train, y_train)
    models['TabNet-CatBoost Ensemble'] = ensemble_model
    training_times['TabNet-CatBoost Ensemble'] = ensemble_time
    
    # Optionally train an optimized ensemble
    if False:  # Set to True to run optimization (computationally expensive)
        print("\nOptimizing ensemble weights...")
        best_ensemble, best_weight, _ = optimize_ensemble_weight(X_train, y_train, X_test, y_test)
        models[f'Optimized Ensemble (weight={best_weight:.1f})'] = best_ensemble
        training_times[f'Optimized Ensemble (weight={best_weight:.1f})'] = ensemble_time  # Reuse time for simplicity
    
    # Evaluate all models
    results_list = []
    
    print("\nEvaluating models...")
    for name, model in models.items():
        try:
            print(f"Evaluating {name}...")
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities
            try:
                y_score = model.predict_proba(X_test)
            except:
                # If predict_proba not available, use placeholder
                y_score = np.zeros((len(y_test), num_classes))
                for i, pred in enumerate(y_pred):
                    y_score[i, int(pred)] = 1
            
            # Evaluate
            result = evaluate_classifier(y_test, y_pred, y_score, name)
            results_list.append(result)
            print(f"Evaluation complete for {name}")
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
    
    # Generate comprehensive evaluation report
    print("\nGenerating comprehensive evaluation report...")
    generate_comprehensive_report(
        results_list, 
        training_times, 
        label_encoder, 
        output_dir=results_dir
    )
    
    # Model explanation using SHAP
    print("\nExplaining models with SHAP...")
    
    # Explain tree-based models with SHAP
    for name in ['Random Forest', 'XGBoost', 'CatBoost']:
        if name in models:
            print(f"Generating SHAP explanations for {name}...")
            try:
                # Use a subset of test data for SHAP analysis
                X_sample = X_test[:500] if len(X_test) > 500 else X_test
                
                # Generate SHAP explanations
                explain_with_shap(
                    models[name], 
                    X_sample, 
                    selected_features, 
                    model_type='tree',
                    plot_type='summary',
                    max_display=20,
                    save_path=f"{results_dir}/shap_{name.lower().replace(' ', '_')}.png"
                )
                
                # Additional SHAP bar plot
                explain_with_shap(
                    models[name], 
                    X_sample, 
                    selected_features, 
                    model_type='tree',
                    plot_type='bar',
                    max_display=20,
                    save_path=f"{results_dir}/shap_bar_{name.lower().replace(' ', '_')}.png"
                )
                
                print(f"SHAP explanations complete for {name}")
            except Exception as e:
                print(f"Error generating SHAP explanations for {name}: {e}")
    
    # Explain TabNet using built-in feature importance
    if 'TabNet' in models:
        print("Explaining TabNet with built-in feature importance...")
        try:
            explain_tabnet(
                models['TabNet'], 
                X_test, 
                selected_features,
                save_path=f"{results_dir}/tabnet_importance.png"
            )
            print("TabNet explanation complete")
        except Exception as e:
            print(f"Error explaining TabNet: {e}")
    
    # Run real-time simulation demo (optional)
    if False:  # Set to True to run simulation
        print("\nRunning real-time threat detection simulation...")
        try:
            # Choose a model for the simulation
            simulation_model = catboost_model  # CatBoost tends to be fast for real-time prediction
            
            # Run the simulation
            sim_stats = run_simulation_demo(
                model=simulation_model,
                data_path=subset_file_paths[0],
                duration=30,  # 30 seconds simulation
                batch_size=32,
                delay=0.1,
                output_file=f"{results_dir}/simulation_animation.gif"
            )
            
            print("Simulation complete")
            print(f"Simulation stats: {sim_stats}")
        except Exception as e:
            print(f"Error running simulation: {e}")
    
    print("\nAll analysis complete. Results saved to the 'results' directory.")

if __name__ == "__main__":
    main() 