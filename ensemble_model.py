import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import necessary model classes
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

class TabNetCatBoostEnsemble(BaseEstimator, ClassifierMixin):
    """
    An ensemble model that combines TabNet and CatBoost predictions
    using a weighted average approach.
    """
    
    def __init__(self, tabnet_weight=0.5, random_state=42):
        """
        Initialize the ensemble model.
        
        Parameters:
        -----------
        tabnet_weight : float, default=0.5
            Weight assigned to TabNet predictions. CatBoost weight will be (1 - tabnet_weight).
        random_state : int, default=42
            Random seed for reproducibility.
        """
        self.tabnet_weight = tabnet_weight
        self.random_state = random_state
        self.tabnet_model = None
        self.catboost_model = None
        
    def fit(self, X, y):
        """
        Fit both TabNet and CatBoost models.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        print("Training TabNet-CatBoost Ensemble...")
        start_time = time.time()
        
        # Split data for validation during TabNet training
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=self.random_state, stratify=y
        )
        
        # Convert data to proper format for TabNet
        X_train_tab = np.array(X_train, dtype=np.float32)
        X_val_tab = np.array(X_val, dtype=np.float32)
        
        # Initialize and train TabNet
        print("Training TabNet component...")
        self.tabnet_model = TabNetClassifier(
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
        
        self.tabnet_model.fit(
            X_train=X_train_tab, y_train=y_train,
            eval_set=[(X_val_tab, y_val)],
            eval_metric=['accuracy'],
            max_epochs=100,
            patience=15,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
        )
        
        # Initialize and train CatBoost
        print("Training CatBoost component...")
        self.catboost_model = CatBoostClassifier(
            iterations=100,
            random_seed=self.random_state,
            verbose=0
        )
        
        self.catboost_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"Total ensemble training time: {training_time:.2f} seconds")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X using weighted average of both models.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        array of shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        # Convert data for TabNet
        X_tab = np.array(X, dtype=np.float32)
        
        # Get predictions from both models
        tabnet_proba = self.tabnet_model.predict_proba(X_tab)
        catboost_proba = self.catboost_model.predict_proba(X)
        
        # Combine predictions using weighted average
        combined_proba = (self.tabnet_weight * tabnet_proba + 
                         (1 - self.tabnet_weight) * catboost_proba)
        
        return combined_proba
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        array of shape (n_samples,)
            The predicted class labels.
        """
        combined_proba = self.predict_proba(X)
        return np.argmax(combined_proba, axis=1)


def optimize_ensemble_weight(X_train, y_train, X_test, y_test, weights_to_try=None):
    """
    Find the optimal weight for combining TabNet and CatBoost predictions.
    
    Parameters:
    -----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    X_test : array-like
        Test features.
    y_test : array-like
        Test labels.
    weights_to_try : list of float, default=None
        TabNet weights to evaluate. If None, uses [0.1, 0.2, ..., 0.9].
        
    Returns:
    --------
    tuple
        (best_ensemble, best_weight, results_df)
    """
    if weights_to_try is None:
        weights_to_try = np.arange(0.1, 1.0, 0.1)
    
    results = []
    best_f1 = 0
    best_ensemble = None
    best_weight = 0
    
    for weight in weights_to_try:
        print(f"\nTrying TabNet weight: {weight:.1f}")
        ensemble = TabNetCatBoostEnsemble(tabnet_weight=weight)
        ensemble.fit(X_train, y_train)
        
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        results.append({
            'TabNet Weight': weight,
            'CatBoost Weight': 1 - weight,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_ensemble = ensemble
            best_weight = weight
    
    results_df = pd.DataFrame(results)
    print("\nWeight Optimization Results:")
    print(results_df)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['TabNet Weight'], results_df['F1 Score'], marker='o')
    plt.xlabel('TabNet Weight')
    plt.ylabel('F1 Score')
    plt.title('Ensemble Performance vs TabNet Weight')
    plt.grid(True)
    plt.savefig('ensemble_weight_optimization.png')
    
    return best_ensemble, best_weight, results_df


if __name__ == "__main__":
    print("Run this file through intrusion_detection.py to use the ensemble model.")
    print("Example:")
    print("from ensemble_model import TabNetCatBoostEnsemble, optimize_ensemble_weight")
    print("# Then in your main() function:")
    print("best_ensemble, best_weight, weight_results = optimize_ensemble_weight(X_train, y_train, X_test, y_test)")
    print("ensemble_results = evaluate_model(y_test, best_ensemble.predict(X_test), f'TabNet-CatBoost Ensemble (TabNet weight: {best_weight:.1f})')")
    print("results.append(ensemble_results)") 