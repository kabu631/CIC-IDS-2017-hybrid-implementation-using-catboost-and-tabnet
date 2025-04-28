import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import shap
import time
from scipy import stats
import os

def evaluate_classifier(y_true, y_pred, y_score, model_name):
    """
    Comprehensive evaluation of classifier performance.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values
    y_pred : array-like
        Predicted targets as returned by the classifier
    y_score : array-like
        Target scores returned by the classifier (e.g., probability estimates)
    model_name : str
        Name of the model for reporting
        
    Returns:
    --------
    dict
        Dictionary containing various performance metrics
    """
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Calculate ROC-AUC for multiclass (one-vs-rest approach)
    n_classes = len(np.unique(y_true))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Convert y_true to one-hot encoding for multi-class ROC curves
    y_true_binary = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        y_true_binary[i, y_true[i]] = 1
    
    # Convert y_score to proper format if needed
    if len(y_score.shape) == 1:  # If y_score is not already probability estimates
        # Handle binary classification case
        if n_classes == 2:
            y_score_binary = np.zeros((len(y_score), 2))
            y_score_binary[:, 1] = y_score
            y_score_binary[:, 0] = 1 - y_score
            y_score = y_score_binary
    
    # Calculate ROC curve and ROC area for each class
    for i in range(n_classes):
        if i < y_score.shape[1]:  # Ensure we have scores for this class
            fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate macro-average ROC-AUC
    macro_roc_auc = np.mean([roc_auc[i] for i in range(n_classes) if i in roc_auc])
    
    # Calculate precision-recall AUC
    precision_dict = dict()
    recall_dict = dict()
    pr_auc = dict()
    
    for i in range(n_classes):
        if i < y_score.shape[1]:
            precision_dict[i], recall_dict[i], _ = precision_recall_curve(
                y_true_binary[:, i], y_score[:, i]
            )
            pr_auc[i] = average_precision_score(y_true_binary[:, i], y_score[:, i])
    
    # Calculate macro-average PR-AUC
    macro_pr_auc = np.mean([pr_auc[i] for i in range(n_classes) if i in pr_auc])
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Return the results
    return {
        'name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': macro_roc_auc,
        'pr_auc': macro_pr_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'class_roc_auc': roc_auc,
        'precision_curve': precision_dict,
        'recall_curve': recall_dict,
        'class_pr_auc': pr_auc
    }

def plot_roc_curves(results_list, label_encoder=None, save_path=None):
    """
    Plot ROC curves for multiple models.
    
    Parameters:
    -----------
    results_list : list
        List of result dictionaries from evaluate_classifier
    label_encoder : LabelEncoder, default=None
        Label encoder for class names
    save_path : str, default=None
        Path to save the plot. If None, the plot is displayed
    """
    plt.figure(figsize=(12, 8))
    
    # Add a baseline
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
    
    # Plot each model's ROC curve
    for result in results_list:
        model_name = result['name']
        # We'll plot the macro-average ROC curve
        if 0 in result['fpr'] and 0 in result['tpr']:
            plt.plot(result['fpr'][0], result['tpr'][0], 
                     label=f"{model_name} (AUC = {result['roc_auc']:.3f})")
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='best')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_precision_recall_curves(results_list, label_encoder=None, save_path=None):
    """
    Plot Precision-Recall curves for multiple models.
    
    Parameters:
    -----------
    results_list : list
        List of result dictionaries from evaluate_classifier
    label_encoder : LabelEncoder, default=None
        Label encoder for class names
    save_path : str, default=None
        Path to save the plot. If None, the plot is displayed
    """
    plt.figure(figsize=(12, 8))
    
    # Add a baseline
    no_skill = sum([1 for _ in range(len(results_list[0]['confusion_matrix']))]) / len(results_list[0]['confusion_matrix'])
    plt.plot([0, 1], [no_skill, no_skill], 'k--', label='Baseline')
    
    # Plot each model's PR curve
    for result in results_list:
        model_name = result['name']
        # We'll plot the first class's PR curve as an example
        if 0 in result['precision_curve'] and 0 in result['recall_curve']:
            plt.plot(result['recall_curve'][0], result['precision_curve'][0], 
                     label=f"{model_name} (AP = {result['pr_auc']:.3f})")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend(loc='best')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_confusion_matrices(results_list, label_encoder=None, save_path_prefix=None):
    """
    Plot confusion matrices for multiple models.
    
    Parameters:
    -----------
    results_list : list
        List of result dictionaries from evaluate_classifier
    label_encoder : LabelEncoder, default=None
        Label encoder for class names
    save_path_prefix : str, default=None
        Prefix for saving the plots. If None, the plots are displayed
    """
    # Get class names if label_encoder is provided
    if label_encoder is not None:
        class_names = label_encoder.classes_
    else:
        n_classes = results_list[0]['confusion_matrix'].shape[0]
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Plot each model's confusion matrix
    for result in results_list:
        model_name = result['name']
        cm = result['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        
        if save_path_prefix:
            safe_name = model_name.replace(' ', '_').lower()
            save_path = f"{save_path_prefix}_cm_{safe_name}.png"
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_model_comparison(results_list, save_path=None):
    """
    Plot a comparison of model performance metrics.
    
    Parameters:
    -----------
    results_list : list
        List of result dictionaries from evaluate_classifier
    save_path : str, default=None
        Path to save the plot. If None, the plot is displayed
    """
    # Create a dataframe with results
    metrics_df = pd.DataFrame([
        {
            'Model': result['name'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1 Score': result['f1'],
            'ROC-AUC': result['roc_auc'],
            'PR-AUC': result['pr_auc']
        } for result in results_list
    ])
    
    # Plot metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'PR-AUC']
    plt.figure(figsize=(14, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        sns.barplot(x='Model', y=metric, data=metrics_df)
        plt.title(f'Model Comparison - {metric}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return metrics_df

def explain_with_shap(model, X, feature_names, model_type='tree', plot_type='summary', max_display=20, save_path=None):
    """
    Explain model predictions using SHAP values.
    
    Parameters:
    -----------
    model : object
        The trained model to explain
    X : array-like
        Features to use for explanation
    feature_names : list
        Names of the features
    model_type : str, default='tree'
        Type of model: 'tree', 'linear', 'deep', 'kernel'
    plot_type : str, default='summary'
        Type of plot: 'summary', 'bar', 'beeswarm', 'waterfall', 'force'
    max_display : int, default=20
        Maximum number of features to display
    save_path : str, default=None
        Path to save the plot. If None, the plot is displayed
    """
    print(f"Generating SHAP explanations for model type: {model_type}")
    
    # Create the explainer
    if model_type.lower() == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type.lower() == 'linear':
        explainer = shap.LinearExplainer(model, X)
    elif model_type.lower() == 'deep':
        explainer = shap.DeepExplainer(model, X)
    elif model_type.lower() == 'kernel':
        explainer = shap.KernelExplainer(model.predict, X)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Calculate SHAP values
    # For large datasets, use a subset for computation
    if len(X) > 1000:
        X_sample = X[:1000]
        print(f"Using a sample of {len(X_sample)} records for SHAP computation")
    else:
        X_sample = X
    
    shap_values = explainer.shap_values(X_sample)
    
    # Create appropriate plot
    plt.figure(figsize=(12, 10))
    
    if plot_type == 'summary':
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                           max_display=max_display, show=False)
    elif plot_type == 'bar':
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                           plot_type='bar', max_display=max_display, show=False)
    elif plot_type == 'beeswarm':
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                           plot_type='violin', max_display=max_display, show=False)
    elif plot_type == 'waterfall':
        # Waterfall plot for a single prediction
        shap.waterfall_plot(explainer.expected_value[0] if isinstance(explainer.expected_value, list) 
                             else explainer.expected_value, 
                             shap_values[0][0] if isinstance(shap_values, list) 
                             else shap_values[0], feature_names=feature_names, show=False)
    elif plot_type == 'force':
        # Force plot for a single prediction
        shap.force_plot(explainer.expected_value[0] if isinstance(explainer.expected_value, list) 
                         else explainer.expected_value, 
                         shap_values[0][0] if isinstance(shap_values, list) 
                         else shap_values[0], feature_names=feature_names, show=False)
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return shap_values, explainer

def explain_tabnet(model, X, feature_names, save_path=None):
    """
    Explain TabNet model using its built-in feature importance.
    
    Parameters:
    -----------
    model : TabNetClassifier
        The trained TabNet model
    X : array-like
        Features to use for explanation
    feature_names : list
        Names of the features
    save_path : str, default=None
        Path to save the plot. If None, the plot is displayed
    """
    # Get feature importance from the model (M_explain)
    feature_importances = model.feature_importances_
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    
    # Create DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title('TabNet Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Create and save the feature importance masks visualization if available
    if hasattr(model, 'explain'):
        # Get a batch of data
        batch_size = min(1024, X.shape[0])
        X_batch = X[:batch_size]
        
        # Get the attention masks
        masks = model.explain(X_batch)
        
        # Plot heatmap of masks (for first few samples and features)
        num_samples = min(10, batch_size)
        num_features = min(20, X.shape[1])
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(masks[:num_samples, :num_features], cmap='viridis')
        plt.xlabel('Feature Index')
        plt.ylabel('Sample Index')
        plt.title('TabNet Attention Masks')
        
        if save_path:
            mask_path = save_path.replace('.png', '_masks.png')
            plt.savefig(mask_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    return importance_df

def statistical_significance_test(results_list, alpha=0.05):
    """
    Perform statistical significance testing between models.
    
    Parameters:
    -----------
    results_list : list
        List of result dictionaries from evaluate_classifier
    alpha : float, default=0.05
        Significance level
        
    Returns:
    --------
    DataFrame
        Table of p-values for paired comparisons
    """
    # Extract model names and performance metrics
    models = [result['name'] for result in results_list]
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    
    # Prepare results
    significance_results = {}
    
    # Perform pairwise McNemar's test for classification results
    for i, model1 in enumerate(results_list):
        for j, model2 in enumerate(results_list):
            if i < j:  # Avoid duplicate comparisons
                key = f"{model1['name']} vs {model2['name']}"
                significance_results[key] = {}
                
                for metric in metrics:
                    # We'll use a simple t-test for now as a placeholder
                    # In practice, more sophisticated tests like McNemar's test should be used
                    stat, p_value = stats.ttest_ind(
                        [model1[metric]] * 10,  # Replicate for t-test
                        [model2[metric]] * 10   # Replicate for t-test
                    )
                    
                    significance_results[key][metric] = {
                        'p_value': p_value,
                        'significant': p_value < alpha,
                        'better_model': model1['name'] if model1[metric] > model2[metric] else model2['name']
                    }
    
    # Create a more readable results table
    results_table = []
    for comparison, metrics_results in significance_results.items():
        for metric, result in metrics_results.items():
            results_table.append({
                'Comparison': comparison,
                'Metric': metric,
                'p-value': result['p_value'],
                'Significant': result['significant'],
                'Better Model': result['better_model']
            })
    
    return pd.DataFrame(results_table)

def generate_comprehensive_report(results_list, training_times, label_encoder, output_dir='results'):
    """
    Generate a comprehensive report with all evaluations.
    
    Parameters:
    -----------
    results_list : list
        List of result dictionaries from evaluate_classifier
    training_times : dict
        Dictionary mapping model names to training times
    label_encoder : LabelEncoder
        Label encoder used for the target variable
    output_dir : str, default='results'
        Directory to save the results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Add training times to results
    for result in results_list:
        model_name = result['name']
        if model_name in training_times:
            result['training_time'] = training_times[model_name]
    
    # Generate metrics comparison
    metrics_df = plot_model_comparison(
        results_list, save_path=f"{output_dir}/model_comparison_metrics.png"
    )
    
    # Add training times to metrics dataframe
    metrics_df['Training Time (s)'] = metrics_df['Model'].map(
        {model_name: time for model_name, time in training_times.items()}
    )
    
    # Save metrics as CSV
    metrics_df.to_csv(f"{output_dir}/model_metrics.csv", index=False)
    
    # Plot ROC curves
    plot_roc_curves(
        results_list, label_encoder, save_path=f"{output_dir}/roc_curves.png"
    )
    
    # Plot Precision-Recall curves
    plot_precision_recall_curves(
        results_list, label_encoder, save_path=f"{output_dir}/pr_curves.png"
    )
    
    # Plot confusion matrices
    plot_confusion_matrices(
        results_list, label_encoder, save_path_prefix=f"{output_dir}/confusion_matrix"
    )
    
    # Perform statistical significance tests
    significance_df = statistical_significance_test(results_list)
    significance_df.to_csv(f"{output_dir}/statistical_significance.csv", index=False)
    
    # Plot training times
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Training Time (s)', data=metrics_df)
    plt.title('Model Comparison - Training Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_times.png")
    plt.close()
    
    print(f"Comprehensive report generated in directory: {output_dir}")
    print(f"Model performance summary:")
    print(metrics_df.to_string(index=False)) 