import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : object
        A fitted model with feature_importances_ attribute.
    feature_names : list
        List of feature names.
    model_name : str
        Name of the model for the plot title.
    top_n : int, default=20
        Number of top features to display.
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} doesn't have feature_importances_ attribute.")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for plotting
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp.head(top_n))
    plt.title(f'Top {top_n} Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.replace(" ", "_").lower()}.png')
    
    return feat_imp

def visualize_attack_patterns(X, y, label_names, sample_size=5000, perplexity=30):
    """
    Visualize attack patterns using t-SNE dimensionality reduction.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix.
    y : array-like
        Target labels.
    label_names : dict or list
        Mapping from label indices to label names.
    sample_size : int, default=5000
        Number of samples to use for visualization (t-SNE can be slow for large datasets).
    perplexity : int, default=30
        Perplexity parameter for t-SNE.
    """
    # Sample data if needed
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    # Apply t-SNE
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_sample)
    
    # Create DataFrame for plotting
    df_tsne = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'label': y_sample
    })
    
    # Map numerical labels to names if label_names is provided
    if isinstance(label_names, dict):
        df_tsne['label_name'] = df_tsne['label'].map(label_names)
    elif isinstance(label_names, list):
        df_tsne['label_name'] = df_tsne['label'].apply(lambda x: label_names[x] if x < len(label_names) else f"Unknown-{x}")
    
    # Plot t-SNE visualization
    plt.figure(figsize=(12, 10))
    if 'label_name' in df_tsne.columns:
        scatter = sns.scatterplot(x='x', y='y', hue='label_name', data=df_tsne, palette='viridis', alpha=0.7)
    else:
        scatter = sns.scatterplot(x='x', y='y', hue='label', data=df_tsne, palette='viridis', alpha=0.7)
    
    plt.title('t-SNE Visualization of Attack Patterns')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('attack_patterns_tsne.png')
    
    return df_tsne

def analyze_attack_distribution(df, label_col=' Label'):
    """
    Analyze the distribution of attack types in the dataset.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame containing the data.
    label_col : str, default=' Label'
        Column name for the label.
    """
    # Count attack types
    attack_counts = df[label_col].value_counts()
    
    # Plot distribution
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x=attack_counts.index, y=attack_counts.values)
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Attack Types')
    plt.ylabel('Count')
    plt.xlabel('Attack Type')
    
    # Add count labels on top of bars
    for i, count in enumerate(attack_counts.values):
        ax.text(i, count + 100, f'{count:,}', ha='center')
    
    plt.tight_layout()
    plt.savefig('attack_distribution.png')
    
    # Calculate percentages
    attack_percentages = (attack_counts / attack_counts.sum() * 100).round(2)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Attack Type': attack_counts.index,
        'Count': attack_counts.values,
        'Percentage (%)': attack_percentages.values
    }).sort_values('Count', ascending=False)
    
    return summary_df

def analyze_correlations(df, target_col='Label', top_n=15):
    """
    Analyze correlations between features and the target variable.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame containing the data.
    target_col : str, default='Label'
        Column name for the target variable.
    top_n : int, default=15
        Number of top correlated features to display.
    """
    # Calculate correlations with target
    numeric_cols = df.select_dtypes(include=np.number).columns
    correlations = df[numeric_cols].corrwith(df[target_col]).sort_values(ascending=False)
    
    # Get top positive and negative correlations
    top_positive = correlations.nlargest(top_n)
    top_negative = correlations.nsmallest(top_n)
    
    # Plot top positive correlations
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_positive.values, y=top_positive.index)
    plt.title(f'Top {top_n} Positive Correlations with Target')
    plt.tight_layout()
    plt.savefig('top_positive_correlations.png')
    
    # Plot top negative correlations
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_negative.values, y=top_negative.index)
    plt.title(f'Top {top_n} Negative Correlations with Target')
    plt.tight_layout()
    plt.savefig('top_negative_correlations.png')
    
    # Return merged DataFrame with all correlations
    corr_df = pd.DataFrame({
        'Feature': correlations.index,
        'Correlation': correlations.values
    }).sort_values('Correlation', ascending=False)
    
    return corr_df 