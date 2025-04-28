import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns

def handle_imbalance(X, y, method='smote', random_state=42):
    """
    Handle class imbalance using different techniques.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target labels
    method : str, default='smote'
        Method to use for handling imbalance:
        - 'smote': Synthetic Minority Over-sampling Technique
        - 'adasyn': Adaptive Synthetic Sampling
        - 'smotetomek': Combined over and under-sampling
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_resampled : array-like
        Resampled features
    y_resampled : array-like
        Resampled target labels
    """
    print(f"\nHandling class imbalance using {method.upper()} method...")
    print(f"Class distribution before resampling:")
    class_counts = pd.Series(y).value_counts().sort_index()
    print(class_counts)
    
    # Apply the selected resampling method
    if method.lower() == 'smote':
        resampler = SMOTE(random_state=random_state)
    elif method.lower() == 'adasyn':
        resampler = ADASYN(random_state=random_state)
    elif method.lower() == 'smotetomek':
        resampler = SMOTETomek(random_state=random_state)
    else:
        raise ValueError("Unsupported resampling method. Choose from 'smote', 'adasyn', or 'smotetomek'.")
    
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    
    print(f"Class distribution after resampling:")
    resampled_counts = pd.Series(y_resampled).value_counts().sort_index()
    print(resampled_counts)
    
    # Plot class distribution before and after resampling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Before resampling
    ax1.bar(range(len(class_counts)), class_counts.values)
    ax1.set_xticks(range(len(class_counts)))
    ax1.set_xticklabels(class_counts.index, rotation=45)
    ax1.set_title('Class Distribution Before Resampling')
    ax1.set_ylabel('Count')
    
    # After resampling
    ax2.bar(range(len(resampled_counts)), resampled_counts.values)
    ax2.set_xticks(range(len(resampled_counts)))
    ax2.set_xticklabels(resampled_counts.index, rotation=45)
    ax2.set_title(f'Class Distribution After {method.upper()} Resampling')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'class_balance_{method.lower()}.png')
    
    return X_resampled, y_resampled

def select_features(X, y, X_test, method='filter', n_features=30, estimator=None, random_state=42):
    """
    Perform feature selection using different methods.
    
    Parameters:
    -----------
    X : array-like
        Features for training
    y : array-like
        Target labels
    X_test : array-like
        Features for testing
    method : str, default='filter'
        Method to use for feature selection:
        - 'filter': SelectKBest with ANOVA F-statistic
        - 'wrapper': Recursive Feature Elimination
    n_features : int, default=30
        Number of features to select
    estimator : object, default=None
        Estimator to use for RFE (only for wrapper method)
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_selected, X_test_selected, selected_features_idx, selected_features_names)
    """
    print(f"\nPerforming feature selection using {method} method...")
    
    # For wrapper method, initialize estimator if not provided
    if method.lower() == 'wrapper' and estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)
    
    # Apply the selected feature selection method
    if method.lower() == 'filter':
        selector = SelectKBest(f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        X_test_selected = selector.transform(X_test)
        selected_features_idx = selector.get_support(indices=True)
        
    elif method.lower() == 'wrapper':
        selector = RFE(estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        X_test_selected = selector.transform(X_test)
        selected_features_idx = np.where(selector.support_)[0]
        
    else:
        raise ValueError("Unsupported feature selection method. Choose from 'filter' or 'wrapper'.")
    
    return X_selected, X_test_selected, selected_features_idx

def visualize_feature_selection(X, feature_names, selected_idx, method):
    """
    Visualize the selected features.
    
    Parameters:
    -----------
    X : DataFrame
        Original feature matrix
    feature_names : list
        Names of all features
    selected_idx : array-like
        Indices of selected features
    method : str
        Method used for feature selection
    """
    # Get names of selected features
    selected_names = [feature_names[i] for i in selected_idx]
    
    # Create dataframe for visualization
    df = pd.DataFrame({
        'Feature': feature_names,
        'Selected': [1 if i in selected_idx else 0 for i in range(len(feature_names))]
    })
    
    # Sort by selection status
    df = df.sort_values('Selected', ascending=False)
    
    # Visualize
    plt.figure(figsize=(12, 8))
    plt.bar(df['Feature'], df['Selected'], color=[
        'green' if selected == 1 else 'lightgray' for selected in df['Selected']
    ])
    plt.xticks(rotation=90)
    plt.title(f'Feature Selection Using {method.upper()} Method')
    plt.ylabel('Selected (1) / Not Selected (0)')
    plt.tight_layout()
    plt.savefig(f'feature_selection_{method.lower()}.png')
    
    print(f"Selected {len(selected_idx)} features out of {len(feature_names)}")
    print(f"Selected features: {selected_names}")
    
    return selected_names

def comprehensive_preprocessing(df, test_size=0.2, balance_method='smote', 
                               feature_select_method='filter', n_features=30, 
                               random_state=42):
    """
    Complete preprocessing pipeline including handling missing values,
    encoding labels, handling imbalance, and feature selection.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    test_size : float, default=0.2
        Test set proportion
    balance_method : str, default='smote'
        Method for handling class imbalance
    feature_select_method : str, default='filter'
        Method for feature selection
    n_features : int, default=30
        Number of features to select
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing preprocessed data and metadata
    """
    print("Starting comprehensive preprocessing...")
    
    # Handle missing values
    print("\nHandling missing and infinite values...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Report missing values
    missing_values = df.isna().sum()
    print(f"Missing values per column:")
    print(missing_values[missing_values > 0])
    
    # Fill missing values with median for numerical columns
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Encode the label column
    print("\nEncoding target labels...")
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df[' Label'])
    
    # Display class distribution
    print("\nClass distribution:")
    class_counts = df['Label'].value_counts().sort_index()
    print(class_counts)
    
    # Define features and target
    X = df.drop([' Label', 'Label'], axis=1)
    y = df['Label']
    
    # Split the data (before handling imbalance to ensure test set is untouched)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Handle class imbalance (only on training data)
    X_train_resampled, y_train_resampled = handle_imbalance(
        X_train, y_train, method=balance_method, random_state=random_state
    )
    
    # Feature scaling
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection
    X_train_selected, X_test_selected, selected_idx = select_features(
        X_train_scaled, y_train_resampled, X_test_scaled, 
        method=feature_select_method, n_features=n_features, random_state=random_state
    )
    
    # Visualize selected features
    selected_feature_names = visualize_feature_selection(
        X, X.columns.tolist(), selected_idx, feature_select_method
    )
    
    print(f"\nPreprocessing complete!")
    print(f"Final training set shape: {X_train_selected.shape}")
    print(f"Final test set shape: {X_test_selected.shape}")
    
    # Return data and metadata
    return {
        'X_train': X_train_selected,
        'X_test': X_test_selected,
        'y_train': y_train_resampled,
        'y_test': y_test,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'selected_features': selected_feature_names,
        'selected_idx': selected_idx,
        'all_features': X.columns.tolist(),
        'preprocessed_df': df
    } 