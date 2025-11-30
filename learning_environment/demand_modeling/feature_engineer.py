"""
Feature engineering for demand probability modeling.

Transforms aggregated sales data into feature vectors suitable for
logistic regression or other probability models.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def quality_tier_to_one_hot(quality_tier: str) -> List[int]:
    """Convert quality tier to one-hot encoding."""
    tiers = ['Low', 'Medium', 'High', 'Premium']
    return [1 if quality_tier == tier else 0 for tier in tiers]


def day_of_week_to_one_hot(day_of_week: str) -> List[int]:
    """Convert day of week to one-hot encoding."""
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    if day_of_week in days:
        idx = days.index(day_of_week)
        return [1 if i == idx else 0 for i in range(7)]
    return [0] * 7  # Unknown day


def time_bin_to_one_hot(time_bin: int) -> List[int]:
    """Convert time bin to one-hot encoding."""
    bins = list(range(6))  # 0-5
    return [1 if time_bin == b else 0 for b in bins]


def build_features(
    df: pd.DataFrame,
    include_interactions: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Build feature vectors from aggregated sales data.
    
    Feature design:
    - Time: log(time_to_event), time_bin one-hot, continuous time
    - Price: log(price_rel), price_rel (continuous)
    - Quality: quality_tier one-hot, quality_score (continuous if available)
    - Context: is_weekend, is_playoff, day_of_week one-hot
    - Interactions: price × time, quality × time (optional)
    
    Args:
        df: DataFrame from data_extractor.extract_sales_data()
        include_interactions: Whether to include interaction terms
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target probabilities (n_samples,)
        sample_weights: Exposure weights (n_samples,)
        feature_names: List of feature names
    """
    # Map time_bin back to approximate time_to_event (bin center)
    bin_centers = {
        0: 12.0,    # 0-24h -> 12h
        1: 48.0,    # 24-72h -> 48h
        2: 120.0,   # 72-168h -> 120h
        3: 252.0,   # 168-336h -> 252h
        4: 528.0,   # 336-720h -> 528h
        5: 1080.0   # 720h+ -> 1080h (approximate)
    }
    
    df = df.copy()
    df['time_approx'] = df['time_bin'].map(bin_centers)
    df['time_log'] = np.log1p(df['time_approx'])
    
    # Price features
    df['price_log_rel'] = np.log(df['price_rel'].clip(0.1, 10.0))  # Clip to avoid log(0)
    
    # Build feature vectors
    features_list = []
    feature_names = []
    
    # 1. Time features (continuous)
    features_list.append(df['time_log'].values)
    feature_names.append('time_log')
    
    # 2. Time bin (one-hot)
    time_bin_onehot = np.array([time_bin_to_one_hot(b) for b in df['time_bin']])
    for i in range(6):
        features_list.append(time_bin_onehot[:, i])
        feature_names.append(f'time_bin_{i}')
    
    # 3. Price features
    features_list.append(df['price_log_rel'].values)
    feature_names.append('price_log_rel')
    
    features_list.append(df['price_rel'].values)
    feature_names.append('price_rel')
    
    # 4. Quality tier (one-hot)
    quality_onehot = np.array([quality_tier_to_one_hot(q) for q in df['quality_tier']])
    for i, tier in enumerate(['Low', 'Medium', 'High', 'Premium']):
        features_list.append(quality_onehot[:, i])
        feature_names.append(f'quality_{tier}')
    
    # 5. Event context
    features_list.append(df['is_weekend'].values)
    feature_names.append('is_weekend')
    
    features_list.append(df['is_playoff'].values)
    feature_names.append('is_playoff')
    
    # 6. Day of week (one-hot)
    day_onehot = np.array([day_of_week_to_one_hot(d) for d in df['day_of_week']])
    for i, day in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
        features_list.append(day_onehot[:, i])
        feature_names.append(f'day_{day}')
    
    # 7. Interaction terms (optional)
    if include_interactions:
        # Price × Time interaction
        price_time = df['price_log_rel'].values * df['time_log'].values
        features_list.append(price_time)
        feature_names.append('price_time_interaction')
        
        # Quality × Time (using quality tier index)
        quality_idx = df['quality_tier'].map({
            'Low': 0, 'Medium': 1, 'High': 2, 'Premium': 3
        }).values
        quality_time = quality_idx * df['time_log'].values
        features_list.append(quality_time)
        feature_names.append('quality_time_interaction')
    
    # Stack into feature matrix
    X = np.column_stack(features_list).astype(np.float32)
    
    # Target: empirical probability
    y = df['empirical_prob'].values
    
    # Sample weights: exposure (more observations = higher weight)
    sample_weights = df['exposure'].values.astype(np.float32)
    
    return X, y, sample_weights, feature_names


def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler.
    
    Note: One-hot encoded features are already in [0,1] range,
    but we normalize them anyway for consistency.
    
    Args:
        X_train: Training features
        X_test: Test features
        feature_names: List of feature names
    
    Returns:
        X_train_scaled: Normalized training features
        X_test_scaled: Normalized test features
        scaler: Fitted StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def get_feature_importance_info(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Categorize features for interpretability.
    
    Returns:
        Dict mapping category -> list of feature names
    """
    categories = {
        'time': [f for f in feature_names if 'time' in f.lower()],
        'price': [f for f in feature_names if 'price' in f.lower()],
        'quality': [f for f in feature_names if 'quality' in f.lower()],
        'context': [f for f in feature_names if f in ['is_weekend', 'is_playoff'] or f.startswith('day_')],
        'interactions': [f for f in feature_names if 'interaction' in f.lower()]
    }
    return categories


if __name__ == '__main__':
    # Test feature engineering
    from data_extractor import extract_sales_data
    from pathlib import Path
    
    db_path = Path(__file__).parent.parent / 'data_generation' / 'db.sqlite'
    df = extract_sales_data(db_path)
    
    X, y, weights, feature_names = build_features(df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Sample weights shape: {weights.shape}")
    print(f"\nNumber of features: {len(feature_names)}")
    print("\nFeature categories:")
    categories = get_feature_importance_info(feature_names)
    for category, features in categories.items():
        print(f"  {category}: {len(features)} features")
        if len(features) <= 10:
            print(f"    {features}")

