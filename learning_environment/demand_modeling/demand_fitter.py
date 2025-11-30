"""
Fit demand probability model using logistic regression.

Uses binomial GLM (logistic regression) to model P(sale | price, time, quality, context).
"""

from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import json
from pathlib import Path


class DemandModel:
    """
    Wrapper for fitted demand probability model.
    
    Provides interface for predicting P(sale | features) and
    storing model metadata.
    """
    
    def __init__(
        self,
        model: LogisticRegression,
        scaler,
        feature_names: list,
        training_metrics: Dict,
        p_ref_lookup: Optional[Dict] = None
    ):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.training_metrics = training_metrics
        self.p_ref_lookup = p_ref_lookup or {}
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict P(sale | features).
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Probability array (n_samples, 2) where [:, 1] is P(sale)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class (0 = no sale, 1 = sale)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients for interpretability."""
        coef = self.model.coef_[0]
        return dict(zip(self.feature_names, coef))


def fit_demand_model(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    feature_names: list,
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
    stratify_by_event: bool = True,
    event_ids: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    weights_test: Optional[np.ndarray] = None
) -> Tuple[DemandModel, Dict]:
    """
    Fit logistic regression model for demand probability.
    
    Args:
        X: Feature matrix (n_samples, n_features) - training data
        y: Target probabilities (n_samples,)
        sample_weights: Exposure weights (n_samples,)
        feature_names: List of feature names
        test_size: Fraction of data for testing (ignored if X_test provided)
        random_state: Random seed
        C: Inverse regularization strength (higher = less regularization)
        max_iter: Maximum iterations for solver
        stratify_by_event: Whether to stratify train/test by event_id
        event_ids: Event IDs for stratification (required if stratify_by_event=True)
        X_test: Optional test features (if provided, skips train/test split)
        y_test: Optional test targets
        weights_test: Optional test weights
    
    Returns:
        DemandModel: Fitted model wrapper
        metrics: Dictionary of evaluation metrics
    """
    # Normalize features
    from feature_engineer import normalize_features
    
    # If test data is provided, use it directly (for CV)
    if X_test is not None and y_test is not None:
        X_train, y_train = X, y
        weights_train = sample_weights
        weights_test = weights_test if weights_test is not None else np.ones(len(y_test))
    elif test_size > 0 and stratify_by_event and event_ids is not None:
        # Stratified split by event to avoid data leakage
        unique_events = np.unique(event_ids)
        train_events, test_events = train_test_split(
            unique_events, test_size=test_size, random_state=random_state
        )
        train_mask = np.isin(event_ids, train_events)
        test_mask = np.isin(event_ids, test_events)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        weights_train, weights_test = sample_weights[train_mask], sample_weights[test_mask]
    elif test_size > 0:
        # Standard random split
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, sample_weights, test_size=test_size, random_state=random_state
        )
    else:
        # No test split - use all data for training (no evaluation metrics)
        X_train, y_train = X, y
        weights_train = sample_weights
        X_test, y_test, weights_test = None, None, None
    
    # Normalize features
    if X_test is not None:
        X_train_scaled, X_test_scaled, scaler = normalize_features(
            X_train, X_test, feature_names
        )
    else:
        # No test set - just fit scaler on training data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = None
    
    # Fit logistic regression
    # Use class_weight='balanced' to handle potential class imbalance
    # But since we're predicting probabilities (not binary), we use sample_weight
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver='lbfgs',  # Good for small-medium datasets
        random_state=random_state,
        n_jobs=-1
    )
    
    # Convert probabilities to binary for training (threshold at 0.5)
    # Actually, for binomial GLM, we should use the probabilities directly
    # But sklearn LogisticRegression expects binary targets, so we threshold
    y_train_binary = (y_train >= 0.5).astype(int)
    
    model.fit(X_train_scaled, y_train_binary, sample_weight=weights_train)
    
    # Evaluate
    y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]
    
    # Compute metrics
    from sklearn.metrics import (
        roc_auc_score, 
        brier_score_loss, 
        log_loss,
        accuracy_score
    )
    
    # Training metrics
    y_train_binary = (y_train >= 0.5).astype(int)
    try:
        train_auc = roc_auc_score(y_train_binary, y_pred_proba_train)
    except ValueError:
        train_auc = 0.5
    
    train_brier = brier_score_loss(y_train_binary, y_pred_proba_train)
    train_logloss = log_loss(y_train_binary, y_pred_proba_train)
    train_acc = accuracy_score(y_train_binary, (y_pred_proba_train >= 0.5).astype(int))
    
    # Test metrics (if test set available)
    if X_test is not None and y_test is not None:
        y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]
        y_test_binary = (y_test >= 0.5).astype(int)
        
        try:
            test_auc = roc_auc_score(y_test_binary, y_pred_proba_test)
        except ValueError:
            test_auc = 0.5
        
        test_brier = brier_score_loss(y_test_binary, y_pred_proba_test)
        test_logloss = log_loss(y_test_binary, y_pred_proba_test)
        test_acc = accuracy_score(y_test_binary, (y_pred_proba_test >= 0.5).astype(int))
        
        # Calibration error
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred_proba_test, bin_edges[1:])
        
        calibration_errors = []
        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                pred_mean = y_pred_proba_test[mask].mean()
                obs_mean = y_test_binary[mask].mean()
                calibration_errors.append(abs(pred_mean - obs_mean))
        
        calibration_error = np.mean(calibration_errors) if calibration_errors else 0.0
        
        metrics = {
            'train_auc': float(train_auc),
            'test_auc': float(test_auc),
            'train_brier': float(train_brier),
            'test_brier': float(test_brier),
            'train_logloss': float(train_logloss),
            'test_logloss': float(test_logloss),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'calibration_error': float(calibration_error),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': len(feature_names)
        }
    else:
        # No test set - only training metrics
        metrics = {
            'train_auc': float(train_auc),
            'test_auc': None,
            'train_brier': float(train_brier),
            'test_brier': None,
            'train_logloss': float(train_logloss),
            'test_logloss': None,
            'train_accuracy': float(train_acc),
            'test_accuracy': None,
            'calibration_error': None,
            'n_train': len(X_train),
            'n_test': 0,
            'n_features': len(feature_names)
        }
    
    # Create model wrapper
    demand_model = DemandModel(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        training_metrics=metrics
    )
    
    return demand_model, metrics


def print_metrics(metrics: Dict):
    """Print formatted metrics."""
    print("=" * 60)
    print("DEMAND MODEL EVALUATION METRICS")
    print("=" * 60)
    print(f"\nTraining Set (n={metrics['n_train']}):")
    print(f"  ROC-AUC:        {metrics['train_auc']:.4f}")
    print(f"  Brier Score:    {metrics['train_brier']:.4f}")
    print(f"  Log Loss:       {metrics['train_logloss']:.4f}")
    print(f"  Accuracy:       {metrics['train_accuracy']:.4f}")
    
    if metrics.get('n_test', 0) > 0 and metrics.get('test_auc') is not None:
        print(f"\nTest Set (n={metrics['n_test']}):")
        print(f"  ROC-AUC:        {metrics['test_auc']:.4f}")
        print(f"  Brier Score:    {metrics['test_brier']:.4f}")
        print(f"  Log Loss:       {metrics['test_logloss']:.4f}")
        print(f"  Accuracy:       {metrics['test_accuracy']:.4f}")
        
        print(f"\nCalibration:")
        if metrics.get('calibration_error') is not None:
            print(f"  Calibration Error: {metrics['calibration_error']:.4f}")
            print(f"    (Lower is better, <0.05 is well-calibrated)")
        
        print(f"\nModel Info:")
        print(f"  Features: {metrics['n_features']}")
        if metrics.get('test_auc') is not None:
            print(f"  Test AUC > 0.65: {'✓' if metrics['test_auc'] > 0.65 else '✗'}")
        if metrics.get('calibration_error') is not None:
            print(f"  Calibration Error < 0.05: {'✓' if metrics['calibration_error'] < 0.05 else '✗'}")
    else:
        print(f"\nModel Info:")
        print(f"  Features: {metrics['n_features']}")
        print(f"  (No test set - training only)")


if __name__ == '__main__':
    # Test fitting
    from data_extractor import extract_sales_data
    from feature_engineer import build_features
    from pathlib import Path
    
    db_path = Path(__file__).parent.parent / 'data_generation' / 'db.sqlite'
    df = extract_sales_data(db_path)
    
    X, y, weights, feature_names = build_features(df)
    event_ids = df['event_id'].values
    
    print(f"Fitting model on {len(X)} samples with {len(feature_names)} features...")
    
    model, metrics = fit_demand_model(
        X, y, weights, feature_names,
        event_ids=event_ids,
        C=1.0,
        max_iter=1000
    )
    
    print_metrics(metrics)
    
    # Show top coefficients
    print("\n" + "=" * 60)
    print("TOP FEATURE COEFFICIENTS (by absolute value)")
    print("=" * 60)
    coefs = model.get_coefficients()
    sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in sorted_coefs[:10]:
        print(f"  {name:30s}: {coef:8.4f}")

