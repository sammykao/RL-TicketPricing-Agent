"""
Model validation and cross-validation utilities.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import KFold
from demand_fitter import fit_demand_model, DemandModel


def cross_validate_demand_model(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    feature_names: list,
    event_ids: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
    max_iter: int = 1000
) -> Dict:
    """
    Perform k-fold cross-validation stratified by event.
    
    Args:
        X: Feature matrix
        y: Target probabilities
        sample_weights: Exposure weights
        feature_names: Feature names
        event_ids: Event IDs for stratification
        n_splits: Number of CV folds
        C: Regularization parameter
        max_iter: Max iterations
    
    Returns:
        Dictionary with CV metrics
    """
    unique_events = np.unique(event_ids)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_metrics = {
        'test_auc': [],
        'test_brier': [],
        'test_logloss': [],
        'calibration_error': []
    }
    
    for fold, (train_event_idx, test_event_idx) in enumerate(kf.split(unique_events)):
        train_events = unique_events[train_event_idx]
        test_events = unique_events[test_event_idx]
        
        train_mask = np.isin(event_ids, train_events)
        test_mask = np.isin(event_ids, test_events)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        weights_train = sample_weights[train_mask]
        
        # Fit model on fold (provide test data directly to avoid splitting)
        model, _ = fit_demand_model(
            X_train, y_train, weights_train, feature_names,
            test_size=0.0,  # Don't split further
            C=C,
            max_iter=max_iter,
            stratify_by_event=False,
            X_test=X_test,
            y_test=y_test,
            weights_test=sample_weights[test_mask]
        )
        
        # Evaluate on test fold
        X_test_scaled = model.scaler.transform(X_test)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_test_binary = (y_test >= 0.5).astype(int)
        
        from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
        
        try:
            auc = roc_auc_score(y_test_binary, y_pred_proba)
        except ValueError:
            auc = 0.5
        
        brier = brier_score_loss(y_test_binary, y_pred_proba)
        logloss = log_loss(y_test_binary, y_pred_proba)
        
        # Calibration error
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bin_edges[1:])
        
        cal_errors = []
        for i in range(1, n_splits + 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                pred_mean = y_pred_proba[mask].mean()
                obs_mean = y_test_binary[mask].mean()
                cal_errors.append(abs(pred_mean - obs_mean))
        
        cal_error = np.mean(cal_errors) if cal_errors else 0.0
        
        cv_metrics['test_auc'].append(auc)
        cv_metrics['test_brier'].append(brier)
        cv_metrics['test_logloss'].append(logloss)
        cv_metrics['calibration_error'].append(cal_error)
        
        print(f"Fold {fold + 1}/{n_splits}: AUC={auc:.4f}, Brier={brier:.4f}, CalError={cal_error:.4f}")
    
    # Aggregate metrics
    summary = {
        'mean_test_auc': float(np.mean(cv_metrics['test_auc'])),
        'std_test_auc': float(np.std(cv_metrics['test_auc'])),
        'mean_test_brier': float(np.mean(cv_metrics['test_brier'])),
        'std_test_brier': float(np.std(cv_metrics['test_brier'])),
        'mean_test_logloss': float(np.mean(cv_metrics['test_logloss'])),
        'std_test_logloss': float(np.std(cv_metrics['test_logloss'])),
        'mean_calibration_error': float(np.mean(cv_metrics['calibration_error'])),
        'std_calibration_error': float(np.std(cv_metrics['calibration_error']))
    }
    
    return summary


def validate_model_quality(metrics: Dict) -> Tuple[bool, List[str]]:
    """
    Validate model meets quality thresholds.
    
    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check AUC
    if metrics.get('test_auc', 0) < 0.60:
        warnings.append(f"Test AUC ({metrics['test_auc']:.3f}) is below threshold (0.60)")
    
    # Check calibration
    if metrics.get('calibration_error', 1.0) > 0.10:
        warnings.append(f"Calibration error ({metrics['calibration_error']:.3f}) is high (>0.10)")
    
    # Check Brier score
    if metrics.get('test_brier', 1.0) > 0.30:
        warnings.append(f"Brier score ({metrics['test_brier']:.3f}) is high (>0.30)")
    
    is_valid = len(warnings) == 0
    
    return is_valid, warnings

