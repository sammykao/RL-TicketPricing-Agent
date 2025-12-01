"""
Main script to train demand model from SQLite database.

Usage:
    python train_model.py [--db-path PATH] [--output-path PATH] [--cv]
"""

import argparse
from pathlib import Path

from demand_modeling.data_extractor import extract_sales_data, get_data_summary
from demand_modeling.feature_engineer import build_features
from demand_modeling.demand_fitter import fit_demand_model, print_metrics
from demand_modeling.model_validator import cross_validate_demand_model, validate_model_quality
from demand_modeling.model_serializer import save_model


def main():
    parser = argparse.ArgumentParser(description='Train demand probability model')
    parser.add_argument(
        '--db-path',
        type=Path,
        default=Path(__file__).parent.parent / 'data_generation' / 'db.sqlite',
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--output-path',
        type=Path,
        default=Path(__file__).parent.parent / 'models' / 'demand_model_v1.pkl',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--cv',
        action='store_true',
        help='Run cross-validation'
    )
    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='Regularization parameter (inverse of C)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DEMAND MODEL TRAINING")
    print("=" * 60)
    print(f"\nDatabase: {args.db_path}")
    print(f"Output: {args.output_path}")
    
    # Step 1: Extract data
    print("\n[1/4] Extracting sales data from database...")
    df = extract_sales_data(args.db_path)
    summary = get_data_summary(df)
    print(f"  Extracted {summary['n_observations']} observations from {summary['n_events']} events")
    print(f"  Overall sale rate: {summary['overall_sale_rate']:.1%}")
    
    # Step 2: Build features
    print("\n[2/4] Building features...")
    X, y, weights, feature_names = build_features(df, include_interactions=True)
    event_ids = df['event_id'].values
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Number of features: {len(feature_names)}")
    
    # Step 3: Cross-validation (optional)
    if args.cv:
        print("\n[3/4] Running cross-validation...")
        cv_results = cross_validate_demand_model(
            X, y, weights, feature_names, event_ids,
            n_splits=5,
            C=args.C
        )
        print(f"\nCV Results:")
        print(f"  Mean Test AUC: {cv_results['mean_test_auc']:.4f} ± {cv_results['std_test_auc']:.4f}")
        print(f"  Mean Brier Score: {cv_results['mean_test_brier']:.4f} ± {cv_results['std_test_brier']:.4f}")
        print(f"  Mean Calibration Error: {cv_results['mean_calibration_error']:.4f} ± {cv_results['std_calibration_error']:.4f}")
    
    # Step 4: Train final model
    print("\n[3/4] Training final model...")
    model, metrics = fit_demand_model(
        X, y, weights, feature_names,
        event_ids=event_ids,
        C=args.C,
        max_iter=1000
    )
    
    print_metrics(metrics)
    
    # Step 5: Validate model quality
    print("\n[4/4] Validating model quality...")
    is_valid, warnings = validate_model_quality(metrics)
    
    if is_valid:
        print("  ✓ Model passes quality thresholds")
    else:
        print("  ✗ Model has quality issues:")
        for warning in warnings:
            print(f"    - {warning}")
    
    # Step 6: Save model
    print(f"\nSaving model to {args.output_path}...")
    save_model(
        model,
        args.output_path,
        metadata={
            'db_path': str(args.db_path),
            'n_events': summary['n_events'],
            'n_observations': summary['n_observations']
        }
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Show top features
    print("\nTop 10 Feature Coefficients (by absolute value):")
    coefs = model.get_coefficients()
    sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in sorted_coefs[:10]:
        print(f"  {name:30s}: {coef:8.4f}")


if __name__ == '__main__':
    main()

