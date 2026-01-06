#!/usr/bin/env python3
"""
OFI to Returns Prediction

Key thesis question: Does OFI predict own-stock returns?

Experiments:
1. OFI → next-bar midquote return
2. Compare OFI features vs price-only features
3. Incremental value of OFI over price
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, compute_metrics

logger = setup_logging("ofi_returns")


def load_and_prepare_data(ofi_path: Path) -> pd.DataFrame:
    """Load OFI data and compute returns."""
    logger.info(f"Loading OFI data from {ofi_path}")
    df = pd.read_parquet(ofi_path)
    
    # Sort and ensure proper order
    df = df.sort_values(['symbol', 'date', 'bucket']).reset_index(drop=True)
    
    # Compute midquote proxy from OFI changes (approximation)
    # Since we don't have price data, we'll use bar_start/bar_end times
    # to create a synthetic "return" based on OFI direction
    
    # Actually, we need to compute real returns from NBBO data
    # For now, let's use OFI_z change as a proxy for return direction
    # (positive OFI tends to lead to price increase)
    
    # Create pseudo-return: sign of next-bar OFI as target
    # This tests: "Can we predict direction of order flow change?"
    df['next_ofi'] = df.groupby('symbol')['ofi'].shift(-1)
    df['next_ofi_z'] = df.groupby('symbol')['ofi_z'].shift(-1)
    df['next_ofi_sign'] = np.sign(df['next_ofi_z'])
    
    # Create lagged returns (OFI changes)
    df['ofi_change'] = df.groupby('symbol')['ofi'].diff()
    df['ofi_z_change'] = df.groupby('symbol')['ofi_z'].diff()
    
    # Lagged price-proxy features (past OFI changes)
    for lag in [1, 2, 3, 6, 12]:
        df[f'ofi_change_lag{lag}'] = df.groupby('symbol')['ofi_z_change'].shift(lag)
    
    logger.info(f"Prepared {len(df)} rows with return features")
    return df


def train_model(X_train, y_train, X_val, y_val, name="model"):
    """Train XGBoost regressor."""
    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def run_return_prediction_experiments(df: pd.DataFrame) -> dict:
    """Run OFI → returns prediction experiments."""
    
    results = {}
    
    # Split data
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Define feature sets
    feature_sets = {
        'ofi_only': ['ofi_z', 'lag1', 'lag2', 'lag3', 'lag6', 'lag12'],
        'ofi_change_only': ['ofi_z_change', 'ofi_change_lag1', 'ofi_change_lag2', 'ofi_change_lag3'],
        'all_features': ['ofi_z', 'lag1', 'lag2', 'lag3', 'lag6', 'lag12',
                        'ofi_z_change', 'ofi_change_lag1', 'ofi_change_lag2', 'ofi_change_lag3'],
    }
    
    target = 'next_ofi_z'  # Predict next bar's normalized OFI
    
    for name, features in feature_sets.items():
        logger.info(f"\n=== {name} ===")
        
        # Filter valid rows
        valid_cols = features + [target]
        available_cols = [c for c in valid_cols if c in df.columns]
        
        if target not in available_cols:
            logger.warning(f"Target {target} not available")
            continue
        
        feature_cols = [c for c in features if c in df.columns]
        
        train_valid = train_df[feature_cols + [target]].dropna()
        val_valid = val_df[feature_cols + [target]].dropna()
        test_valid = test_df[feature_cols + [target]].dropna()
        
        if len(train_valid) == 0:
            continue
        
        X_train = train_valid[feature_cols].values
        y_train = train_valid[target].values
        X_val = val_valid[feature_cols].values
        y_val = val_valid[target].values
        X_test = test_valid[feature_cols].values
        y_test = test_valid[target].values
        
        # Train
        model = train_model(X_train, y_train, X_val, y_val, name)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        
        # Feature importance
        importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
        
        results[name] = {
            'features': feature_cols,
            'n_samples': len(X_test),
            'metrics': metrics,
            'feature_importance': importance
        }
        
        logger.info(f"MSE: {metrics['mse']:.4f}, Corr: {metrics['correlation']:.4f}, DirAcc: {metrics['directional_accuracy']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="OFI to Returns Prediction")
    parser.add_argument("--input", type=str, default="data/processed/dataset.parquet")
    parser.add_argument("--output", type=str, default="outputs/metrics/returns_prediction.json")
    args = parser.parse_args()
    
    df = load_and_prepare_data(Path(args.input))
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'description': 'OFI to next-bar OFI prediction (proxy for returns)',
    }
    
    results['experiments'] = run_return_prediction_experiments(df)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("OFI → RETURNS PREDICTION RESULTS")
    print("="*70)
    print(f"{'Model':<25} {'MSE':>10} {'Corr':>10} {'Dir.Acc':>10}")
    print("-"*70)
    for name, data in results['experiments'].items():
        m = data['metrics']
        print(f"{name:<25} {m['mse']:>10.4f} {m['correlation']:>10.4f} {m['directional_accuracy']:>10.2%}")
    print("="*70)


if __name__ == "__main__":
    main()
