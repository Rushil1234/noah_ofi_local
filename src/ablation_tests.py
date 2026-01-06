#!/usr/bin/env python3
"""
Ablation Tests for OFI Prediction

Tests different feature configurations to understand what drives predictability:
1. Full model (ofi_z + all lags)
2. No current ofi_z (lags only) - tests if structure beyond instant persistence
3. Only ofi_z (no lags) - tests if just AR(1)-like
4. AR(1) baseline - pure statistical comparison
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, compute_metrics

logger = setup_logging("ablation")


def load_dataset(path: Path) -> pd.DataFrame:
    logger.info(f"Loading dataset from {path}")
    df = pd.read_parquet(path)
    return df


def prepare_data(df: pd.DataFrame, feature_cols: list, target_col: str = 'y_next'):
    """Prepare X, y for training."""
    valid_cols = feature_cols + [target_col]
    valid_mask = df[valid_cols].notna().all(axis=1)
    df_valid = df[valid_mask].copy()
    
    X = df_valid[feature_cols].values
    y = df_valid[target_col].values
    
    return X, y


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, model_name: str):
    """Train XGBoost and return metrics."""
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
    
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    
    logger.info(f"{model_name}: MSE={metrics['mse']:.4f}, Corr={metrics['correlation']:.4f}, DirAcc={metrics['directional_accuracy']:.4f}")
    
    return metrics, model


def run_ar_baseline(df: pd.DataFrame):
    """Run AR(1) and AR(p) baselines using linear regression."""
    logger.info("Running AR baselines...")
    
    results = {}
    
    for p in [1, 3, 6]:
        # AR(p) features
        feature_cols = [f'lag{i}' for i in range(1, p+1)]
        
        for split in ['train', 'val', 'test']:
            if split == 'train':
                continue  # Don't evaluate on training
        
        # Prepare data
        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']
        
        # Check which lag columns exist
        available_cols = [c for c in feature_cols if c in df.columns]
        if not available_cols:
            continue
        
        X_train, y_train = prepare_data(train_df, available_cols)
        X_test, y_test = prepare_data(test_df, available_cols)
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        
        # Fit linear AR model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        
        results[f'AR({p})'] = metrics
        logger.info(f"AR({p}): MSE={metrics['mse']:.4f}, Corr={metrics['correlation']:.4f}, DirAcc={metrics['directional_accuracy']:.4f}")
    
    return results


def run_autocorrelation_analysis(df: pd.DataFrame):
    """Compute autocorrelation of ofi_z for each symbol."""
    logger.info("\nAutocorrelation Analysis:")
    
    results = {}
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].sort_values(['date', 'bucket'])
        ofi_z = symbol_df['ofi_z'].dropna().values
        
        if len(ofi_z) < 100:
            continue
        
        # Compute autocorrelations at different lags
        acf = {}
        for lag in [1, 2, 3, 6, 12, 24]:
            if len(ofi_z) > lag:
                corr = np.corrcoef(ofi_z[lag:], ofi_z[:-lag])[0, 1]
                acf[f'lag{lag}'] = float(corr)
        
        results[symbol] = acf
        logger.info(f"{symbol} ACF: lag1={acf.get('lag1', 0):.4f}, lag6={acf.get('lag6', 0):.4f}, lag12={acf.get('lag12', 0):.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation tests for OFI prediction")
    parser.add_argument("--input", type=str, default="data/processed/dataset.parquet")
    parser.add_argument("--output", type=str, default="outputs/metrics/ablation_results.json")
    args = parser.parse_args()
    
    df = load_dataset(Path(args.input))
    
    # Split data
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    results = {'timestamp': datetime.now().isoformat(), 'ablations': {}}
    
    # Define ablation configurations
    ablations = {
        'full_model': ['ofi_z', 'lag1', 'lag2', 'lag3', 'lag6', 'lag12'],
        'no_current_ofi': ['lag1', 'lag2', 'lag3', 'lag6', 'lag12'],
        'only_current_ofi': ['ofi_z'],
        'only_lag1': ['lag1'],
        'lags_1_2_3': ['lag1', 'lag2', 'lag3'],
    }
    
    logger.info("\n=== Ablation Tests ===")
    
    for name, features in ablations.items():
        logger.info(f"\nRunning: {name} with features {features}")
        
        X_train, y_train = prepare_data(train_df, features)
        X_val, y_val = prepare_data(val_df, features)
        X_test, y_test = prepare_data(test_df, features)
        
        if len(X_train) == 0:
            logger.warning(f"No valid data for {name}")
            continue
        
        metrics, _ = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, name)
        results['ablations'][name] = {
            'features': features,
            'test_metrics': metrics
        }
    
    # Run AR baselines
    logger.info("\n=== AR Baselines ===")
    ar_results = run_ar_baseline(df)
    results['ar_baselines'] = ar_results
    
    # Autocorrelation analysis
    logger.info("\n=== Autocorrelation Analysis ===")
    acf_results = run_autocorrelation_analysis(df)
    results['autocorrelation'] = acf_results
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")
    
    # Print summary table
    print("\n" + "="*70)
    print("ABLATION TEST RESULTS")
    print("="*70)
    print(f"{'Configuration':<25} {'MSE':>10} {'Corr':>10} {'Dir.Acc':>10}")
    print("-"*70)
    
    for name, data in results['ablations'].items():
        m = data['test_metrics']
        print(f"{name:<25} {m['mse']:>10.4f} {m['correlation']:>10.4f} {m['directional_accuracy']:>10.2%}")
    
    print("-"*70)
    for name, m in results.get('ar_baselines', {}).items():
        print(f"{name:<25} {m['mse']:>10.4f} {m['correlation']:>10.4f} {m['directional_accuracy']:>10.2%}")
    
    print("="*70)


if __name__ == "__main__":
    main()
