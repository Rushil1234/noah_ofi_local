#!/usr/bin/env python3
"""
Cross-Asset OFI Analysis

Key thesis question: Does AAPL OFI predict MSFT returns (and vice versa)?

This tests the "correlated stock movements" hypothesis.

Experiments:
1. AAPL OFI → MSFT next-bar OFI (cross-predictability)
2. MSFT OFI → AAPL next-bar OFI
3. VAR-style: Combined model with both symbols
4. Granger causality tests
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

logger = setup_logging("cross_asset")


def load_and_pivot_data(input_path: Path) -> pd.DataFrame:
    """Load OFI data and pivot to wide format (AAPL/MSFT side by side)."""
    df = pd.read_parquet(input_path)
    df = df.sort_values(['symbol', 'date', 'bucket']).reset_index(drop=True)
    
    # Pivot to wide format
    aapl = df[df['symbol'] == 'AAPL'][['date', 'bucket', 'ofi', 'ofi_z', 'split']].copy()
    msft = df[df['symbol'] == 'MSFT'][['date', 'bucket', 'ofi', 'ofi_z', 'split']].copy()
    
    aapl = aapl.rename(columns={'ofi': 'aapl_ofi', 'ofi_z': 'aapl_ofi_z'})
    msft = msft.rename(columns={'ofi': 'msft_ofi', 'ofi_z': 'msft_ofi_z', 'split': 'split_msft'})
    
    # Merge on date + bucket
    wide = pd.merge(aapl, msft, on=['date', 'bucket'], how='inner')
    wide = wide.sort_values(['date', 'bucket']).reset_index(drop=True)
    
    logger.info(f"Wide format: {len(wide)} rows (matched AAPL-MSFT bars)")
    
    # Create lag features for both
    for lag in [1, 2, 3, 6]:
        wide[f'aapl_lag{lag}'] = wide['aapl_ofi_z'].shift(lag)
        wide[f'msft_lag{lag}'] = wide['msft_ofi_z'].shift(lag)
    
    # Create targets (next bar)
    wide['aapl_next'] = wide['aapl_ofi_z'].shift(-1)
    wide['msft_next'] = wide['msft_ofi_z'].shift(-1)
    
    return wide


def run_cross_prediction(df: pd.DataFrame) -> dict:
    """Run cross-asset prediction experiments."""
    results = {}
    
    # Split data
    train = df[df['split'] == 'train'].copy()
    val = df[df['split'] == 'val'].copy()
    test = df[df['split'] == 'test'].copy()
    
    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    experiments = {
        # Predict MSFT from AAPL
        'aapl_to_msft': {
            'features': ['aapl_ofi_z', 'aapl_lag1', 'aapl_lag2', 'aapl_lag3'],
            'target': 'msft_next'
        },
        # Predict AAPL from MSFT
        'msft_to_aapl': {
            'features': ['msft_ofi_z', 'msft_lag1', 'msft_lag2', 'msft_lag3'],
            'target': 'aapl_next'
        },
        # Predict MSFT from own lags (baseline)
        'msft_own': {
            'features': ['msft_ofi_z', 'msft_lag1', 'msft_lag2', 'msft_lag3'],
            'target': 'msft_next'
        },
        # Predict AAPL from own lags (baseline)
        'aapl_own': {
            'features': ['aapl_ofi_z', 'aapl_lag1', 'aapl_lag2', 'aapl_lag3'],
            'target': 'aapl_next'
        },
        # Combined: MSFT from MSFT + AAPL
        'msft_from_both': {
            'features': ['msft_ofi_z', 'msft_lag1', 'msft_lag2', 'aapl_ofi_z', 'aapl_lag1', 'aapl_lag2'],
            'target': 'msft_next'
        },
        # Combined: AAPL from AAPL + MSFT
        'aapl_from_both': {
            'features': ['aapl_ofi_z', 'aapl_lag1', 'aapl_lag2', 'msft_ofi_z', 'msft_lag1', 'msft_lag2'],
            'target': 'aapl_next'
        },
    }
    
    for name, config in experiments.items():
        logger.info(f"\n=== {name} ===")
        
        features = config['features']
        target = config['target']
        
        # Filter valid
        valid_cols = features + [target]
        available = [c for c in valid_cols if c in df.columns]
        
        if target not in available:
            continue
        
        feature_cols = [c for c in features if c in df.columns]
        
        train_valid = train[feature_cols + [target]].dropna()
        val_valid = val[feature_cols + [target]].dropna()
        test_valid = test[feature_cols + [target]].dropna()
        
        if len(train_valid) < 100:
            continue
        
        X_train = train_valid[feature_cols].values
        y_train = train_valid[target].values
        X_val = val_valid[feature_cols].values
        y_val = val_valid[target].values
        X_test = test_valid[feature_cols].values
        y_test = test_valid[target].values
        
        # Train XGBoost
        model = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        
        importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
        
        results[name] = {
            'features': feature_cols,
            'target': target,
            'n_test': len(X_test),
            'metrics': metrics,
            'feature_importance': importance
        }
        
        logger.info(f"Corr: {metrics['correlation']:.4f}, DirAcc: {metrics['directional_accuracy']:.4f}")
    
    return results


def compute_correlation_matrix(df: pd.DataFrame) -> dict:
    """Compute contemporaneous and lagged correlations between AAPL and MSFT OFI."""
    logger.info("\n=== Correlation Analysis ===")
    
    correlations = {}
    
    # Contemporaneous correlation
    valid = df[['aapl_ofi_z', 'msft_ofi_z']].dropna()
    if len(valid) > 100:
        corr = np.corrcoef(valid['aapl_ofi_z'], valid['msft_ofi_z'])[0, 1]
        correlations['contemporaneous'] = float(corr)
        logger.info(f"Contemporaneous AAPL-MSFT OFI correlation: {corr:.4f}")
    
    # Lead-lag correlations
    for lag in [1, 2, 3, 6]:
        # AAPL leads MSFT
        df_temp = df.copy()
        df_temp['aapl_lagged'] = df_temp['aapl_ofi_z'].shift(lag)
        valid = df_temp[['aapl_lagged', 'msft_ofi_z']].dropna()
        if len(valid) > 100:
            corr = np.corrcoef(valid['aapl_lagged'], valid['msft_ofi_z'])[0, 1]
            correlations[f'aapl_leads_msft_lag{lag}'] = float(corr)
        
        # MSFT leads AAPL
        df_temp['msft_lagged'] = df_temp['msft_ofi_z'].shift(lag)
        valid = df_temp[['msft_lagged', 'aapl_ofi_z']].dropna()
        if len(valid) > 100:
            corr = np.corrcoef(valid['msft_lagged'], valid['aapl_ofi_z'])[0, 1]
            correlations[f'msft_leads_aapl_lag{lag}'] = float(corr)
    
    return correlations


def granger_causality_test(df: pd.DataFrame, max_lag: int = 6) -> dict:
    """Simple Granger causality test between AAPL and MSFT OFI."""
    logger.info("\n=== Granger Causality Tests ===")
    
    results = {}
    
    # Test: Does AAPL Granger-cause MSFT?
    # Compare: MSFT ~ MSFT_lags vs MSFT ~ MSFT_lags + AAPL_lags
    
    for direction in ['aapl_to_msft', 'msft_to_aapl']:
        if direction == 'aapl_to_msft':
            y_col = 'msft_ofi_z'
            own_lags = [f'msft_lag{i}' for i in range(1, max_lag+1) if f'msft_lag{i}' in df.columns]
            other_lags = [f'aapl_lag{i}' for i in range(1, max_lag+1) if f'aapl_lag{i}' in df.columns]
        else:
            y_col = 'aapl_ofi_z'
            own_lags = [f'aapl_lag{i}' for i in range(1, max_lag+1) if f'aapl_lag{i}' in df.columns]
            other_lags = [f'msft_lag{i}' for i in range(1, max_lag+1) if f'msft_lag{i}' in df.columns]
        
        # Restricted model (own lags only)
        restricted_cols = own_lags + [y_col]
        valid_r = df[restricted_cols].dropna()
        X_r = valid_r[own_lags].values
        y_r = valid_r[y_col].values
        
        model_r = LinearRegression()
        model_r.fit(X_r, y_r)
        rss_r = np.sum((y_r - model_r.predict(X_r))**2)
        
        # Unrestricted model (own + other lags)
        unrestricted_cols = own_lags + other_lags + [y_col]
        valid_u = df[unrestricted_cols].dropna()
        X_u = valid_u[own_lags + other_lags].values
        y_u = valid_u[y_col].values
        
        model_u = LinearRegression()
        model_u.fit(X_u, y_u)
        rss_u = np.sum((y_u - model_u.predict(X_u))**2)
        
        # F-test
        n = len(y_u)
        k_r = len(own_lags)
        k_u = len(own_lags + other_lags)
        df1 = k_u - k_r
        df2 = n - k_u
        
        if rss_u > 0 and df2 > 0:
            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
            p_value = 1 - scipy_stats.f.cdf(f_stat, df1, df2)
            
            results[direction] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant_at_05': bool(p_value < 0.05),
                'rss_reduction': float((rss_r - rss_u) / rss_r)
            }
            
            sig = "***" if p_value < 0.01 else ("**" if p_value < 0.05 else "")
            logger.info(f"{direction}: F={f_stat:.2f}, p={p_value:.4f} {sig}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-Asset OFI Analysis")
    parser.add_argument("--input", type=str, default="data/processed/dataset.parquet")
    parser.add_argument("--output", type=str, default="outputs/metrics/cross_asset_results.json")
    args = parser.parse_args()
    
    wide = load_and_pivot_data(Path(args.input))
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Cross-asset OFI prediction and causality analysis',
    }
    
    # Run experiments
    results['cross_prediction'] = run_cross_prediction(wide)
    results['correlations'] = compute_correlation_matrix(wide)
    results['granger_causality'] = granger_causality_test(wide)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("CROSS-ASSET OFI ANALYSIS")
    print("="*80)
    
    print("\n### Cross-Prediction Performance ###")
    print(f"{'Experiment':<25} {'Target':<15} {'Corr':>10} {'Dir.Acc':>10}")
    print("-"*80)
    for name, data in results['cross_prediction'].items():
        m = data['metrics']
        print(f"{name:<25} {data['target']:<15} {m['correlation']:>10.4f} {m['directional_accuracy']:>10.2%}")
    
    print("\n### Lead-Lag Correlations ###")
    for k, v in results['correlations'].items():
        print(f"{k}: {v:.4f}")
    
    print("\n### Granger Causality ###")
    for k, v in results['granger_causality'].items():
        sig = "SIGNIFICANT" if v['significant_at_05'] else "not significant"
        print(f"{k}: F={v['f_statistic']:.2f}, p={v['p_value']:.4f} ({sig})")
    
    print("="*80)


if __name__ == "__main__":
    main()
