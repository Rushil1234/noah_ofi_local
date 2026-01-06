#!/usr/bin/env python3
"""
OFI vs Price Return Prediction Experiment

THE KEY THESIS EXPERIMENT:
"Is OFI a better driver of correlated stock movements than price?"

Predicts MSFT return at t+1 using:
1. Price-only model: MSFT past returns
2. OFI-only model: MSFT OFI + AAPL OFI
3. Price+OFI model: All features combined

Shows incremental value of OFI over price.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, compute_metrics

logger = setup_logging("return_prediction")

# Date splits
TRAIN_END = "2023-09-30"
VAL_START = "2023-10-01"
VAL_END = "2023-11-30"
TEST_START = "2023-12-01"


def load_and_merge_data(ofi_path: Path, returns_path: Path) -> pd.DataFrame:
    """Load OFI and returns data, merge into wide format."""
    
    # Load OFI data
    ofi_df = pd.read_parquet(ofi_path)
    ofi_df['date'] = pd.to_datetime(ofi_df['date']).dt.date
    
    # Load returns data
    returns_df = pd.read_parquet(returns_path)
    returns_df['date'] = pd.to_datetime(returns_df['date']).dt.date
    
    # Pivot OFI to wide format
    aapl_ofi = ofi_df[ofi_df['symbol'] == 'AAPL'][['date', 'bucket', 'ofi', 'ofi_z']].copy()
    msft_ofi = ofi_df[ofi_df['symbol'] == 'MSFT'][['date', 'bucket', 'ofi', 'ofi_z']].copy()
    
    aapl_ofi = aapl_ofi.rename(columns={'ofi': 'aapl_ofi', 'ofi_z': 'aapl_ofi_z'})
    msft_ofi = msft_ofi.rename(columns={'ofi': 'msft_ofi', 'ofi_z': 'msft_ofi_z'})
    
    # Pivot returns to wide format
    aapl_ret = returns_df[returns_df['symbol'] == 'AAPL'][['date', 'bucket', 'bar_return', 'avg_spread']].copy()
    msft_ret = returns_df[returns_df['symbol'] == 'MSFT'][['date', 'bucket', 'bar_return', 'avg_spread']].copy()
    
    aapl_ret = aapl_ret.rename(columns={'bar_return': 'aapl_ret', 'avg_spread': 'aapl_spread'})
    msft_ret = msft_ret.rename(columns={'bar_return': 'msft_ret', 'avg_spread': 'msft_spread'})
    
    # Merge all on date + bucket
    wide = pd.merge(aapl_ofi, msft_ofi, on=['date', 'bucket'], how='inner')
    wide = pd.merge(wide, aapl_ret, on=['date', 'bucket'], how='inner')
    wide = pd.merge(wide, msft_ret, on=['date', 'bucket'], how='inner')
    
    wide = wide.sort_values(['date', 'bucket']).reset_index(drop=True)
    
    logger.info(f"Merged dataset: {len(wide)} rows")
    
    return wide


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all features for prediction."""
    df = df.copy()
    
    # Lag features for returns (price momentum)
    for lag in [1, 2, 3, 6]:
        df[f'aapl_ret_lag{lag}'] = df['aapl_ret'].shift(lag)
        df[f'msft_ret_lag{lag}'] = df['msft_ret'].shift(lag)
    
    # Lag features for OFI
    for lag in [1, 2, 3, 6]:
        df[f'aapl_ofi_lag{lag}'] = df['aapl_ofi_z'].shift(lag)
        df[f'msft_ofi_lag{lag}'] = df['msft_ofi_z'].shift(lag)
    
    # Targets: next-bar returns
    df['aapl_ret_next'] = df['aapl_ret'].shift(-1)
    df['msft_ret_next'] = df['msft_ret'].shift(-1)
    
    # Add split labels
    df['date_dt'] = pd.to_datetime(df['date'])
    train_end = pd.to_datetime(TRAIN_END)
    val_end = pd.to_datetime(VAL_END)
    
    df['split'] = 'unknown'
    df.loc[df['date_dt'] <= train_end, 'split'] = 'train'
    df.loc[(df['date_dt'] > train_end) & (df['date_dt'] <= val_end), 'split'] = 'val'
    df.loc[df['date_dt'] > val_end, 'split'] = 'test'
    
    return df


def run_experiment(df: pd.DataFrame, target_symbol: str = 'MSFT') -> dict:
    """Run the main price vs OFI experiment."""
    
    results = {}
    
    if target_symbol == 'MSFT':
        target = 'msft_ret_next'
        own_ret_lags = [f'msft_ret_lag{i}' for i in [1, 2, 3, 6]]
        cross_ret_lags = [f'aapl_ret_lag{i}' for i in [1, 2, 3, 6]]
        own_ofi = ['msft_ofi_z'] + [f'msft_ofi_lag{i}' for i in [1, 2, 3]]
        cross_ofi = ['aapl_ofi_z'] + [f'aapl_ofi_lag{i}' for i in [1, 2, 3]]
    else:
        target = 'aapl_ret_next'
        own_ret_lags = [f'aapl_ret_lag{i}' for i in [1, 2, 3, 6]]
        cross_ret_lags = [f'msft_ret_lag{i}' for i in [1, 2, 3, 6]]
        own_ofi = ['aapl_ofi_z'] + [f'aapl_ofi_lag{i}' for i in [1, 2, 3]]
        cross_ofi = ['msft_ofi_z'] + [f'msft_ofi_lag{i}' for i in [1, 2, 3]]
    
    # Define model configurations
    models = {
        'price_only_own': own_ret_lags,
        'price_only_both': own_ret_lags + cross_ret_lags,
        'ofi_only_own': own_ofi,
        'ofi_only_both': own_ofi + cross_ofi,
        'price_plus_ofi_own': own_ret_lags + own_ofi,
        'price_plus_ofi_both': own_ret_lags + cross_ret_lags + own_ofi + cross_ofi,
    }
    
    # Split data
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    logger.info(f"\n=== Predicting {target_symbol} returns ===")
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    for model_name, features in models.items():
        # Filter available columns
        available = [c for c in features if c in df.columns]
        if not available:
            continue
        
        valid_cols = available + [target]
        
        train_valid = train_df[valid_cols].dropna()
        val_valid = val_df[valid_cols].dropna()
        test_valid = test_df[valid_cols].dropna()
        
        if len(train_valid) < 100:
            continue
        
        X_train = train_valid[available].values
        y_train = train_valid[target].values
        X_val = val_valid[available].values
        y_val = val_valid[target].values
        X_test = test_valid[available].values
        y_test = test_valid[target].values
        
        # Train XGBoost
        model = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred = model.predict(X_test)
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred)
        
        # Also compute Information Coefficient (rank correlation)
        from scipy.stats import spearmanr
        ic, _ = spearmanr(y_test, y_pred)
        metrics['information_coefficient'] = float(ic if not np.isnan(ic) else 0)
        
        importance = dict(zip(available, model.feature_importances_.tolist()))
        
        results[model_name] = {
            'features': available,
            'n_test': len(X_test),
            'metrics': metrics,
            'feature_importance': importance
        }
        
        logger.info(f"{model_name}: Corr={metrics['correlation']:.4f}, IC={metrics['information_coefficient']:.4f}, DirAcc={metrics['directional_accuracy']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="OFI vs Price Return Prediction")
    parser.add_argument("--ofi", type=str, default="data/processed/dataset.parquet")
    parser.add_argument("--returns", type=str, default="data/raw/returns_2023.parquet")
    parser.add_argument("--output", type=str, default="outputs/metrics/return_prediction.json")
    args = parser.parse_args()
    
    ofi_path = Path(args.ofi)
    returns_path = Path(args.returns)
    
    if not returns_path.exists():
        logger.error(f"Returns file not found: {returns_path}")
        logger.error("Run extract_returns.py first!")
        return
    
    # Load and merge data
    df = load_and_merge_data(ofi_path, returns_path)
    
    # Build features
    df = build_features(df)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'description': 'OFI vs Price for return prediction',
    }
    
    # Run experiments for both targets
    results['MSFT'] = run_experiment(df, 'MSFT')
    results['AAPL'] = run_experiment(df, 'AAPL')
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Print summary table
    print("\n" + "="*90)
    print("OFI vs PRICE FOR RETURN PREDICTION")
    print("="*90)
    
    for symbol in ['MSFT', 'AAPL']:
        print(f"\n### Predicting {symbol} Returns ###")
        print(f"{'Model':<30} {'Corr':>10} {'IC':>10} {'Dir.Acc':>10}")
        print("-"*70)
        
        for name, data in results[symbol].items():
            m = data['metrics']
            print(f"{name:<30} {m['correlation']:>10.4f} {m['information_coefficient']:>10.4f} {m['directional_accuracy']:>10.2%}")
    
    print("\n" + "="*90)
    print("KEY FINDING: Compare 'price_only_both' vs 'price_plus_ofi_both'")
    print("If OFI adds value, 'price_plus_ofi_both' should outperform.")
    print("="*90)


if __name__ == "__main__":
    main()
