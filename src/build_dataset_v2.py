#!/usr/bin/env python3
"""
Build Dataset Script - Fixed Version (No Leakage)

Key fix: Uses EXPANDING z-score within each day to avoid look-ahead bias.
Only uses data available up to time t to normalize bar t.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging

logger = setup_logging("build_dataset_v2")

# Date splits for train/val/test
TRAIN_END = "2023-09-30"
VAL_START = "2023-10-01"
VAL_END = "2023-11-30"
TEST_START = "2023-12-01"


def load_raw_parquet_files(input_dir: Path) -> pd.DataFrame:
    """Load all raw OFI parquet files from a directory."""
    parquet_files = sorted(input_dir.glob("ofi5m_*.parquet"))
    parquet_files = [f for f in parquet_files if 'combined' not in f.name]
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    dfs = []
    for f in tqdm(parquet_files, desc="Loading files"):
        df = pd.read_parquet(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date']).dt.date
    combined['symbol'] = combined['symbol'].astype(str)
    combined['bucket'] = combined['bucket'].astype(int)
    combined['ofi'] = combined['ofi'].astype(float)
    combined = combined.sort_values(['symbol', 'date', 'bucket']).reset_index(drop=True)
    
    logger.info(f"Loaded {len(combined)} total rows")
    return combined


def add_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert bucket to proper datetime."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['bucket'] * 300, unit='s')
    df['bar_time'] = df['datetime'].dt.strftime('%H:%M')
    df['datetime'] = pd.to_datetime(df['date'].astype(str)) + pd.to_timedelta(
        df['datetime'].dt.hour * 3600 + df['datetime'].dt.minute * 60, unit='s'
    )
    return df


def compute_expanding_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EXPANDING z-score within each day to avoid look-ahead bias.
    
    At bar t, we use mean/std of bars 1...t (within the same day).
    This ensures no future information is used.
    """
    df = df.copy()
    df = df.sort_values(['symbol', 'date', 'bucket']).reset_index(drop=True)
    
    def expanding_zscore_day(group):
        """Apply expanding z-score within a single day."""
        ofi = group['ofi'].values
        n = len(ofi)
        ofi_z = np.zeros(n)
        
        for i in range(n):
            if i == 0:
                # First bar: can't normalize, use 0
                ofi_z[i] = 0.0
            else:
                # Use all bars up to and including current
                past_ofi = ofi[:i+1]
                mean = past_ofi.mean()
                std = past_ofi.std()
                if std == 0 or np.isnan(std):
                    ofi_z[i] = 0.0
                else:
                    ofi_z[i] = (ofi[i] - mean) / std
        
        group['ofi_z'] = ofi_z
        return group
    
    logger.info("Computing expanding z-score (no look-ahead)...")
    df = df.groupby(['symbol', 'date'], group_keys=False).apply(
        expanding_zscore_day, include_groups=False
    )
    
    # Re-add the grouping columns that were excluded
    df = df.reset_index()
    
    return df


def compute_training_set_zscore(df: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """
    Alternative: Normalize using training set statistics only.
    
    Compute mean/std from training period, apply to all data.
    This is the safest approach for train/val/test splits.
    """
    df = df.copy()
    train_end_dt = pd.to_datetime(train_end).date()
    
    # Compute stats from training set only
    train_mask = df['date'] <= train_end_dt
    
    stats = {}
    for symbol in df['symbol'].unique():
        symbol_train = df[(df['symbol'] == symbol) & train_mask]
        stats[symbol] = {
            'mean': symbol_train['ofi'].mean(),
            'std': symbol_train['ofi'].std()
        }
        logger.info(f"{symbol} training stats: mean={stats[symbol]['mean']:.2f}, std={stats[symbol]['std']:.2f}")
    
    # Apply to all data
    def normalize_row(row):
        s = stats[row['symbol']]
        if s['std'] == 0 or np.isnan(s['std']):
            return 0.0
        return (row['ofi'] - s['mean']) / s['std']
    
    df['ofi_z'] = df.apply(normalize_row, axis=1)
    
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [1, 2, 3, 6, 12]) -> pd.DataFrame:
    """Add lagged OFI features."""
    df = df.copy()
    df = df.sort_values(['symbol', 'date', 'bucket']).reset_index(drop=True)
    
    for lag in lags:
        col_name = f'lag{lag}'
        df[col_name] = df.groupby('symbol')['ofi_z'].shift(lag)
    
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add prediction target: next bar's ofi_z."""
    df = df.copy()
    df['y_next'] = df.groupby('symbol')['ofi_z'].shift(-1)
    return df


def add_split_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add train/val/test split labels."""
    df = df.copy()
    df['date_dt'] = pd.to_datetime(df['date'])
    
    train_end = pd.to_datetime(TRAIN_END)
    val_start = pd.to_datetime(VAL_START)
    val_end = pd.to_datetime(VAL_END)
    test_start = pd.to_datetime(TEST_START)
    
    conditions = [
        df['date_dt'] <= train_end,
        (df['date_dt'] >= val_start) & (df['date_dt'] <= val_end),
        df['date_dt'] >= test_start,
    ]
    choices = ['train', 'val', 'test']
    df['split'] = np.select(conditions, choices, default='unknown')
    df = df.drop(columns=['date_dt'])
    
    return df


def build_dataset(input_dir: Path, output_path: Path, normalization: str = 'training_set') -> pd.DataFrame:
    """
    Build complete dataset from raw parquet files.
    
    Args:
        normalization: 'training_set' (recommended) or 'expanding'
    """
    # Load raw data
    logger.info("Loading raw parquet files...")
    df = load_raw_parquet_files(input_dir)
    
    # Add time column
    logger.info("Adding time columns...")
    df = add_time_column(df)
    
    # Compute z-score (leak-free)
    if normalization == 'expanding':
        df = compute_expanding_zscore(df)
    else:
        df = compute_training_set_zscore(df, TRAIN_END)
    
    # Add lag features
    logger.info("Adding lag features...")
    df = add_lag_features(df)
    
    # Add target
    logger.info("Adding prediction target...")
    df = add_target(df)
    
    # Add split labels
    logger.info("Adding split labels...")
    df = add_split_labels(df)
    
    # Summary statistics
    logger.info("\n=== Dataset Summary ===")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Normalization method: {normalization}")
    
    for split in ['train', 'val', 'test']:
        n = len(df[df['split'] == split])
        logger.info(f"  {split}: {n} rows")
    
    # Count valid samples
    feature_cols = ['ofi_z', 'lag1', 'lag2', 'lag3', 'lag6', 'lag12', 'y_next']
    valid_mask = df[feature_cols].notna().all(axis=1)
    logger.info(f"\nValid samples (no NaN): {valid_mask.sum()}")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"\nSaved processed dataset to {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Build dataset with leak-free normalization")
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed/dataset_v2.parquet")
    parser.add_argument(
        "--normalization",
        type=str,
        choices=['training_set', 'expanding'],
        default='training_set',
        help="Normalization method: 'training_set' (recommended) or 'expanding'"
    )
    args = parser.parse_args()
    
    build_dataset(Path(args.input), Path(args.output), args.normalization)


if __name__ == "__main__":
    main()
