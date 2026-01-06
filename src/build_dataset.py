#!/usr/bin/env python3
"""
Build Dataset Script

Loads raw OFI parquet files from WRDS extraction and builds a modeling dataset with:
- Daily z-score normalization
- Lag features
- Train/val/test splits by date

Usage:
    python build_dataset.py --input data/raw/ --output data/processed/dataset.parquet
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

# Initialize logger
logger = setup_logging("build_dataset")

# Date splits for train/val/test
TRAIN_END = "2023-09-30"
VAL_START = "2023-10-01"
VAL_END = "2023-11-30"
TEST_START = "2023-12-01"


def load_raw_parquet_files(input_dir: Path) -> pd.DataFrame:
    """
    Load all raw OFI parquet files from a directory.
    
    Args:
        input_dir: Directory containing ofi5m_*.parquet files
    
    Returns:
        Combined DataFrame
    """
    parquet_files = sorted(input_dir.glob("ofi5m_*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    dfs = []
    for f in tqdm(parquet_files, desc="Loading files"):
        df = pd.read_parquet(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Ensure proper types
    combined['date'] = pd.to_datetime(combined['date']).dt.date
    combined['symbol'] = combined['symbol'].astype(str)
    combined['bucket'] = combined['bucket'].astype(int)
    combined['ofi'] = combined['ofi'].astype(float)
    
    # Sort by symbol, date, bucket
    combined = combined.sort_values(['symbol', 'date', 'bucket']).reset_index(drop=True)
    
    logger.info(f"Loaded {len(combined)} total rows")
    logger.info(f"Symbols: {combined['symbol'].unique().tolist()}")
    logger.info(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    
    return combined


def add_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert bucket to proper datetime.
    
    Args:
        df: DataFrame with bucket column
    
    Returns:
        DataFrame with datetime column
    """
    # bucket = floor(epoch / 300)
    # datetime = bucket * 300 seconds from epoch
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['bucket'] * 300, unit='s')
    
    # Extract bar time (HH:MM)
    df['bar_time'] = df['datetime'].dt.strftime('%H:%M')
    
    # Create proper datetime with actual date
    df['datetime'] = pd.to_datetime(df['date'].astype(str)) + pd.to_timedelta(
        df['datetime'].dt.hour * 3600 + df['datetime'].dt.minute * 60, unit='s'
    )
    
    return df


def compute_daily_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily z-score normalized OFI.
    
    Args:
        df: DataFrame with ofi column
    
    Returns:
        DataFrame with ofi_z column
    """
    df = df.copy()
    
    # Group by symbol and date to compute daily stats
    def zscore_group(group):
        mean = group['ofi'].mean()
        std = group['ofi'].std()
        if std == 0 or np.isnan(std):
            group['ofi_z'] = 0.0
        else:
            group['ofi_z'] = (group['ofi'] - mean) / std
        return group
    
    df = df.groupby(['symbol', 'date'], group_keys=False).apply(zscore_group)
    
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [1, 2, 3, 6, 12]) -> pd.DataFrame:
    """
    Add lagged OFI features.
    
    Args:
        df: DataFrame with ofi_z column
        lags: List of lag periods
    
    Returns:
        DataFrame with lag columns
    """
    df = df.copy()
    
    # Sort to ensure proper lag computation
    df = df.sort_values(['symbol', 'date', 'bucket']).reset_index(drop=True)
    
    for lag in lags:
        col_name = f'lag{lag}'
        df[col_name] = df.groupby('symbol')['ofi_z'].shift(lag)
    
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add prediction target: next bar's ofi_z.
    
    Args:
        df: DataFrame with ofi_z column
    
    Returns:
        DataFrame with y_next column
    """
    df = df.copy()
    
    # Target is the next bar's ofi_z
    df['y_next'] = df.groupby('symbol')['ofi_z'].shift(-1)
    
    return df


def split_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test by date.
    
    Args:
        df: Full DataFrame
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    train_end = pd.to_datetime(TRAIN_END)
    val_start = pd.to_datetime(VAL_START)
    val_end = pd.to_datetime(VAL_END)
    test_start = pd.to_datetime(TEST_START)
    
    train_df = df[df['date'] <= train_end].copy()
    val_df = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()
    test_df = df[df['date'] >= test_start].copy()
    
    # Convert date back to date type
    for split_df in [train_df, val_df, test_df]:
        split_df['date'] = split_df['date'].dt.date
    
    logger.info(f"Train: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
    logger.info(f"Val: {len(val_df)} rows ({val_df['date'].min()} to {val_df['date'].max()})")
    logger.info(f"Test: {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})")
    
    return train_df, val_df, test_df


def build_dataset(input_dir: Path, output_path: Path) -> pd.DataFrame:
    """
    Build complete dataset from raw parquet files.
    
    Args:
        input_dir: Directory with raw parquet files
        output_path: Path to save processed dataset
    
    Returns:
        Processed DataFrame
    """
    # Load raw data
    logger.info("Loading raw parquet files...")
    df = load_raw_parquet_files(input_dir)
    
    # Add time column
    logger.info("Adding time columns...")
    df = add_time_column(df)
    
    # Compute daily z-score
    logger.info("Computing daily z-score normalization...")
    df = compute_daily_zscore(df)
    
    # Add lag features
    logger.info("Adding lag features...")
    df = add_lag_features(df)
    
    # Add target
    logger.info("Adding prediction target...")
    df = add_target(df)
    
    # Add split column
    logger.info("Adding split labels...")
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
    
    # Summary statistics
    logger.info("\n=== Dataset Summary ===")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"\nSplit distribution:")
    for split in ['train', 'val', 'test']:
        n = len(df[df['split'] == split])
        logger.info(f"  {split}: {n} rows")
    
    logger.info(f"\nSymbol distribution:")
    for sym in df['symbol'].unique():
        n = len(df[df['symbol'] == sym])
        logger.info(f"  {sym}: {n} rows")
    
    # Count valid samples (no NaN in features or target)
    feature_cols = ['ofi_z', 'lag1', 'lag2', 'lag3', 'lag6', 'lag12', 'y_next']
    valid_mask = df[feature_cols].notna().all(axis=1)
    logger.info(f"\nValid samples (no NaN): {valid_mask.sum()}")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"\nSaved processed dataset to {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build modeling dataset from raw OFI parquet files"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Input directory with raw parquet files (default: data/raw)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/dataset.parquet",
        help="Output path for processed dataset (default: data/processed/dataset.parquet)"
    )
    
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't save"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_path = Path(args.output)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if args.stats_only:
        # Just load and show stats
        df = load_raw_parquet_files(input_dir)
        print("\n=== Raw Data Statistics ===")
        print(df.describe())
        print(f"\nSymbols: {df['symbol'].unique().tolist()}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    else:
        build_dataset(input_dir, output_path)


if __name__ == "__main__":
    main()
