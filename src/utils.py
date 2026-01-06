"""
Utility functions for OFI prediction project.
"""

import logging
import sys
from datetime import datetime, date
from typing import List, Optional
import pandas as pd
import numpy as np

# Try to import torch for device detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def setup_logging(name: str = "ofi", level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging with timestamps and return a logger.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    return logger


def get_device() -> str:
    """
    Get the best available device for PyTorch.
    
    Returns:
        'mps' if Apple Silicon GPU is available, else 'cpu'
    """
    if not TORCH_AVAILABLE:
        return "cpu"
    
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_trading_dates(year: int) -> List[date]:
    """
    Generate a list of potential trading dates for a given year.
    Excludes weekends. Actual holidays are handled by catching
    missing table errors.
    
    Args:
        year: The year to generate dates for
    
    Returns:
        List of date objects (weekdays only)
    """
    dates = []
    current = date(year, 1, 1)
    end = date(year, 12, 31)
    
    while current <= end:
        # Exclude weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5:
            dates.append(current)
        current = pd.Timestamp(current) + pd.Timedelta(days=1)
        current = current.date()
    
    return dates


def parse_date(date_str: str) -> date:
    """
    Parse date string in various formats.
    
    Args:
        date_str: Date string (YYYY-MM-DD or YYYYMMDD)
    
    Returns:
        date object
    """
    for fmt in ['%Y-%m-%d', '%Y%m%d']:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


def format_date_for_table(d: date) -> str:
    """
    Format date for WRDS table name (YYYYMMDD).
    
    Args:
        d: date object
    
    Returns:
        String in YYYYMMDD format
    """
    return d.strftime('%Y%m%d')


def bucket_to_time(bucket: int) -> str:
    """
    Convert epoch bucket (floor(epoch/300)) back to HH:MM time string.
    
    Args:
        bucket: Epoch bucket number
    
    Returns:
        Time string in HH:MM format
    """
    epoch_seconds = bucket * 300
    dt = datetime.utcfromtimestamp(epoch_seconds)
    return dt.strftime('%H:%M')


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute evaluation metrics for predictions.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary with MSE, MAE, correlation, directional accuracy
    """
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {
            'mse': np.nan,
            'mae': np.nan,
            'correlation': np.nan,
            'directional_accuracy': np.nan,
            'n_samples': 0
        }
    
    # MSE
    mse = float(np.mean((y_true - y_pred) ** 2))
    
    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # Pearson correlation
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        correlation = np.nan
    
    # Directional accuracy (sign correctness)
    # Only count where both values are non-zero
    nonzero_mask = (y_true != 0) & (y_pred != 0)
    if np.sum(nonzero_mask) > 0:
        signs_match = np.sign(y_true[nonzero_mask]) == np.sign(y_pred[nonzero_mask])
        directional_accuracy = float(np.mean(signs_match))
    else:
        directional_accuracy = np.nan
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'directional_accuracy': directional_accuracy,
        'n_samples': len(y_true)
    }


def create_sequences(data: np.ndarray, seq_length: int) -> tuple:
    """
    Create sequences for LSTM training.
    
    Args:
        data: 2D array of shape (n_samples, n_features + 1)
              Last column is target, rest are features.
        seq_length: Length of input sequences
    
    Returns:
        Tuple of (X, y) where X is (n_sequences, seq_length, n_features)
        and y is (n_sequences,)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # Features: all columns except last
        X.append(data[i:i + seq_length, :-1])
        # Target: last column of the next row after sequence
        y.append(data[i + seq_length, -1])
    
    return np.array(X), np.array(y)


# US Market holidays 2023 (approximate - used for informational purposes)
US_HOLIDAYS_2023 = [
    date(2023, 1, 2),   # New Year's Day (observed)
    date(2023, 1, 16),  # MLK Day
    date(2023, 2, 20),  # Presidents Day
    date(2023, 4, 7),   # Good Friday
    date(2023, 5, 29),  # Memorial Day
    date(2023, 6, 19),  # Juneteenth
    date(2023, 7, 4),   # Independence Day
    date(2023, 9, 4),   # Labor Day
    date(2023, 11, 23), # Thanksgiving
    date(2023, 12, 25), # Christmas
]


def is_likely_holiday(d: date) -> bool:
    """
    Check if a date is likely a US market holiday.
    
    Args:
        d: date to check
    
    Returns:
        True if likely a holiday
    """
    return d in US_HOLIDAYS_2023
