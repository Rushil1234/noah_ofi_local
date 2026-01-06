#!/usr/bin/env python3
"""
XGBoost Training Script

Trains an XGBoost regressor to predict next-bar OFI from current features.

Usage:
    python train_xgb.py --input data/processed/dataset.parquet
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, compute_metrics

# Initialize logger
logger = setup_logging("train_xgb")

# Feature columns for training
FEATURE_COLS = ['ofi_z', 'lag1', 'lag2', 'lag3', 'lag6', 'lag12']
TARGET_COL = 'y_next'


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the processed dataset."""
    logger.info(f"Loading dataset from {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix and target vector.
    
    Args:
        df: DataFrame with features and target
    
    Returns:
        Tuple of (X, y) as numpy arrays
    """
    # Drop rows with NaN in features or target
    valid_mask = df[FEATURE_COLS + [TARGET_COL]].notna().all(axis=1)
    df_valid = df[valid_mask].copy()
    
    X = df_valid[FEATURE_COLS].values
    y = df_valid[TARGET_COL].values
    
    logger.info(f"Prepared {len(X)} samples with {len(FEATURE_COLS)} features")
    
    return X, y


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict = None
) -> XGBRegressor:
    """
    Train XGBoost regressor.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        params: XGBoost parameters (optional)
    
    Returns:
        Trained model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'tree_method': 'hist',  # CPU-optimized
            'random_state': 42,
            'n_jobs': -1,
        }
    
    logger.info(f"Training XGBoost with params: {params}")
    
    model = XGBRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    logger.info("Training complete!")
    
    return model


def evaluate_model(model: XGBRegressor, X: np.ndarray, y: np.ndarray, split_name: str) -> dict:
    """
    Evaluate model on a dataset split.
    
    Args:
        model: Trained model
        X, y: Features and targets
        split_name: Name of the split (for logging)
    
    Returns:
        Metrics dictionary
    """
    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred)
    
    logger.info(f"\n=== {split_name} Metrics ===")
    logger.info(f"MSE: {metrics['mse']:.6f}")
    logger.info(f"MAE: {metrics['mae']:.6f}")
    logger.info(f"Correlation: {metrics['correlation']:.4f}")
    logger.info(f"Directional Accuracy: {metrics['directional_accuracy']:.4f}")
    
    return metrics


def get_feature_importance(model: XGBRegressor, feature_names: list) -> dict:
    """Get feature importance from trained model."""
    importance = model.feature_importances_
    return {name: float(imp) for name, imp in zip(feature_names, importance)}


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost regressor for OFI prediction"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/dataset.parquet",
        help="Path to processed dataset (default: data/processed/dataset.parquet)"
    )
    
    parser.add_argument(
        "--model-output",
        type=str,
        default="outputs/models/xgb_model.json",
        help="Path to save model (default: outputs/models/xgb_model.json)"
    )
    
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="outputs/metrics/xgb_metrics.json",
        help="Path to save metrics (default: outputs/metrics/xgb_metrics.json)"
    )
    
    # Hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Load dataset
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Dataset not found: {input_path}")
        sys.exit(1)
    
    df = load_dataset(input_path)
    
    # Split by the split column
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)
    
    # Train model
    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
    }
    
    model = train_xgboost(X_train, y_train, X_val, y_val, params)
    
    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Feature importance
    importance = get_feature_importance(model, FEATURE_COLS)
    logger.info(f"\nFeature Importance: {importance}")
    
    # Save model
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    all_metrics = {
        'model': 'xgboost',
        'timestamp': datetime.now().isoformat(),
        'params': params,
        'feature_importance': importance,
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
    }
    
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("XGBOOST TRAINING COMPLETE")
    print("="*50)
    print(f"Test MSE: {test_metrics['mse']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test Correlation: {test_metrics['correlation']:.4f}")
    print(f"Test Directional Accuracy: {test_metrics['directional_accuracy']:.2%}")
    print("="*50)


if __name__ == "__main__":
    main()
