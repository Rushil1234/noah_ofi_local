#!/usr/bin/env python3
"""
Evaluation Script

Computes and compares metrics across trained models.

Usage:
    python eval.py --metrics-dir outputs/metrics/
    python eval.py --xgb-metrics outputs/metrics/xgb_metrics.json --lstm-metrics outputs/metrics/lstm_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging

# Initialize logger
logger = setup_logging("eval")


def load_metrics(path: Path) -> dict:
    """Load metrics from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def format_metrics_table(metrics_list: List[Dict]) -> str:
    """Format metrics as a comparison table."""
    if not metrics_list:
        return "No metrics to display"
    
    # Extract test metrics
    rows = []
    for m in metrics_list:
        model_name = m.get('model', 'unknown')
        test = m.get('test', {})
        rows.append({
            'Model': model_name.upper(),
            'MSE': test.get('mse', float('nan')),
            'MAE': test.get('mae', float('nan')),
            'Correlation': test.get('correlation', float('nan')),
            'Directional Acc': test.get('directional_accuracy', float('nan')),
            'N Samples': test.get('n_samples', 0),
        })
    
    df = pd.DataFrame(rows)
    
    # Format numeric columns
    df['MSE'] = df['MSE'].apply(lambda x: f"{x:.6f}")
    df['MAE'] = df['MAE'].apply(lambda x: f"{x:.6f}")
    df['Correlation'] = df['Correlation'].apply(lambda x: f"{x:.4f}")
    df['Directional Acc'] = df['Directional Acc'].apply(lambda x: f"{x:.2%}")
    
    return df.to_string(index=False)


def compare_models(metrics_list: List[Dict]) -> Dict:
    """
    Compare models and determine best performer for each metric.
    
    Args:
        metrics_list: List of metrics dictionaries
    
    Returns:
        Dictionary with best model for each metric
    """
    if not metrics_list:
        return {}
    
    comparison = {
        'mse': {'best': None, 'value': float('inf')},
        'mae': {'best': None, 'value': float('inf')},
        'correlation': {'best': None, 'value': float('-inf')},
        'directional_accuracy': {'best': None, 'value': float('-inf')},
    }
    
    for m in metrics_list:
        model_name = m.get('model', 'unknown')
        test = m.get('test', {})
        
        # Lower is better for MSE/MAE
        if test.get('mse', float('inf')) < comparison['mse']['value']:
            comparison['mse']['best'] = model_name
            comparison['mse']['value'] = test['mse']
        
        if test.get('mae', float('inf')) < comparison['mae']['value']:
            comparison['mae']['best'] = model_name
            comparison['mae']['value'] = test['mae']
        
        # Higher is better for correlation/directional accuracy
        if test.get('correlation', float('-inf')) > comparison['correlation']['value']:
            comparison['correlation']['best'] = model_name
            comparison['correlation']['value'] = test['correlation']
        
        if test.get('directional_accuracy', float('-inf')) > comparison['directional_accuracy']['value']:
            comparison['directional_accuracy']['best'] = model_name
            comparison['directional_accuracy']['value'] = test['directional_accuracy']
    
    return comparison


def print_detailed_report(metrics_list: List[Dict]):
    """Print detailed evaluation report."""
    
    print("\n" + "="*70)
    print("OFI PREDICTION MODEL EVALUATION REPORT")
    print("="*70)
    
    # Test metrics comparison
    print("\n### Test Set Performance ###\n")
    print(format_metrics_table(metrics_list))
    
    # Best model analysis
    comparison = compare_models(metrics_list)
    
    print("\n### Best Model by Metric ###\n")
    for metric, info in comparison.items():
        metric_name = metric.replace('_', ' ').title()
        if info['best']:
            print(f"  {metric_name}: {info['best'].upper()} ({info['value']:.4f})")
    
    # Individual model details
    print("\n### Model Details ###\n")
    for m in metrics_list:
        model_name = m.get('model', 'unknown').upper()
        print(f"\n{model_name}:")
        print(f"  Timestamp: {m.get('timestamp', 'N/A')}")
        
        params = m.get('params', {})
        if params:
            print(f"  Parameters:")
            for k, v in params.items():
                if not isinstance(v, dict):
                    print(f"    {k}: {v}")
        
        # Feature importance for XGBoost
        if 'feature_importance' in m:
            print(f"  Feature Importance:")
            imp = m['feature_importance']
            sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
            for feat, val in sorted_imp:
                print(f"    {feat}: {val:.4f}")
        
        # Training history for LSTM
        if 'history' in m:
            hist = m['history']
            if 'train_loss' in hist and hist['train_loss']:
                print(f"  Training:")
                print(f"    Final train loss: {hist['train_loss'][-1]:.6f}")
                print(f"    Final val loss: {hist['val_loss'][-1]:.6f}")
                print(f"    Best epoch: {m.get('params', {}).get('best_epoch', 'N/A')}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare trained models"
    )
    
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="outputs/metrics",
        help="Directory containing metrics JSON files (default: outputs/metrics)"
    )
    
    parser.add_argument(
        "--xgb-metrics",
        type=str,
        help="Path to XGBoost metrics JSON"
    )
    
    parser.add_argument(
        "--lstm-metrics",
        type=str,
        help="Path to LSTM metrics JSON"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save comparison report (optional)"
    )
    
    args = parser.parse_args()
    
    metrics_list = []
    
    # Load from specific files if provided
    if args.xgb_metrics:
        path = Path(args.xgb_metrics)
        if path.exists():
            metrics_list.append(load_metrics(path))
            logger.info(f"Loaded XGB metrics from {path}")
    
    if args.lstm_metrics:
        path = Path(args.lstm_metrics)
        if path.exists():
            metrics_list.append(load_metrics(path))
            logger.info(f"Loaded LSTM metrics from {path}")
    
    # Otherwise, load from directory
    if not metrics_list:
        metrics_dir = Path(args.metrics_dir)
        if not metrics_dir.exists():
            logger.error(f"Metrics directory not found: {metrics_dir}")
            sys.exit(1)
        
        for json_file in metrics_dir.glob("*_metrics.json"):
            metrics_list.append(load_metrics(json_file))
            logger.info(f"Loaded metrics from {json_file}")
    
    if not metrics_list:
        logger.error("No metrics files found!")
        sys.exit(1)
    
    # Print report
    print_detailed_report(metrics_list)
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'models': metrics_list,
            'comparison': compare_models(metrics_list),
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
