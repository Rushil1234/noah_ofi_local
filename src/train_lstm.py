#!/usr/bin/env python3
"""
LSTM Training Script

Trains an LSTM sequence model to predict next-bar OFI.

Usage:
    python train_lstm.py --input data/processed/dataset.parquet
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, compute_metrics, get_device, create_sequences

# Initialize logger
logger = setup_logging("train_lstm")

# Feature columns for training
FEATURE_COLS = ['ofi_z', 'lag1', 'lag2', 'lag3', 'lag6', 'lag12']
TARGET_COL = 'y_next'

# Default hyperparameters
DEFAULT_SEQ_LENGTH = 24  # 24 bars = 2 hours
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.2
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
DEFAULT_LR = 0.001
DEFAULT_PATIENCE = 10


class OFILSTMModel(nn.Module):
    """LSTM model for OFI prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_hidden = lstm_out[:, -1, :]
        # Predict
        output = self.fc(last_hidden)
        return output.squeeze(-1)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the processed dataset."""
    logger.info(f"Loading dataset from {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows")
    return df


def prepare_sequences_by_symbol(
    df: pd.DataFrame,
    seq_length: int,
    split: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for a specific split, respecting symbol boundaries.
    
    Args:
        df: Full DataFrame
        seq_length: Sequence length
        split: 'train', 'val', or 'test'
    
    Returns:
        Tuple of (X, y) arrays
    """
    split_df = df[df['split'] == split].copy()
    
    # Drop rows with NaN in features or target
    valid_mask = split_df[FEATURE_COLS + [TARGET_COL]].notna().all(axis=1)
    split_df = split_df[valid_mask]
    
    all_X = []
    all_y = []
    
    for symbol in split_df['symbol'].unique():
        symbol_df = split_df[split_df['symbol'] == symbol].sort_values(['date', 'bucket'])
        
        # Create feature matrix with target
        features = symbol_df[FEATURE_COLS].values
        targets = symbol_df[TARGET_COL].values
        data = np.column_stack([features, targets])
        
        # Create sequences
        if len(data) > seq_length:
            X, y = create_sequences(data, seq_length)
            all_X.append(X)
            all_y.append(y)
    
    if not all_X:
        return np.array([]), np.array([])
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    logger.info(f"{split}: {len(X)} sequences of length {seq_length}")
    
    return X, y


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    device: str,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders."""
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(X_batch)
    
    return total_loss / len(loader.dataset)


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            total_loss += loss.item() * len(X_batch)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    return avg_loss, preds, targets


def train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
) -> dict:
    """
    Train LSTM model with early stopping.
    
    Returns:
        Training history dict
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 2
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
    }
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Epochs: {epochs}, LR: {lr}, Patience: {patience}")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            history['best_epoch'] = epoch + 1
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""
        
        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}{marker}"
        )
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model from epoch {history['best_epoch']}")
    
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM model for OFI prediction"
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
        default="outputs/models/lstm_model.pt",
        help="Path to save model (default: outputs/models/lstm_model.pt)"
    )
    
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="outputs/metrics/lstm_metrics.json",
        help="Path to save metrics (default: outputs/metrics/lstm_metrics.json)"
    )
    
    # Hyperparameters
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load dataset
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Dataset not found: {input_path}")
        sys.exit(1)
    
    df = load_dataset(input_path)
    
    # Prepare sequences
    logger.info("Preparing sequences...")
    X_train, y_train = prepare_sequences_by_symbol(df, args.seq_length, 'train')
    X_val, y_val = prepare_sequences_by_symbol(df, args.seq_length, 'val')
    X_test, y_test = prepare_sequences_by_symbol(df, args.seq_length, 'test')
    
    if len(X_train) == 0:
        logger.error("No training sequences created. Check your data.")
        sys.exit(1)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, args.batch_size, device
    )
    
    # Create model
    input_size = X_train.shape[2]  # Number of features
    model = OFILSTMModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    
    logger.info(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Train
    history = train_lstm(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, patience=args.patience
    )
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = model.to(device)
    criterion = nn.MSELoss()
    test_loss, y_pred, y_true = evaluate_epoch(model, test_loader, criterion, device)
    
    test_metrics = compute_metrics(y_true, y_pred)
    
    logger.info("\n=== Test Metrics ===")
    logger.info(f"MSE: {test_metrics['mse']:.6f}")
    logger.info(f"MAE: {test_metrics['mae']:.6f}")
    logger.info(f"Correlation: {test_metrics['correlation']:.4f}")
    logger.info(f"Directional Accuracy: {test_metrics['directional_accuracy']:.4f}")
    
    # Save model
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'seq_length': args.seq_length,
    }, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    all_metrics = {
        'model': 'lstm',
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'params': {
            'seq_length': args.seq_length,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epochs_trained': len(history['train_loss']),
            'best_epoch': history['best_epoch'],
            'total_params': total_params,
        },
        'history': {
            'train_loss': [float(l) for l in history['train_loss']],
            'val_loss': [float(l) for l in history['val_loss']],
        },
        'test': test_metrics,
    }
    
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("LSTM TRAINING COMPLETE")
    print("="*50)
    print(f"Device: {device}")
    print(f"Best Epoch: {history['best_epoch']}")
    print(f"Test MSE: {test_metrics['mse']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test Correlation: {test_metrics['correlation']:.4f}")
    print(f"Test Directional Accuracy: {test_metrics['directional_accuracy']:.2%}")
    print("="*50)


if __name__ == "__main__":
    main()
