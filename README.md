# Thread A: OFI Prediction from WRDS TAQM NBBO Data

Predict Order Flow Imbalance (OFI) from microstructure data using WRDS TAQM NBBO millisecond tables. Train XGBoost and LSTM models locally on Mac.

## Overview

This project:
1. **Extracts** OFI from WRDS TAQM NBBO data (computed server-side via SQL)
2. **Aggregates** to 5-minute bars (~78 bars/day per ticker)
3. **Builds** a modeling dataset with z-score normalization and lag features
4. **Trains** XGBoost baseline and LSTM sequence models
5. **Evaluates** with MSE, MAE, correlation, and directional accuracy

**Tickers**: AAPL, MSFT  
**Period**: 2023 (full year)

---

## Quick Start

### 1. Setup Environment

```bash
cd noah_ofi_local

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
make setup
```

### 2. Extract Data from WRDS

⚠️ **Data extraction must be done on WRDS** (either JupyterHub or via `wrds` package with network access).

#### Plan A: WRDS JupyterHub (Recommended)

1. Log into [WRDS JupyterHub](https://wrds-cloud.wharton.upenn.edu/jupyter/)
2. Upload `src/wrds_extract.py` and `src/utils.py` to your WRDS home directory
3. Open a terminal in JupyterHub and run:

```bash
# Install dependencies
pip install --user pandas pyarrow tqdm

# Extract all 2023 data for AAPL and MSFT
python wrds_extract.py --year 2023 --tickers AAPL,MSFT --output ofi_data/

# Or extract a specific date range
python wrds_extract.py --start-date 2023-01-03 --end-date 2023-01-31 --tickers AAPL
```

4. Download the `ofi_data/` folder to your local `data/raw/` directory:
   - Use JupyterHub file browser to download as ZIP
   - Or use `scp` if you have SSH access

#### Plan B: Remote Connection from Mac (if available)

If you have network access to WRDS from your Mac:

```bash
# Set up ~/.pgpass for password-less login (optional)
# Format: wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD

# Run extraction
python src/wrds_extract.py --year 2023 --tickers AAPL,MSFT --output data/raw/
```

### 3. Build Dataset

After downloading parquet files to `data/raw/`:

```bash
make build
```

This creates `data/processed/dataset.parquet` with:
- Daily z-score normalized OFI (`ofi_z`)
- Lag features (`lag1`, `lag2`, `lag3`, `lag6`, `lag12`)
- Target (`y_next` = next bar's `ofi_z`)
- Train/val/test splits by date

### 4. Train Models

```bash
# Train XGBoost
make train-xgb

# Train LSTM
make train-lstm

# Or train both
make train-all
```

### 5. Evaluate

```bash
make eval
```

---

## Project Structure

```
noah_ofi_local/
├── README.md
├── requirements.txt
├── Makefile
├── src/
│   ├── __init__.py
│   ├── utils.py           # Shared utilities
│   ├── wrds_extract.py    # WRDS data extraction
│   ├── build_dataset.py   # Dataset building
│   ├── train_xgb.py       # XGBoost training
│   ├── train_lstm.py      # LSTM training
│   └── eval.py            # Model evaluation
├── data/
│   ├── raw/               # Downloaded parquet files from WRDS
│   └── processed/         # Processed dataset
└── outputs/
    ├── models/            # Saved model files
    └── metrics/           # Evaluation metrics JSON
```

---

## OFI Computation

Order Flow Imbalance captures the net buying/selling pressure from NBBO quote updates.

### Event-Level OFI Formula

For each NBBO update at time $t$:

**Bid Contribution ($e^{bid}_t$):**
| Condition | Value |
|-----------|-------|
| `best_bid[t] > best_bid[t-1]` (price ↑) | `+best_bidsiz[t]` |
| `best_bid[t] < best_bid[t-1]` (price ↓) | `-best_bidsiz[t-1]` |
| `best_bid[t] = best_bid[t-1]` (unchanged) | `+(best_bidsiz[t] - best_bidsiz[t-1])` |

**Ask Contribution ($e^{ask}_t$):**
| Condition | Value |
|-----------|-------|
| `best_ask[t] < best_ask[t-1]` (price ↓) | `-best_asksiz[t]` |
| `best_ask[t] > best_ask[t-1]` (price ↑) | `+best_asksiz[t-1]` |
| `best_ask[t] = best_ask[t-1]` (unchanged) | `-(best_asksiz[t] - best_asksiz[t-1])` |

**Event OFI:** $OFI_t = e^{bid}_t + e^{ask}_t$

**5-Minute Bar OFI:** $OFI_{bar} = \sum_{t \in bar} OFI_t$

### SQL Implementation

The extraction script computes OFI server-side using PostgreSQL window functions:

```sql
WITH nbbo AS (
    SELECT time_m, best_bid, best_bidsiz, best_ask, best_asksiz,
           LAG(best_bid) OVER (ORDER BY time_m, time_m_nano) AS prev_bid,
           LAG(best_bidsiz) OVER (ORDER BY time_m, time_m_nano) AS prev_bidsiz,
           ...
    FROM taqm_2023.nbbom_YYYYMMDD
    WHERE sym_root = 'AAPL' AND time_m BETWEEN '09:30:00' AND '16:00:00'
),
ofi_events AS (
    SELECT FLOOR(EXTRACT(EPOCH FROM time_m) / 300) AS bucket,
           CASE WHEN best_bid > prev_bid THEN best_bidsiz ... END AS e_bid,
           CASE WHEN best_ask < prev_ask THEN -best_asksiz ... END AS e_ask
    FROM nbbo WHERE prev_bid IS NOT NULL
)
SELECT bucket, SUM(e_bid + e_ask) AS ofi FROM ofi_events GROUP BY bucket;
```

This returns only ~78 rows per day per ticker, minimizing data transfer.

---

## Dataset Splits

| Split | Date Range | Description |
|-------|------------|-------------|
| Train | Jan 1 – Sep 30, 2023 | Model fitting |
| Validation | Oct 1 – Nov 30, 2023 | Hyperparameter tuning / early stopping |
| Test | Dec 1 – Dec 31, 2023 | Final evaluation |

---

## Model Details

### XGBoost

- **Type**: Gradient boosted trees (regression)
- **Method**: `tree_method="hist"` (CPU-optimized)
- **Features**: `ofi_z`, `lag1`, `lag2`, `lag3`, `lag6`, `lag12`
- **Target**: `y_next` (next bar's z-scored OFI)

Default hyperparameters:
```python
n_estimators=100, max_depth=4, learning_rate=0.1
```

### LSTM

- **Type**: Sequence-to-one regression
- **Architecture**: LSTM(64, 2 layers) → FC(64) → FC(1)
- **Sequence Length**: 24 bars (2 hours)
- **Device**: Apple MPS if available, else CPU
- **Training**: Adam optimizer, MSE loss, early stopping

Default hyperparameters:
```python
hidden_size=64, num_layers=2, dropout=0.2, lr=0.001, patience=10
```

---

## CLI Reference

### wrds_extract.py

```bash
python src/wrds_extract.py [OPTIONS]

Options:
  --year INT              Year to extract (default: 2023)
  --start-date STR        Start date (YYYY-MM-DD)
  --end-date STR          End date (YYYY-MM-DD)
  --tickers STR           Comma-separated tickers (default: AAPL,MSFT)
  --output STR            Output directory (default: data/raw)
  --combined-output STR   Path to save combined parquet
  --no-individual         Don't save individual day files
  --discover-only         Only list available dates
```

### build_dataset.py

```bash
python src/build_dataset.py [OPTIONS]

Options:
  --input STR       Input directory with raw parquet (default: data/raw)
  --output STR      Output path (default: data/processed/dataset.parquet)
  --stats-only      Only show statistics, don't save
```

### train_xgb.py

```bash
python src/train_xgb.py [OPTIONS]

Options:
  --input STR           Dataset path (default: data/processed/dataset.parquet)
  --model-output STR    Model save path (default: outputs/models/xgb_model.json)
  --metrics-output STR  Metrics save path (default: outputs/metrics/xgb_metrics.json)
  --n-estimators INT    Number of trees (default: 100)
  --max-depth INT       Tree depth (default: 4)
  --learning-rate FLOAT Learning rate (default: 0.1)
```

### train_lstm.py

```bash
python src/train_lstm.py [OPTIONS]

Options:
  --input STR           Dataset path
  --model-output STR    Model save path
  --metrics-output STR  Metrics save path
  --seq-length INT      Sequence length (default: 24)
  --hidden-size INT     LSTM hidden size (default: 64)
  --num-layers INT      LSTM layers (default: 2)
  --dropout FLOAT       Dropout rate (default: 0.2)
  --batch-size INT      Batch size (default: 64)
  --epochs INT          Max epochs (default: 50)
  --lr FLOAT            Learning rate (default: 0.001)
  --patience INT        Early stopping patience (default: 10)
```

### eval.py

```bash
python src/eval.py [OPTIONS]

Options:
  --metrics-dir STR     Directory with metrics JSON files (default: outputs/metrics)
  --xgb-metrics STR     Path to XGBoost metrics JSON
  --lstm-metrics STR    Path to LSTM metrics JSON
  --output STR          Save comparison report to file
```

---

## Expected Outputs

### Model Files
- `outputs/models/xgb_model.json` - XGBoost model (XGBoost native format)
- `outputs/models/lstm_model.pt` - LSTM model (PyTorch state dict)

### Metrics Files
- `outputs/metrics/xgb_metrics.json` - XGBoost evaluation metrics
- `outputs/metrics/lstm_metrics.json` - LSTM evaluation metrics

### Sample Metrics JSON

```json
{
  "model": "xgboost",
  "timestamp": "2024-01-15T10:30:00",
  "params": { ... },
  "test": {
    "mse": 0.85,
    "mae": 0.72,
    "correlation": 0.12,
    "directional_accuracy": 0.52,
    "n_samples": 1200
  }
}
```

---

## Troubleshooting

### WRDS Connection Issues

```
Error: Failed to connect to WRDS
```

- Ensure you have a valid WRDS account
- Set up `~/.pgpass` file for password-less login:
  ```
  wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD
  ```
- Or run the extraction script on WRDS JupyterHub

### Missing Tables (Holidays)

The script handles missing tables (market holidays) gracefully. You'll see:
```
Skipping 2023-01-16 (holiday)
```

### MPS Not Available

If Apple MPS is not detected, the LSTM will train on CPU:
```
Using device: cpu
```

This is expected on older Macs or when PyTorch MPS support is disabled.

### Empty Dataset

If `build_dataset.py` reports no files:
```
FileNotFoundError: No parquet files found in data/raw
```

Ensure you've downloaded the WRDS extraction output to `data/raw/`.

---

## References

- Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*.
- WRDS TAQ Documentation: https://wrds-www.wharton.upenn.edu/pages/get-data/nyse-trade-and-quote-taq/

---

## License

MIT License - see LICENSE file for details.
