# Order Flow Imbalance (OFI) Prediction Analysis
## Comprehensive Technical Report

**Author:** Rushil Kakkad  
**Date:** January 6, 2026  
**Repository:** [github.com/Rushil1234/noah_ofi_local](https://github.com/Rushil1234/noah_ofi_local)

---

## Executive Summary

This report documents the complete analysis of Order Flow Imbalance (OFI) predictability using WRDS TAQM NBBO millisecond data for AAPL and MSFT during 2023. The key findings are:

1. **OFI is highly predictable** at the 5-minute horizon (79% directional accuracy)
2. **OFI exhibits AR(1)-like structure** with lag-1 autocorrelation of ~56%
3. **Cross-asset dynamics exist**: AAPL and MSFT OFI are 35% correlated
4. **Granger causality is bidirectional**: MSFT→AAPL is statistically stronger (p=0.0002)
5. **XGBoost significantly outperforms LSTM** for this task

---

## 1. Data Extraction

### 1.1 Data Source
- **Database:** WRDS TAQM (TAQ Millisecond)
- **Tables:** `taqm_2023.nbbom_YYYYMMDD` (NBBO updates)
- **Symbols:** AAPL, MSFT
- **Period:** January 3, 2023 – December 29, 2023
- **Trading Hours:** 09:30–16:00 (Regular Trading Hours)

### 1.2 Extraction Statistics
| Metric | Value |
|--------|-------|
| Total parquet files | 500 |
| Total 5-min bars | 38,998 (per symbol) |
| Combined dataset | 77,996 rows |
| Date range | 250 trading days |
| Bars per day | ~78 (6.5 hours × 12 bars/hour) |

### 1.3 OFI Computation
Order Flow Imbalance is computed at the event level using NBBO quote changes:

**Bid Contribution (e_bid):**
- If bid price ↑: +bid_size_t
- If bid price ↓: -bid_size_{t-1}
- If bid price unchanged: +(bid_size_t - bid_size_{t-1})

**Ask Contribution (e_ask):**
- If ask price ↓: -ask_size_t
- If ask price ↑: +ask_size_{t-1}
- If ask price unchanged: -(ask_size_t - ask_size_{t-1})

**Event OFI:** `ofi_event = e_bid + e_ask`

**5-Minute Bar OFI:** Sum of event OFI within each 5-minute bucket

### 1.4 SQL Implementation
The extraction uses PostgreSQL window functions for efficiency:

```sql
WITH nbbo AS (
    SELECT 
        time_m,
        best_bid, best_bidsiz, best_ask, best_asksiz,
        LAG(best_bid) OVER (ORDER BY time_m) AS prev_bid,
        LAG(best_bidsiz) OVER (ORDER BY time_m) AS prev_bidsiz,
        -- ... similar for ask
        FLOOR(EXTRACT(EPOCH FROM time_m) / 300) AS bucket
    FROM taqm_2023.nbbom_YYYYMMDD
    WHERE sym_root = 'AAPL' AND time_m >= '09:30:00' AND time_m < '16:00:00'
),
ofi_events AS (
    SELECT bucket,
        CASE WHEN best_bid > prev_bid THEN best_bidsiz
             WHEN best_bid < prev_bid THEN -prev_bidsiz
             ELSE best_bidsiz - prev_bidsiz END AS e_bid,
        -- ... similar for e_ask
    FROM nbbo
)
SELECT bucket, SUM(e_bid + e_ask) AS ofi FROM ofi_events GROUP BY bucket
```

---

## 2. Dataset Preparation

### 2.1 Feature Engineering

| Feature | Description |
|---------|-------------|
| `ofi_z` | Z-score normalized OFI (using training set statistics) |
| `lag1` | Previous bar's ofi_z |
| `lag2` | 2 bars ago ofi_z |
| `lag3` | 3 bars ago ofi_z |
| `lag6` | 6 bars ago ofi_z |
| `lag12` | 12 bars ago ofi_z |
| `y_next` | Target: next bar's ofi_z |

### 2.2 Data Splits

| Split | Date Range | Rows | Percentage |
|-------|------------|------|------------|
| Train | Jan–Sep 2023 | 58,340 | 74.8% |
| Validation | Oct–Nov 2023 | 13,416 | 17.2% |
| Test | Dec 2023 | 6,240 | 8.0% |

### 2.3 Normalization (Leak-Free)
To avoid look-ahead bias, normalization uses **training set statistics only**:

```python
# Compute from training period
train_mean = train_df['ofi'].mean()  
train_std = train_df['ofi'].std()

# Apply to all data
df['ofi_z'] = (df['ofi'] - train_mean) / train_std
```

AAPL training statistics: mean=10.93, std=815.07
MSFT training statistics: mean=-10.24, std=475.63

---

## 3. Model Training Results

### 3.1 XGBoost Regressor

**Hyperparameters:**
- n_estimators: 100
- max_depth: 4
- learning_rate: 0.1
- tree_method: hist (CPU-optimized)

**Performance:**

| Split | MSE | MAE | Correlation | Directional Accuracy |
|-------|-----|-----|-------------|---------------------|
| Train | 0.489 | 0.365 | 71.29% | 78.25% |
| Validation | 0.501 | 0.373 | 70.36% | 77.78% |
| **Test** | **0.499** | **0.339** | **70.65%** | **79.16%** |

**Feature Importance:**

| Feature | Importance | Interpretation |
|---------|------------|----------------|
| ofi_z | 62.01% | Current bar OFI is dominant predictor |
| lag1 | 23.27% | Recent momentum matters |
| lag2 | 8.17% | Decaying importance |
| lag3 | 4.02% | Minor contribution |
| lag6 | 1.32% | Near-zero |
| lag12 | 1.20% | Near-zero |

**Key Insight:** Current OFI alone explains 62% of predictive power. The model is largely exploiting short-term persistence.

### 3.2 LSTM Neural Network

**Architecture:**
```
OFILSTMModel(
  (lstm): LSTM(6, 64, num_layers=2, batch_first=True, dropout=0.2)
  (fc): Sequential(
    Linear(64, 32), ReLU(), Dropout(0.2), Linear(32, 1)
  )
)
Total parameters: 53,825
```

**Training:**
- Device: Apple MPS (GPU)
- Epochs: 9 (early stopping at epoch 4)
- Sequence length: 24 bars (2 hours)

**Performance:**

| Metric | Value |
|--------|-------|
| Test MSE | 0.975 |
| Test MAE | 0.603 |
| Test Correlation | 15.26% |
| Test Directional Accuracy | 59.05% |

**Key Insight:** LSTM significantly underperforms XGBoost. The sequential structure adds noise rather than signal—OFI doesn't have complex temporal dependencies beyond lag-1.

### 3.3 Model Comparison

| Model | MSE | Correlation | Dir. Accuracy |
|-------|-----|-------------|---------------|
| **XGBoost** | **0.499** | **70.65%** | **79.16%** |
| LSTM | 0.975 | 15.26% | 59.05% |

**Winner: XGBoost** (simpler is better for this task)

---

## 4. Ablation Tests

### 4.1 Feature Configuration Analysis

| Configuration | Features | MSE | Corr | Dir.Acc |
|--------------|----------|-----|------|---------|
| **Full model** | ofi_z + all lags | **0.499** | **70.65%** | **79.16%** |
| Only ofi_z | Current bar only | 0.659 | 58.19% | 78.95% |
| Lags only | lag1-12 (no current) | 0.963 | 17.66% | 58.96% |
| Only lag1 | Single lag | 0.964 | 17.70% | 59.27% |
| Lags 1-3 | Short-term lags | 0.964 | 17.39% | 59.25% |

### 4.2 AR Baseline Comparison

| Model | MSE | Correlation | Dir. Accuracy |
|-------|-----|-------------|---------------|
| AR(1) | 0.969 | 16.15% | 59.11% |
| AR(3) | 0.969 | 16.14% | 59.68% |
| AR(6) | 0.968 | 16.15% | 59.84% |

**Key Findings:**

1. **Current OFI dominates**: Removing ofi_z drops correlation from 70.6% to 17.7%
2. **XGBoost marginally beats AR**: 17.7% vs 16.2% correlation for lag-only models
3. **Nonlinearity helps slightly**: XGBoost captures minor regime effects
4. **Directional accuracy stable**: ~79% whether using current OFI or full model

---

## 5. Autocorrelation Analysis

### 5.1 Lag-1 Autocorrelation

| Symbol | Lag-1 ACF | Lag-6 ACF | Lag-12 ACF |
|--------|-----------|-----------|------------|
| AAPL | 0.5653 | 0.0195 | -0.0079 |
| MSFT | 0.5697 | 0.0543 | 0.0165 |

### 5.2 Interpretation
OFI exhibits:
- **Strong lag-1 persistence**: ~56% autocorrelation
- **Rapid decay**: Near-zero after lag-1
- **AR(1)-like structure**: Consistent with mean reversion

This explains why current OFI (ofi_z) is the dominant predictor—it captures most of the predictable persistence.

---

## 6. Cross-Asset Analysis

### 6.1 Contemporaneous Correlation
AAPL-MSFT OFI correlation: **35.33%**

This suggests common factor exposure (e.g., market-wide order flow, sector effects).

### 6.2 Lead-Lag Correlations

| Lag | AAPL leads MSFT | MSFT leads AAPL |
|-----|-----------------|-----------------|
| 1 | 27.44% | 28.31% |
| 2 | 19.55% | 21.28% |
| 3 | 11.65% | 14.25% |
| 6 | 3.54% | 4.72% |

**Observation:** Symmetric decay suggests common factor rather than price leadership.

### 6.3 Cross-Prediction Performance

| Experiment | Target | Correlation | Dir. Accuracy |
|-----------|--------|-------------|---------------|
| **AAPL → MSFT** | msft_next | 23.25% | 58.37% |
| **MSFT → AAPL** | aapl_next | 25.91% | 57.88% |
| MSFT own | msft_next | 85.35% | 89.97% |
| AAPL own | aapl_next | 84.04% | 88.16% |

**Key Finding:** Cross-stock prediction has limited power (23-26% correlation) compared to own-stock (85%).

### 6.4 Granger Causality Tests

| Direction | F-statistic | p-value | Significant |
|-----------|-------------|---------|-------------|
| AAPL → MSFT | 2.74 | 0.0272 | Yes (5%) |
| **MSFT → AAPL** | **5.46** | **0.0002** | **Yes (0.1%)** |

**Key Finding:** MSFT OFI Granger-causes AAPL OFI with high significance. This suggests MSFT may lead AAPL in order flow dynamics.

---

## 7. Thesis Implications

### 7.1 For the Paper

**Claim 1: OFI is predictable**
> "OFI exhibits strong short-memory autocorrelation (lag-1 ACF ≈ 0.56), enabling 79% directional accuracy in next-bar prediction using XGBoost regression."

**Claim 2: Structure is AR(1)-like**
> "Feature importance analysis reveals current-bar OFI accounts for 62% of predictive power, with lag features contributing marginally. This AR(1)-like structure limits gains from deep learning approaches."

**Claim 3: Cross-asset dynamics exist**
> "Contemporaneous AAPL-MSFT OFI correlation of 35% suggests common factor exposure. Granger causality tests indicate bidirectional predictability, with MSFT→AAPL being statistically stronger (p=0.0002)."

**Claim 4: Simple models suffice**
> "XGBoost with 6 features outperforms 2-layer LSTM (MSE 0.50 vs 0.98), demonstrating that OFI prediction does not benefit from complex sequence modeling."

### 7.2 Limitations
1. **OFI → OFI, not OFI → returns**: Current analysis predicts order flow, not prices
2. **Two assets only**: Results may not generalize across sectors
3. **Single year**: 2023 market conditions may be specific

### 7.3 Next Steps (Pending WRDS Connection)
1. Extract midquote returns for actual return prediction
2. Compare: Price-only vs OFI-only vs Price+OFI for return prediction
3. Test if OFI adds alpha over price for trading signals

---

## 8. Repository Structure

```
noah_ofi_local/
├── src/
│   ├── utils.py              # Shared utilities
│   ├── wrds_extract.py       # Sequential OFI extraction
│   ├── wrds_parallel.py      # Parallel OFI extraction
│   ├── build_dataset.py      # Original dataset builder
│   ├── build_dataset_v2.py   # Leak-free version
│   ├── train_xgb.py          # XGBoost training
│   ├── train_lstm.py         # LSTM training
│   ├── eval.py               # Model evaluation
│   ├── ablation_tests.py     # Feature ablation
│   ├── cross_asset.py        # Cross-asset analysis
│   ├── ofi_returns.py        # OFI-level prediction
│   ├── extract_returns.py    # Return extraction (parallel)
│   ├── extract_returns_seq.py # Return extraction (sequential)
│   └── return_prediction.py  # Price vs OFI experiment
├── data/
│   ├── raw/                  # 500 parquet files
│   └── processed/            # Processed datasets
├── outputs/
│   ├── models/               # Trained model files
│   └── metrics/              # JSON result files
├── Makefile                  # CLI automation
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

---

## 9. Reproducibility

### 9.1 Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 9.2 Running the Pipeline
```bash
# Extract OFI (requires WRDS credentials)
python src/wrds_parallel.py --year 2023 --tickers AAPL,MSFT

# Build dataset
python src/build_dataset_v2.py

# Train models
python src/train_xgb.py
python src/train_lstm.py

# Run ablation tests
python src/ablation_tests.py

# Run cross-asset analysis
python src/cross_asset.py
```

### 9.3 WRDS Credentials
Username: rkk5541
Connection: wrds-pgdata.wharton.upenn.edu:9737

---

## 10. Appendix: Key Metrics Summary

| Metric | Value | Source |
|--------|-------|--------|
| Total bars extracted | 77,996 | WRDS TAQM |
| XGBoost test correlation | 70.65% | train_xgb.py |
| XGBoost directional accuracy | 79.16% | train_xgb.py |
| LSTM test correlation | 15.26% | train_lstm.py |
| Lag-1 autocorrelation (AAPL) | 56.53% | ablation_tests.py |
| Lag-1 autocorrelation (MSFT) | 56.97% | ablation_tests.py |
| AAPL-MSFT contemporaneous corr | 35.33% | cross_asset.py |
| MSFT→AAPL Granger p-value | 0.0002 | cross_asset.py |
| AAPL→MSFT Granger p-value | 0.0272 | cross_asset.py |

---

*Report generated: January 6, 2026*
