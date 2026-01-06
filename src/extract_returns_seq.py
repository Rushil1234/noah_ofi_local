#!/usr/bin/env python3
"""
Extract Returns - Sequential Version (avoids connection limits)
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import wrds
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging, get_trading_dates, format_date_for_table

logger = setup_logging("extract_returns_seq")

SIMPLE_MIDQUOTE_QUERY = """
WITH nbbo AS (
    SELECT 
        time_m,
        (best_bid + best_ask) / 2.0 AS midquote,
        best_ask - best_bid AS spread,
        FLOOR(EXTRACT(EPOCH FROM time_m) / 300) AS bucket
    FROM taqm_2023.nbbom_{date_str}
    WHERE sym_root = '{ticker}'
      AND time_m >= '09:30:00'
      AND time_m < '16:00:00'
      AND best_bid > 0 
      AND best_ask > 0
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY bucket ORDER BY time_m ASC) AS rn_first,
        ROW_NUMBER() OVER (PARTITION BY bucket ORDER BY time_m DESC) AS rn_last
    FROM nbbo
)
SELECT 
    bucket,
    MAX(CASE WHEN rn_first = 1 THEN midquote END) AS bar_open,
    MAX(CASE WHEN rn_last = 1 THEN midquote END) AS bar_close,
    AVG(midquote) AS bar_vwap,
    AVG(spread) AS avg_spread,
    COUNT(*) AS n_quotes
FROM ranked
GROUP BY bucket
ORDER BY bucket
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT")
    parser.add_argument("--output", type=str, default="data/raw/returns_2023.parquet")
    args = parser.parse_args()
    
    tickers = [t.strip().upper() for t in args.tickers.split(',')]
    
    logger.info("Connecting to WRDS (single connection)...")
    conn = wrds.Connection(wrds_username='rkk5541')
    
    dates = get_trading_dates(args.year)
    logger.info(f"Processing {len(dates)} days × {len(tickers)} tickers")
    
    all_data = []
    
    for d in tqdm(dates, desc="Extracting"):
        for ticker in tickers:
            date_str = format_date_for_table(d)
            query = SIMPLE_MIDQUOTE_QUERY.format(date_str=date_str, ticker=ticker)
            
            try:
                df = conn.raw_sql(query)
                if df.empty or len(df) < 2:
                    continue
                    
                df = df.sort_values('bucket').reset_index(drop=True)
                df['bar_return'] = df['bar_close'].pct_change()
                df['date'] = d
                df['symbol'] = ticker
                all_data.append(df)
            except:
                continue
    
    conn.close()
    
    if not all_data:
        logger.error("No data extracted!")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['symbol', 'date', 'bucket']).reset_index(drop=True)
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    
    logger.info(f"\nExtracted {len(combined)} bars, saved to {args.output}")
    for ticker in tickers:
        rets = combined[combined['symbol'] == ticker]['bar_return'].dropna()
        logger.info(f"{ticker}: n={len(rets)}, mean={rets.mean()*10000:.2f}bps, std={rets.std()*10000:.1f}bps")

if __name__ == "__main__":
    main()
