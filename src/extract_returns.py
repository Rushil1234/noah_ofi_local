#!/usr/bin/env python3
"""
Extract Midquote Returns from WRDS TAQM NBBO

Computes 5-minute bar returns from midquote prices.
Midquote = (best_bid + best_ask) / 2

This creates the actual RETURN data needed to answer:
"Is OFI a better driver of correlated stock movements than price?"
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd
import wrds
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_trading_dates, format_date_for_table

logger = setup_logging("extract_returns")


# SQL query to compute 5-min bar close midquote
MIDQUOTE_QUERY = """
WITH nbbo AS (
    SELECT 
        time_m,
        best_bid,
        best_ask,
        (best_bid + best_ask) / 2.0 AS midquote,
        FLOOR(EXTRACT(EPOCH FROM time_m) / 300) AS bucket
    FROM taqm_2023.nbbom_{date_str}
    WHERE sym_root = '{ticker}'
      AND time_m >= '09:30:00'
      AND time_m < '16:00:00'
      AND best_bid > 0 
      AND best_ask > 0
),
bar_prices AS (
    SELECT 
        bucket,
        FIRST_VALUE(midquote) OVER (PARTITION BY bucket ORDER BY time_m) AS bar_open,
        LAST_VALUE(midquote) OVER (PARTITION BY bucket ORDER BY time_m 
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS bar_close,
        AVG(midquote) AS bar_vwap,
        MIN(midquote) AS bar_low,
        MAX(midquote) AS bar_high,
        COUNT(*) AS n_quotes,
        AVG(best_ask - best_bid) AS avg_spread
    FROM nbbo
    GROUP BY bucket, midquote, time_m
)
SELECT DISTINCT
    bucket,
    FIRST_VALUE(bar_open) OVER (PARTITION BY bucket ORDER BY bucket) AS bar_open,
    LAST_VALUE(bar_close) OVER (PARTITION BY bucket ORDER BY bucket) AS bar_close,
    AVG(bar_vwap) OVER (PARTITION BY bucket) AS bar_vwap,
    MIN(bar_low) OVER (PARTITION BY bucket) AS bar_low,
    MAX(bar_high) OVER (PARTITION BY bucket) AS bar_high,
    SUM(n_quotes) OVER (PARTITION BY bucket) AS n_quotes,
    AVG(avg_spread) OVER (PARTITION BY bucket) AS avg_spread
FROM bar_prices
ORDER BY bucket
"""

# Simpler query that just gets first/last midquote per bucket
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


def extract_returns_for_day(conn, d: date, ticker: str) -> Optional[pd.DataFrame]:
    """Extract midquote returns for a single day and ticker."""
    date_str = format_date_for_table(d)
    
    query = SIMPLE_MIDQUOTE_QUERY.format(date_str=date_str, ticker=ticker)
    
    try:
        df = conn.raw_sql(query)
        
        if df.empty or len(df) < 2:
            return None
        
        # Compute returns
        df = df.sort_values('bucket').reset_index(drop=True)
        df['bar_return'] = df['bar_close'].pct_change()
        df['log_return'] = np.log(df['bar_close'] / df['bar_close'].shift(1))
        
        # Add metadata
        df['date'] = d
        df['symbol'] = ticker
        
        return df
        
    except Exception as e:
        logger.debug(f"Error {ticker} {d}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract midquote returns from WRDS")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT")
    parser.add_argument("--output", type=str, default="data/raw/returns_2023.parquet")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    
    import numpy as np
    global np  # Make numpy available after import check
    
    tickers = [t.strip().upper() for t in args.tickers.split(',')]
    
    logger.info(f"Extracting returns for {tickers}")
    logger.info("Connecting to WRDS...")
    
    conn = wrds.Connection(wrds_username='rkk5541')
    
    dates = get_trading_dates(args.year)
    logger.info(f"Processing {len(dates)} days × {len(tickers)} tickers")
    
    all_data = []
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    thread_local = threading.local()
    
    def get_thread_conn():
        if not hasattr(thread_local, 'conn'):
            thread_local.conn = wrds.Connection(wrds_username='rkk5541')
        return thread_local.conn
    
    def extract_task(args_tuple):
        d, ticker = args_tuple
        conn = get_thread_conn()
        return extract_returns_for_day(conn, d, ticker)
    
    tasks = [(d, ticker) for d in dates for ticker in tickers]
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_task, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Extracting"):
            result = future.result()
            if result is not None:
                all_data.append(result)
    
    conn.close()
    
    if not all_data:
        logger.error("No data extracted!")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['symbol', 'date', 'bucket']).reset_index(drop=True)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    
    logger.info(f"\nExtracted {len(combined)} bars")
    logger.info(f"Saved to {output_path}")
    
    # Stats
    logger.info("\n=== Return Statistics ===")
    for ticker in tickers:
        ticker_data = combined[combined['symbol'] == ticker]
        rets = ticker_data['bar_return'].dropna()
        logger.info(f"{ticker}: n={len(rets)}, mean={rets.mean()*10000:.2f}bps, std={rets.std()*10000:.1f}bps")


if __name__ == "__main__":
    import numpy as np
    main()
