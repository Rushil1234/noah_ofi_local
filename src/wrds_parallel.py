#!/usr/bin/env python3
"""
WRDS TAQM NBBO OFI Extraction Script - Parallel Version

Uses concurrent database connections to maximize throughput.
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
from tqdm import tqdm
import wrds

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    setup_logging,
    get_trading_dates,
    parse_date,
    format_date_for_table,
    is_likely_holiday,
)

logger = setup_logging("wrds_parallel")

# Thread-local storage for connections
thread_local = threading.local()

OFI_QUERY_TEMPLATE = """
WITH nbbo AS (
    SELECT 
        time_m,
        best_bid,
        best_bidsiz,
        best_ask,
        best_asksiz,
        LAG(best_bid) OVER (ORDER BY time_m, time_m_nano) AS prev_bid,
        LAG(best_bidsiz) OVER (ORDER BY time_m, time_m_nano) AS prev_bidsiz,
        LAG(best_ask) OVER (ORDER BY time_m, time_m_nano) AS prev_ask,
        LAG(best_asksiz) OVER (ORDER BY time_m, time_m_nano) AS prev_asksiz
    FROM taqm_2023.nbbom_{date_str}
    WHERE sym_root = '{ticker}'
      AND time_m >= '09:30:00'
      AND time_m < '16:00:00'
),
ofi_events AS (
    SELECT 
        time_m,
        FLOOR(EXTRACT(EPOCH FROM time_m) / 300) AS bucket,
        CASE 
            WHEN best_bid > prev_bid THEN best_bidsiz
            WHEN best_bid < prev_bid THEN -prev_bidsiz
            ELSE best_bidsiz - prev_bidsiz
        END AS e_bid,
        CASE 
            WHEN best_ask < prev_ask THEN -best_asksiz
            WHEN best_ask > prev_ask THEN prev_asksiz
            ELSE -(best_asksiz - prev_asksiz)
        END AS e_ask
    FROM nbbo
    WHERE prev_bid IS NOT NULL
      AND prev_ask IS NOT NULL
)
SELECT 
    bucket,
    SUM(e_bid + e_ask) AS ofi,
    COUNT(*) AS n_events,
    MIN(time_m) AS bar_start,
    MAX(time_m) AS bar_end
FROM ofi_events
GROUP BY bucket
ORDER BY bucket
"""


def get_connection():
    """Get thread-local connection."""
    if not hasattr(thread_local, 'conn'):
        thread_local.conn = wrds.Connection(wrds_username='rkk5541')
    return thread_local.conn


def check_table_exists(conn, table_name: str) -> bool:
    schema, table = table_name.split('.')
    query = f"""
    SELECT EXISTS (
        SELECT FROM pg_tables WHERE schemaname = '{schema}' AND tablename = '{table}'
    )
    """
    try:
        result = conn.raw_sql(query)
        return result.iloc[0, 0]
    except:
        return False


def extract_single_day_ticker(args: Tuple[date, str, Path]) -> Optional[pd.DataFrame]:
    """Extract OFI for a single day and ticker - designed for parallel execution."""
    d, ticker, output_dir = args
    
    try:
        conn = get_connection()
        date_str = format_date_for_table(d)
        table_name = f"taqm_2023.nbbom_{date_str}"
        
        if not check_table_exists(conn, table_name):
            return None
        
        query = OFI_QUERY_TEMPLATE.format(date_str=date_str, ticker=ticker)
        df = conn.raw_sql(query)
        
        if df.empty:
            return None
        
        df['date'] = d
        df['symbol'] = ticker
        df = df[['date', 'symbol', 'bucket', 'ofi', 'n_events', 'bar_start', 'bar_end']]
        
        # Save individual file
        date_str_fmt = d.strftime('%Y-%m-%d')
        filename = f"ofi5m_{ticker}_{date_str_fmt}.parquet"
        filepath = output_dir / filename
        df.to_parquet(filepath, index=False)
        
        return df
        
    except Exception as e:
        logger.error(f"Error {ticker} {d}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Parallel OFI extraction from WRDS")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT")
    parser.add_argument("--output", type=str, default="data/raw")
    parser.add_argument("--combined-output", type=str)
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()
    
    tickers = [t.strip().upper() for t in args.tickers.split(',')]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Parallel extraction with {args.workers} workers")
    logger.info(f"Tickers: {tickers}")
    
    # Initialize first connection to handle auth
    logger.info("Initializing WRDS connection...")
    conn = wrds.Connection(wrds_username='rkk5541')
    conn.close()
    
    # Get trading dates
    dates = get_trading_dates(args.year)
    logger.info(f"Processing {len(dates)} days × {len(tickers)} tickers = {len(dates) * len(tickers)} tasks")
    
    # Build task list
    tasks = [(d, ticker, output_dir) for d in dates for ticker in tickers]
    
    all_data = []
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_single_day_ticker, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="Extracting OFI") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_data.append(result)
                    success_count += 1
                pbar.update(1)
    
    logger.info(f"Extracted {success_count} day/ticker combinations")
    
    # Save combined file
    if all_data and args.combined_output:
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['symbol', 'date', 'bucket']).reset_index(drop=True)
        combined.to_parquet(args.combined_output, index=False)
        logger.info(f"Saved combined file: {args.combined_output} ({len(combined)} rows)")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
