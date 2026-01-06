#!/usr/bin/env python3
"""
WRDS JupyterHub OFI Extraction Script - Standalone Version

Run this script directly on WRDS JupyterHub to extract OFI data.
No external dependencies beyond standard WRDS JupyterHub environment.

Usage on WRDS JupyterHub:
    1. Upload this file to your WRDS home directory
    2. Open a terminal in JupyterHub
    3. Run: python wrds_jupyterhub_extract.py

Output:
    Creates ofi_data/ directory with parquet files for each day/ticker.
"""

import os
import sys
from datetime import date, datetime
from typing import List, Optional

# These are available on WRDS JupyterHub
import pandas as pd
import wrds

# Configuration
YEAR = 2023
TICKERS = ['AAPL', 'MSFT']
OUTPUT_DIR = 'ofi_data'

# SQL query template for OFI computation
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
    FROM taqm_{year}.nbbom_{date_str}
    WHERE sym_root = '{ticker}'
      AND time_m >= '09:30:00'
      AND time_m < '16:00:00'
),
ofi_events AS (
    SELECT 
        time_m,
        FLOOR(EXTRACT(EPOCH FROM time_m) / 300) AS bucket,
        -- Bid contribution
        CASE 
            WHEN best_bid > prev_bid THEN best_bidsiz
            WHEN best_bid < prev_bid THEN -prev_bidsiz
            ELSE best_bidsiz - prev_bidsiz
        END AS e_bid,
        -- Ask contribution
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


def get_trading_dates(year: int) -> List[date]:
    """Generate weekday dates for a year."""
    dates = []
    current = date(year, 1, 1)
    end = date(year, 12, 31)
    
    while current <= end:
        if current.weekday() < 5:  # Monday-Friday
            dates.append(current)
        current = pd.Timestamp(current) + pd.Timedelta(days=1)
        current = current.date()
    
    return dates


def check_table_exists(conn, schema: str, table: str) -> bool:
    """Check if a table exists in WRDS."""
    query = f"""
    SELECT EXISTS (
        SELECT FROM pg_tables
        WHERE schemaname = '{schema}'
        AND tablename = '{table}'
    )
    """
    try:
        result = conn.raw_sql(query)
        return result.iloc[0, 0]
    except:
        return False


def extract_ofi_for_day(conn, d: date, ticker: str, year: int) -> Optional[pd.DataFrame]:
    """Extract OFI data for a single day and ticker."""
    date_str = d.strftime('%Y%m%d')
    table_name = f"nbbom_{date_str}"
    schema = f"taqm_{year}"
    
    # Check if table exists
    if not check_table_exists(conn, schema, table_name):
        return None
    
    # Build and execute query
    query = OFI_QUERY_TEMPLATE.format(year=year, date_str=date_str, ticker=ticker)
    
    try:
        df = conn.raw_sql(query)
        
        if df.empty:
            return None
        
        # Add metadata columns
        df['date'] = d
        df['symbol'] = ticker
        
        # Reorder columns
        df = df[['date', 'symbol', 'bucket', 'ofi', 'n_events', 'bar_start', 'bar_end']]
        
        return df
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    print("=" * 60)
    print("WRDS TAQM NBBO OFI Extraction")
    print("=" * 60)
    print(f"Year: {YEAR}")
    print(f"Tickers: {TICKERS}")
    print(f"Output: {OUTPUT_DIR}/")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Connect to WRDS
    print("Connecting to WRDS...")
    conn = wrds.Connection()
    print("Connected!\n")
    
    # Get trading dates
    dates = get_trading_dates(YEAR)
    print(f"Processing {len(dates)} potential trading days...\n")
    
    all_data = []
    success_count = 0
    skip_count = 0
    
    try:
        for i, d in enumerate(dates):
            date_str = d.strftime('%Y-%m-%d')
            
            for ticker in TICKERS:
                df = extract_ofi_for_day(conn, d, ticker, YEAR)
                
                if df is not None:
                    all_data.append(df)
                    
                    # Save individual file
                    filename = f"ofi5m_{ticker}_{date_str}.parquet"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    df.to_parquet(filepath, index=False)
                    success_count += 1
                else:
                    skip_count += 1
            
            # Progress update every 10 days
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(dates)} days ({success_count} files saved, {skip_count} skipped)")
        
        print(f"\nDone! {success_count} files saved, {skip_count} skipped")
        
        # Save combined file
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined_path = os.path.join(OUTPUT_DIR, 'ofi5m_combined_2023.parquet')
            combined.to_parquet(combined_path, index=False)
            print(f"\nCombined file saved: {combined_path}")
            print(f"Total rows: {len(combined)}")
            print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
        
    finally:
        conn.close()
        print("\nWRDS connection closed.")
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"\nDownload the '{OUTPUT_DIR}/' folder to your local machine:")
    print("  1. In JupyterHub file browser, right-click the folder")
    print("  2. Select 'Download as archive'")
    print("  3. Extract to your local noah_ofi_local/data/raw/ directory")


if __name__ == "__main__":
    main()
