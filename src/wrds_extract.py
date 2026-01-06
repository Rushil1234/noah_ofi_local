#!/usr/bin/env python3
"""
WRDS TAQM NBBO OFI Extraction Script

Computes Order Flow Imbalance (OFI) from WRDS TAQM NBBO millisecond data
and aggregates to 5-minute bars.

Usage:
    Plan A (WRDS JupyterHub): Run this script directly on WRDS
    Plan B (Remote): Run from Mac with wrds package and network access

Examples:
    python wrds_extract.py --year 2023 --tickers AAPL,MSFT --output data/raw/
    python wrds_extract.py --start-date 2023-01-03 --end-date 2023-01-10 --tickers AAPL
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    setup_logging,
    get_trading_dates,
    parse_date,
    format_date_for_table,
    is_likely_holiday,
)

# Initialize logger
logger = setup_logging("wrds_extract")


# SQL query template for OFI computation
# Uses LAG window function to compute event-level OFI, then aggregates to 5-min buckets
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


def get_wrds_connection():
    """
    Establish connection to WRDS.
    Uses ~/.pgpass for credentials if available.
    
    Returns:
        wrds.Connection object
    """
    try:
        import wrds
        logger.info("Connecting to WRDS...")
        conn = wrds.Connection()
        logger.info("Connected successfully!")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to WRDS: {e}")
        logger.error("Make sure you have ~/.pgpass configured or run on WRDS JupyterHub")
        raise


def check_table_exists(conn, table_name: str) -> bool:
    """
    Check if a WRDS table exists.
    
    Args:
        conn: WRDS connection
        table_name: Full table name (e.g., 'taqm_2023.nbbom_20230103')
    
    Returns:
        True if table exists
    """
    schema, table = table_name.split('.')
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
    except Exception:
        return False


def extract_ofi_for_day(conn, d: date, ticker: str) -> Optional[pd.DataFrame]:
    """
    Extract OFI data for a single day and ticker.
    
    Args:
        conn: WRDS connection
        d: Trading date
        ticker: Stock symbol (e.g., 'AAPL')
    
    Returns:
        DataFrame with OFI data or None if table doesn't exist
    """
    date_str = format_date_for_table(d)
    table_name = f"taqm_2023.nbbom_{date_str}"
    
    # Check if table exists (handles holidays)
    if not check_table_exists(conn, table_name):
        if is_likely_holiday(d):
            logger.debug(f"Skipping {d} (holiday)")
        else:
            logger.warning(f"Table {table_name} does not exist")
        return None
    
    # Build and execute query
    query = OFI_QUERY_TEMPLATE.format(date_str=date_str, ticker=ticker)
    
    try:
        df = conn.raw_sql(query)
        
        if df.empty:
            logger.warning(f"No data for {ticker} on {d}")
            return None
        
        # Add metadata columns
        df['date'] = d
        df['symbol'] = ticker
        
        # Reorder columns
        df = df[['date', 'symbol', 'bucket', 'ofi', 'n_events', 'bar_start', 'bar_end']]
        
        logger.info(f"Extracted {len(df)} bars for {ticker} on {d}")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting {ticker} on {d}: {e}")
        return None


def extract_ofi_batch(
    conn,
    dates: List[date],
    tickers: List[str],
    output_dir: Path,
    save_individual: bool = True,
) -> pd.DataFrame:
    """
    Extract OFI for multiple days and tickers.
    
    Args:
        conn: WRDS connection
        dates: List of trading dates
        tickers: List of stock symbols
        output_dir: Directory to save parquet files
        save_individual: Whether to save individual day files
    
    Returns:
        Combined DataFrame with all OFI data
    """
    all_data = []
    
    total_tasks = len(dates) * len(tickers)
    progress = tqdm(total=total_tasks, desc="Extracting OFI")
    
    for d in dates:
        for ticker in tickers:
            df = extract_ofi_for_day(conn, d, ticker)
            
            if df is not None:
                all_data.append(df)
                
                # Save individual file
                if save_individual:
                    date_str = d.strftime('%Y-%m-%d')
                    filename = f"ofi5m_{ticker}_{date_str}.parquet"
                    filepath = output_dir / filename
                    df.to_parquet(filepath, index=False)
            
            progress.update(1)
    
    progress.close()
    
    if not all_data:
        logger.warning("No data extracted!")
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def discover_available_dates(conn, year: int) -> List[date]:
    """
    Discover which trading dates have NBBO data available.
    
    Args:
        conn: WRDS connection
        year: Year to check
    
    Returns:
        List of dates with available data
    """
    logger.info(f"Discovering available dates for {year}...")
    
    # Query pg_tables for all nbbom tables in the schema
    query = f"""
    SELECT tablename 
    FROM pg_tables 
    WHERE schemaname = 'taqm_{year}'
    AND tablename LIKE 'nbbom_%'
    ORDER BY tablename
    """
    
    try:
        result = conn.raw_sql(query)
        
        dates = []
        for _, row in result.iterrows():
            table_name = row['tablename']
            # Extract date from table name (nbbom_YYYYMMDD)
            date_str = table_name.replace('nbbom_', '')
            try:
                d = parse_date(date_str)
                dates.append(d)
            except ValueError:
                continue
        
        logger.info(f"Found {len(dates)} available trading dates")
        return sorted(dates)
        
    except Exception as e:
        logger.error(f"Error discovering dates: {e}")
        # Fall back to weekday dates
        logger.info("Falling back to weekday date list")
        return get_trading_dates(year)


def main():
    parser = argparse.ArgumentParser(
        description="Extract OFI from WRDS TAQM NBBO data"
    )
    
    # Date range options
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Year to extract (default: 2023)"
    )
    date_group.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD), required if --start-date is used"
    )
    
    # Ticker options
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL,MSFT",
        help="Comma-separated list of tickers (default: AAPL,MSFT)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for parquet files (default: data/raw)"
    )
    
    parser.add_argument(
        "--combined-output",
        type=str,
        help="Path to save combined parquet file (optional)"
    )
    
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Don't save individual day files"
    )
    
    # Discovery mode
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover available dates, don't extract"
    )
    
    args = parser.parse_args()
    
    # Parse tickers
    tickers = [t.strip().upper() for t in args.tickers.split(',')]
    logger.info(f"Tickers: {tickers}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Connect to WRDS
    conn = get_wrds_connection()
    
    try:
        # Determine date range
        if args.start_date:
            if not args.end_date:
                logger.error("--end-date required when using --start-date")
                sys.exit(1)
            
            start = parse_date(args.start_date)
            end = parse_date(args.end_date)
            
            # Get weekdays in range
            all_dates = get_trading_dates(start.year)
            dates = [d for d in all_dates if start <= d <= end]
        else:
            # Use full year, discover available dates
            dates = discover_available_dates(conn, args.year)
        
        logger.info(f"Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")
        
        if args.discover_only:
            print("\nAvailable trading dates:")
            for d in dates:
                print(f"  {d}")
            print(f"\nTotal: {len(dates)} days")
            return
        
        # Extract OFI
        combined_df = extract_ofi_batch(
            conn,
            dates,
            tickers,
            output_dir,
            save_individual=not args.no_individual
        )
        
        # Save combined file if requested
        if args.combined_output and not combined_df.empty:
            combined_path = Path(args.combined_output)
            combined_path.parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_parquet(combined_path, index=False)
            logger.info(f"Saved combined data to {combined_path}")
        
        # Summary
        logger.info("\n=== Extraction Summary ===")
        logger.info(f"Total rows: {len(combined_df)}")
        if not combined_df.empty:
            logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
            logger.info(f"Tickers: {combined_df['symbol'].unique().tolist()}")
            logger.info(f"Files saved to: {output_dir}")
        
    finally:
        conn.close()
        logger.info("WRDS connection closed")


if __name__ == "__main__":
    main()
