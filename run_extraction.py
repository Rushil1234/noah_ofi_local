#!/usr/bin/env python3
"""
WRDS Extraction Runner - Uses pexpect to handle interactive prompts.
"""
import pexpect
import sys
import os

# Configuration
SCRIPT = "src/wrds_extract.py"
ARGS = "--year 2023 --tickers AAPL,MSFT --output data/raw/ --combined-output data/raw/ofi5m_combined_2023.parquet"
USERNAME = "rkk5541"
PASSWORD = "Mayurikakkad@1582"

def run_extraction():
    cmd = f"python {SCRIPT} {ARGS}"
    print(f"Running: {cmd}")
    print("=" * 60)
    
    # Spawn the process
    child = pexpect.spawn(cmd, timeout=3600, encoding='utf-8')
    child.logfile = sys.stdout
    
    try:
        # Wait for username prompt
        child.expect("Enter your WRDS username.*:", timeout=60)
        child.sendline(USERNAME)
        
        # Wait for password prompt
        child.expect("Enter your password:", timeout=30)
        child.sendline(PASSWORD)
        
        # Let it run
        child.expect(pexpect.EOF, timeout=3600)
        
    except pexpect.TIMEOUT:
        print("\nTimeout waiting for response")
        child.close()
        sys.exit(1)
    except pexpect.EOF:
        pass
    
    child.close()
    print(f"\nProcess exited with code: {child.exitstatus}")
    return child.exitstatus

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
    sys.exit(run_extraction() or 0)
