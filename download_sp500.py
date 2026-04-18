#!/usr/bin/env python
"""
Download all S&P 500 constituent stock data (2000–2026) from Yahoo Finance
and save each as a CSV in the datas/ folder.

Usage:
    python download_sp500.py
    python download_sp500.py --workers 8
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
DATAS_DIR = os.path.join(HERE, 'datas')


def get_sp500_tickers() -> list[str]:
    """Return current S&P 500 tickers (as of April 2026).
    Tickers use Yahoo Finance format (dots replaced with dashes).
    """
    tickers = [
        'A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI',
        'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ',
        'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD',
        'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS',
        'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'ATVI', 'AVB', 'AVGO',
        'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BAX', 'BBWI', 'BBY',
        'BDX', 'BEN', 'BF-B', 'BG', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR',
        'BLDR', 'BLK', 'BMY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BWA', 'BX',
        'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE',
        'CCI', 'CCL', 'CDAY', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CFG',
        'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA',
        'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP',
        'COR', 'COST', 'CPAY', 'CPB', 'CPRT', 'CPT', 'CRM', 'CSCO', 'CSGP',
        'CSX', 'CTAS', 'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR',
        'D', 'DAL', 'DAY', 'DD', 'DE', 'DECK', 'DELL', 'DFS', 'DG', 'DGX',
        'DHI', 'DHR', 'DIS', 'DLTR', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE',
        'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX',
        'EIX', 'EL', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR',
        'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY', 'EVRG', 'EW', 'EXC',
        'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCNCA', 'FCX', 'FDS',
        'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FISV', 'FITB', 'FLT',
        'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GDDY',
        'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM',
        'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW',
        'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HOLX', 'HON', 'HPE', 'HPQ',
        'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE',
        'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'INVH',
        'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ',
        'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K',
        'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX',
        'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN',
        'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS',
        'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP',
        'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC',
        'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC',
        'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB',
        'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX',
        'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE',
        'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC',
        'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PARA', 'PAYC', 'PAYX', 'PCAR',
        'PCG', 'PEAK', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH',
        'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD',
        'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PVH', 'PWR',
        'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RJF',
        'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY',
        'SBAC', 'SBUX', 'SCHW', 'SEE', 'SHW', 'SIVB', 'SJM', 'SLB', 'SMCI',
        'SNA', 'SNPS', 'SO', 'SOLV', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD',
        'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T',
        'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT',
        'TJX', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO',
        'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER',
        'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V',
        'VICI', 'VLO', 'VLTO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VST', 'VTR',
        'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL',
        'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WRK', 'WST', 'WTW', 'WY',
        'WYNN', 'XEL', 'XOM', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS',
    ]
    return sorted(tickers)


def download_ticker(ticker: str) -> str:
    """Download one ticker and save to CSV. Returns status message."""
    filename = f'{ticker.lower()}-2000-2026.csv'
    csv_path = os.path.join(DATAS_DIR, filename)

    if os.path.exists(csv_path):
        return f'{ticker:<6} SKIP (already exists)'

    try:
        df = yf.download(ticker, start='2000-01-01', end='2026-12-31',
                         auto_adjust=True, progress=False)
        if df.empty:
            return f'{ticker:<6} EMPTY (no data returned)'

        # Flatten MultiIndex columns if present (yfinance >= 0.2)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Keep standard OHLCV columns in the same order as existing files
        cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        for c in cols:
            if c not in df.columns:
                return f'{ticker:<6} ERROR (missing column {c})'
        df = df[cols]
        df.index.name = 'Date'
        df.to_csv(csv_path)
        return f'{ticker:<6} OK    ({len(df)} rows → {filename})'
    except Exception as exc:
        return f'{ticker:<6} ERROR ({exc})'


def main():
    parser = argparse.ArgumentParser(description='Download S&P 500 data')
    parser.add_argument('--workers', type=int, default=5,
                        help='Parallel download threads (default: 5)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        dest='skip_existing',
                        help='Skip tickers that already have a CSV (default)')
    args = parser.parse_args()

    os.makedirs(DATAS_DIR, exist_ok=True)

    print('Fetching S&P 500 ticker list from Wikipedia …')
    tickers = get_sp500_tickers()
    print(f'Found {len(tickers)} tickers\n')

    t0 = time.perf_counter()
    ok = 0
    skip = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_ticker, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            msg = fut.result()
            print(f'  [{i:>3}/{len(tickers)}] {msg}')
            if 'OK' in msg:
                ok += 1
            elif 'SKIP' in msg:
                skip += 1
            else:
                fail += 1

    elapsed = time.perf_counter() - t0
    print(f'\nDone in {elapsed:.1f}s')
    print(f'  Downloaded: {ok}  Skipped: {skip}  Failed: {fail}')
    print(f'  Data dir: {os.path.abspath(DATAS_DIR)}')


if __name__ == '__main__':
    main()
