"""
Binance historical data downloader - M0 infrastructure
Downloads spot klines from Binance Vision API
"""
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
from typing import Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/api/v3"


def download_klines(
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime,
    output_path: Path,
    limit: int = 1000
) -> pd.DataFrame:
    """Download klines from Binance Vision API."""
    all_klines = []
    
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    current_start = start_ts
    total_requests = 0
    max_requests = 100
    
    while current_start < end_ts and total_requests < max_requests:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_ts,
            'limit': limit
        }
        
        try:
            response = requests.get(f"{BASE_URL}/klines", params=params, timeout=30)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
                
            all_klines.extend(klines)
            current_start = klines[-1][0] + 1
            time.sleep(0.1)
            total_requests += 1
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    if not all_klines:
        logger.warning(f"No klines returned for {symbol}")
        return pd.DataFrame()
    
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ]
    
    df = pd.DataFrame(all_klines, columns=columns)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                    'quote_asset_volume', 'taker_buy_base_volume', 'taker_buy_quote_volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} klines to {output_path}")
    
    return df


def download_asset_history(
    symbol: str,
    interval: str = "5m",
    raw_dir: Path = Path("cryptoformer/data/raw")
) -> Path:
    """Download complete history for an asset."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = raw_dir / f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.parquet"
    
    if output_path.exists():
        logger.info(f"Data exists at {output_path}, checking freshness")
        df = pd.read_parquet(output_path)
        if len(df) > 0:
            return output_path
    
    logger.info(f"Downloading {symbol} {interval} data...")
    
    # Simple download from earliest to now
    early_date = datetime(2015, 1, 1)
    now = datetime.now()
    
    df = download_klines(symbol, interval, early_date, now, output_path)
    
    if df.empty:
        logger.warning(f"No data downloaded for {symbol}")
    
    return output_path


if __name__ == "__main__":
    output = download_asset_history("BTCUSDT", "5m")
    print(f"Data saved to: {output}")