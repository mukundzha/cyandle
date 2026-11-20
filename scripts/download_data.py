#!/usr/bin/env python3
"""
Download cryptocurrency historical data from Binance.
Supports spot klines with multiple intervals.
"""
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/api/v3"


def download_klines(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download kline data from Binance Vision API."""
    all_klines = []
    
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    
    # Binance limit is 1000 per request
    current = start_ms
    request_count = 0
    max_requests = 200
    
    while current < end_ms and request_count < max_requests:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current,
            'endTime': end_ms,
            'limit': 1000
        }
        
        try:
            response = requests.get(f"{BASE_URL}/klines", params=params, timeout=30)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                break
                
            all_klines.extend(klines)
            current = klines[-1][0] + 1
            
            # Rate limiting
            time.sleep(0.1)
            request_count += 1
            
            if request_count % 50 == 0:
                logger.info(f"Downloaded {len(all_klines)} candlesticks...")
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    if not all_klines:
        logger.warning(f"No data returned for {symbol}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ]
    
    df = pd.DataFrame(all_klines, columns=columns)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    # Convert numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort and remove duplicates
    df = df.sort_values('open_time').reset_index(drop=True)
    df = df.drop_duplicates(subset=['open_time']).reset_index(drop=True)
    
    logger.info(f"Downloaded {len(df)} candlesticks for {symbol}")
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical features to raw data."""
    df = df.copy()
    
    # Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['open_return'] = np.log(df['open'] / df['close'].shift(1))
    df['high_return'] = np.log(df['high'] / df['close'].shift(1))
    df['low_return'] = np.log(df['low'] / df['close'].shift(1))
    
    # Candle features
    df['candle_body'] = df['close'] - df['open']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['candle_range'] = df['high'] - df['low']
    
    # Volume features
    df['log_volume'] = np.log1p(df['volume'])
    df['vol_ma_5'] = df['volume'].rolling(5).mean()
    df['vol_ma_20'] = df['volume'].rolling(20).mean()
    df['rel_volume'] = df['volume'] / (df['vol_ma_20'] + 1e-8)
    
    # Volatility
    df['rv_5'] = df['log_return'].rolling(5).std()
    df['rv_20'] = df['log_return'].rolling(20).std()
    
    # Time features (assuming 5m intervals)
    df['hour'] = df['open_time'].dt.hour
    df['dow'] = df['open_time'].dt.dayofweek
    import math
    df['hour_sin'] = np.sin(2 * math.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * math.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * math.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * math.pi * df['dow'] / 7)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Download crypto data from Binance")
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair')
    parser.add_argument('--interval', default='5m', help='Kline interval')
    parser.add_argument('--days', type=int, default=730, help='Days of historical data')
    parser.add_argument('--output', default=None, help='Output path')
    
    args = parser.parse_args()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    logger.info(f"Downloading {args.symbol} {args.interval} for {args.days} days...")
    logger.info(f"Period: {start_date} to {end_date}")
    
    # Download
    df = download_klines(args.symbol, args.interval, start_date, end_date)
    
    if df.empty:
        logger.error("Failed to download data")
        sys.exit(1)
    
    # Save raw data
    output_path = Path(args.output) if args.output else Path(f"data/raw/{args.symbol}_{args.interval}.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved raw data to {output_path}")
    
    # Prepare features
    logger.info("Preparing features...")
    featured_df = prepare_features(df)
    featured_path = output_path.with_name(f"{args.symbol}_{args.interval}_features.parquet")
    featured_df.to_parquet(featured_path, index=False)
    logger.info(f"Saved featured data to {featured_path}")
    
    print(f"\n✅ Data preparation complete!")
    print(f"  Total candles: {len(df)}")
    print(f"  Date range: {df['open_time'].min()} to {df['open_time'].max()}")
    print(f"  Features: {len(featured_df.columns)}")
    print(f"  Output: {featured_path}")


if __name__ == "__main__":
    import numpy as np
    main()