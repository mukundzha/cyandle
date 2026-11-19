"""
Feature engineering for cryptocurrency time series - M2
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineers features for cryptocurrency price prediction."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df = self.df.sort_values('open_time').reset_index(drop=True)
    
    def compute_returns(self) -> pd.DataFrame:
        """Compute various return features."""
        df = self.df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['open_return'] = np.log(df['open'] / df['close'].shift(1))
        df['high_return'] = np.log(df['high'] / df['close'].shift(1))
        df['low_return'] = np.log(df['low'] / df['close'].shift(1))
        return df
    
    def compute_candle_features(self) -> pd.DataFrame:
        """Compute candle-specific features."""
        df = self.df.copy()
        df['candle_body'] = df['close'] - df['open']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']
        return df
    
    def compute_volume_features(self) -> pd.DataFrame:
        """Compute volume-related features."""
        df = self.df.copy()
        df['log_volume'] = np.log1p(df['volume'])
        df['vol_ma_5'] = df['volume'].rolling(5).mean()
        df['vol_ma_20'] = df['volume'].rolling(20).mean()
        df['rel_volume'] = df['volume'] / (df['vol_ma_20'] + 1e-8)
        return df
    
    def compute_volatility_features(self) -> pd.DataFrame:
        """Compute volatility features."""
        df = self.df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['rv_5'] = df['log_return'].rolling(5).std()
        df['rv_20'] = df['log_return'].rolling(20).std()
        df['rv_50'] = df['log_return'].rolling(50).std()
        return df
    
    def compute_temporal_features(self) -> pd.DataFrame:
        """Compute temporal features."""
        df = self.df.copy()
        df['hour'] = df['open_time'].dt.hour
        df['day_of_week'] = df['open_time'].dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        return df
    
    def add_all_features(self) -> pd.DataFrame:
        """Add all engineered features."""
        logger.info("Starting feature engineering...")
        
        df = self.compute_returns()
        df = self.compute_candle_features(df)
        df = self.compute_volume_features(df)
        df = self.compute_volatility_features(df)
        df = self.compute_temporal_features()
        
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped = initial_len - len(df)
        
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with NaN values")
        
        logger.info(f"Final features: {len(df.columns)} columns, {len(df)} rows")
        return df
    
    def compute_returns(self) -> pd.DataFrame:
        """Compute various return features."""
        df = self.df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['open_return'] = np.log(df['open'] / df['close'].shift(1))
        df['high_return'] = np.log(df['high'] / df['close'].shift(1))
        df['low_return'] = np.log(df['low'] / df['close'].shift(1))
        return df
    
    def compute_candle_features(self) -> pd.DataFrame:
        """Compute candle-specific features."""
        df = self.df.copy()
        df['candle_body'] = df['close'] - df['open']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']
        return df


def prepare_features(
    input_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Load raw data, engineer features, and save processed data."""
    df = pd.read_parquet(input_path)
    
    engineer = FeatureEngineer(df)
    featured_df = engineer.add_all_features()
    
    featured_df.to_parquet(output_path, index=False)
    logger.info(f"Saved processed features to {output_path}")
    
    return featured_df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) >= 3:
        prepare_features(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python features.py <input.parquet> <output.parquet>")