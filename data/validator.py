"""
Data validation and quality checks - M1
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates cryptocurrency market data quality."""
    
    def __init__(self, df: pd.DataFrame, symbol: str):
        self.df = df.copy()
        self.symbol = symbol
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_all(self) -> bool:
        """Run all validation checks."""
        checks = [
            self.check_empty,
            self.check_duplicates,
            self.check_missing_timestamps,
            self.check_ohlc_relationships,
            self.check_negative_prices,
            self.check_volume,
            self.check_ordering,
            self.check_timestamp_gaps,
            self.check_data_types,
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                self.errors.append(f"{check.__name__}: {str(e)}")
        
        if self.errors:
            logger.error(f"Validation failed with {len(self.errors)} errors")
        
        if self.warnings:
            logger.warning(f"Warnings: {len(self.warnings)}")
        
        return len(self.errors) == 0
    
    def check_empty(self) -> None:
        if self.df.empty:
            self.errors.append("DataFrame is empty")
    
    def check_duplicates(self) -> None:
        dup_count = self.df['open_time'].duplicated().sum()
        if dup_count > 0:
            self.errors.append(f"Found {dup_count} duplicate timestamps")
    
    def check_missing_timestamps(self) -> None:
        if len(self.df) < 2:
            return
        timestamps = pd.to_datetime(self.df['open_time'])
        diffs = timestamps.diff().dropna()
        expected_diff = pd.Timedelta(minutes=5)
        gaps = diffs[diffs != expected_diff]
        if len(gaps) > 0:
            gap_pct = len(gaps) / len(diffs) * 100
            if gap_pct > 0.1:
                self.warnings.append(f"Found {len(gaps)} timestamp gaps ({gap_pct:.2f}%)")
    
    def check_ohlc_relationships(self) -> None:
        if 'high' not in self.df.columns or 'low' not in self.df.columns:
            self.errors.append("Missing high or low column")
            return
        invalid = (self.df['high'] < self.df['low']).sum()
        if invalid > 0:
            self.errors.append(f"Found {invalid} candles with high < low")
    
    def check_negative_prices(self) -> None:
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in self.df.columns:
                invalid = (self.df[col] <= 0).sum()
                if invalid > 0:
                    self.errors.append(f"Found {invalid} non-positive {col} values")
    
    def check_volume(self) -> None:
        if 'volume' in self.df.columns:
            negative = (self.df['volume'] < 0).sum()
            if negative > 0:
                self.errors.append(f"Found {negative} negative volume values")
            zero = (self.df['volume'] == 0).sum()
            if zero > 0:
                self.warnings.append(f"Found {zero} zero volume candles")
    
    def check_ordering(self) -> None:
        if len(self.df) > 1:
            if not self.df['open_time'].is_monotonic_increasing:
                self.errors.append("Data is not ordered by timestamp (ascending)")
    
    def check_timestamp_gaps(self) -> None:
        if len(self.df) < 2:
            return
        timestamps = pd.to_datetime(self.df['open_time'])
        diffs = timestamps.diff().dropna()
        large_gaps = diffs[diffs > pd.Timedelta(hours=1)]
        if len(large_gaps) > 10:
            self.warnings.append(f"Found {len(large_gaps)} large gaps (>1hr)")
        elif len(large_gaps) > 0:
            self.warnings.append(f"Found {len(large_gaps)} large gaps")
    
    def check_data_types(self) -> None:
        if 'open_time' in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df['open_time']):
                self.warnings.append("open_time is not datetime type")
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in self.df.columns:
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    self.errors.append(f"{col} is not numeric type")
    
    def get_report(self) -> Dict:
        """Get a summary report of the data."""
        if self.df.empty:
            return {"error": "Empty dataframe"}
        
        report = {
            "symbol": self.symbol,
            "start_time": str(self.df['open_time'].min()),
            "end_time": str(self.df['open_time'].max()),
            "total_candles": len(self.df),
            "duration_days": (self.df['open_time'].max() - self.df['open_time'].min()).days,
            "columns": list(self.df.columns),
            "errors": self.errors,
            "warnings": self.warnings,
            "valid": len(self.errors) == 0
        }
        
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in price_cols:
            if col in self.df.columns:
                report[f"{col}_min"] = float(self.df[col].min())
                report[f"{col}_max"] = float(self.df[col].max())
                report[f"{col}_mean"] = float(self.df[col].mean())
        
        return report


def validate_parquet_data(file_path: Path) -> Tuple[bool, Dict]:
    """Validate a parquet file containing market data."""
    df = pd.read_parquet(file_path)
    symbol = file_path.stem.split('_')[0]
    
    validator = DataValidator(df, symbol)
    is_valid = validator.validate_all()
    
    return is_valid, validator.get_report()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        is_valid, report = validate_parquet_data(file_path)
        print(f"Valid: {is_valid}")
        print(f"Report: {report}")
    else:
        print("Usage: python validator.py <parquet_file>")