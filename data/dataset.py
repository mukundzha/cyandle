"""
Sliding-window dataset for multi-horizon prediction - M3
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataset(Dataset):
    """
    Sliding window dataset for multi-horizon cryptocurrency prediction.
    
    Input: Historical observations ending at time t
    Target: Future returns and directions at multiple horizons
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        context_length: int = 512,
        horizons: List[int] = None,
        asset: str = "BTCUSDT",
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ):
        self.df = df.copy()
        self.df = self.df.sort_values('open_time').reset_index(drop=True)
        self.feature_cols = feature_cols
        self.context_length = context_length
        self.asset = asset
        
        if horizons is None:
            horizons = [1, 3, 6, 12, 48, 144, 288, 2016]
        self.horizons = horizons
        
        if normalize:
            self.mean = mean if mean is not None else df[feature_cols].mean().values
            self.std = std if std is not None else df[feature_cols].std().values
            self.std[self.std == 0] = 1.0
        else:
            self.mean = np.zeros(len(feature_cols))
            self.std = np.ones(len(feature_cols))
        
        self.samples = self._build_samples()
        logger.info(f"Created {len(self.samples)} samples for {asset}")
    
    def _build_samples(self) -> List[Dict]:
        """Build list of valid samples."""
        samples = []
        max_horizon = max(self.horizons)
        
        min_valid_idx = self.context_length + max_horizon
        
        for idx in range(min_valid_idx, len(self.df)):
            start_idx = idx - self.context_length
            
            input_features = self.df[self.feature_cols].iloc[start_idx:idx].values
            
            targets = {}
            for h in self.horizons:
                future_idx = idx + h
                
                if future_idx < len(self.df):
                    future_price = self.df['close'].iloc[future_idx]
                    current_price = self.df['close'].iloc[idx - 1]
                    log_ret = np.log(future_price / current_price)
                    direction = 1.0 if log_ret > 0 else 0.0
                    targets[f'h{h}'] = {'return': log_ret, 'direction': direction}
                else:
                    targets[f'h{h}'] = {'return': np.nan, 'direction': np.nan}
            
            samples.append({
                'idx': idx,
                'input': input_features,
                'targets': targets
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        input_normalized = (sample['input'] - self.mean) / self.std
        
        returns = []
        directions = []
        valid_masks = []
        
        for h in self.horizons:
            target = sample['targets'][f'h{h}']
            returns.append(target['return'])
            directions.append(target['direction'])
            valid_masks.append(not np.isnan(target['return']))
        
        return {
            'input': torch.FloatTensor(input_normalized),
            'returns': torch.FloatTensor(returns),
            'directions': torch.FloatTensor(directions),
            'valid_mask': torch.BoolTensor(valid_masks),
            'asset': torch.tensor(hash(self.asset) % 100, dtype=torch.long),
            'timestamp_idx': torch.tensor(sample['idx'], dtype=torch.long)
        }
    
    def get_normalization_params(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mean, self.std


def create_dataloader(
    dataset: CryptoDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False
) -> DataLoader:
    """Create a DataLoader for the dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available()
    )


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split of time series data."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    
    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    n = 1000
    df = pd.DataFrame({
        'open_time': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'open': 50000 + np.random.randn(n).cumsum() * 100,
        'high': 50000 + np.random.randn(n).cumsum() * 100 + 500,
        'low': 50000 + np.random.randn(n).cumsum() * 100 - 500,
        'close': 50000 + np.random.randn(n).cumsum() * 100,
        'volume': 1000 + np.random.randn(n).cumsum() * 10
    })
    
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    
    dataset = CryptoDataset(df, feature_cols, context_length=64)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[100]
    print(f"Input shape: {sample['input'].shape}")
    print(f"Returns: {sample['returns']}")