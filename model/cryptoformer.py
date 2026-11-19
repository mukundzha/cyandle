"""
CryptoFormer: Custom Transformer for cryptocurrency prediction - M4

Main model class combining all components.
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Optional, Dict, Tuple, Union
import logging

from .embeddings import CryptoEmbedding
from .transformer import TransformerBackbone
from .forecasting_heads import MultiHeadPrediction, ForecastLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoFormer(nn.Module):
    """
    CryptoFormer: Transformer-based cryptocurrency predictive model.
    """
    
    def __init__(
        self,
        feature_dim: int = 14,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        context_length: int = 512,
        num_assets: int = 10,
        num_horizons: int = 8,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        use_time_features: bool = False,
        use_rope: bool = False,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length
        self.num_assets = num_assets
        self.num_horizons = num_horizons
        
        params = self._count_parameters()
        logger.info(f"CryptoFormer: {params:,} parameters")
        
        self.embedding = CryptoEmbedding(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_assets=num_assets,
            max_seq_len=context_length,
            use_time_features=use_time_features,
            use_rope=use_rope
        )
        
        self.backbone = TransformerBackbone(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            activation=activation
        )
        
        self.prediction_head = MultiHeadPrediction(
            hidden_dim=hidden_dim,
            num_horizons=num_horizons,
            quantiles=quantiles
        )
        
        self.loss_fn = ForecastLoss(
            return_weight=1.0,
            direction_weight=1.0,
            quantile_weight=0.5
        )
    
    def forward(
        self,
        x: torch.Tensor,
        asset_ids: torch.Tensor,
        hour: Optional[torch.Tensor] = None,
        day_of_week: Optional[torch.Tensor] = None,
        targets: Optional[Dict] = None
    ) -> Tuple[Dict, Optional[Dict]]:
        hidden = self.embedding(x, asset_ids, hour, day_of_week)
        hidden = self.backbone(hidden)
        predictions = self.prediction_head(hidden)
        
        loss_dict = None
        if targets is not None:
            loss_dict = self.loss_fn(predictions, targets)
        
        return predictions, loss_dict
    
    def predict(
        self,
        x: torch.Tensor,
        asset_ids: torch.Tensor,
        hour: Optional[torch.Tensor] = None,
        day_of_week: Optional[torch.Tensor] = None
    ) -> Dict:
        predictions, _ = self.forward(x, asset_ids, hour, day_of_week, targets=None)
        return predictions
    
    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_parameters(self) -> int:
        return self._count_parameters()
    
    def get_config(self) -> Dict:
        return {
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'context_length': self.context_length,
            'num_assets': self.num_assets,
            'num_horizons': self.num_horizons
        }
    
    def save_checkpoint(self, path: str, optimizer: Optional[Union[Optimizer, dict]] = None):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.get_config()
        }
        if optimizer is not None:
            if isinstance(optimizer, Optimizer):
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            else:
                checkpoint['optimizer_state_dict'] = optimizer
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str, optimizer: Optional[Union[Optimizer, dict]] = None) -> Dict:
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            if isinstance(optimizer, Optimizer):
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('config', {})


def create_model(config: Dict) -> CryptoFormer:
    return CryptoFormer(
        feature_dim=config.get('feature_dim', 14),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 4),
        ffn_dim=config.get('ffn_dim', 256),
        dropout=config.get('dropout', 0.1),
        context_length=config.get('context_length', 512),
        num_assets=config.get('num_assets', 10),
        num_horizons=len(config.get('horizons', [1, 3, 6, 12, 48, 144, 288, 2016])),
        quantiles=config.get('quantiles', [0.1, 0.5, 0.9]),
        use_time_features=config.get('use_time_features', True),
        use_rope=config.get('use_rope', False),
        activation=config.get('activation', 'gelu')
    )


def create_tiny_model(num_assets: int = 10, horizons: Optional[List[int]] = None) -> CryptoFormer:
    if horizons is None:
        horizons = [1, 3, 6, 12, 48, 144, 288, 2016]
    elif isinstance(horizons, int):
        horizons = list(range(1, horizons + 1))
    return CryptoFormer(
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        ffn_dim=128,
        num_assets=num_assets,
        num_horizons=len(horizons)
    )


def create_small_model(num_assets: int = 10, horizons: Optional[List[int]] = None) -> CryptoFormer:
    if horizons is None:
        horizons = [1, 3, 6, 12, 48, 144, 288, 2016]
    elif isinstance(horizons, int):
        horizons = list(range(1, horizons + 1))
    return CryptoFormer(
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        ffn_dim=256,
        num_assets=num_assets,
        num_horizons=len(horizons),
        feature_dim=14
    )


def create_medium_model(num_assets: int = 10, horizons: Optional[List[int]] = None) -> CryptoFormer:
    if horizons is None:
        horizons = [1, 3, 6, 12, 48, 144, 288, 2016]
    elif isinstance(horizons, int):
        horizons = list(range(1, horizons + 1))
    return CryptoFormer(
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=512,
        num_assets=num_assets,
        num_horizons=len(horizons)
    )


if __name__ == "__main__":
    batch_size = 4
    seq_len = 128
    feature_dim = 30
    num_assets = 5
    num_horizons = 8
    
    model = create_small_model(num_assets=num_assets)
    print(f"Config: {model.get_config()}")
    print(f"Parameters: {model.count_parameters():,}")
    
    x = torch.randn(batch_size, seq_len, 30)
    asset_ids = torch.randint(0, num_assets, (batch_size,))
    
    predictions, loss_dict = model(
        x, asset_ids,
        targets={
            'returns': torch.randn(batch_size, num_horizons),
            'directions': torch.randint(0, 2, (batch_size, num_horizons)).float(),
            'valid_mask': torch.ones(batch_size, num_horizons, dtype=torch.bool)
        }
    )
    
    print(f"Direction: {predictions['direction'].shape}")
    print(f"Returns: {predictions['returns'].shape}")
    print(f"Quantiles: {predictions['quantiles'].shape}")
    print(f"Loss keys: {list(loss_dict.keys())}")