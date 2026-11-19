"""
Embedding layers for CryptoFormer - M4
"""
import torch
import torch.nn as nn
import math
from typing import Optional


class FeatureEmbedding(nn.Module):
    """Projects raw features to hidden dimension: e_t = W_x x_t + b_x"""
    
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.norm = nn.RMSNorm(hidden_dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.norm(x)
        return x


class PositionalEmbedding(nn.Module):
    """Learned positional embeddings."""
    
    def __init__(self, max_seq_len: int, hidden_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(positions)
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        return x + pos_emb


class AssetEmbedding(nn.Module):
    """Embeds asset identifiers."""
    
    def __init__(self, num_assets: int, hidden_dim: int):
        super().__init__()
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.asset_embedding = nn.Embedding(num_assets, hidden_dim)
    
    def forward(self, asset_ids: torch.Tensor) -> torch.Tensor:
        return self.asset_embedding(asset_ids)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""
    
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        inv_freq = inv_freq.to(x.device)
        t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        
        x1 = x[..., :self.dim]
        x2 = x[..., self.dim:] if self.dim * 2 <= hidden_dim else None
        
        rotated = torch.cat([
            x1[..., :self.dim//2] * freqs.cos() - x1[..., self.dim//2:] * freqs.sin(),
            x1[..., self.dim//2:] * freqs.cos() + x1[..., :self.dim//2] * freqs.sin()
        ], dim=-1)
        
        if x2 is not None:
            return torch.cat([rotated, x2], dim=-1)
        return rotated


class TimeEmbedding(nn.Module):
    """Cyclical time embeddings."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_sin = nn.Linear(1, hidden_dim)
        self.proj_cos = nn.Linear(1, hidden_dim)
    
    def forward(self, hour: torch.Tensor, day_of_week: torch.Tensor) -> torch.Tensor:
        import math
        h = hour.float() / 24.0
        d = day_of_week.float() / 7.0
        h = h.unsqueeze(-1)
        d = d.unsqueeze(-1)
        
        h_sin = self.proj_sin(torch.sin(2 * math.pi * h))
        h_cos = self.proj_cos(torch.cos(2 * math.pi * h))
        d_sin = self.proj_sin(torch.sin(2 * math.pi * d))
        d_cos = self.proj_cos(torch.cos(2 * math.pi * d))
        
        return h_sin + h_cos + d_sin + d_cos


class CryptoEmbedding(nn.Module):
    """Combined embedding module for CryptoFormer."""
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_assets: int,
        max_seq_len: int,
        use_time_features: bool = False,
        use_rope: bool = False
    ):
        super().__init__()
        
        self.use_time_features = use_time_features
        self.use_rope = use_rope
        
        self.feature_embedding = FeatureEmbedding(feature_dim, hidden_dim)
        
        if use_rope:
            self.pos_embedding = RotaryPositionalEmbedding(hidden_dim // 2)
        else:
            self.pos_embedding = PositionalEmbedding(max_seq_len, hidden_dim)
        
        self.asset_embedding = AssetEmbedding(num_assets, hidden_dim)
        
        if use_time_features:
            self.time_embedding = TimeEmbedding(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        asset_ids: torch.Tensor,
        hour: Optional[torch.Tensor] = None,
        day_of_week: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        emb = self.feature_embedding(x)
        emb = self.pos_embedding(emb)
        
        asset_emb = self.asset_embedding(asset_ids)
        asset_emb = asset_emb.unsqueeze(1).expand(-1, emb.shape[1], -1)
        emb = emb + asset_emb
        
        if self.use_time_features and hour is not None and day_of_week is not None:
            time_emb = self.time_embedding(hour, day_of_week)
            emb = emb + time_emb
        
        return emb