"""
Transformer blocks and backbone for CryptoFormer - M4
"""
import torch
import torch.nn as nn
import math
from .attention import CausalSelfAttention


class SiGLU(nn.Module):
    """SwiGLU activation: SiLU(x) * G(x)"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * torch.nn.functional.silu(b)


class TransformerBlock(nn.Module):
    """Standard Transformer block with causal self-attention."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        use_rmsnorm: bool = False,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.use_rmsnorm = use_rmsnorm
        self.norm1 = self._get_norm(hidden_dim)
        self.norm2 = self._get_norm(hidden_dim)
        
        self.attention = CausalSelfAttention(hidden_dim, num_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def _get_norm(self, dim: int) -> nn.Module:
        if hasattr(nn, 'RMSNorm'):
            return nn.RMSNorm(dim)
        return nn.LayerNorm(dim)
    
    def _get_activation(self, name: str) -> nn.Module:
        if name == "gelu":
            return nn.GELU()
        elif name == "relu":
            return nn.ReLU()
        elif name == "swish":
            return nn.SiLU()
        elif name == "swiglu":
            return SiGLU(2 * hidden_dim)
        else:
            return nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_rmsnorm:
            residual = x
            x = self.norm1(x)
            x = self.attention(x)
            x = x + residual
        else:
            residual = x
            x = self.attention(self.norm1(x))
            x = x + residual
        
        residual = x
        x = self.ffn(self.norm2(x))
        x = x + residual
        
        return x


class TransformerBackbone(nn.Module):
    """Stack of Transformer blocks forming the CryptoFormer backbone."""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        use_rmsnorm: bool = False,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_rmsnorm=use_rmsnorm,
                activation=activation
            ) for _ in range(num_layers)
        ])
        
        self.final_norm = self._get_norm(hidden_dim)
    
    def _get_norm(self, dim: int) -> nn.Module:
        if hasattr(nn, 'RMSNorm'):
            return nn.RMSNorm(dim)
        return nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_backbone(config: dict) -> TransformerBackbone:
    return TransformerBackbone(
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 4),
        ffn_dim=config.get('ffn_dim', 256),
        dropout=config.get('dropout', 0.1),
        use_rmsnorm=config.get('use_rmsnorm', False),
        activation=config.get('activation', 'gelu')
    )


if __name__ == "__main__":
    batch_size, seq_len, hidden_dim = 2, 512, 128
    
    backbone = TransformerBackbone(
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        ffn_dim=256
    )
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    output = backbone(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Parameters: {backbone.count_parameters():,}")