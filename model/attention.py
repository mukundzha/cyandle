"""
Causal attention mechanism for CryptoFormer - M4
"""
import torch
import torch.nn as nn
import math
from typing import Optional


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with proper masking."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_flash_attention: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Check if flash attention is available
        self.has_flash_attn = False
        if use_flash_attention:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn = flash_attn_func
                self.has_flash_attn = True
            except ImportError:
                import logging
                logging.warning("Flash attention not available, using standard attention")
        
        self.cache = {}
    
    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build a causal mask for attention. Shape: (1, 1, seq_len, seq_len)"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.has_flash_attn and not self.training:
            attn_output = self.flash_attn(q, k, v, causal=True)
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            
            causal_mask = self._build_causal_mask(seq_len, x.device)
            attn_weights = attn_weights * causal_mask
            attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))
            
            attn_probs = torch.softmax(attn_weights, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output
    
    def enable_kv_cache(self):
        self.use_cache = True
    
    def disable_kv_cache(self):
        self.use_cache = False


class CrossAttention(nn.Module):
    """Multi-head cross-attention."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key_value.shape[1]
        
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_weights = attn_weights * mask
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, self.hidden_dim)
        
        return self.o_proj(attn_output)


if __name__ == "__main__":
    batch_size, seq_len, hidden_dim = 2, 64, 128
    
    attn = CausalSelfAttention(hidden_dim, num_heads=4)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    output = attn(x)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    
    causal_mask = attn._build_causal_mask(seq_len, x.device)
    print(f"Causal mask: {causal_mask.shape}")