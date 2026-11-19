"""
Unit tests for CryptoFormer model components - M4
"""
import pytest
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.embeddings import CryptoEmbedding
from model.attention import CausalSelfAttention
from model.transformer import TransformerBackbone
from model.forecasting_heads import MultiHeadPrediction
from model.cryptoformer import CryptoFormer, create_small_model


class TestEmbeddings:
    """Tests for embedding modules."""
    
    def test_feature_embedding_shape(self):
        emb = CryptoEmbedding(feature_dim=30, hidden_dim=128, num_assets=10, max_seq_len=512)
        x = torch.randn(4, 128, 30)
        asset_ids = torch.tensor([0, 1, 2, 3])
        out = emb(x, asset_ids)
        assert out.shape == (4, 128, 128)
    
    def test_crypto_embedding_shape(self):
        emb = CryptoEmbedding(30, 128, 10, 512)
        x = torch.randn(4, 128, 30)
        asset_ids = torch.tensor([0, 1, 2, 3])
        out = emb(x, asset_ids)
        assert out.shape == (4, 128, 128)


class TestAttention:
    """Tests for attention modules."""
    
    def test_causal_attention_shape(self):
        attn = CausalSelfAttention(hidden_dim=128, num_heads=4)
        x = torch.randn(4, 128, 128)
        out = attn(x)
        assert out.shape == (4, 128, 128)
    
    def test_causal_mask_correctness(self):
        attn = CausalSelfAttention(hidden_dim=128, num_heads=4)
        mask = attn._build_causal_mask(8, torch.device('cpu'))
        expected = torch.tril(torch.ones(8, 8)).unsqueeze(0).unsqueeze(0)
        assert torch.allclose(mask, expected)


class TestTransformer:
    """Tests for Transformer blocks."""
    
    def test_transformer_block_shape(self):
        from model.transformer import TransformerBlock
        block = TransformerBlock(hidden_dim=128, num_heads=4, ffn_dim=256)
        x = torch.randn(4, 128, 128)
        out = block(x)
        assert out.shape == (4, 128, 128)
    
    def test_transformer_backbone_shape(self):
        backbone = TransformerBackbone(hidden_dim=128, num_layers=4, num_heads=4, ffn_dim=256)
        x = torch.randn(4, 128, 128)
        out = backbone(x)
        assert out.shape == (4, 128, 128)


class TestForecastingHeads:
    """Tests for prediction heads."""
    
    def test_multi_head_prediction_shape(self):
        head = MultiHeadPrediction(hidden_dim=128, num_horizons=8)
        x = torch.randn(4, 128, 128)
        out = head(x)
        
        assert out['direction'].shape == (4, 8)
        assert out['returns'].shape == (4, 8)
        assert out['quantiles'].shape == (4, 8, 3)


class TestCryptoFormer:
    """Tests for the full model."""
    
    def test_model_creation(self):
        model = create_small_model(num_assets=5, horizons=8)
        assert model is not None
    
    def test_model_parameter_count(self):
        model = create_small_model(num_assets=5, horizons=8)
        params = model.count_parameters()
        assert 500000 < params < 5000000
    
    def test_forward_pass_shape(self):
        model = create_small_model(num_assets=5)
        
        batch_size = 4
        seq_len = 128
        
        x = torch.randn(batch_size, seq_len, 30)
        asset_ids = torch.randint(0, 5, (batch_size,))
        
        predictions, loss_dict = model(
            x, asset_ids,
            targets={
                'returns': torch.randn(batch_size, 8),
                'directions': torch.randint(0, 2, (batch_size, 8)).float(),
                'valid_mask': torch.ones(batch_size, 8, dtype=torch.bool)
            }
        )
        
        assert predictions['direction'].shape == (batch_size, 8)
        assert predictions['returns'].shape == (batch_size, 8)
        assert 'total_loss' in loss_dict
    
    def test_backward_pass(self):
        model = create_small_model(num_assets=2)
        model.train()
        
        x = torch.randn(2, 64, 30, requires_grad=True)
        asset_ids = torch.tensor([0, 1])
        
        predictions, loss_dict = model(
            x, asset_ids,
            targets={
                'returns': torch.randn(2, 8),
                'directions': torch.randint(0, 2, (2, 8)).float(),
                'valid_mask': torch.ones(2, 8, dtype=torch.bool)
            }
        )
        
        loss = loss_dict['total_loss']
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])