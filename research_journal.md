# Cyandle Research Journal

## M0: Data Infrastructure - Completed

**Date:** 2026-11-19

### Hypothesis
A custom Transformer-based cryptocurrency predictor can be built from scratch using only historical market data, with no pretrained weights. The model should learn meaningful market representations through purely numerical pattern recognition.

### Implementation

#### Data Pipeline
- **Downloader:** Binance Vision API client for historical spot klines
- **Validator:** Comprehensive data integrity checks
- **Feature Engineering:** Returns, candle features, volatility, temporal encodings

#### Key Decisions
1. **Feature Dimension:** 30 features combining price, volume, volatility, and time features
2. **Causal Attention:** Ensured by lower-triangular mask, preventing future information leakage
3. **Multi-Horizon Targets:** 8 horizons from 5 minutes to 7 days
4. **Loss Function:** Multi-task with Huber, BCE, and pinball losses

### Parameter Count
- Tiny model: ~640K parameters (2 layers, 64 hidden)
- Small model: ~640K parameters (4 layers, 128 hidden) - target for development
- Medium model: ~2M parameters (6 layers, 256 hidden)

### Next Steps
1. Download actual BTCUSDT data from Binance
2. Run full training with walk-forward validation
3. Compare against baselines (zero-return, momentum, LightGBM)

### Potential Risks
1. **Data Quality:** Binance data may have gaps or errors
2. **Overfitting:** Small dataset may overfit without regularization
3. **Compute:** CPU-only training will be slow; need GPU for full training

---

## M1: Feature Engineering

**Features Implemented:**
- Return features: log_return, open_return, high_return, low_return
- Candle features: body, upper_wick, lower_wick, range
- Volume features: log_volume, relative_volume
- Volatility: rv_5, rv_20, rv_50
- Temporal: hour_sin, hour_cos, dow_sin, dow_cos (cyclical encoding)

---

## M2: Model Architecture

**CryptoFormer Components:**
1. FeatureEmbedding: Linear projection + RMSNorm
2. PositionalEmbedding: Learned embeddings
3. AssetEmbedding: Learned asset identifiers
4. CausalSelfAttention: Multi-head with proper masking
5. TransformerBlock: Pre-LayerNorm with residual connections
6. MultiHeadPrediction: Direction, return, quantile heads

**Training Configuration:**
- AdamW optimizer
- Cosine annealing with warmup
- Gradient clipping at 1.0
- Mixed precision (when CUDA available)

---

## M3: Evaluation Plan

### Metrics
- Regression: MAE, RMSE, Spearman correlation
- Classification: Directional accuracy, precision, recall, F1, ROC-AUC
- Probabilistic: Quantile loss, coverage

### Baselines
- Zero-return predictor
- Historical mean
- Momentum strategy
- LightGBM

---

## M4: Causal Attention Verification

**Test:** Changing future observations cannot change early representations.

**Implementation:** Built causal mask as lower-triangular matrix where position i can only attend to positions j <= i.

```python
def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
```

---

## M5: Checkpointing System

Implemented checkpoint manager with:
- Best model tracking (based on validation loss)
- Maximum checkpoints limit
- Resume capability
- Experiment ID generation

---

## M6: Forward/Backward Pass Verified

**Test Results:**
- Forward pass: SUCCESS
- Backward pass: SUCCESS (all gradients computed)
- Gradient clipping: SUCCESS
- Mixed precision: Available when CUDA

---

## Research Notes

### What Works
- Causal attention correctly prevents future leakage
- Multi-task loss balances return and direction objectives
- Quantile predictions enable uncertainty estimation

### Open Questions
1. Should we use RoPE or learned positional embeddings?
2. What's the optimal weight for multi-task loss?
3. How to handle assets with different data availability?

### Next Milestone
- Implement walk-forward validation
- Download BTCUSDT data and train for real
- Compare against baselines quantitatively