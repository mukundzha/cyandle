# Cyandle: Cryptocurrency Predictive Foundation Model

A research-focused Transformer-based cryptocurrency time-series forecasting model built from scratch.

## Overview

Cyandle (Crypto Foundation Model) is a custom causal Transformer that predicts:
- Future market direction
- Cumulative log returns
- Return distribution quantiles (uncertainty)

## Architecture

**CryptoFormer** - Custom Transformer architecture:
- 4 Transformer layers (configurable)
- 128 hidden dimensions (small model: 128, medium: 256)
- 4 attention heads
- Causal self-attention (no future information leakage)
- RMSNorm / LayerNorm
- GELU activation

## Project Structure

```
cyandle/
├── configs/           # Model configurations
│   ├── tiny.yaml      # CPU test config (~1M params)
│   ├── small.yaml     # Development config (~2-3M params)
│   └── medium.yaml    # Full training config (~5-8M params)
│
├── data/              # Data infrastructure
│   ├── raw/           # Raw downloads from Binance
│   └── downloader.py  # Binance Vision API client
├── model/             # Model components
│   ├── embeddings.py  # Feature, positional, asset embeddings
│   ├── attention.py   # Causal self-attention
│   ├── transformer.py  # Transformer blocks
│   ├── forecasting_heads.py # Multi-horizon prediction heads
│   └── cryptoformer.py # Main model class
│
├── training/          # Training utilities
│   ├── trainer.py     # Training loop with AMP
│   ├── losses.py      # Multi-task loss functions
│   ├── optimizer.py   # Optimizer & scheduler setup
│   └── checkpoint.py  # Checkpoint management
│
├── evaluation/        # Evaluation tools
│   ├── metrics.py     # Evaluation metrics
│   └── benchmarks.py  # Baseline models
│
├── tests/             # Unit tests
│   └── test_model.py
│
├── scripts/           # CLI scripts
│   └── train.py
│
└── README.md
```

## Installation

```bash
pip install torch pandas numpy pyarrow scikit-learn lightgbm
```

## Quick Start

### Test the model (CPU)

```bash
python scripts/train.py --data data/sample.parquet --dry-run
```

### Train a model

```bash
# Generate sample data first
python -c "
import pandas as pd
import numpy as np
dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
df = pd.DataFrame({
    'open_time': dates,
    'open': 50000 + np.random.randn(1000).cumsum() * 100,
    'high': 50000 + np.random.randn(1000).cumsum() * 100 + 500,
    'low': 50000 + np.random.randn(1000).cumsum() * 100 - 500,
    'close': 50000 + np.random.randn(1000).cumsum() * 100,
    'volume': 1000 + np.random.randn(1000).cumsum() * 10
})
df.to_parquet('sample.parquet')
"

python scripts/train.py --data sample.parquet
```

## Model Features

### Input Features
- Price returns (open, high, low, close relative to previous close)
- Log returns
- Candle body, wicks, range
- Volume features (log volume, relative volume)
- Volatility features (realized volatility)
- Temporal features (hour, day-of-week with cyclical encoding)

### Prediction Horizons
- 5 min (1 candle) → 7 days (2016 candles)

### Output
For each horizon:
- `direction`: P(returns > 0) via sigmoid
- `returns`: Expected cumulative log returns
- `quantiles`: 10th, 50th, 90th percentiles for uncertainty

## Training Objectives

Multi-task loss:
```
L = λ1 * L_return + λ2 * L_direction + λ3 * L_quantile

where:
- L_return: Huber loss (robust to outliers)
- L_direction: Binary cross-entropy with logits
- L_quantile: Pinball loss
```

## Key Principles

1. **Causal**: Only uses past data for predictions
2. **Multi-asset**: Trains across multiple cryptocurrencies
3. **Multi-horizon**: Predicts multiple time frames simultaneously
4. **Uncertainty**: Provides quantile predictions
5. **Reproducible**: All experiments tracked with config and checkpoints

## Baseline Models

Compared against:
- Zero-return predictor
- Historical mean
- Momentum strategy
- LightGBM

## Development Milestones

- [x] M0: Data downloader infrastructure
- [x] M1: Data validation and quality checks
- [x] M2: Feature engineering pipeline
- [x] M3: Sliding window dataset
- [x] M4: Transformer architecture from scratch
- [x] M5: Training infrastructure
- [ ] M6: Evaluation benchmarks
- [ ] M7: Walk-forward validation
- [ ] M8: Full training and evaluation

## License

MIT - Research purposes