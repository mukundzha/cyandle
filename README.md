# cyandle: cryptocurrency predictive foundation model

hey there! cyandle (short for crypto foundation model) is a transformer-based model i built from the ground up to predict cryptocurrency market movements. no pretrained weights, no shortcuts - just raw market patterns learned from historical data.

## what this thing does

the model predicts three things for each time horizon:

- **direction**: the probability that price will go up
- **returns**: how much it'll move (in log returns)
- **quantiles**: uncertainty bands (10th, 50th, 90th percentiles)

you can predict from 5 minutes all the way to 7 days ahead.

## the architecture

**cryptformer** is basically a standard transformer with some crypto-specific tweaks:

- 4 layers by default (configurable)
- 128 hidden dimensions for the small version
- 4 attention heads
- fully causal - it only looks at past data, never future
- rmsnorm and gelu activations
- about 640k parameters for the small model

## project structure

```
cyandle/
├── configs/           # model configs: tiny/small/medium
├── data/              # downloader, features, dataset, validator
├── model/             # embeddings, attention, transformer, heads
├── training/          # trainer, losses, optimizer, checkpoint
├── evaluation/        # metrics, benchmarks
├── tests/             # unit tests
├── scripts/           # train.py, etc.
└── README.md
```

## quick setup

just install the basics:

```bash
pip install torch pandas numpy pyarrow scikit-learn lightgbm
```

## trying it out

first, make some fake data:

```bash
python -c "
import pandas as pd, numpy as np
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
```

then test the model:

```bash
python scripts/train.py --data sample.parquet --dry-run
```

the dry-run flag just tests forward/backward pass without actually training.

## what goes in, what comes out

**input features** (30 total):
- returns based on open/high/low/close vs previous close
- candle body, wicks, range
- volume metrics
- volatility (realized std of returns)
- time features (hour of day, day of week - cyclically encoded)

**prediction horizons**:
- 1, 3, 6, 12, 48, 144, 288, 2016 candles ahead
- that's 5m, 15m, 30m, 1h, 4h, 12h, 24h, 7d

**output per horizon**:
- direction: 0-1 probability price goes up
- returns: expected cumulative log return
- quantiles: uncertainty estimates

## training setup

multi-task learning with a weighted sum:

```
loss = λ1 * return_loss + λ2 * direction_loss + λ3 * quantile_loss
```

- return_loss: huber (handles outliers well)
- direction_loss: binary cross-entropy
- quantile_loss: pinball loss

default weights are 1.0, 1.0, 0.5.

## baseline comparison

before claiming the transformer is useful, we compare against:
- always predict zero return
- historical average
- momentum strategy
- lightgbm

the model has to beat these to be worth using.

## the commits

this repo is built commit-by-commit:

- m0: binance data downloader
- m1: data validation
- m2: feature engineering
- m3: sliding window dataset
- m4: transformer from scratch
- m5: training infrastructure

each commit represents a working piece you can test independently.

## getting started locally

if you want to run a quick test:

1. download some btcusdt data from binance
2. run `scripts/train.py` with your data
3. check the `experiments/` folder for outputs

for full training you'll want a gpu with 16gb+ vram, but the tiny model works on cpu for testing.

## license

mit license - do what you want, just don't blame me if it loses money :)