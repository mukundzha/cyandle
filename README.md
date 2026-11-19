# cyandle

cryptocurrency transformer from scratch. no pretrained weights.

## what

predicts crypto direction, returns, quantiles for 5m→7d horizons.

## quick test

```bash
pip install torch pandas numpy
python -c "import pandas as pd, numpy as np; d=pd.date_range('2024-01-01',100); df=pd.DataFrame({'open_time':d,'open':50000,'high':50100,'low':49900,'close':50000,'volume':1000}); df.to_parquet('s.parquet')"
python scripts/train.py --data s.parquet --dry-run
```

## model sizes

- tiny: ~60k params (cpu testing)
- small: ~640k params (default)
- medium: ~3.5m params

## structure

```
cyandle/
├── data/          # downloader, features, dataset, validator
├── model/         # transformer, attention, heads
├── training/      # trainer, losses, optimizer
├── evaluation/    # metrics, benchmarks
├── configs/       # tiny/small/medium yaml
└── scripts/       # train.py
```

## license

mit license.

**do what the hell you want with it** - but if you make money from it, i want a cut. if it blows up the stock market, not my problem.

**disclaimer:** this is for research purposes only. dont trade with real money based on this model. if you do, youre dumb.

**fun fact:** this readme is shorter than most crypto twitter posts.