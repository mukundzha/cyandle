#!/usr/bin/env python3
"""
Train CryptoFormer model.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
from model.cryptoformer import create_small_model
from training.trainer import Trainer
from training.optimizer import create_optimizer, create_scheduler, get_lr
from data.dataset import CryptoDataset, create_dataloader, chronological_split


def main():
    parser = argparse.ArgumentParser(description="Train CryptoFormer")
    parser.add_argument('--data', required=True, help='Path to processed training data')
    parser.add_argument('--config', default='small', help='Model size config')
    parser.add_argument('--output', default='cryptoformer/experiments', help='Output directory')
    parser.add_argument('--dry-run', action='store_true', help='Test forward/backward pass only')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'open_return', 'high_return', 'low_return', 'log_return',
        'candle_body', 'upper_wick', 'lower_wick', 'candle_range',
        'log_volume', 'rel_volume', 'rv_5', 'rv_20',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]
    
    available_features = [c for c in feature_cols if c in df.columns]
    if len(available_features) < len(feature_cols):
        import warnings
        warnings.warn(f"Some features not available: {set(feature_cols) - set(available_features)}")
        available_features = list(df.select_dtypes(include=['float64', 'int64']).columns)
    
    print(f"Using {len(available_features)} features")
    
    train_df, val_df, test_df = chronological_split(df)
    
    horizons = [1, 3, 6, 12, 48, 144, 288, 2016]
    
    num_assets = 5
    
    train_dataset = CryptoDataset(
        train_df, available_features,
        context_length=512,
        horizons=horizons
    )
    
    val_dataset = CryptoDataset(
        val_df, available_features,
        context_length=512,
        horizons=horizons
    )
    
    test_dataset = CryptoDataset(
        test_df, available_features,
        context_length=512,
        horizons=horizons
    )
    
    print(f"Datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    if args.dry_run:
        print("\nDry run: Testing forward and backward pass...")
        model = create_small_model(num_assets=num_assets)
        sample = train_dataset[0]
        
        x = sample['input'].unsqueeze(0)
        asset_ids = sample['asset'].unsqueeze(0)
        targets = {
            'returns': sample['returns'].unsqueeze(0),
            'directions': sample['directions'].unsqueeze(0),
            'valid_mask': sample['valid_mask'].unsqueeze(0)
        }
        
        predictions, loss_dict = model(x, asset_ids, targets=targets)
        
        print(f"Input: {x.shape}")
        print(f"Loss: {loss_dict['total_loss'].item():.4f}")
        print("Dry run successful!")
        return
    
    model = create_small_model(num_assets=num_assets)
    print(f"Model created with {model.count_parameters():,} parameters")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'epochs': 10,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'warmup_steps': 100,
        'patience': 5
    }
    
    train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=32)
    test_loader = create_dataloader(test_dataset, batch_size=32)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        output_dir=str(output_dir)
    )
    
    print(f"\nStarting training on {trainer.device}...")
    stats = trainer.train()
    
    print(f"\nTraining complete. Best val loss: {stats['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()