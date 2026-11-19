#!/usr/bin/env python3
"""
Train CryptoFormer model - Colab T4 GPU ready.

Usage in Colab:
1. Set runtime to GPU (T4)
2. Run this script or !python train.py --data your_data.parquet
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
from model.cryptoformer import create_small_model, create_medium_model
from training.trainer import Trainer
from training.optimizer import create_optimizer, create_scheduler, get_lr
from data.dataset import CryptoDataset, create_dataloader, chronological_split


def main():
    parser = argparse.ArgumentParser(description="Train CryptoFormer - Colab T4 GPU Ready")
    parser.add_argument('--data', required=True, help='Path to processed training data')
    parser.add_argument('--model', choices=['small', 'medium'], default='medium', help='Model size')
    parser.add_argument('--config', default='medium', help='Config file name (without .yaml)')
    parser.add_argument('--output', default='cryptoformer/experiments', help='Output directory')
    parser.add_argument('--dry-run', action='store_true', help='Test forward/backward pass only')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Auto-detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"\nLoading data from {args.data}...")
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
    print(f"Data shape: {df.shape}")
    
    # Temporal split
    train_df, val_df, test_df = chronological_split(df)
    
    horizons = [1, 3, 6, 12, 48, 144, 288, 2016]
    num_assets = 5
    
    # Create datasets
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
        model = create_medium_model(num_assets=num_assets)
        sample = train_dataset[0]
        
        x = sample['input'].unsqueeze(0).to(device)
        asset_ids = sample['asset'].unsqueeze(0)
        targets = {
            'returns': sample['returns'].unsqueeze(0),
            'directions': sample['directions'].unsqueeze(0),
            'valid_mask': sample['valid_mask'].unsqueeze(0)
        }
        
        predictions, loss_dict = model(x, asset_ids, targets=targets)
        
        print(f"Input: {x.shape}")
        print(f"Loss: {loss_dict['total_loss'].item():.4f}")
        print(f"Model params: {model.count_parameters():,}")
        print("Dry run successful!")
        return
    
    # Create model based on choice
    if args.model == 'medium':
        model = create_medium_model(num_assets=num_assets)
    else:
        model = create_small_model(num_assets=num_assets)
    
    print(f"Model params: {model.count_parameters():,}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'warmup_steps': 100,
        'patience': 5,
        'use_amp': torch.cuda.is_available(),  # Use mixed precision on GPU
    }
    
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size)
    test_loader = create_dataloader(test_dataset, batch_size=args.batch_size)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        output_dir=str(output_dir)
    )
    
    print(f"\nStarting training on {device}...")
    if torch.cuda.is_available():
        print("Using mixed precision (AMP)")
    
    stats = trainer.train()
    
    print(f"\nTraining complete. Best val loss: {stats['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()