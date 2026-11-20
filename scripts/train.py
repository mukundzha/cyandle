#!/usr/bin/env python3
"""
Train CryptoFormer model - Colab T4 GPU optimized for 5M parameter models.

Features:
- Mixed precision training (AMP)
- Gradient accumulation for large models
- Progress bars with tqdm
- Google Drive integration
- Automatic checkpoint management
- Comprehensive logging

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
from tqdm import tqdm
import logging

from model.cryptoformer import create_small_model, create_medium_model, create_large_model
from training.trainer import Trainer
from training.optimizer import create_optimizer, create_scheduler, get_lr
from data.dataset import CryptoDataset, create_dataloader, chronological_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mount_drive_if_needed():
    """Mount Google Drive for persistent storage."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        return True
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train CryptoFormer - Colab T4 GPU Ready (5M Parameters)"
    )
    parser.add_argument('--data', required=True, help='Path to processed training data')
    parser.add_argument('--model', choices=['small', 'medium', 'large'], default='medium', 
                       help='Model size (small=3M, medium=5M, large=12M params)')
    parser.add_argument('--config', default='medium', help='Config file name (without .yaml)')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--dry-run', action='store_true', help='Test forward/backward pass only')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--gradient-accumulation', type=int, default=1, 
                       help='Gradient accumulation steps for large batch effective size')
    parser.add_argument('--num-assets', type=int, default=100, 
                       help='Number of assets to support in embedding')
    
    args = parser.parse_args()
    
    # Auto-detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  CRYPTOFORMER TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Compute: {torch.cuda.get_device_properties(0).multi_process_memory // 1024} MB multi-process memory")
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path('cryptoformer/experiments')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  MODEL: {args.model.upper()}")
    print(f"{'='*60}")
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    df = pd.read_parquet(args.data)
    print(f"Data shape: {df.shape}")
    
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'open_return', 'high_return', 'low_return', 'log_return',
        'candle_body', 'upper_wick', 'lower_wick', 'candle_range',
        'log_volume', 'rel_volume', 'rv_5', 'rv_20',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]
    
    available_features = [c for c in feature_cols if c in df.columns and c != 'open_time']
    if len(available_features) < len(feature_cols):
        import warnings
        warnings.warn(f"Some features not available: {set(feature_cols) - set(available_features)}")
        # Get all numeric columns except open_time
        numeric_cols = df.select_dtypes(include=['float64', 'int64', 'uint64']).columns.tolist()
        available_features = [c for c in numeric_cols if c != 'open_time']
    
    print(f"Using {len(available_features)} features")
    
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
    
    # Create model with auto-detected feature count
    feature_dim = len([c for c in available_features if c != 'open_time'])
    # Use 100 as max assets for embedding table (sufficient for crypto market)
    model_num_assets = max(args.num_assets, 100)
    if args.model == 'small':
        model = create_small_model(num_assets=model_num_assets, feature_dim=feature_dim)
    elif args.model == 'large':
        model = create_large_model(num_assets=model_num_assets, feature_dim=feature_dim)
    else:  # medium (5M params)
        model = create_medium_model(num_assets=model_num_assets, feature_dim=feature_dim)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,} (~{num_params/1e6:.1f}M)")
    
    # Determine batch size based on VRAM
    if args.batch_size:
        batch_size = args.batch_size
    else:
        # Auto-scale batch size for T4 GPU
        if num_params > 8_000_000:
            batch_size = 16  # Large model
        elif num_params > 5_000_000:
            batch_size = 24  # Medium model (5M)
        else:
            batch_size = 32  # Small model
    
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation}")
    print(f"Effective batch size: {batch_size * args.gradient_accumulation}")
    
    # Dry run
    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"  DRY RUN - Testing forward/backward pass")
        print(f"{'='*60}")
        
        model = model.to(device)
        sample = train_dataset[0]
        
        x = sample['input'].unsqueeze(0).to(device)
        asset_ids = sample['asset'].unsqueeze(0)
        targets = {
            'returns': sample['returns'].unsqueeze(0).to(device),
            'directions': sample['directions'].unsqueeze(0).to(device),
            'valid_mask': sample['valid_mask'].unsqueeze(0).to(device)
        }
        
        # Enable AMP for dry run too
        use_amp = torch.cuda.is_available()
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        model.train()
        with torch.cuda.amp.autocast(enabled=use_amp):
            predictions, loss_dict = model(x, asset_ids, targets=targets)
            loss = loss_dict['total_loss']
        
        scaler.scale(loss).backward()
        
        print(f"Input: {x.shape}")
        print(f"Loss: {loss.item():.4f}")
        print(f"Direction shape: {predictions['direction'].shape}")
        print(f"Returns shape: {predictions['returns'].shape}")
        print("✓ Dry run successful!")
        return
    
    # Training configuration
    config = {
        'epochs': args.epochs if args.epochs else 10,
        'learning_rate': args.lr if args.lr else 1e-4,
        'batch_size': batch_size,
        'warmup_steps': 200,
        'patience': 8,
        'use_amp': torch.cuda.is_available(),
        'gradient_accumulation_steps': args.gradient_accumulation,
        'max_grad_norm': 1.0,
        'model_size': args.model
    }
    
    # Override for large models
    if args.model == 'large':
        config['learning_rate'] = 5e-5
        config['patience'] = 10
        config['warmup_steps'] = 300
    
    # Create dataloaders with progress bars
    print(f"\n{'='*60}")
    print(f"  Creating dataloaders...")
    print(f"{'='*60}")
    
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=min(batch_size, 16))
    test_loader = create_dataloader(test_dataset, batch_size=min(batch_size, 16))
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Calculate training time estimate
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config['epochs']
    print(f"Total training steps: {total_steps}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        output_dir=str(output_dir)
    )
    
    # Print training summary
    print(f"\n{'='*60}")
    print(f"  TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model size: {num_params:,} parameters ({num_params/1e6:.1f}M)")
    print(f"Device: {device}")
    print(f"Mixed precision: {config['use_amp']}")
    print(f"Gradient accumulation: {config['gradient_accumulation_steps']} steps")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Warmup steps: {config['warmup_steps']}")
    print(f"Patience (early stopping): {config['patience']}")
    
    # Estimate training time
    if torch.cuda.is_available():
        print(f"\nEstimated training time: ~{(total_steps * 0.015):.1f} seconds (~{total_steps * 0.015 / 60:.1f} minutes)")
    
    # Start training
    print(f"\n{'='*60}")
    print(f"  STARTING TRAINING...")
    print(f"{'='*60}\n")
    
    stats = trainer.train()
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation loss: {stats['best_val_loss']:.4f}")
    if stats['train_losses']:
        print(f"Final train loss: {stats['train_losses'][-1]:.4f}")
        print(f"Final val loss: {stats['val_losses'][-1]:.4f}")
    print(f"Results saved to: {output_dir}")
    
    # Quick evaluation
    print(f"\n{'='*60}")
    print(f"  QUICK EVALUATION")
    print(f"{'='*60}")
    
    try:
        metrics = trainer.evaluate()
        print(f"Directional accuracy: {metrics.get('directional_accuracy', 0):.4f}")
        print(f"MAE: {metrics.get('mae', 0):.6f}")
        print(f"RMSE: {metrics.get('rmse', 0):.6f}")
    except Exception as e:
        print(f"Evaluation skipped: {e}")


if __name__ == "__main__":
    main()