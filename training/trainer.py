"""
Training loop and utilities for CryptoFormer - M5
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import json
import time
from pathlib import Path
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestration for CryptoFormer."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: dict = None,
        device: torch.device = None,
        output_dir: str = "cryptocurrency/experiments"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or {}
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        self.epochs = config.get('epochs', 50)
        self.lr = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95)
        )
        
        warmup_steps = config.get('warmup_steps', 100)
        total_steps = len(self.train_loader) * self.epochs
        
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=warmup_steps
        ) if warmup_steps > 0 else None
        
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.lr * 0.01
        )
        
        if warmup_scheduler is not None:
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        else:
            self.scheduler = cosine_scheduler
        
        self.best_val_loss = float('inf')
        self.patience = config.get('patience', 10)
        self.patience_counter = 0


    def train(self) -> Dict:
        """Execute full training loop."""
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Using mixed precision: {self.use_amp}")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(epoch)
            train_losses.append(train_loss)
            
            val_loss = self._validate_epoch(epoch)
            val_losses.append(val_loss)
            
            self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                       f"Train: {train_loss:.4f}, "
                       f"Val: {val_loss:.4f}, "
                       f"LR: {current_lr:.2e}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_best_model()
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            x = batch['input'].to(self.device)
            asset_ids = batch['asset'].to(self.device)
            targets = {
                'returns': batch['returns'].to(self.device),
                'directions': batch['directions'].to(self.device),
                'valid_mask': batch['valid_mask'].to(self.device)
            }
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                predictions, loss_dict = self.model(x, asset_ids, targets=targets)
                loss = loss_dict['total_loss']
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            x = batch['input'].to(self.device)
            asset_ids = batch['asset'].to(self.device)
            targets = {
                'returns': batch['returns'].to(self.device),
                'directions': batch['directions'].to(self.device),
                'valid_mask': batch['valid_mask'].to(self.device)
            }
            
            predictions, loss_dict = self.model(x, asset_ids, targets=targets)
            total_loss += loss_dict['total_loss'].item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _save_best_model(self):
        path = self.output_dir / "best_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss
        }, path)


if __name__ == "__main__":
    print("Trainer module loaded")