"""
Checkpoint management for CryptoFormer - M5
"""
import torch
from pathlib import Path
from typing import Optional, Dict, Union
from torch.optim import Optimizer
import time


class CheckpointManager:
    """Manages model checkpoints with versioning."""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 10,
        keep_best_only: bool = False
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.keep_best_only = keep_best_only
        self.best_val_loss = float('inf')
        self.checkpoint_history = []
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[Union[Optimizer, dict]] = None,
        epoch: int = 0,
        train_loss: float = 0.0,
        val_loss: float = 0.0,
        config: Optional[Dict] = None
    ) -> Path:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer and isinstance(optimizer, Optimizer) else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'timestamp': time.time(),
            'config': config or {}
        }
        
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        
        self.checkpoint_history.append({
            'epoch': epoch,
            'path': str(path),
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model at epoch {epoch}: val_loss={val_loss:.4f}")
        
        if len(self.checkpoint_history) > self.max_checkpoints:
            oldest = self.checkpoint_history.pop(0)
            old_path = Path(oldest['path'])
            if old_path.exists():
                old_path.unlink()
        
        return path
    
    def load(
        self,
        path: Path,
        model: torch.nn.Module,
        optimizer: Optional[Optimizer] = None
    ) -> Dict:
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint


if __name__ == "__main__":
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        for epoch in range(3):
            path = manager.save(model, optimizer, epoch, 1.0 - epoch * 0.1, 1.2 - epoch * 0.1)
            print(f"Saved: {path}")
        
        print(f"Best val loss: {manager.best_val_loss}")