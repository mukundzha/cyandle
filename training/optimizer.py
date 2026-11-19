"""
Optimizer utilities and learning rate schedules - M5
"""
import torch
from torch.optim import Optimizer, SGD, Adam, AdamW
from typing import Dict, Optional


def create_optimizer(model: torch.nn.Module, config: Dict) -> Optimizer:
    """Create optimizer from configuration."""
    optimizer_name = config.get('optimizer', 'adamw').lower()
    lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)
    betas = tuple(config.get('betas', [0.9, 0.95]))
    eps = config.get('eps', 1e-8)
    
    if optimizer_name == 'adam':
        return Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer: Optimizer, config: Dict, num_training_steps: int):
    """Create learning rate scheduler from configuration."""
    scheduler_name = config.get('scheduler', 'cosine').lower()
    warmup_steps = config.get('warmup_steps', 100)
    
    if scheduler_name == 'cosine':
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps - warmup_steps, eta_min=config.get('learning_rate', 1e-4) * 0.01
        )
        
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_steps
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
            )
        return cosine_scheduler
    
    elif scheduler_name == 'step':
        step_size = config.get('step_size', 50)
        gamma = config.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)


def get_lr(optimizer: Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


if __name__ == "__main__":
    model = torch.nn.Linear(100, 100)
    
    config = {
        'optimizer': 'adamw',
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'scheduler': 'cosine',
        'warmup_steps': 100
    }
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, num_training_steps=1000)
    
    print(f"Initial LR: {get_lr(optimizer):.2e}")
    
    for i in range(10):
        scheduler.step()
    
    print(f"After 10 steps LR: {get_lr(optimizer):.2e}")