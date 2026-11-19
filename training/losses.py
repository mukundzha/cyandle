"""
Loss functions for multi-task cryptocurrency forecasting - M5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class QuantileLoss(nn.Module):
    """Quantile (pinball) loss for quantile regression."""
    
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size, num_horizons, num_quantiles = predictions.shape
        targets_expanded = targets.unsqueeze(-1).expand(-1, -1, num_quantiles)
        errors = targets_expanded - predictions
        
        losses = []
        for q in self.quantiles:
            q_tensor = torch.tensor(q, device=predictions.device)
            loss = torch.max(q_tensor * errors, (q_tensor - 1) * errors)
            losses.append(loss)
        
        return torch.mean(torch.stack(losses, dim=-1))


class MultiTaskLoss(nn.Module):
    """Combined multi-task loss for cryptocurrency prediction."""
    
    def __init__(
        self,
        return_weight: float = 1.0,
        direction_weight: float = 1.0,
        quantile_weight: float = 0.5,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        delta_huber: float = 1.0
    ):
        super().__init__()
        self.return_weight = return_weight
        self.direction_weight = direction_weight
        self.quantile_weight = quantile_weight
        self.huber = nn.HuberLoss(delta=delta_huber)
        self.quantile_loss = QuantileLoss(quantiles)
        self.quantiles = quantiles
    
    def forward(self, predictions: dict, targets: dict) -> dict:
        return_loss = self.huber(predictions['returns'], targets['returns'])
        
        logits = torch.logit(predictions['direction'].clamp(min=1e-7, max=1-1e-7))
        direction_loss = F.binary_cross_entropy_with_logits(logits, targets['directions'])
        
        quantile_loss = self.quantile_loss(predictions['quantiles'], targets['returns'])
        
        total_loss = (
            self.return_weight * return_loss +
            self.direction_weight * direction_loss +
            self.quantile_weight * quantile_loss
        )
        
        return {
            'total_loss': total_loss,
            'return_loss': return_loss,
            'direction_loss': direction_loss,
            'quantile_loss': quantile_loss
        }