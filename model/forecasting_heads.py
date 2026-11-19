"""
Multi-horizon forecasting heads for CryptoFormer.

Provides three prediction heads for multi-task learning:
1. Direction head: P(returns > 0) via sigmoid
2. Return head: Expected cumulative log returns
3. Quantile head: Uncertainty estimation via quantiles

Uses pinball loss for quantile regression to ensure valid probabilistic forecasts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DirectionHead(nn.Module):
    """
    Predicts direction probability using sigmoid.
    
    Output: P(r_h > 0) - probability that return at horizon h is positive.
    """
    
    def __init__(self, hidden_dim: int, num_horizons: int):
        super().__init__()
        self.num_horizons = num_horizons
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_horizons),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ReturnHead(nn.Module):
    """
    Predicts expected cumulative log returns.
    
    Output: E[r_h] - expected sum of log returns from t to t+h.
    """
    
    def __init__(self, hidden_dim: int, num_horizons: int):
        super().__init__()
        self.num_horizons = num_horizons
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_horizons)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class QuantileHead(nn.Module):
    """
    Predicts return quantiles for uncertainty estimation.
    
    Output: [Q_0.1(r_h), Q_0.5(r_h), Q_0.9(r_h)] - 10th, 50th, 90th percentiles.
    
    Enables prediction interval coverage assessment and risk management.
    """
    
    def __init__(self, hidden_dim: int, num_horizons: int, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.num_horizons = num_horizons
        self.quantiles = quantiles
        num_quantiles = len(quantiles)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_horizons * num_quantiles)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        out = self.fc(x)
        out = out.view(batch_size, self.num_horizons, len(self.quantiles))
        return out


class MultiHeadPrediction(nn.Module):
    """Multi-task prediction head combining all objectives."""
    
    def __init__(self, hidden_dim: int, num_horizons: int = 8, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        
        self.num_horizons = num_horizons
        self.quantiles = quantiles
        
        self.direction_head = DirectionHead(hidden_dim, num_horizons)
        self.return_head = ReturnHead(hidden_dim, num_horizons)
        self.quantile_head = QuantileHead(hidden_dim, num_horizons, quantiles)
        
        self.cls_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> dict:
        cls_token = x.mean(dim=1)
        cls_token = self.cls_projection(cls_token)
        
        return {
            'direction': self.direction_head(cls_token),
            'returns': self.return_head(cls_token),
            'quantiles': self.quantile_head(cls_token)
        }


def quantile_loss(predictions: torch.Tensor, targets: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    """
    Compute pinball loss for quantile regression.
    
    The pinball loss is defined as:
    L_q(y, ŷ) = max(q(y - ŷ), (q-1)(ŷ - y))
    
    This encourages the predicted quantiles to match the true quantile function.
    """
    batch_size, num_horizons, num_quantiles = predictions.shape
    targets_expanded = targets.unsqueeze(-1).expand(-1, -1, num_quantiles)
    errors = targets_expanded - predictions
    
    losses = []
    for i, q in enumerate(quantiles):
        q_tensor = torch.tensor(q, device=predictions.device)
        loss = torch.max(q_tensor * errors, (q_tensor - 1) * errors)
        losses.append(loss)
    
    return torch.mean(torch.stack(losses, dim=-1))


class ForecastLoss(nn.Module):
    """Combined multi-task loss for forecasting."""
    
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
        self.quantile_loss = quantile_loss
        self.quantiles = quantiles
    
    def forward(self, predictions: dict, targets: dict) -> dict:
        return_loss = self.huber(predictions['returns'], targets['returns'])
        
        logits = torch.logit(predictions['direction'].clamp(min=1e-7, max=1-1e-7))
        direction_loss = F.binary_cross_entropy_with_logits(logits, targets['directions'])
        
        q_loss = self.quantile_loss(predictions['quantiles'], targets['returns'], self.quantiles)
        
        total_loss = (
            self.return_weight * return_loss +
            self.direction_weight * direction_loss +
            self.quantile_weight * q_loss
        )
        
        return {
            'total_loss': total_loss,
            'return_loss': return_loss,
            'direction_loss': direction_loss,
            'quantile_loss': q_loss
        }


if __name__ == "__main__":
    batch_size, hidden_dim, num_horizons = 4, 128, 8
    
    head = MultiHeadPrediction(hidden_dim, num_horizons)
    
    x = torch.randn(batch_size, 64, hidden_dim)
    
    predictions = head(x)
    print(f"Direction shape: {predictions['direction'].shape}")
    print(f"Returns shape: {predictions['returns'].shape}")
    print(f"Quantiles shape: {predictions['quantiles'].shape}")
    
    targets = {
        'returns': torch.randn(batch_size, num_horizons),
        'directions': torch.randint(0, 2, (batch_size, num_horizons)).float(),
        'valid_mask': torch.ones(batch_size, num_horizons, dtype=torch.bool)
    }
    
    loss_fn = ForecastLoss()
    loss_dict = loss_fn(predictions, targets)
    
    print(f"\nLoss values:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value.item():.4f}")