"""
Evaluation metrics for cryptocurrency forecasting - M6
"""
import torch
import numpy as np
from typing import Dict, List, Optional


def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return torch.mean(torch.abs(predictions - targets)).item()


def root_mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def directional_accuracy(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0) -> float:
    """Directional accuracy (up/down prediction)."""
    pred_direction = (predictions > threshold).float()
    true_direction = (targets > threshold).float()
    return (pred_direction == true_direction).float().mean().item()


def precision_recall_f1(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0) -> tuple:
    """Compute precision, recall, and F1 for direction prediction."""
    pred_binary = (predictions > threshold).float()
    true_binary = (targets > threshold).float()
    
    true_positives = (pred_binary * true_binary).sum().item()
    predicted_positives = pred_binary.sum().item()
    actual_positives = true_binary.sum().item()
    
    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (actual_positives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f1


def roc_auc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """ROC-AUC for binary classification."""
    sorted_indices = torch.argsort(predictions.view(-1), descending=True)
    sorted_targets = targets.view(-1)[sorted_indices]
    
    total_pos = sorted_targets.sum()
    total_neg = len(sorted_targets) - total_pos
    
    tp_cumsum = torch.cumsum(sorted_targets, dim=0)
    fp_cumsum = torch.cumsum(1 - sorted_targets, dim=0)
    
    tpr = tp_cumsum / (total_pos + 1e-8)
    fpr = fp_cumsum / (total_neg + 1e-8)
    
    auc = torch.trapz(tpr, fpr).item()
    return auc


def quantile_loss(predictions: torch.Tensor, targets: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    """Pinball loss for quantile regression."""
    errors = targets.unsqueeze(-1) - predictions
    
    losses = []
    for q in quantiles:
        q_tensor = torch.tensor(q, device=predictions.device)
        loss = torch.max(q_tensor * errors, (q_tensor - 1) * errors)
        losses.append(loss)
    
    return torch.mean(torch.stack(losses, dim=-1))


def evaluate_predictions(predictions_list: List[Dict], targets_list: List[Dict], quantiles: List[float] = [0.1, 0.5, 0.9]) -> Dict:
    """Evaluate all predictions and return metrics."""
    all_pred_returns = torch.cat([p['returns'] for p in predictions_list], dim=0)
    all_pred_directions = torch.cat([p['direction'] for p in predictions_list], dim=0)
    
    all_targets_returns = torch.cat([t['returns'] for t in targets_list], dim=0)
    all_targets_directions = torch.cat([t['directions'] for t in targets_list], dim=0)
    
    metrics = {}
    
    metrics['mae'] = mean_absolute_error(all_pred_returns, all_targets_returns)
    metrics['rmse'] = root_mean_squared_error(all_pred_returns, all_targets_returns)
    metrics['directional_accuracy'] = directional_accuracy(all_pred_directions, all_targets_directions)
    
    precision, recall, f1 = precision_recall_f1(all_pred_directions, all_targets_directions)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    metrics['auc'] = roc_auc(all_pred_directions, all_targets_directions)
    
    if 'quantiles' in predictions_list[0]:
        all_pred_quantiles = torch.cat([p['quantiles'] for p in predictions_list], dim=0)
        metrics['quantile_loss'] = quantile_loss(all_pred_quantiles, all_targets_returns, quantiles).item()
    
    return metrics


if __name__ == "__main__":
    batch_size, num_horizons = 16, 8
    
    predictions = {
        'returns': torch.randn(batch_size, num_horizons),
        'direction': torch.rand(batch_size, num_horizons),
    }
    
    targets = {
        'returns': torch.randn(batch_size, num_horizons),
        'directions': torch.randint(0, 2, (batch_size, num_horizons)).float(),
        'valid_mask': torch.ones(batch_size, num_horizons, dtype=torch.bool)
    }
    
    metrics = evaluate_predictions([predictions], [targets])
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")