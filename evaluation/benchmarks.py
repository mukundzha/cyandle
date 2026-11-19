"""
Baseline and benchmark models for evaluation - M6
"""
import numpy as np
import torch
from typing import Dict, List, Optional


class ZeroReturnBaseline:
    """Baseline that always predicts zero return."""
    
    def predict_returns(self, historical_data: np.ndarray) -> np.ndarray:
        return np.zeros(historical_data.shape[0])
    
    def predict_directions(self, historical_data: np.ndarray) -> np.ndarray:
        return np.full(historical_data.shape[0], 0.5)


class HistoricalMeanBaseline:
    """Baseline using historical mean return."""
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.mean_return = 0.0
        self.std_return = 1.0
        self.fitted = False
    
    def fit(self, returns: np.ndarray):
        self.mean_return = np.mean(returns)
        self.std_return = np.std(returns)
        self.fitted = True
    
    def predict_returns(self, historical_data: np.ndarray) -> np.ndarray:
        return np.full(historical_data.shape[0], self.mean_return)
    
    def predict_directions(self, historical_data: np.ndarray) -> np.ndarray:
        prob = 1 / (1 + np.exp(-self.mean_return / (self.std_return + 1e-8)))
        return np.full(historical_data.shape[0], prob)


class MomentumBaseline:
    """Simple momentum-based baseline."""
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
    
    def predict_returns(self, historical_data: np.ndarray) -> np.ndarray:
        n_samples = historical_data.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(self.lookback, n_samples):
            recent_returns = np.diff(historical_data[i-self.lookback:i])
            predictions[i] = np.mean(recent_returns)
        
        return predictions
    
    def predict_directions(self, historical_data: np.ndarray) -> np.ndarray:
        returns = self.predict_returns(historical_data)
        prob = 1 / (1 + np.exp(-returns))
        return prob


def run_all_benchmarks(
    historical_prices: np.ndarray,
    test_returns: np.ndarray,
    test_directions: np.ndarray
) -> Dict:
    """Run baseline benchmarks and return their performance."""
    results = {}
    
    # Zero return baseline
    zero_baseline = ZeroReturnBaseline()
    zero_returns = zero_baseline.predict_returns(test_returns.reshape(-1, 1)).flatten()
    results['zero_return'] = {
        'mae': np.mean(np.abs(zero_returns - test_returns)),
        'rmse': np.sqrt(np.mean((zero_returns - test_returns) ** 2)),
        'directional_accuracy': 0.5
    }
    
    # Historical mean
    mean_baseline = HistoricalMeanBaseline()
    mean_baseline.fit(test_returns)
    mean_returns = mean_baseline.predict_returns(np.zeros_like(test_returns)).flatten()
    results['historical_mean'] = {
        'mae': np.mean(np.abs(mean_returns - test_returns)),
        'rmse': np.sqrt(np.mean((mean_returns - test_returns) ** 2)),
        'directional_accuracy': np.mean((mean_returns > 0) == (test_returns > 0))
    }
    
    # Momentum
    mom_baseline = MomentumBaseline(lookback=20)
    mom_returns = mom_baseline.predict_returns(historical_prices)
    valid_len = len(mom_returns)
    results['momentum'] = {
        'mae': np.mean(np.abs(mom_returns[-valid_len:] - test_returns[-valid_len:])),
        'rmse': np.sqrt(np.mean((mom_returns[-valid_len:] - test_returns[-valid_len:]) ** 2)),
        'directional_accuracy': np.mean((mom_returns[-valid_len:] > 0) == (test_returns[-valid_len:] > 0))
    }
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    prices = 50000 + np.cumsum(np.random.randn(n) * 50)
    returns = np.diff(prices) / prices[:-1]
    directions = (returns > 0).astype(float)
    
    benchmarks = run_all_benchmarks(prices[:-1], returns, directions)
    
    print("Benchmark results:")
    for name, metrics in benchmarks.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")