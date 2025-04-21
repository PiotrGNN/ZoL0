"""
strategy_optimizer.py
---------------------
Moduł analizujący wyniki strategii handlowych i proponujący modyfikacje parametrów.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.optimize import minimize

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)

def optimize_portfolio(returns: pd.DataFrame) -> Dict[str, Any]:
    """
    Optymalizuje portfel używając podejścia Markowitza.
    
    Args:
        returns: DataFrame ze zwrotami aktywów
        
    Returns:
        Dict z wynikami optymalizacji
    """
    # Obliczenie podstawowych statystyk
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    n_assets = len(returns.columns)
    
    # Funkcje pomocnicze
    def portfolio_performance(weights):
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = port_return / port_std if port_std > 0 else 0
        return port_return, port_std, sharpe

    def negative_sharpe(weights):
        return -portfolio_performance(weights)[2]
    
    # Ograniczenia
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Optymalizacja
    initial_weights = np.array([1.0/n_assets] * n_assets)
    result = minimize(negative_sharpe, 
                     initial_weights,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)
    
    if not result.success:
        raise ValueError("Optymalizacja portfela nie powiodła się")
        
    # Obliczenie finalnych metryk
    opt_weights = result.x
    port_return, port_std, sharpe = portfolio_performance(opt_weights)
    
    return {
        "weights": opt_weights,
        "expected_return": port_return,
        "risk": port_std,
        "sharpe_ratio": sharpe
    }