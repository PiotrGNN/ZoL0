"""
risk_metrics.py
-------------
Moduł odpowiedzialny za obliczanie metryk ryzyka dla portfela.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

def calculate_risk_metrics(returns: pd.Series, portfolio_value: float, risk_free_rate: float = 0.0) -> Dict[str, Any]:
    """
    Oblicza podstawowe metryki ryzyka dla portfela.

    Parameters:
        returns (pd.Series): Szereg zwrotów z portfela
        portfolio_value (float): Aktualna wartość portfela
        risk_free_rate (float): Stopa wolna od ryzyka (domyślnie 0)

    Returns:
        Dict[str, Any]: Słownik zawierający metryki ryzyka
    """
    try:
        # Upewnij się, że mamy wystarczająco danych
        if len(returns) < 2:
            return {
                "error": "Niewystarczająca ilość danych do obliczenia metryk"
            }

        # Oblicz podstawowe statystyki
        mean_return = returns.mean()
        std_dev = returns.std()
        
        # Annualizacja (zakładamy dane dzienne)
        annual_return = (1 + mean_return) ** 252 - 1
        annual_std = std_dev * np.sqrt(252)

        # Sharpe Ratio
        excess_returns = returns - risk_free_rate
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0

        # Sortino Ratio (używa tylko negatywnych zwrotów)
        downside_returns = returns[returns < risk_free_rate]
        sortino_ratio = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() != 0 else 0

        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()

        # Value at Risk (95% poziom ufności)
        var_95 = np.percentile(returns, 5)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()

        # Win Rate
        win_rate = len(returns[returns > 0]) / len(returns)

        metrics = {
            "portfolio_value": portfolio_value,
            "annual_return": annual_return,
            "annual_volatility": annual_std,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "win_rate": win_rate,
            "risk_level": _determine_risk_level(sharpe_ratio, max_drawdown, annual_std)
        }

        logging.info("Obliczono metryki ryzyka: %s", metrics)
        return metrics

    except Exception as e:
        logging.error("Błąd podczas obliczania metryk ryzyka: %s", e)
        return {"error": str(e)}

def _determine_risk_level(sharpe_ratio: float, max_drawdown: float, volatility: float) -> str:
    """
    Określa poziom ryzyka na podstawie kluczowych metryk.
    """
    # Progi dla poszczególnych metryk
    SHARPE_THRESHOLDS = {"low": 1.5, "medium": 0.5}
    DRAWDOWN_THRESHOLDS = {"high": -0.2, "medium": -0.1}
    VOLATILITY_THRESHOLDS = {"high": 0.25, "medium": 0.15}

    # Punktacja ryzyka
    risk_score = 0

    # Ocena Sharpe Ratio
    if sharpe_ratio >= SHARPE_THRESHOLDS["low"]:
        risk_score += 1
    elif sharpe_ratio >= SHARPE_THRESHOLDS["medium"]:
        risk_score += 2
    else:
        risk_score += 3

    # Ocena Max Drawdown
    if max_drawdown <= DRAWDOWN_THRESHOLDS["high"]:
        risk_score += 3
    elif max_drawdown <= DRAWDOWN_THRESHOLDS["medium"]:
        risk_score += 2
    else:
        risk_score += 1

    # Ocena zmienności
    if volatility >= VOLATILITY_THRESHOLDS["high"]:
        risk_score += 3
    elif volatility >= VOLATILITY_THRESHOLDS["medium"]:
        risk_score += 2
    else:
        risk_score += 1

    # Określenie finalnego poziomu ryzyka
    avg_score = risk_score / 3
    if avg_score <= 1.5:
        return "low"
    elif avg_score <= 2.5:
        return "medium"
    else:
        return "high"