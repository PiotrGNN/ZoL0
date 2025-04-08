"""
portfolio_risk.py
-----------------
Zaawansowany moduł zarządzania ryzykiem portfela inwestycyjnego.

Funkcjonalności:
- Obliczanie współczynników Sharpe'a i Sortino.
- Analiza korelacji aktywów i optymalizacja dywersyfikacji.
- Monitorowanie wartości narażonej na ryzyko (Value at Risk - VaR).
- Dynamiczna alokacja kapitału na podstawie zmienności i ryzyka rynkowego.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class PortfolioRiskManager:
    """
    Klasa do zarządzania ryzykiem portfela. Oblicza kluczowe wskaźniki ryzyka i dostosowuje alokację kapitału.
    """

    def __init__(self, confidence_level: float = 0.95, risk_free_rate: float = 0.01):
        """
        Inicjalizuje menedżera ryzyka portfela.

        Parameters:
            confidence_level (float): Poziom ufności dla obliczeń VaR (np. 95%).
            risk_free_rate (float): Stopa wolna od ryzyka dla obliczeń Sharpe'a i Sortino.
        """
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Oblicza współczynnik Sharpe'a.

        Parameters:
            returns (pd.Series): Dzienna stopa zwrotu portfela.

        Returns:
            float: Wartość współczynnika Sharpe'a.
        """
        returns = returns.dropna()
        if returns.empty:
            logging.warning("Brak danych do obliczenia Sharpe Ratio.")
            return 0.0

        excess_return = returns.mean() - self.risk_free_rate
        std_dev = returns.std()
        sharpe_ratio = excess_return / std_dev if std_dev > 0 else 0.0
        logging.info("Sharpe Ratio: %.4f", sharpe_ratio)
        return sharpe_ratio

    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """
        Oblicza współczynnik Sortino.

        Parameters:
            returns (pd.Series): Dzienna stopa zwrotu portfela.

        Returns:
            float: Wartość współczynnika Sortino.
        """
        returns = returns.dropna()
        if returns.empty:
            logging.warning("Brak danych do obliczenia Sortino Ratio.")
            return 0.0

        excess_return = returns.mean() - self.risk_free_rate
        downside_deviation = returns[returns < self.risk_free_rate].std()
        sortino_ratio = (
            excess_return / downside_deviation if downside_deviation > 0 else 0.0
        )
        logging.info("Sortino Ratio: %.4f", sortino_ratio)
        return sortino_ratio

    def calculate_var(self, returns: pd.Series, portfolio_value: float) -> float:
        """
        Oblicza wartość narażoną na ryzyko (Value at Risk - VaR) metodą parametryczną.

        Parameters:
            returns (pd.Series): Dzienna stopa zwrotu portfela.
            portfolio_value (float): Aktualna wartość portfela.

        Returns:
            float: Wartość VaR (ujemna wartość oznacza stratę).
        """
        returns = returns.dropna()
        if returns.empty:
            logging.warning("Brak danych do obliczenia VaR.")
            return 0.0

        std_dev = returns.std()
        z_score = norm.ppf(1 - self.confidence_level)
        var = portfolio_value * (z_score * std_dev)

        logging.info(
            "Value at Risk (%.1f%% confidence): %.2f", self.confidence_level * 100, var
        )
        return var

    def calculate_correlation_matrix(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Oblicza macierz korelacji między aktywami w portfelu.

        Parameters:
            price_data (pd.DataFrame): Dane cenowe aktywów (w kolumnach).

        Returns:
            pd.DataFrame: Macierz korelacji.
        """
        if not isinstance(price_data, pd.DataFrame) or price_data.empty:
            raise ValueError("price_data musi być niepustym DataFrame!")

        returns = price_data.pct_change().dropna()
        if returns.empty:
            logging.warning(
                "Brak wystarczających danych do obliczenia macierzy korelacji."
            )
            return pd.DataFrame()

        correlation_matrix = returns.corr()
        logging.info("Macierz korelacji:\n%s", correlation_matrix)
        return correlation_matrix

    def optimal_allocation(
        self, asset_volatility: pd.Series, target_risk: float = 0.02
    ) -> pd.Series:
        """
        Oblicza optymalną alokację kapitału w portfelu na podstawie zmienności aktywów.

        Parameters:
            asset_volatility (pd.Series): Zmienność każdego aktywa.
            target_risk (float): Poziom docelowego ryzyka portfela.

        Returns:
            pd.Series: Proporcjonalna alokacja kapitału w poszczególne aktywa.
        """
        if not isinstance(asset_volatility, pd.Series) or asset_volatility.empty:
            raise ValueError("asset_volatility musi być niepustym Series!")

        asset_volatility = asset_volatility.replace(
            0, np.nan
        ).dropna()  # Zapobiegaj dzieleniu przez 0

        if asset_volatility.empty:
            logging.warning("Brak ważnych danych do obliczenia alokacji kapitału.")
            return pd.Series()

        inv_volatility = 1 / asset_volatility
        weights = inv_volatility / inv_volatility.sum()
        scaled_weights = weights * (
            target_risk / np.sqrt(np.sum(weights**2 * asset_volatility**2))
        )

        logging.info("Optymalna alokacja kapitału:\n%s", scaled_weights)
        return scaled_weights


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=252)
        price_data = pd.DataFrame(
            {
                "Asset_A": np.cumprod(1 + np.random.normal(0, 0.02, 252)),
                "Asset_B": np.cumprod(1 + np.random.normal(0, 0.015, 252)),
                "Asset_C": np.cumprod(1 + np.random.normal(0, 0.01, 252)),
            },
            index=dates,
        )

        portfolio_manager = PortfolioRiskManager(
            confidence_level=0.95, risk_free_rate=0.01
        )

        returns = price_data.pct_change().dropna()
        portfolio_value = 100000

        sharpe_ratio = portfolio_manager.calculate_sharpe_ratio(returns.mean(axis=1))
        sortino_ratio = portfolio_manager.calculate_sortino_ratio(returns.mean(axis=1))
        var = portfolio_manager.calculate_var(returns.mean(axis=1), portfolio_value)
        correlation_matrix = portfolio_manager.calculate_correlation_matrix(price_data)

        asset_volatility = returns.std()
        allocation = portfolio_manager.optimal_allocation(
            asset_volatility, target_risk=0.02
        )

        logging.info("Finalna analiza ryzyka portfela zakończona sukcesem.")
    except Exception as e:
        logging.error("Błąd w PortfolioRiskManager: %s", e)
        raise
"""
portfolio_risk.py
----------------
Moduł do oceny i zarządzania ryzykiem portfela.
"""

import logging
from typing import Dict, Any, List

# Konfiguracja logowania
logger = logging.getLogger("portfolio_risk")
if not logger.handlers:
    log_dir = "logs"
    import os
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "portfolio_risk.log"))
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

class PortfolioRiskManager:
    """
    Klasa do zarządzania ryzykiem portfela.
    """

    def __init__(self, max_risk_per_trade: float = 0.02, max_open_positions: int = 5):
        """
        Inicjalizacja menedżera ryzyka portfela.

        Parameters:
            max_risk_per_trade (float): Maksymalny procent kapitału ryzyka na transakcję
            max_open_positions (int): Maksymalna liczba otwartych pozycji
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_open_positions = max_open_positions
        self.current_positions = []
        logger.info(f"Inicjalizacja menedżera ryzyka portfela: max_risk={max_risk_per_trade}, max_positions={max_open_positions}")

    def calculate_position_size(self, account_balance: float, risk_per_trade: float = None, 
                                stop_loss_pct: float = 0.02) -> float:
        """
        Oblicza rozmiar pozycji na podstawie ryzyka.

        Parameters:
            account_balance (float): Saldo konta
            risk_per_trade (float): Procent kapitału do zaryzykowania (jeśli None, używa max_risk_per_trade)
            stop_loss_pct (float): Procent stop loss

        Returns:
            float: Zalecany rozmiar pozycji
        """
        if risk_per_trade is None:
            risk_per_trade = self.max_risk_per_trade
            
        risk_amount = account_balance * risk_per_trade
        position_size = risk_amount / stop_loss_pct
        
        logger.debug(f"Obliczono rozmiar pozycji: {position_size} dla salda {account_balance}")
        return position_size

    def evaluate_portfolio_risk(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ocenia ryzyko całego portfela.

        Parameters:
            portfolio (Dict[str, Any]): Dane portfela

        Returns:
            Dict[str, Any]: Metryki ryzyka portfela
        """
        # Implementacja stub - zwraca podstawowe metryki ryzyka
        risk_metrics = {
            "total_risk": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "risk_level": "low"
        }
        
        logger.info("Ocena ryzyka portfela")
        return risk_metrics

    def can_open_new_position(self) -> bool:
        """
        Sprawdza, czy można otworzyć nową pozycję.

        Returns:
            bool: True jeśli można otworzyć nową pozycję, False w przeciwnym przypadku
        """
        return len(self.current_positions) < self.max_open_positions
