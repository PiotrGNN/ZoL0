
"""
portfolio_risk.py
---------------
Moduł do zarządzania ryzykiem portfela.
"""

import logging
import os
import random
from typing import Dict, List, Any, Optional, Tuple

# Konfiguracja logowania
logger = logging.getLogger("portfolio_risk")
if not logger.handlers:
    log_dir = "logs"
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

    def __init__(
        self,
        max_portfolio_risk: float = 0.05,
        max_position_size_pct: float = 0.2,
        max_drawdown_pct: float = 0.1,
        correlation_threshold: float = 0.7
    ):
        """
        Inicjalizuje zarządcę ryzyka portfela.

        Parameters:
            max_portfolio_risk (float): Maksymalne ryzyko portfela (0.05 = 5%).
            max_position_size_pct (float): Maksymalny rozmiar pozycji jako procent portfela (0.2 = 20%).
            max_drawdown_pct (float): Maksymalny dozwolony drawdown (0.1 = 10%).
            correlation_threshold (float): Próg korelacji dla dywersyfikacji.
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size_pct = max_position_size_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.correlation_threshold = correlation_threshold

        self.portfolio = {}
        self.current_drawdown = 0.0
        self.peak_value = 0.0

        logger.info(f"Zainicjalizowano zarządcę ryzyka portfela. Max ryzyko: {max_portfolio_risk}, "
                    f"Max rozmiar pozycji: {max_position_size_pct}, Max drawdown: {max_drawdown_pct}")

    def calculate_portfolio_risk(self) -> float:
        """
        Oblicza całkowite ryzyko portfela.

        Returns:
            float: Całkowite ryzyko portfela.
        """
        # Implementacja obliczania ryzyka portfela
        # W szablonie zwracamy dummy wartość
        return random.uniform(0, self.max_portfolio_risk)

    def calculate_position_size(self, symbol: str, risk_per_trade: float, stop_loss_pct: float) -> float:
        """
        Oblicza optymalny rozmiar pozycji.

        Parameters:
            symbol (str): Symbol pary handlowej.
            risk_per_trade (float): Ryzyko na transakcję jako procent kapitału.
            stop_loss_pct (float): Procent stop loss.

        Returns:
            float: Optymalny rozmiar pozycji.
        """
        # Implementacja obliczania rozmiaru pozycji
        # W szablonie zwracamy dummy wartość
        total_capital = self.get_total_capital()
        max_position = total_capital * self.max_position_size_pct

        if stop_loss_pct == 0:
            return 0

        risk_based_position = (total_capital * risk_per_trade) / stop_loss_pct
        return min(risk_based_position, max_position)

    def update_portfolio(self, portfolio_data: Dict[str, Any]) -> None:
        """
        Aktualizuje dane portfela.

        Parameters:
            portfolio_data (Dict[str, Any]): Dane portfela.
        """
        self.portfolio = portfolio_data

        # Aktualizacja drawdown
        total_value = self.get_total_capital()
        if total_value > self.peak_value:
            self.peak_value = total_value

        if self.peak_value > 0:
            self.current_drawdown = 1 - (total_value / self.peak_value)
            if self.current_drawdown > self.max_drawdown_pct:
                logger.warning(f"Przekroczono maksymalny drawdown: {self.current_drawdown:.2%} > {self.max_drawdown_pct:.2%}")

    def get_total_capital(self) -> float:
        """
        Zwraca całkowity kapitał portfela.

        Returns:
            float: Całkowity kapitał.
        """
        # W szablonie zwracamy dummy wartość
        if not self.portfolio:
            return 10000.0

        try:
            # Próba wyciągnięcia danych z faktycznego portfela
            total = 0.0
            if 'balances' in self.portfolio:
                for coin, data in self.portfolio['balances'].items():
                    if isinstance(data, dict) and 'equity' in data:
                        # Dla uproszczenia zakładamy, że wszystkie wartości są w USD
                        total += float(data['equity'])
            return total if total > 0 else 10000.0
        except Exception as e:
            logger.error(f"Błąd podczas obliczania całkowitego kapitału: {e}")
            return 10000.0

    def check_risk_limits(self) -> Dict[str, Any]:
        """
        Sprawdza limity ryzyka portfela.

        Returns:
            Dict[str, Any]: Status limitów ryzyka.
        """
        portfolio_risk = self.calculate_portfolio_risk()
        risk_status = {
            "portfolio_risk": portfolio_risk,
            "portfolio_risk_pct": f"{portfolio_risk:.2%}",
            "max_portfolio_risk": self.max_portfolio_risk,
            "max_portfolio_risk_pct": f"{self.max_portfolio_risk:.2%}",
            "current_drawdown": self.current_drawdown,
            "current_drawdown_pct": f"{self.current_drawdown:.2%}",
            "max_drawdown": self.max_drawdown_pct,
            "max_drawdown_pct": f"{self.max_drawdown_pct:.2%}",
            "risk_exceeded": portfolio_risk > self.max_portfolio_risk or self.current_drawdown > self.max_drawdown_pct
        }

        if risk_status["risk_exceeded"]:
            logger.warning(f"Przekroczono limity ryzyka: {risk_status}")

        return risk_status

    def calculate_correlation(self, symbols: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Oblicza korelację między parami handlowymi.

        Parameters:
            symbols (List[str]): Lista symboli par handlowych.

        Returns:
            Dict[Tuple[str, str], float]: Słownik korelacji.
        """
        # Implementacja obliczania korelacji
        # W szablonie zwracamy dummy wartości
        correlations = {}
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                correlations[(symbol1, symbol2)] = random.uniform(-1, 1)

        return correlations

    def get_diversification_recommendations(self) -> List[Dict[str, Any]]:
        """
        Zwraca rekomendacje dotyczące dywersyfikacji portfela.

        Returns:
            List[Dict[str, Any]]: Lista rekomendacji.
        """
        # Implementacja rekomendacji dywersyfikacji
        # W szablonie zwracamy dummy dane
        return [
            {
                "message": "Zwiększ ekspozycję na rynki o niskiej korelacji",
                "severity": "medium",
                "action": "diversify"
            }
        ]


# Singleton instancja dla łatwego dostępu z różnych modułów
portfolio_risk_manager = PortfolioRiskManager()


def calculate_position_size(symbol: str, risk_per_trade: float, stop_loss_pct: float) -> float:
    """
    Funkcja pomocnicza do obliczania rozmiaru pozycji.

    Parameters:
        symbol (str): Symbol pary handlowej.
        risk_per_trade (float): Ryzyko na transakcję jako procent kapitału.
        stop_loss_pct (float): Procent stop loss.

    Returns:
        float: Optymalny rozmiar pozycji.
    """
    return portfolio_risk_manager.calculate_position_size(symbol, risk_per_trade, stop_loss_pct)
