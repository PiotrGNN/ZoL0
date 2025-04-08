"""
simplified_risk_manager.py
--------------------------
Uproszczony moduł zarządzania ryzykiem, kompatybilny zarówno z lokalnym środowiskiem, jak i Replit.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/risk_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimplifiedRiskManager:
    """
    Uproszczony manager ryzyka dla systemu tradingowego.
    """

    def __init__(self, max_risk: float = 0.02, max_position_size: float = 0.2, max_drawdown: float = 0.1):
        """
        Inicjalizacja managera ryzyka.

        Args:
            max_risk: Maksymalne ryzyko na pojedynczą transakcję (wyrażone jako % kapitału)
            max_position_size: Maksymalny rozmiar pojedynczej pozycji (wyrażony jako % kapitału)
            max_drawdown: Maksymalny akceptowalny drawdown (spadek kapitału)
        """
        self.max_risk = max_risk
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.current_drawdown = 0.0
        self.initial_capital = 0.0
        self.current_capital = 0.0
        logger.info(f"Zainicjalizowano SimplifiedRiskManager (max_risk={max_risk}, max_position_size={max_position_size}, max_drawdown={max_drawdown})")

    def set_capital(self, capital: float) -> None:
        """
        Ustawia kapitał początkowy i bieżący.

        Args:
            capital: Wartość kapitału
        """
        if self.initial_capital == 0.0:
            self.initial_capital = capital
        self.current_capital = capital
        self.update_drawdown()

    def update_drawdown(self) -> float:
        """
        Aktualizuje i zwraca bieżący drawdown.

        Returns:
            float: Bieżący drawdown (0.0 - 1.0)
        """
        if self.initial_capital > 0:
            self.current_drawdown = max(0, (self.initial_capital - self.current_capital) / self.initial_capital)
        else:
            self.current_drawdown = 0.0
        return self.current_drawdown

    def can_take_new_positions(self) -> bool:
        """
        Sprawdza, czy można otwierać nowe pozycje w oparciu o bieżący drawdown.

        Returns:
            bool: True jeśli można otwierać nowe pozycje, False w przeciwnym razie
        """
        return self.current_drawdown < self.max_drawdown

    def determine_position_size(self, account_balance: float, stop_loss: float, asset_price: float) -> float:
        """
        Określa rozmiar pozycji na podstawie balansu konta, poziomu stop-loss i ceny aktywa.

        Args:
            account_balance: Bilans konta
            stop_loss: Poziom stop-loss (jako % od ceny wejścia)
            asset_price: Cena aktywa

        Returns:
            float: Zalecany rozmiar pozycji (w jednostkach aktywa)
        """
        if stop_loss <= 0 or asset_price <= 0:
            return 0.0

        # Obliczenie ryzyka w jednostkach waluty
        risk_amount = account_balance * self.max_risk

        # Obliczenie straty na jednostkę, jeśli stop-loss zostanie aktywowany
        loss_per_unit = asset_price * stop_loss

        # Obliczenie ilości jednostek, które można nabyć przy danym poziomie ryzyka
        if loss_per_unit > 0:
            position_size = risk_amount / loss_per_unit
        else:
            position_size = 0.0

        # Upewnienie się, że rozmiar pozycji nie przekracza maksymalnego dopuszczalnego rozmiaru
        max_allowed_size = account_balance * self.max_position_size / asset_price
        position_size = min(position_size, max_allowed_size)

        return position_size

    def calculate_stop_loss(self, entry_price: float, position_type: str, volatility: float = 0.01) -> float:
        """
        Oblicza poziom stop-loss na podstawie ceny wejścia, typu pozycji i zmienności.

        Args:
            entry_price: Cena wejścia
            position_type: Typ pozycji ('long' lub 'short')
            volatility: Zmienność (jako % od ceny wejścia)

        Returns:
            float: Poziom stop-loss
        """
        volatility_factor = max(0.005, volatility)  # Minimum 0.5% zmienności

        if position_type.lower() == 'long':
            stop_loss = entry_price * (1 - 2 * volatility_factor)
        else:  # short
            stop_loss = entry_price * (1 + 2 * volatility_factor)

        return stop_loss

    def calculate_take_profit(self, stop_loss: float, entry_price: float = None, position_type: str = None, risk_reward: float = 2.0) -> float:
        """
        Oblicza poziom take-profit na podstawie poziomu stop-loss i stosunku ryzyka do zysku.

        Args:
            stop_loss: Poziom stop-loss
            entry_price: Cena wejścia (opcjonalne)
            position_type: Typ pozycji ('long' lub 'short') (opcjonalne)
            risk_reward: Stosunek ryzyka do zysku (domyślnie 2.0)

        Returns:
            float: Poziom take-profit
        """
        if entry_price is None or position_type is None:
            return 0.0

        sl_distance = abs(entry_price - stop_loss)
        tp_distance = sl_distance * risk_reward

        if position_type.lower() == 'long':
            take_profit = entry_price + tp_distance
        else:  # short
            take_profit = entry_price - tp_distance

        return take_profit

    def adjust_leverage(self, market_conditions: Dict[str, Any]) -> float:
        """
        Dostosowuje poziom dźwigni w oparciu o warunki rynkowe.

        Args:
            market_conditions: Słownik zawierający informacje o warunkach rynkowych

        Returns:
            float: Zalecany poziom dźwigni
        """
        # Przykładowa logika dostosowywania dźwigni
        # W rzeczywistej implementacji byłaby bardziej złożona logika

        volatility = market_conditions.get('volatility', 0.02)
        trend_strength = market_conditions.get('trend_strength', 0.5)

        base_leverage = 3.0  # Domyślna dźwignia

        # Zmniejsz dźwignię przy dużej zmienności
        if volatility > 0.05:  # Wysoka zmienność
            base_leverage *= 0.5

        # Zwiększ dźwignię przy silnym trendzie
        if trend_strength > 0.7:  # Silny trend
            base_leverage *= 1.2

        # Upewnij się, że dźwignia jest w rozsądnych granicach
        return max(1.0, min(10.0, base_leverage))

# Przykład użycia
if __name__ == "__main__":
    # Inicjalizacja managera ryzyka
    risk_manager = SimplifiedRiskManager(
        max_risk=0.02,  # 2% kapitału na transakcję
        max_position_size=0.2,  # Maksymalnie 20% kapitału w jednej pozycji
        max_drawdown=0.1  # Maksymalny drawdown 10%
    )

    # Przykładowe użycie
    capital = 10000  # 10,000 USD
    risk_manager.set_capital(capital)

    # Sprawdzenie czy można otwierać nowe pozycje
    can_trade = risk_manager.can_take_new_positions()
    print(f"Można otwierać nowe pozycje: {can_trade}")

    # Obliczenie rozmiaru pozycji
    entry_price = 50000  # 50,000 USD za BTC
    stop_loss_pct = 0.02  # 2% stop-loss
    position_size = risk_manager.determine_position_size(capital, stop_loss_pct, entry_price)
    print(f"Zalecany rozmiar pozycji: {position_size:.6f} BTC (${position_size * entry_price:.2f})")

    # Obliczenie poziomów stop-loss i take-profit
    stop_loss = risk_manager.calculate_stop_loss(entry_price, 'long', 0.02)
    take_profit = risk_manager.calculate_take_profit(stop_loss, entry_price, 'long', 2.5)
    print(f"Stop-Loss: ${stop_loss:.2f}, Take-Profit: ${take_profit:.2f}")

    # Dostosowanie dźwigni
    market_conditions = {'volatility': 0.03, 'trend_strength': 0.8}
    leverage = risk_manager.adjust_leverage(market_conditions)
    print(f"Zalecana dźwignia: {leverage:.1f}x")