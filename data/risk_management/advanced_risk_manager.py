"""
advanced_risk_manager.py
------------------------
Zaawansowany moduł zarządzania ryzykiem dla systemów tradingowych.

Funkcjonalności:
- Dynamiczne dostosowywanie poziomów stop-loss i take-profit.
- Monitorowanie drawdown i limitów strat.
- Analiza zmienności rynku i adaptacja strategii ryzyka.
- Obsługa hedgingu i dynamiczna dźwignia w zależności od warunków rynkowych.
- Możliwość integracji z innymi modułami zarządzania portfelem.
"""

import logging
import numpy as np

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

class AdvancedRiskManager:
    """
    Klasa zaawansowanego zarządzania ryzykiem, dostosowująca parametry stop-loss,
    take-profit oraz wielkość pozycji na podstawie warunków rynkowych.
    """

    def __init__(self, max_risk_per_trade=0.02, max_drawdown=0.2, volatility_factor=1.5,
                 min_stop_loss=0.005, max_stop_loss=0.02, take_profit_factor=2.0,
                 leverage_adjustment=True):
        """
        Inicjalizacja menedżera ryzyka.

        Parameters:
            max_risk_per_trade (float): Maksymalne ryzyko na pojedynczą transakcję (np. 2% kapitału).
            max_drawdown (float): Maksymalny dopuszczalny drawdown na koncie.
            volatility_factor (float): Współczynnik dostosowania SL do zmienności.
            min_stop_loss (float): Minimalny poziom stop-loss.
            max_stop_loss (float): Maksymalny poziom stop-loss.
            take_profit_factor (float): Współczynnik określający TP na podstawie SL.
            leverage_adjustment (bool): Czy dynamicznie dostosowywać dźwignię.
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown = max_drawdown
        self.volatility_factor = volatility_factor
        self.min_stop_loss = min_stop_loss
        self.max_stop_loss = max_stop_loss
        self.take_profit_factor = take_profit_factor
        self.leverage_adjustment = leverage_adjustment

        self.initial_capital = None
        self.current_capital = None
        self.highest_capital = None

    def initialize(self, initial_capital):
        """
        Inicjalizuje system zarządzania ryzykiem.

        Parameters:
            initial_capital (float): Początkowy kapitał tradera.
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.highest_capital = initial_capital
        logging.info("Zainicjalizowano menedżera ryzyka. Kapitał początkowy: %.2f", initial_capital)

    def update_capital(self, new_capital):
        """
        Aktualizuje bieżący kapitał i monitoruje drawdown.

        Parameters:
            new_capital (float): Aktualna wartość kapitału pozycjonowanego.
        """
        self.current_capital = new_capital
        if new_capital > self.highest_capital:
            self.highest_capital = new_capital

        drawdown = 1 - (new_capital / self.highest_capital)
        if drawdown > self.max_drawdown:
            logging.warning("Przekroczono maksymalny drawdown! Aktualny drawdown: %.2f%%", drawdown * 100)

    def calculate_stop_loss(self, volatility):
        """
        Oblicza dynamiczny stop-loss na podstawie zmienności rynku.

        Parameters:
            volatility (float): Aktualna zmienność rynku.

        Returns:
            float: Poziom stop-loss w procentach.
        """
        dynamic_sl = self.volatility_factor * volatility
        sl = np.clip(dynamic_sl, self.min_stop_loss, self.max_stop_loss)
        logging.info("Obliczony poziom stop-loss: %.4f", sl)
        return sl

    def calculate_take_profit(self, stop_loss):
        """
        Oblicza poziom take-profit na podstawie stop-loss i ustalonego współczynnika.

        Parameters:
            stop_loss (float): Poziom stop-loss w procentach.

        Returns:
            float: Poziom take-profit w procentach.
        """
        tp = stop_loss * self.take_profit_factor
        logging.info("Obliczony poziom take-profit: %.4f", tp)
        return tp

    def determine_position_size(self, account_balance, stop_loss, asset_price):
        """
        Oblicza wielkość pozycji w oparciu o kapitał, stop-loss i maksymalne ryzyko na transakcję.

        Parameters:
            account_balance (float): Aktualny kapitał tradera.
            stop_loss (float): Poziom stop-loss w procentach.
            asset_price (float): Aktualna cena aktywa.

        Returns:
            float: Maksymalna wielkość pozycji.
        """
        risk_amount = account_balance * self.max_risk_per_trade
        position_size = risk_amount / (stop_loss * asset_price)
        logging.info("Obliczona wielkość pozycji: %.4f jednostek", position_size)
        return position_size

    def adjust_leverage(self, market_conditions):
        """
        Dostosowuje dźwignię w zależności od warunków rynkowych.

        Parameters:
            market_conditions (dict): Słownik zawierający np. zmienność, trend itp.

        Returns:
            float: Nowy poziom dźwigni.
        """
        if not self.leverage_adjustment:
            return 1.0

        volatility = market_conditions.get("volatility", 0.01)
        trend_strength = market_conditions.get("trend_strength", 1.0)

        new_leverage = max(1.0, min(5.0, (1.0 / volatility) * trend_strength))
        logging.info("Dostosowana dźwignia: %.2f", new_leverage)
        return new_leverage

# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Inicjalizacja menedżera ryzyka
        risk_manager = AdvancedRiskManager(max_risk_per_trade=0.02, max_drawdown=0.15,
                                           volatility_factor=1.2, min_stop_loss=0.005, max_stop_loss=0.03,
                                           take_profit_factor=2.5, leverage_adjustment=True)
        
        risk_manager.initialize(initial_capital=10000)

        # Przykładowe warunki rynkowe
        market_conditions = {"volatility": 0.015, "trend_strength": 1.2}

        # Obliczenie stop-loss i take-profit
        stop_loss = risk_manager.calculate_stop_loss(market_conditions["volatility"])
        take_profit = risk_manager.calculate_take_profit(stop_loss)

        # Obliczenie wielkości pozycji
        position_size = risk_manager.determine_position_size(account_balance=10000, stop_loss=stop_loss, asset_price=50)

        # Dynamiczna dźwignia
        leverage = risk_manager.adjust_leverage(market_conditions)

        logging.info("Finalna konfiguracja: SL=%.4f, TP=%.4f, Wielkość pozycji=%.2f, Dźwignia=%.2f",
                     stop_loss, take_profit, position_size, leverage)

    except Exception as e:
        logging.error("Błąd w AdvancedRiskManager: %s", e)
        raise
