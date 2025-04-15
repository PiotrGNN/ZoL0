"""
simplified_strategy.py
---------------------
Uproszczony menedżer strategii dla platformy tradingowej.
"""

import logging
import time
import random
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

class StrategyManager:
    """Menedżer strategii do zarządzania strategiami tradingowymi."""

    def __init__(self, strategies: Dict[str, Dict[str, Any]] = None, exposure_limits: Dict[str, float] = None):
        """
        Inicjalizuje menedżera strategii.

        Parameters:
            strategies (Dict[str, Dict[str, Any]]): Słownik strategii
            exposure_limits (Dict[str, float]): Limity ekspozycji dla strategii
        """
        self.strategies = strategies or {
            "trend_following": {"name": "Trend Following", "enabled": False},
            "mean_reversion": {"name": "Mean Reversion", "enabled": False},
            "breakout": {"name": "Breakout", "enabled": False}
        }

        self.exposure_limits = exposure_limits or {
            "trend_following": 0.5,
            "mean_reversion": 0.3,
            "breakout": 0.4
        }

        self.active_strategies = []
        self.strategy_performance = {}

        # Domyślne definicje strategii
        self.strategy_definitions = {
            "trend_following": self._trend_following_strategy,
            "mean_reversion": self._mean_reversion_strategy,
            "breakout": self._breakout_strategy
        }

        logger.info(f"Zainicjalizowano StrategyManager z {len(self.strategies)} strategiami")

    def activate_strategy(self, strategy_id: str) -> bool:
        """
        Aktywuje strategię.

        Parameters:
            strategy_id (str): ID strategii

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            if strategy_id not in self.strategies:
                logger.warning(f"Nieznana strategia: {strategy_id}")
                return False

            self.strategies[strategy_id]["enabled"] = True

            if strategy_id not in self.active_strategies:
                self.active_strategies.append(strategy_id)

            logger.info(f"Aktywowano strategię: {strategy_id}")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas aktywacji strategii {strategy_id}: {e}")
            return False

    def deactivate_strategy(self, strategy_id: str) -> bool:
        """
        Deaktywuje strategię.

        Parameters:
            strategy_id (str): ID strategii

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            if strategy_id not in self.strategies:
                logger.warning(f"Nieznana strategia: {strategy_id}")
                return False

            self.strategies[strategy_id]["enabled"] = False

            if strategy_id in self.active_strategies:
                self.active_strategies.remove(strategy_id)

            logger.info(f"Deaktywowano strategię: {strategy_id}")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas deaktywacji strategii {strategy_id}: {e}")
            return False

    def evaluate_strategies(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ocenia wszystkie aktywne strategie na podstawie danych rynkowych.

        Parameters:
            market_data (Dict[str, Any]): Dane rynkowe

        Returns:
            Dict[str, Any]: Wyniki oceny strategii
        """
        try:
            results = {}
            combined_signal = 0.0
            weights_sum = 0.0

            for strategy_id in self.active_strategies:
                if strategy_id in self.strategy_definitions:
                    # Wykonaj strategię
                    strategy_result = self.strategy_definitions[strategy_id](market_data)

                    # Zapisz wynik
                    results[strategy_id] = strategy_result

                    # Dodaj do sygnału łączonego
                    weight = self.exposure_limits.get(strategy_id, 0.0)
                    combined_signal += strategy_result["signal"] * weight
                    weights_sum += weight

            # Normalizuj sygnał łączony
            if weights_sum > 0:
                combined_signal /= weights_sum

            # Określ decyzję
            if combined_signal > 0.5:
                decision = "buy"
            elif combined_signal < -0.5:
                decision = "sell"
            else:
                decision = "hold"

            return {
                "strategy_results": results,
                "combined_signal": combined_signal,
                "decision": decision,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Błąd podczas oceny strategii: {e}")
            return {"error": str(e)}

    def _trend_following_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strategia podążania za trendem.

        Parameters:
            market_data (Dict[str, Any]): Dane rynkowe

        Returns:
            Dict[str, Any]: Wynik strategii
        """
        # Symulacja dla celów demonstracyjnych
        trend_strength = random.uniform(-1.0, 1.0)
        signal = trend_strength

        return {
            "signal": signal,
            "indicators": {
                "trend_strength": trend_strength,
                "ma_cross": random.choice([True, False]),
                "adx": random.uniform(10, 40)
            },
            "confidence": abs(signal) * 0.8 + 0.1
        }

    def _mean_reversion_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strategia powrotu do średniej.

        Parameters:
            market_data (Dict[str, Any]): Dane rynkowe

        Returns:
            Dict[str, Any]: Wynik strategii
        """
        # Symulacja dla celów demonstracyjnych
        overbought = random.random() > 0.7
        oversold = random.random() > 0.7

        if overbought:
            signal = -random.uniform(0.5, 1.0)
        elif oversold:
            signal = random.uniform(0.5, 1.0)
        else:
            signal = random.uniform(-0.3, 0.3)

        return {
            "signal": signal,
            "indicators": {
                "rsi": random.uniform(0, 100),
                "bollinger_bands": {
                    "upper": random.uniform(30000, 40000),
                    "middle": random.uniform(28000, 35000),
                    "lower": random.uniform(25000, 30000)
                }
            },
            "confidence": abs(signal) * 0.7 + 0.2
        }

    def _breakout_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strategia breakoutu.

        Parameters:
            market_data (Dict[str, Any]): Dane rynkowe

        Returns:
            Dict[str, Any]: Wynik strategii
        """
        # Symulacja dla celów demonstracyjnych
        breakout_up = random.random() > 0.8
        breakout_down = random.random() > 0.8

        if breakout_up:
            signal = random.uniform(0.7, 1.0)
        elif breakout_down:
            signal = random.uniform(-1.0, -0.7)
        else:
            signal = random.uniform(-0.2, 0.2)

        return {
            "signal": signal,
            "indicators": {
                "support_resistance": {
                    "support": random.uniform(25000, 30000),
                    "resistance": random.uniform(35000, 40000)
                },
                "volume_increase": random.uniform(0, 200)
            },
            "confidence": abs(signal) * 0.9 + 0.1
        }

    def get_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Zwraca dostępne strategie.

        Returns:
            Dict[str, Dict[str, Any]]: Dostępne strategie
        """
        return self.strategies

    def get_active_strategies(self) -> List[str]:
        """
        Zwraca aktywne strategie.

        Returns:
            List[str]: Aktywne strategie
        """
        return self.active_strategies

    def add_strategy(self, strategy_id: str, strategy_name: str, strategy_function: Callable) -> bool:
        """
        Dodaje nową strategię.

        Parameters:
            strategy_id (str): ID strategii
            strategy_name (str): Nazwa strategii
            strategy_function (Callable): Funkcja implementująca strategię

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            if strategy_id in self.strategies:
                logger.warning(f"Strategia o ID {strategy_id} już istnieje")
                return False

            self.strategies[strategy_id] = {"name": strategy_name, "enabled": False}
            self.strategy_definitions[strategy_id] = strategy_function
            self.exposure_limits[strategy_id] = 0.3  # Domyślny limit ekspozycji

            logger.info(f"Dodano nową strategię: {strategy_id} ({strategy_name})")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas dodawania strategii {strategy_id}: {e}")
            return False

    def update_strategy_exposure(self, strategy_id: str, exposure: float) -> bool:
        """
        Aktualizuje limit ekspozycji dla strategii.

        Parameters:
            strategy_id (str): ID strategii
            exposure (float): Limit ekspozycji

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            if strategy_id not in self.strategies:
                logger.warning(f"Nieznana strategia: {strategy_id}")
                return False

            self.exposure_limits[strategy_id] = exposure
            logger.info(f"Zaktualizowano limit ekspozycji dla strategii {strategy_id}: {exposure}")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas aktualizacji limitu ekspozycji dla strategii {strategy_id}: {e}")
            return False

# Przykład użycia
if __name__ == "__main__":
    # Inicjalizacja managera strategii
    strategy_manager = StrategyManager()

    # Aktywacja strategii
    strategy_manager.activate_strategy("trend_following")
    strategy_manager.activate_strategy("breakout")

    # Wyświetlenie aktywnych strategii
    active_strategies = strategy_manager.get_active_strategies()
    print(f"Aktywne strategie: {active_strategies}")

    # Przykładowa analiza danych rynkowych
    market_data = {"symbol": "BTCUSDT", "price": 50000, "volume": 100, "timestamp": time.time()}
    results = strategy_manager.evaluate_strategies(market_data)

    # Wyświetlenie wyników
    print(f"Wyniki analizy: {results}")
"""
simplified_strategy.py
--------------------
Uproszczony manager strategii dla platformy tradingowej.
"""

import logging
from typing import Dict, List, Any, Optional, Callable

class Strategy:
    """Podstawowa klasa strategii tradingowej."""
    
    def __init__(self, name: str):
        """
        Inicjalizacja strategii.
        
        Args:
            name: Nazwa strategii
        """
        self.name = name
        self.logger = logging.getLogger(f'strategy.{name}')
        self.enabled = True
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generuje sygnały handlowe na podstawie danych rynkowych.
        
        Args:
            market_data: Dane rynkowe
            
        Returns:
            List[Dict[str, Any]]: Lista sygnałów handlowych
        """
        # Ta metoda powinna być nadpisana przez konkretne strategie
        return []

class TrendFollowingStrategy(Strategy):
    """Strategia podążania za trendem."""
    
    def __init__(self):
        """Inicjalizacja strategii podążania za trendem."""
        super().__init__("Trend Following")
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generuje sygnały handlowe na podstawie trendu.
        
        Args:
            market_data: Dane rynkowe
            
        Returns:
            List[Dict[str, Any]]: Lista sygnałów handlowych
        """
        signals = []
        
        try:
            symbol = market_data.get('symbol', '')
            candles = market_data.get('candles', [])
            
            if not candles or len(candles) < 10:
                return signals
            
            # Uproszczona analiza trendu: porównanie pierwszej i ostatniej świecy
            first_price = float(candles[0].get('close', 0))
            last_price = float(candles[-1].get('close', 0))
            
            # Trend wzrostowy
            if last_price > first_price * 1.02:  # Wzrost o co najmniej 2%
                signals.append({
                    'type': 'LIMIT',
                    'side': 'BUY',
                    'price': last_price,
                    'quantity': 0.01,  # Domyślna ilość (zostanie dostosowana przez risk manager)
                    'stop_loss': last_price * 0.97,  # Stop loss 3% poniżej ceny
                    'confidence': 0.75
                })
            
            # Trend spadkowy
            elif last_price < first_price * 0.98:  # Spadek o co najmniej 2%
                signals.append({
                    'type': 'LIMIT',
                    'side': 'SELL',
                    'price': last_price,
                    'quantity': 0.01,  # Domyślna ilość
                    'stop_loss': last_price * 1.03,  # Stop loss 3% powyżej ceny
                    'confidence': 0.75
                })
        except Exception as e:
            self.logger.error(f"Błąd w strategii podążania za trendem: {e}")
        
        return signals

class MeanReversionStrategy(Strategy):
    """Strategia powrotu do średniej."""
    
    def __init__(self):
        """Inicjalizacja strategii powrotu do średniej."""
        super().__init__("Mean Reversion")
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generuje sygnały handlowe na podstawie powrotu do średniej.
        
        Args:
            market_data: Dane rynkowe
            
        Returns:
            List[Dict[str, Any]]: Lista sygnałów handlowych
        """
        signals = []
        
        try:
            symbol = market_data.get('symbol', '')
            candles = market_data.get('candles', [])
            
            if not candles or len(candles) < 20:
                return signals
            
            # Oblicz średnią z 20 świec
            closes = [float(candle.get('close', 0)) for candle in candles]
            avg_price = sum(closes) / len(closes)
            last_price = closes[-1]
            
            # Sprawdź, czy cena jest znacząco poniżej średniej (potencjalny sygnał kupna)
            if last_price < avg_price * 0.95:  # Cena 5% poniżej średniej
                signals.append({
                    'type': 'LIMIT',
                    'side': 'BUY',
                    'price': last_price,
                    'quantity': 0.01,  # Domyślna ilość
                    'stop_loss': last_price * 0.97,  # Stop loss 3% poniżej ceny
                    'confidence': 0.70
                })
            
            # Sprawdź, czy cena jest znacząco powyżej średniej (potencjalny sygnał sprzedaży)
            elif last_price > avg_price * 1.05:  # Cena 5% powyżej średniej
                signals.append({
                    'type': 'LIMIT',
                    'side': 'SELL',
                    'price': last_price,
                    'quantity': 0.01,  # Domyślna ilość
                    'stop_loss': last_price * 1.03,  # Stop loss 3% powyżej ceny
                    'confidence': 0.70
                })
        except Exception as e:
            self.logger.error(f"Błąd w strategii powrotu do średniej: {e}")
        
        return signals

class BreakoutStrategy(Strategy):
    """Strategia przełamania."""
    
    def __init__(self):
        """Inicjalizacja strategii przełamania."""
        super().__init__("Breakout")
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generuje sygnały handlowe na podstawie przełamań.
        
        Args:
            market_data: Dane rynkowe
            
        Returns:
            List[Dict[str, Any]]: Lista sygnałów handlowych
        """
        signals = []
        
        try:
            symbol = market_data.get('symbol', '')
            candles = market_data.get('candles', [])
            
            if not candles or len(candles) < 20:
                return signals
            
            # Znajdź maksima i minima z ostatnich 20 świec
            highs = [float(candle.get('high', 0)) for candle in candles[:-1]]  # Bez ostatniej świecy
            lows = [float(candle.get('low', 0)) for candle in candles[:-1]]  # Bez ostatniej świecy
            
            highest = max(highs)
            lowest = min(lows)
            
            last_price = float(candles[-1].get('close', 0))
            
            # Sprawdź, czy ostatnia cena przełamała ostatnie maksimum
            if last_price > highest * 1.01:  # Przełamanie 1% powyżej maksimum
                signals.append({
                    'type': 'LIMIT',
                    'side': 'BUY',
                    'price': last_price,
                    'quantity': 0.01,  # Domyślna ilość
                    'stop_loss': last_price * 0.97,  # Stop loss 3% poniżej ceny
                    'confidence': 0.80
                })
            
            # Sprawdź, czy ostatnia cena przełamała ostatnie minimum
            elif last_price < lowest * 0.99:  # Przełamanie 1% poniżej minimum
                signals.append({
                    'type': 'LIMIT',
                    'side': 'SELL',
                    'price': last_price,
                    'quantity': 0.01,  # Domyślna ilość
                    'stop_loss': last_price * 1.03,  # Stop loss 3% powyżej ceny
                    'confidence': 0.80
                })
        except Exception as e:
            self.logger.error(f"Błąd w strategii przełamania: {e}")
        
        return signals

class StrategyManager:
    """Manager strategii tradingowych."""
    
    def __init__(self, strategies: Dict[str, Dict[str, Any]] = None, exposure_limits: Dict[str, float] = None):
        """
        Inicjalizacja managera strategii.
        
        Args:
            strategies: Słownik strategii
            exposure_limits: Limity ekspozycji dla każdej strategii
        """
        self.logger = logging.getLogger('strategy_manager')
        
        # Inicjalizacja strategii
        self.strategies = {}
        self.exposure_limits = exposure_limits or {}
        
        # Utworzenie strategii z konfiguracji
        if strategies:
            for strategy_id, config in strategies.items():
                self._create_strategy(strategy_id, config)
        
        # Jeśli nie ma żadnych strategii, utwórz domyślne
        if not self.strategies:
            self._create_default_strategies()
        
        self.logger.info(f"Zainicjalizowano managera strategii z {len(self.strategies)} strategiami")
    
    def _create_strategy(self, strategy_id: str, config: Dict[str, Any]) -> bool:
        """
        Tworzy instancję strategii na podstawie konfiguracji.
        
        Args:
            strategy_id: Identyfikator strategii
            config: Konfiguracja strategii
            
        Returns:
            bool: Czy utworzenie się powiodło
        """
        try:
            strategy_class = None
            
            # Wybór klasy strategii na podstawie identyfikatora
            if 'trend' in strategy_id.lower():
                strategy_class = TrendFollowingStrategy
            elif 'mean' in strategy_id.lower() or 'reversion' in strategy_id.lower():
                strategy_class = MeanReversionStrategy
            elif 'break' in strategy_id.lower():
                strategy_class = BreakoutStrategy
            else:
                strategy_class = Strategy
            
            # Utworzenie instancji
            strategy = strategy_class()
            
            # Konfiguracja strategii
            strategy.enabled = config.get('enabled', True)
            
            # Dodanie do managera
            self.strategies[strategy_id] = strategy
            
            # Ustawienie limitu ekspozycji, jeśli nie istnieje
            if strategy_id not in self.exposure_limits:
                self.exposure_limits[strategy_id] = 0.3  # Domyślny limit 30%
            
            self.logger.info(f"Utworzono strategię: {strategy_id} (enabled: {strategy.enabled})")
            return True
        except Exception as e:
            self.logger.error(f"Błąd podczas tworzenia strategii {strategy_id}: {e}")
            return False
    
    def _create_default_strategies(self):
        """Tworzy domyślne strategie."""
        self.strategies['trend_following'] = TrendFollowingStrategy()
        self.strategies['mean_reversion'] = MeanReversionStrategy()
        self.strategies['breakout'] = BreakoutStrategy()
        
        # Domyślne limity ekspozycji
        self.exposure_limits['trend_following'] = 0.5
        self.exposure_limits['mean_reversion'] = 0.3
        self.exposure_limits['breakout'] = 0.4
        
        self.logger.info("Utworzono domyślne strategie")
    
    def activate_strategy(self, strategy_id: str) -> bool:
        """
        Aktywuje strategię.
        
        Args:
            strategy_id: Identyfikator strategii
            
        Returns:
            bool: Czy aktywacja się powiodła
        """
        if strategy_id in self.strategies:
            self.strategies[strategy_id].enabled = True
            self.logger.info(f"Aktywowano strategię: {strategy_id}")
            return True
        
        self.logger.warning(f"Nie można aktywować strategii {strategy_id}: nie znaleziono")
        return False
    
    def deactivate_strategy(self, strategy_id: str) -> bool:
        """
        Dezaktywuje strategię.
        
        Args:
            strategy_id: Identyfikator strategii
            
        Returns:
            bool: Czy dezaktywacja się powiodła
        """
        if strategy_id in self.strategies:
            self.strategies[strategy_id].enabled = False
            self.logger.info(f"Dezaktywowano strategię: {strategy_id}")
            return True
        
        self.logger.warning(f"Nie można dezaktywować strategii {strategy_id}: nie znaleziono")
        return False
    
    def get_active_strategies(self) -> Dict[str, Strategy]:
        """
        Zwraca słownik aktywnych strategii.
        
        Returns:
            Dict[str, Strategy]: Słownik aktywnych strategii
        """
        return {strategy_id: strategy for strategy_id, strategy in self.strategies.items() if strategy.enabled}
    
    def get_all_strategies(self) -> Dict[str, Strategy]:
        """
        Zwraca słownik wszystkich strategii.
        
        Returns:
            Dict[str, Strategy]: Słownik wszystkich strategii
        """
        return self.strategies.copy()
    
    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """
        Zwraca strategię o podanym identyfikatorze.
        
        Args:
            strategy_id: Identyfikator strategii
            
        Returns:
            Optional[Strategy]: Strategia lub None
        """
        return self.strategies.get(strategy_id)
    
    def register_strategy(self, strategy_id: str, strategy: Strategy, exposure_limit: float = 0.3) -> bool:
        """
        Rejestruje nową strategię.
        
        Args:
            strategy_id: Identyfikator strategii
            strategy: Instancja strategii
            exposure_limit: Limit ekspozycji
            
        Returns:
            bool: Czy rejestracja się powiodła
        """
        if strategy_id in self.strategies:
            self.logger.warning(f"Strategia {strategy_id} już istnieje")
            return False
        
        self.strategies[strategy_id] = strategy
        self.exposure_limits[strategy_id] = exposure_limit
        
        self.logger.info(f"Zarejestrowano strategię: {strategy_id} (exposure limit: {exposure_limit})")
        return True
    
    def set_exposure_limit(self, strategy_id: str, limit: float) -> bool:
        """
        Ustawia limit ekspozycji dla strategii.
        
        Args:
            strategy_id: Identyfikator strategii
            limit: Limit ekspozycji
            
        Returns:
            bool: Czy operacja się powiodła
        """
        if strategy_id not in self.strategies:
            self.logger.warning(f"Nie można ustawić limitu ekspozycji dla {strategy_id}: nie znaleziono")
            return False
        
        self.exposure_limits[strategy_id] = limit
        self.logger.info(f"Ustawiono limit ekspozycji dla {strategy_id}: {limit}")
        return True
    
    def execute_strategy(self, strategy_id: str, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Wykonuje strategię na danych rynkowych.
        
        Args:
            strategy_id: Identyfikator strategii
            market_data: Dane rynkowe
            
        Returns:
            List[Dict[str, Any]]: Lista sygnałów handlowych
        """
        if strategy_id not in self.strategies:
            self.logger.warning(f"Nie można wykonać strategii {strategy_id}: nie znaleziono")
            return []
        
        strategy = self.strategies[strategy_id]
        
        if not strategy.enabled:
            self.logger.debug(f"Strategia {strategy_id} jest nieaktywna")
            return []
        
        try:
            # Generowanie sygnałów
            signals = strategy.generate_signals(market_data)
            
            # Zastosowanie limitu ekspozycji
            exposure_limit = self.exposure_limits.get(strategy_id, 0.3)
            
            # Dostosowanie sygnałów do limitu ekspozycji
            for signal in signals:
                if 'confidence' in signal:
                    signal['exposure_factor'] = min(signal['confidence'], exposure_limit)
                else:
                    signal['exposure_factor'] = exposure_limit
            
            return signals
        except Exception as e:
            self.logger.error(f"Błąd podczas wykonywania strategii {strategy_id}: {e}")
            return []
"""
simplified_strategy.py - Uproszczone strategie tradingowe
"""

import logging
import random
from typing import Dict, List, Any, Optional

class Strategy:
    """Bazowa klasa strategii."""
    
    def __init__(self, name: str):
        """
        Inicjalizacja strategii.
        
        Args:
            name: Nazwa strategii
        """
        self.name = name
        self.logger = logging.getLogger(f"Strategy.{name}")
        self.enabled = True
        self.parameters = {}
        
        self.logger.info(f"Zainicjalizowano strategię: {name}")
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generuje sygnały handlowe na podstawie danych rynkowych.
        
        Args:
            market_data: Dane rynkowe
            
        Returns:
            List[Dict[str, Any]]: Lista sygnałów handlowych
        """
        self.logger.warning(f"Metoda generate_signals nie została zaimplementowana w {self.name}")
        return []
    
    def update_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Aktualizuje parametry strategii.
        
        Args:
            parameters: Nowe parametry
            
        Returns:
            bool: Czy operacja się powiodła
        """
        try:
            for key, value in parameters.items():
                self.parameters[key] = value
            
            self.logger.info(f"Zaktualizowano parametry strategii {self.name}: {parameters}")
            return True
        except Exception as e:
            self.logger.error(f"Błąd podczas aktualizacji parametrów strategii {self.name}: {e}")
            return False
    
    def enable(self):
        """Włącza strategię."""
        self.enabled = True
        self.logger.info(f"Strategia {self.name} została włączona")
    
    def disable(self):
        """Wyłącza strategię."""
        self.enabled = False
        self.logger.info(f"Strategia {self.name} została wyłączona")
    
    def is_enabled(self) -> bool:
        """
        Sprawdza, czy strategia jest włączona.
        
        Returns:
            bool: Czy strategia jest włączona
        """
        return self.enabled

class TrendFollowingStrategy(Strategy):
    """Strategia podążająca za trendem."""
    
    def __init__(self):
        """Inicjalizacja strategii podążającej za trendem."""
        super().__init__("Trend Following")
        
        # Domyślne parametry
        self.parameters = {
            "short_period": 10,
            "long_period": 30,
            "signal_threshold": 0.01,
            "position_size": 0.1
        }
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generuje sygnały handlowe na podstawie danych rynkowych.
        
        Args:
            market_data: Dane rynkowe
            
        Returns:
            List[Dict[str, Any]]: Lista sygnałów handlowych
        """
        if not self.enabled:
            return []
        
        signals = []
        
        try:
            # Pobierz dane
            symbol = market_data.get("symbol", "BTCUSDT")
            
            # Symulacja logiki strategii
            # W rzeczywistej implementacji używalibyśmy tu odpowiednich wskaźników
            
            # Generowanie losowego sygnału dla celów demonstracyjnych
            if random.random() < 0.2:  # 20% szans na sygnał
                side = "BUY" if random.random() > 0.5 else "SELL"
                
                signal = {
                    "symbol": symbol,
                    "side": side,
                    "type": "LIMIT",
                    "quantity": round(random.uniform(0.001, 0.01), 4),
                    "price": round(random.uniform(30000, 40000), 2),
                    "strategy": self.name,
                    "reason": f"Wykryto trend {'wzrostowy' if side == 'BUY' else 'spadkowy'}"
                }
                
                signals.append(signal)
                self.logger.info(f"Wygenerowano sygnał {side} dla {symbol}")
        except Exception as e:
            self.logger.error(f"Błąd podczas generowania sygnałów dla {self.name}: {e}")
        
        return signals

class MeanReversionStrategy(Strategy):
    """Strategia powrotu do średniej."""
    
    def __init__(self):
        """Inicjalizacja strategii powrotu do średniej."""
        super().__init__("Mean Reversion")
        
        # Domyślne parametry
        self.parameters = {
            "lookback_period": 20,
            "std_dev_threshold": 2.0,
            "position_size": 0.05
        }
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generuje sygnały handlowe na podstawie danych rynkowych.
        
        Args:
            market_data: Dane rynkowe
            
        Returns:
            List[Dict[str, Any]]: Lista sygnałów handlowych
        """
        if not self.enabled:
            return []
        
        signals = []
        
        try:
            # Pobierz dane
            symbol = market_data.get("symbol", "BTCUSDT")
            
            # Symulacja logiki strategii
            # W rzeczywistej implementacji używalibyśmy tu odpowiednich wskaźników
            
            # Generowanie losowego sygnału dla celów demonstracyjnych
            if random.random() < 0.15:  # 15% szans na sygnał
                is_overbought = random.random() > 0.5
                side = "SELL" if is_overbought else "BUY"
                
                signal = {
                    "symbol": symbol,
                    "side": side,
                    "type": "LIMIT",
                    "quantity": round(random.uniform(0.001, 0.01), 4),
                    "price": round(random.uniform(30000, 40000), 2),
                    "strategy": self.name,
                    "reason": f"Rynek {'wykupiony' if is_overbought else 'wyprzedany'}"
                }
                
                signals.append(signal)
                self.logger.info(f"Wygenerowano sygnał {side} dla {symbol}")
        except Exception as e:
            self.logger.error(f"Błąd podczas generowania sygnałów dla {self.name}: {e}")
        
        return signals

class BreakoutStrategy(Strategy):
    """Strategia breakout."""
    
    def __init__(self):
        """Inicjalizacja strategii breakout."""
        super().__init__("Breakout")
        
        # Domyślne parametry
        self.parameters = {
            "lookback_period": 50,
            "breakout_threshold": 0.03,
            "position_size": 0.07
        }
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generuje sygnały handlowe na podstawie danych rynkowych.
        
        Args:
            market_data: Dane rynkowe
            
        Returns:
            List[Dict[str, Any]]: Lista sygnałów handlowych
        """
        if not self.enabled:
            return []
        
        signals = []
        
        try:
            # Pobierz dane
            symbol = market_data.get("symbol", "BTCUSDT")
            
            # Symulacja logiki strategii
            # W rzeczywistej implementacji używalibyśmy tu odpowiednich wskaźników
            
            # Generowanie losowego sygnału dla celów demonstracyjnych
            if random.random() < 0.1:  # 10% szans na sygnał
                is_upper_breakout = random.random() > 0.5
                side = "BUY" if is_upper_breakout else "SELL"
                
                signal = {
                    "symbol": symbol,
                    "side": side,
                    "type": "MARKET",  # Breakout to zwykle zlecenia rynkowe
                    "quantity": round(random.uniform(0.001, 0.01), 4),
                    "strategy": self.name,
                    "reason": f"Wykryto breakout {'powyżej oporu' if is_upper_breakout else 'poniżej wsparcia'}"
                }
                
                signals.append(signal)
                self.logger.info(f"Wygenerowano sygnał {side} dla {symbol}")
        except Exception as e:
            self.logger.error(f"Błąd podczas generowania sygnałów dla {self.name}: {e}")
        
        return signals

class StrategyManager:
    """Menedżer strategii tradingowych."""
    
    def __init__(self, strategies: Dict[str, Dict[str, Any]] = None, exposure_limits: Dict[str, float] = None):
        """
        Inicjalizacja menedżera strategii.
        
        Args:
            strategies: Słownik z konfiguracją strategii
            exposure_limits: Limity ekspozycji dla każdej strategii
        """
        self.logger = logging.getLogger("StrategyManager")
        
        # Inicjalizacja dostępnych strategii
        self.available_strategies = {}
        self.active_strategies = {}
        self.exposure_limits = exposure_limits or {}
        
        # Dodanie wbudowanych strategii
        self._add_builtin_strategies()
        
        # Konfiguracja na podstawie parametrów
        if strategies:
            self._configure_strategies(strategies)
        
        self.logger.info(f"Zainicjalizowano StrategyManager z {len(self.available_strategies)} strategiami")
    
    def _add_builtin_strategies(self):
        """Dodaje wbudowane strategie."""
        self.available_strategies["trend_following"] = TrendFollowingStrategy()
        self.available_strategies["mean_reversion"] = MeanReversionStrategy()
        self.available_strategies["breakout"] = BreakoutStrategy()
        
        # Domyślne limity ekspozycji
        if not self.exposure_limits:
            self.exposure_limits = {
                "trend_following": 0.5,
                "mean_reversion": 0.3,
                "breakout": 0.4
            }
    
    def _configure_strategies(self, strategies: Dict[str, Dict[str, Any]]):
        """
        Konfiguruje strategie na podstawie słownika konfiguracji.
        
        Args:
            strategies: Słownik z konfiguracją strategii
        """
        for strategy_id, config in strategies.items():
            if strategy_id in self.available_strategies:
                # Aktualizuj istniejącą strategię
                if config.get("enabled", False):
                    self.available_strategies[strategy_id].enable()
                else:
                    self.available_strategies[strategy_id].disable()
                
                # Aktualizuj parametry, jeśli są
                if "parameters" in config:
                    self.available_strategies[strategy_id].update_parameters(config["parameters"])
            else:
                self.logger.warning(f"Nieznana strategia: {strategy_id}")
    
    def activate_strategy(self, strategy_id: str) -> bool:
        """
        Aktywuje strategię o podanym ID.
        
        Args:
            strategy_id: ID strategii
            
        Returns:
            bool: Czy operacja się powiodła
        """
        if strategy_id in self.available_strategies:
            self.available_strategies[strategy_id].enable()
            self.active_strategies[strategy_id] = self.available_strategies[strategy_id]
            self.logger.info(f"Aktywowano strategię: {strategy_id}")
            return True
        else:
            self.logger.warning(f"Próba aktywacji nieznanej strategii: {strategy_id}")
            return False
    
    def deactivate_strategy(self, strategy_id: str) -> bool:
        """
        Dezaktywuje strategię o podanym ID.
        
        Args:
            strategy_id: ID strategii
            
        Returns:
            bool: Czy operacja się powiodła
        """
        if strategy_id in self.active_strategies:
            self.available_strategies[strategy_id].disable()
            del self.active_strategies[strategy_id]
            self.logger.info(f"Dezaktywowano strategię: {strategy_id}")
            return True
        else:
            self.logger.warning(f"Próba dezaktywacji nieaktywnej lub nieznanej strategii: {strategy_id}")
            return False
    
    def get_active_strategies(self) -> Dict[str, Strategy]:
        """
        Zwraca słownik aktywnych strategii.
        
        Returns:
            Dict[str, Strategy]: Aktywne strategie
        """
        # Aktualizuj listę aktywnych strategii
        self.active_strategies = {
            strategy_id: strategy 
            for strategy_id, strategy in self.available_strategies.items() 
            if strategy.is_enabled()
        }
        
        return self.active_strategies
    
    def get_strategy_info(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Zwraca informacje o strategii o podanym ID.
        
        Args:
            strategy_id: ID strategii
            
        Returns:
            Optional[Dict[str, Any]]: Informacje o strategii lub None, jeśli nie znaleziono
        """
        if strategy_id in self.available_strategies:
            strategy = self.available_strategies[strategy_id]
            return {
                "id": strategy_id,
                "name": strategy.name,
                "enabled": strategy.is_enabled(),
                "parameters": strategy.parameters,
                "exposure_limit": self.exposure_limits.get(strategy_id, 0.0)
            }
        else:
            return None
    
    def get_all_strategies_info(self) -> List[Dict[str, Any]]:
        """
        Zwraca informacje o wszystkich dostępnych strategiach.
        
        Returns:
            List[Dict[str, Any]]: Lista informacji o strategiach
        """
        return [
            {
                "id": strategy_id,
                "name": strategy.name,
                "enabled": strategy.is_enabled(),
                "parameters": strategy.parameters,
                "exposure_limit": self.exposure_limits.get(strategy_id, 0.0)
            }
            for strategy_id, strategy in self.available_strategies.items()
        ]
    
    def update_strategy_parameters(self, strategy_id: str, parameters: Dict[str, Any]) -> bool:
        """
        Aktualizuje parametry strategii o podanym ID.
        
        Args:
            strategy_id: ID strategii
            parameters: Nowe parametry
            
        Returns:
            bool: Czy operacja się powiodła
        """
        if strategy_id in self.available_strategies:
            success = self.available_strategies[strategy_id].update_parameters(parameters)
            if success:
                self.logger.info(f"Zaktualizowano parametry strategii {strategy_id}: {parameters}")
            return success
        else:
            self.logger.warning(f"Próba aktualizacji parametrów nieznanej strategii: {strategy_id}")
            return False
    
    def update_exposure_limit(self, strategy_id: str, limit: float) -> bool:
        """
        Aktualizuje limit ekspozycji dla strategii.
        
        Args:
            strategy_id: ID strategii
            limit: Nowy limit ekspozycji
            
        Returns:
            bool: Czy operacja się powiodła
        """
        if strategy_id in self.available_strategies:
            self.exposure_limits[strategy_id] = limit
            self.logger.info(f"Zaktualizowano limit ekspozycji dla strategii {strategy_id}: {limit}")
            return True
        else:
            self.logger.warning(f"Próba aktualizacji limitu ekspozycji nieznanej strategii: {strategy_id}")
            return False
