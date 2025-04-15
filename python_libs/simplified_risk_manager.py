"""
simplified_risk_manager.py
-------------------------
Uproszczony menedżer ryzyka dla platformy tradingowej.
"""

import logging
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SimplifiedRiskManager:
    """Uproszczony menedżer ryzyka do zarządzania ryzykiem w handlu."""

    def __init__(self, max_risk: float = 0.05, max_position_size: float = 0.2, max_drawdown: float = 0.1):
        """
        Inicjalizuje menedżera ryzyka.

        Parameters:
            max_risk (float): Maksymalne ryzyko na transakcję jako % kapitału
            max_position_size (float): Maksymalny rozmiar pozycji jako % kapitału
            max_drawdown (float): Maksymalny dopuszczalny drawdown jako % kapitału
        """
        self.max_risk = max_risk
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown

        self.risk_metrics = {
            "current_drawdown": 0.0,
            "max_drawdown_reached": 0.0,
            "total_risk_exposure": 0.0,
            "positions_risk": {}
        }

        self.risk_limits_reached = {
            "max_risk": False,
            "max_position_size": False,
            "max_drawdown": False
        }

        logger.info(f"Zainicjalizowano zarządcę ryzyka portfela. Max ryzyko: {max_risk}, Max rozmiar pozycji: {max_position_size}, Max drawdown: {max_drawdown}")

    def check_trade_risk(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Sprawdza ryzyko związane z transakcją.

        Parameters:
            symbol (str): Symbol instrumentu
            side (str): Strona transakcji ('buy', 'sell')
            quantity (float): Ilość instrumentu
            price (Optional[float]): Cena transakcji

        Returns:
            Dict[str, Any]: Wynik sprawdzenia ryzyka
        """
        try:
            # Sprawdź, czy osiągnięto limity ryzyka
            if self.risk_limits_reached["max_drawdown"]:
                return {
                    "success": False,
                    "error": f"Osiągnięto maksymalny drawdown ({self.max_drawdown * 100}%)",
                    "risk_level": "critical"
                }

            if self.risk_limits_reached["max_risk"]:
                return {
                    "success": False,
                    "error": f"Osiągnięto maksymalne ryzyko całkowite ({self.max_risk * 100}%)",
                    "risk_level": "high"
                }

            # Symulowany poziom ryzyka dla transakcji
            transaction_risk = quantity * (price or 1.0) * 0.01  # Przykładowe ryzyko 1%

            # Przykładowa walidacja ryzyka
            if transaction_risk > self.max_risk:
                return {
                    "success": False,
                    "error": f"Ryzyko transakcji ({transaction_risk:.2f}) przekracza maksymalne ryzyko ({self.max_risk:.2f})",
                    "risk_level": "high"
                }

            position_size = quantity * (price or 1.0)
            if position_size > self.max_position_size:
                return {
                    "success": False,
                    "error": f"Rozmiar pozycji ({position_size:.2f}) przekracza maksymalny rozmiar ({self.max_position_size:.2f})",
                    "risk_level": "medium"
                }

            # Aktualizuj metryki ryzyka
            self.risk_metrics["total_risk_exposure"] += transaction_risk

            if symbol not in self.risk_metrics["positions_risk"]:
                self.risk_metrics["positions_risk"][symbol] = 0.0

            self.risk_metrics["positions_risk"][symbol] += transaction_risk

            # Aktualizuj flagi limitów ryzyka
            self.risk_limits_reached["max_risk"] = self.risk_metrics["total_risk_exposure"] >= self.max_risk

            return {
                "success": True,
                "risk": transaction_risk,
                "risk_level": "low" if transaction_risk < self.max_risk / 2 else "medium"
            }
        except Exception as e:
            logger.error(f"Błąd podczas sprawdzania ryzyka transakcji: {e}")
            return {"success": False, "error": str(e), "risk_level": "unknown"}

    def update_drawdown(self, current_drawdown: float) -> Dict[str, Any]:
        """
        Aktualizuje bieżący drawdown.

        Parameters:
            current_drawdown (float): Bieżący drawdown jako % kapitału

        Returns:
            Dict[str, Any]: Zaktualizowane metryki ryzyka
        """
        try:
            self.risk_metrics["current_drawdown"] = current_drawdown

            if current_drawdown > self.risk_metrics["max_drawdown_reached"]:
                self.risk_metrics["max_drawdown_reached"] = current_drawdown

            self.risk_limits_reached["max_drawdown"] = current_drawdown >= self.max_drawdown

            return {
                "success": True,
                "metrics": self.risk_metrics,
                "limits_reached": self.risk_limits_reached
            }
        except Exception as e:
            logger.error(f"Błąd podczas aktualizacji drawdown: {e}")
            return {"success": False, "error": str(e)}

    def reset_risk_metrics(self) -> Dict[str, Any]:
        """
        Resetuje metryki ryzyka.

        Returns:
            Dict[str, Any]: Zresetowane metryki ryzyka
        """
        try:
            self.risk_metrics = {
                "current_drawdown": 0.0,
                "max_drawdown_reached": 0.0,
                "total_risk_exposure": 0.0,
                "positions_risk": {}
            }

            self.risk_limits_reached = {
                "max_risk": False,
                "max_position_size": False,
                "max_drawdown": False
            }

            logger.info("Zresetowano metryki ryzyka")

            return {
                "success": True,
                "metrics": self.risk_metrics,
                "limits_reached": self.risk_limits_reached
            }
        except Exception as e:
            logger.error(f"Błąd podczas resetowania metryk ryzyka: {e}")
            return {"success": False, "error": str(e)}

    def update_settings(self, settings: Dict[str, float]) -> bool:
        """
        Aktualizuje ustawienia menedżera ryzyka.

        Parameters:
            settings (Dict[str, float]): Nowe ustawienia

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            if "max_risk" in settings:
                self.max_risk = settings["max_risk"]

            if "max_position_size" in settings:
                self.max_position_size = settings["max_position_size"]

            if "max_drawdown" in settings:
                self.max_drawdown = settings["max_drawdown"]

            # Aktualizuj flagi limitów ryzyka
            self.risk_limits_reached["max_risk"] = self.risk_metrics["total_risk_exposure"] >= self.max_risk
            self.risk_limits_reached["max_drawdown"] = self.risk_metrics["current_drawdown"] >= self.max_drawdown

            logger.info(f"Zaktualizowano ustawienia zarządcy ryzyka: {settings}")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas aktualizacji ustawień zarządcy ryzyka: {e}")
            return False

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Zwraca metryki ryzyka.

        Returns:
            Dict[str, Any]: Metryki ryzyka
        """
        return {
            "metrics": self.risk_metrics,
            "limits": {
                "max_risk": self.max_risk,
                "max_position_size": self.max_position_size,
                "max_drawdown": self.max_drawdown
            },
            "limits_reached": self.risk_limits_reached
        }
"""
simplified_risk_manager.py
------------------------
Uproszczony manager ryzyka dla platformy tradingowej.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

class SimplifiedRiskManager:
    """
    Uproszczony manager ryzyka, który filtruje sygnały handlowe w oparciu o zarządzanie ryzykiem.
    """
    
    def __init__(self, max_risk: float = 0.05, max_position_size: float = 0.2, max_drawdown: float = 0.1):
        """
        Inicjalizacja managera ryzyka.
        
        Args:
            max_risk: Maksymalne ryzyko na pojedynczą transakcję (jako procent kapitału)
            max_position_size: Maksymalny rozmiar pozycji (jako procent kapitału)
            max_drawdown: Maksymalny dopuszczalny drawdown
        """
        self.logger = logging.getLogger('risk_manager')
        self.max_risk = max_risk
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        
        self.current_drawdown = 0.0
        self.portfolio_value = 10000.0  # Domyślna wartość portfela
        self.open_positions = {}
        
        self.logger.info(f"Zainicjalizowano uproszczony manager ryzyka (max_risk: {max_risk}, max_position_size: {max_position_size})")
    
    def filter_signals(self, signals: List[Dict[str, Any]], symbol: str, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filtruje sygnały handlowe w oparciu o zarządzanie ryzykiem.
        
        Args:
            signals: Lista sygnałów handlowych
            symbol: Symbol
            market_data: Dane rynkowe
            
        Returns:
            List[Dict[str, Any]]: Przefiltrowane sygnały
        """
        filtered_signals = []
        
        for signal in signals:
            try:
                # Sprawdź czy sygnał przechodzi filtry ryzyka
                if self._check_risk_filters(signal, symbol, market_data):
                    # Dostosuj wielkość pozycji
                    adjusted_signal = self._adjust_position_size(signal, symbol, market_data)
                    filtered_signals.append(adjusted_signal)
            except Exception as e:
                self.logger.error(f"Błąd podczas filtrowania sygnału: {e}")
        
        return filtered_signals
    
    def _check_risk_filters(self, signal: Dict[str, Any], symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Sprawdza, czy sygnał przechodzi filtry ryzyka.
        
        Args:
            signal: Sygnał handlowy
            symbol: Symbol
            market_data: Dane rynkowe
            
        Returns:
            bool: Czy sygnał przechodzi filtry
        """
        # Sprawdź czy przekroczyliśmy maksymalny drawdown
        if self.current_drawdown >= self.max_drawdown:
            self.logger.warning(f"Odrzucono sygnał: przekroczono maksymalny drawdown ({self.current_drawdown:.2%})")
            return False
        
        # Sprawdź czy możemy otworzyć kolejną pozycję
        if len(self.open_positions) >= 5:  # Maksymalnie 5 pozycji
            self.logger.warning(f"Odrzucono sygnał: przekroczono maksymalną liczbę pozycji")
            return False
        
        # Sprawdź czy sygnał ma określone stop loss
        if 'stop_loss' not in signal and signal.get('side') == 'BUY':
            # Automatycznie ustaw stop loss na 5% poniżej ceny wejścia
            price = signal.get('price', 0.0)
            signal['stop_loss'] = price * 0.95
            self.logger.info(f"Automatycznie ustawiono stop loss na {signal['stop_loss']}")
        
        return True
    
    def _adjust_position_size(self, signal: Dict[str, Any], symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dostosowuje wielkość pozycji w sygnale.
        
        Args:
            signal: Sygnał handlowy
            symbol: Symbol
            market_data: Dane rynkowe
            
        Returns:
            Dict[str, Any]: Dostosowany sygnał
        """
        adjusted_signal = signal.copy()
        
        # Oblicz maksymalną wartość pozycji
        max_position_value = self.portfolio_value * self.max_position_size
        
        # Oblicz maksymalne ryzyko na transakcję
        max_risk_value = self.portfolio_value * self.max_risk
        
        # Pobierz cenę wejścia i stop loss
        entry_price = signal.get('price', 0.0)
        stop_loss = signal.get('stop_loss', entry_price * 0.95)
        
        # Oblicz ryzyko na jednostkę
        risk_per_unit = abs(entry_price - stop_loss)
        
        # Jeśli brak ryzyka na jednostkę, zastosuj domyślną wielkość
        if risk_per_unit <= 0:
            adjusted_quantity = max_position_value / entry_price
        else:
            # Oblicz wielkość pozycji w oparciu o ryzyko
            units_for_max_risk = max_risk_value / risk_per_unit
            units_for_max_position = max_position_value / entry_price
            
            # Wybierz mniejszą wartość
            adjusted_quantity = min(units_for_max_risk, units_for_max_position)
        
        # Zastosuj obliczoną wielkość
        adjusted_signal['quantity'] = adjusted_quantity
        
        # Dodaj informacje o ryzyku
        adjusted_signal['risk_info'] = {
            'max_risk': self.max_risk,
            'max_position_size': self.max_position_size,
            'portfolio_value': self.portfolio_value,
            'risk_per_unit': risk_per_unit,
            'position_value': entry_price * adjusted_quantity,
            'risk_value': risk_per_unit * adjusted_quantity
        }
        
        return adjusted_signal
    
    def update_portfolio_value(self, value: float):
        """
        Aktualizuje wartość portfela.
        
        Args:
            value: Nowa wartość portfela
        """
        old_value = self.portfolio_value
        self.portfolio_value = value
        
        # Aktualizuj drawdown
        if old_value > 0:
            current_dd = 1.0 - (value / old_value)
            if current_dd > self.current_drawdown:
                self.current_drawdown = current_dd
                self.logger.info(f"Zaktualizowano drawdown: {self.current_drawdown:.2%}")
    
    def register_position(self, symbol: str, side: str, quantity: float, entry_price: float, stop_loss: float = None):
        """
        Rejestruje nową pozycję.
        
        Args:
            symbol: Symbol
            side: Strona pozycji ('BUY' lub 'SELL')
            quantity: Ilość
            entry_price: Cena wejścia
            stop_loss: Poziom stop loss
        """
        self.open_positions[symbol] = {
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'entry_time': datetime.now().isoformat()
        }
        
        self.logger.info(f"Zarejestrowano pozycję: {side} {quantity} {symbol} @ {entry_price}")
    
    def close_position(self, symbol: str):
        """
        Zamyka pozycję.
        
        Args:
            symbol: Symbol
        """
        if symbol in self.open_positions:
            self.open_positions.pop(symbol)
            self.logger.info(f"Zamknięto pozycję: {symbol}")
    
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Zwraca otwarte pozycje.
        
        Returns:
            Dict[str, Dict[str, Any]]: Otwarte pozycje
        """
        return self.open_positions
