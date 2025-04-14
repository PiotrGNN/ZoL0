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