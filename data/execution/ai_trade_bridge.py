
"""
AI Trade Bridge - moduł łączący modele AI z systemem tradingowym.
Odpowiedzialny za przekształcanie predykcji AI na konkretne decyzje handlowe.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import json

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_bridge.log", mode="a"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("AITradeBridge")

# Upewniamy się, że folder logs istnieje
os.makedirs("logs", exist_ok=True)


class AITradeBridge:
    """
    Most pomiędzy modelami AI a systemem wykonywania zleceń.
    
    Funkcje:
    - Interpretacja sygnałów z modeli AI
    - Konwersja predykcji na konkretne parametry zleceń
    - Walidacja wiarygodności sygnałów AI
    - Łączenie sygnałów z wielu modeli
    - Uruchamianie strategii awaryjnych gdy AI zawodzi
    """
    
    def __init__(
        self,
        connector,
        fallback_strategy,
        min_confidence: float = 0.6,
        use_fallback: bool = True,
        max_position_size: float = 0.1,
        risk_per_trade: float = 0.02,
        simulation_mode: bool = False,
    ):
        """
        Inicjalizacja mostu AI/Trading.
        
        Args:
            connector: Instancja konektora giełdowego
            fallback_strategy: Strategia awaryjna gdy AI nie dostarcza sygnałów
            min_confidence: Minimalny próg pewności dla sygnałów AI (0-1)
            use_fallback: Czy używać strategii awaryjnej
            max_position_size: Maksymalny rozmiar pozycji jako część kapitału
            risk_per_trade: Ryzyko na transakcję jako część kapitału
            simulation_mode: Czy działać w trybie symulacji
        """
        self.connector = connector
        self.fallback_strategy = fallback_strategy
        self.min_confidence = min_confidence
        self.use_fallback = use_fallback
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        self.simulation_mode = simulation_mode
        
        # Aktualny stan portfela i sygnałów
        self.last_signals = {}
        self.active_positions = {}
        self.prediction_history = []
        
        # Inicjalizacja dziennika decyzji
        self.decision_log = []
        
        logger.info(
            f"AITradeBridge zainicjowany: min_confidence={min_confidence}, "
            f"use_fallback={use_fallback}, max_position_size={max_position_size}, "
            f"risk_per_trade={risk_per_trade}, simulation_mode={simulation_mode}"
        )
    
    def process_ai_signal(
        self,
        model_id: str,
        signal: Dict,
        market_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Przetwarza sygnał z modelu AI i podejmuje decyzję handlową.
        
        Args:
            model_id: Identyfikator modelu AI
            signal: Słownik z sygnałem AI (wymagane pola: action, confidence, price)
            market_data: Opcjonalne dane rynkowe do podjęcia lepszej decyzji
            
        Returns:
            Dict: Decyzja handlowa zawierająca szczegóły zlecenia
        """
        try:
            logger.info(f"Otrzymano sygnał z modelu {model_id}: {signal}")
            
            # Zapisujemy sygnał do historii
            self.last_signals[model_id] = signal
            
            # Sprawdzamy czy sygnał spełnia minimalne wymogi
            if not self._validate_signal(signal):
                logger.warning(f"Sygnał z modelu {model_id} nie przeszedł walidacji")
                return self._get_fallback_decision(market_data, reason="invalid_signal")
            
            # Sprawdzamy czy pewność predykcji jest wystarczająca
            if signal.get("confidence", 0) < self.min_confidence:
                logger.info(
                    f"Pewność sygnału ({signal.get('confidence', 0):.2f}) poniżej progu ({self.min_confidence})"
                )
                return self._get_fallback_decision(market_data, reason="low_confidence")
            
            # Konwertujemy sygnał na konkretne parametry zlecenia
            decision = self._convert_signal_to_order(signal, market_data)
            
            # Logujemy decyzję
            self._log_decision(decision, model_id)
            
            return decision
            
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania sygnału AI: {e}", exc_info=True)
            return self._get_fallback_decision(market_data, reason=f"error: {str(e)}")
    
    def execute_decision(self, decision: Dict) -> Dict:
        """
        Wykonuje decyzję handlową, składając odpowiednie zlecenie.
        
        Args:
            decision: Decyzja handlowa (konwertowany sygnał AI)
            
        Returns:
            Dict: Status wykonania ze szczegółami zlecenia
        """
        try:
            # Jeśli akcja to HOLD, po prostu zwracamy status
            if decision["action"] == "HOLD":
                logger.info("Decyzja: HOLD - brak zlecenia do wykonania")
                return {
                    "status": "success",
                    "action": "HOLD",
                    "order_id": None,
                    "message": "No action taken as per decision"
                }
            
            # W trybie symulacji tylko logujemy decyzję
            if self.simulation_mode:
                logger.info(f"Symulacja: Wykonanie decyzji {decision['action']} dla {decision.get('symbol')}")
                return {
                    "status": "success",
                    "action": decision["action"],
                    "order_id": f"sim_{int(time.time())}",
                    "message": "Order simulated successfully"
                }
            
            # Sprawdzamy czy mamy wszystkie potrzebne dane
            required_fields = ["action", "symbol", "quantity"]
            for field in required_fields:
                if field not in decision:
                    raise ValueError(f"Brak wymaganego pola w decyzji: {field}")
            
            # Przygotowujemy parametry zlecenia
            order_params = {
                "symbol": decision["symbol"],
                "side": "Buy" if decision["action"] == "BUY" else "Sell",
                "order_type": decision.get("order_type", "MARKET"),
                "qty": decision["quantity"],
            }
            
            # Dodajemy opcjonalne parametry
            if "price" in decision and decision.get("order_type") == "LIMIT":
                order_params["price"] = decision["price"]
            
            if "stop_loss" in decision:
                order_params["sl_price"] = decision["stop_loss"]
                
            if "take_profit" in decision:
                order_params["tp_price"] = decision["take_profit"]
            
            # Wykonujemy zlecenie
            logger.info(f"Wykonywanie zlecenia: {order_params}")
            order_result = self.connector.place_order(**order_params)
            
            # Sprawdzamy czy zlecenie powiodło się
            if order_result.get("ret_code") == 0:
                logger.info(f"Zlecenie wykonane pomyślnie: {order_result}")
                return {
                    "status": "success",
                    "action": decision["action"],
                    "order_id": order_result.get("result", {}).get("orderId"),
                    "message": "Order executed successfully",
                    "order_details": order_result.get("result", {})
                }
            else:
                logger.error(f"Błąd podczas wykonywania zlecenia: {order_result}")
                return {
                    "status": "error",
                    "action": decision["action"],
                    "message": f"Order failed: {order_result.get('ret_msg', 'Unknown error')}",
                    "error_code": order_result.get("ret_code")
                }
                
        except Exception as e:
            logger.error(f"Błąd podczas wykonywania decyzji: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Execution error: {str(e)}"
            }
    
    def _validate_signal(self, signal: Dict) -> bool:
        """
        Sprawdza, czy sygnał zawiera wszystkie wymagane pola i spełnia kryteria.
        
        Args:
            signal: Sygnał do walidacji
            
        Returns:
            bool: True jeśli sygnał jest poprawny
        """
        # Sprawdzamy wymagane pola
        required_fields = ["action", "confidence"]
        for field in required_fields:
            if field not in signal:
                logger.warning(f"Brak wymaganego pola w sygnale: {field}")
                return False
        
        # Sprawdzamy wartości
        if signal["action"] not in ["BUY", "SELL", "HOLD"]:
            logger.warning(f"Nieprawidłowa akcja w sygnale: {signal['action']}")
            return False
            
        if not isinstance(signal["confidence"], (int, float)) or signal["confidence"] < 0 or signal["confidence"] > 1:
            logger.warning(f"Nieprawidłowa wartość pewności w sygnale: {signal['confidence']}")
            return False
            
        # Sprawdzamy czy sygnał zawiera cenę (wymagana dla BUY/SELL)
        if signal["action"] in ["BUY", "SELL"] and "price" not in signal:
            logger.warning("Brak ceny w sygnale BUY/SELL")
            return False
            
        return True
    
    def _convert_signal_to_order(self, signal: Dict, market_data: Optional[Dict] = None) -> Dict:
        """
        Konwertuje sygnał AI na parametry zlecenia.
        
        Args:
            signal: Sygnał od modelu AI
            market_data: Dane rynkowe (opcjonalne)
            
        Returns:
            Dict: Kompletna decyzja handlowa
        """
        # Jeśli sygnał to HOLD, zwracamy prostą decyzję
        if signal["action"] == "HOLD":
            return {
                "action": "HOLD",
                "confidence": signal["confidence"],
                "reasons": signal.get("reasons", ["AI predicted HOLD"]),
                "timestamp": int(time.time())
            }
        
        # Podstawowe dane decyzji
        decision = {
            "action": signal["action"],
            "confidence": signal["confidence"],
            "reasons": signal.get("reasons", [f"AI predicted {signal['action']}"]),
            "timestamp": int(time.time()),
            "symbol": signal.get("symbol", "BTCUSDT"),  # domyślnie BTC
            "price": signal.get("price"),
            "order_type": "MARKET"  # domyślnie zlecenie rynkowe
        }
        
        # Jeśli mamy cenę, możemy użyć zlecenia LIMIT
        if signal.get("price"):
            decision["order_type"] = "LIMIT"
        
        # Jeśli mamy Stop Loss i Take Profit w sygnale
        if signal.get("stop_loss"):
            decision["stop_loss"] = signal["stop_loss"]
            
        if signal.get("take_profit"):
            decision["take_profit"] = signal["take_profit"]
        
        # Pobieramy bilans portfela do określenia wielkości pozycji
        try:
            # W trybie symulacji używamy pozornych danych
            if self.simulation_mode:
                wallet_balance = 10000.0
            else:
                wallet_data = self.connector.get_wallet_balance()
                # Pobieramy saldo USDT
                wallet_balance = 0
                for coin in wallet_data.get("result", {}).get("list", [{}])[0].get("coin", []):
                    if coin.get("coin") == "USDT":
                        wallet_balance = float(coin.get("walletBalance", 0))
                        break
            
            # Obliczamy wielkość pozycji
            position_value = wallet_balance * min(self.risk_per_trade, self.max_position_size)
            
            # Obliczamy ilość na podstawie ceny (market price jeśli nie mamy ceny)
            price = signal.get("price")
            if not price and market_data and "last_price" in market_data:
                price = market_data["last_price"]
            elif not price:
                # W przypadku braku ceny, pobieramy z API lub używamy symulacji
                if self.simulation_mode:
                    price = 65000.0  # przykładowa cena BTC
                else:
                    ticker_data = self.connector.get_ticker(decision["symbol"])
                    price = float(ticker_data.get("result", {}).get("list", [{}])[0].get("lastPrice", 0))
            
            # Obliczamy ilość
            quantity = position_value / float(price) if float(price) > 0 else 0
            
            # Zaokrąglamy zgodnie z wymaganiami giełdy (tu uproszczone)
            quantity = round(quantity, 6)
            
            decision["quantity"] = quantity
            decision["estimated_value"] = quantity * float(price)
            
            logger.info(
                f"Konwersja sygnału na zlecenie: {decision['action']} {decision['symbol']} "
                f"ilość: {quantity:.6f}, wartość: ${decision['estimated_value']:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Błąd podczas określania wielkości pozycji: {e}", exc_info=True)
            # W przypadku błędu, używamy bezpiecznej małej wielkości
            decision["quantity"] = 0.001  # Minimalna ilość
            decision["estimated_value"] = decision["quantity"] * (float(signal.get("price", 50000)))
        
        return decision
    
    def _get_fallback_decision(self, market_data: Optional[Dict] = None, reason: str = "unknown") -> Dict:
        """
        Generuje decyzję awaryjną gdy sygnał AI jest niedostępny lub niewiarygodny.
        
        Args:
            market_data: Dane rynkowe do przekazania do strategii awaryjnej
            reason: Powód użycia strategii awaryjnej
            
        Returns:
            Dict: Decyzja awaryjna
        """
        # Jeśli nie używamy strategii awaryjnej, zwracamy HOLD
        if not self.use_fallback:
            logger.info(f"Strategia awaryjna wyłączona, zwracam HOLD. Powód: {reason}")
            return {
                "action": "HOLD",
                "confidence": 0.3,
                "reasons": [f"Fallback strategy disabled. Original issue: {reason}"],
                "timestamp": int(time.time())
            }
        
        # Używamy strategii awaryjnej aby wygenerować sygnał
        try:
            if market_data is None:
                market_data = {}
            
            logger.info(f"Używam strategii awaryjnej. Powód: {reason}")
            fallback_signal = self.fallback_strategy.generate_signal(market_data)
            
            # Konwertujemy sygnał na decyzję
            decision = self._convert_signal_to_order(fallback_signal, market_data)
            
            # Dodajemy informację, że to decyzja awaryjna
            if "reasons" not in decision:
                decision["reasons"] = []
            decision["reasons"].insert(0, f"Fallback strategy activated. Reason: {reason}")
            decision["fallback"] = True
            
            # Logujemy decyzję
            self._log_decision(decision, "fallback_strategy")
            
            return decision
            
        except Exception as e:
            logger.error(f"Błąd podczas generowania decyzji awaryjnej: {e}", exc_info=True)
            # W przypadku błędu zwracamy bezpieczny HOLD
            return {
                "action": "HOLD",
                "confidence": 0.1,
                "reasons": [f"Error in fallback strategy: {str(e)}. Original issue: {reason}"],
                "timestamp": int(time.time()),
                "fallback": True
            }
    
    def _log_decision(self, decision: Dict, source: str) -> None:
        """
        Zapisuje decyzję do historii dla przyszłej analizy.
        
        Args:
            decision: Decyzja handlowa
            source: Źródło decyzji (ID modelu lub 'fallback')
        """
        log_entry = decision.copy()
        log_entry["source"] = source
        log_entry["timestamp"] = int(time.time())
        
        self.decision_log.append(log_entry)
        
        # Maksymalnie przechowujemy 1000 ostatnich decyzji
        if len(self.decision_log) > 1000:
            self.decision_log.pop(0)
        
        # Zapisujemy decyzje do pliku (append-only)
        try:
            # Upewniamy się, że folder logs istnieje
            os.makedirs("logs", exist_ok=True)
            
            with open("logs/trading_decisions.log", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania decyzji do pliku: {e}")
    
    def get_decision_history(self, limit: int = 100) -> List[Dict]:
        """
        Zwraca historię decyzji handlowych.
        
        Args:
            limit: Maksymalna liczba decyzji do zwrócenia
            
        Returns:
            List[Dict]: Lista decyzji (od najnowszej do najstarszej)
        """
        return self.decision_log[-limit:][::-1]
    
    def analyze_model_performance(self, model_id: str, days: int = 30) -> Dict:
        """
        Analizuje skuteczność sygnałów z konkretnego modelu AI.
        
        Args:
            model_id: ID modelu do analizy
            days: Liczba dni historii do analizy
            
        Returns:
            Dict: Statystyki skuteczności modelu
        """
        # Funkcja zaślepka - w rzeczywistej implementacji analizowałaby historię transakcji
        return {
            "model_id": model_id,
            "period_days": days,
            "total_signals": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "max_drawdown": 0.0,
            "simulation_only": True
        }


# Przykład użycia (testowy)
if __name__ == "__main__":
    # Symulowany konektor giełdy i strategia awaryjna
    from data.execution.bybit_connector import BybitConnector
    from data.strategies.fallback_strategy import FallbackStrategy
    import pandas as pd
    import numpy as np
    
    # Inicjalizacja symulowanych komponentów
    connector = BybitConnector(simulation_mode=True)
    
    # Symulowane dane rynkowe do strategii awaryjnej
    dates = pd.date_range(start="2025-01-01", periods=100, freq="1h")
    prices = np.random.normal(50000, 1000, 100).cumsum()
    volumes = np.random.normal(100, 20, 100)
    
    market_data = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": volumes
    })
    
    # Inicjalizacja strategii awaryjnej
    fallback = FallbackStrategy()
    
    # Inicjalizacja mostu AI/Trading
    bridge = AITradeBridge(
        connector=connector,
        fallback_strategy=fallback,
        min_confidence=0.6,
        use_fallback=True,
        simulation_mode=True
    )
    
    # Symulowany sygnał AI
    ai_signal = {
        "action": "BUY",
        "confidence": 0.85,
        "price": 65432.10,
        "stop_loss": 64000.0,
        "take_profit": 67000.0,
        "reasons": ["Wykryto silny trend wzrostowy", "Wsparcie na poziomie 64000"]
    }
    
    # Przetwarzamy sygnał
    decision = bridge.process_ai_signal("price_predictor_v1", ai_signal)
    print("Decyzja handlowa:", json.dumps(decision, indent=2))
    
    # Wykonujemy decyzję (symulacja)
    result = bridge.execute_decision(decision)
    print("Wynik wykonania:", json.dumps(result, indent=2))
    
    # Testujemy przypadek niskiej pewności
    low_conf_signal = {
        "action": "SELL",
        "confidence": 0.4,  # poniżej progu
        "price": 65000.0
    }
    
    fallback_decision = bridge.process_ai_signal("uncertain_model", low_conf_signal, market_data)
    print("Decyzja awaryjna:", json.dumps(fallback_decision, indent=2))
