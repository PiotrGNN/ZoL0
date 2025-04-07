
"""
AI Trade Bridge - most między decyzjami AI a wykonywaniem zleceń na giełdzie.
Przetwarza sygnały z modeli AI na konkretne zlecenia giełdowe.
"""

import logging
import math
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd

from data.execution.bybit_connector import BybitConnector
from data.risk_management.position_sizing import dynamic_position_size
from data.strategies.fallback_strategy import FallbackStrategy

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/ai_trade_bridge.log"), logging.StreamHandler()],
)
logger = logging.getLogger("AITradeBridge")


class AITradeBridge:
    """
    Most między AI a systemem tradingowym.
    
    Odpowiada za:
    - Interpretację sygnałów z modeli AI
    - Konwersję sygnałów na konkretne zlecenia
    - Zarządzanie ryzykiem dla każdego zlecenia
    - Wykonywanie zleceń przez exchange connector
    - Logowanie decyzji i zleceń
    """

    def __init__(
        self,
        exchange_connector: BybitConnector,
        min_confidence: float = 0.6,
        risk_per_trade: float = 0.02,
        capital_allocation: float = 0.8,
        default_pair: str = "BTCUSDT",
        use_fallback: bool = True,
    ):
        """
        Inicjalizacja mostu AI-Trading.
        
        Args:
            exchange_connector: Konektor do giełdy
            min_confidence: Minimalna pewność predykcji do wykonania zlecenia (0-1)
            risk_per_trade: Procent kapitału ryzykowany na transakcję (0-1)
            capital_allocation: Procent kapitału dostępny dla systemu (0-1)
            default_pair: Domyślna para handlowa
            use_fallback: Czy używać strategii awaryjnej w razie braku sygnału AI
        """
        self.exchange = exchange_connector
        self.min_confidence = min_confidence
        self.risk_per_trade = risk_per_trade
        self.capital_allocation = capital_allocation
        self.default_pair = default_pair
        self.use_fallback = use_fallback
        
        # Inicjalizacja strategii awaryjnej
        self.fallback_strategy = FallbackStrategy() if use_fallback else None
        
        # Historia transakcji
        self.trade_history = []
        
        logger.info(
            f"AITradeBridge zainicjalizowany. Min. pewność: {min_confidence}, "
            f"Ryzyko na transakcję: {risk_per_trade*100}%, "
            f"Domyślna para: {default_pair}"
        )

    def execute_ai_signal(self, signal: Dict, market_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Wykonuje zlecenie na podstawie sygnału z modelu AI.
        
        Args:
            signal: Słownik zawierający sygnał z modelu AI z kluczami:
                - 'action': 'BUY', 'SELL' lub 'HOLD'
                - 'confidence': (opcjonalnie) pewność predykcji (0-1)
                - 'symbol': (opcjonalnie) para handlowa
                - 'price': (opcjonalnie) sugerowana cena wejścia
                - 'stop_loss': (opcjonalnie) poziom stop loss
                - 'take_profit': (opcjonalnie) poziom take profit
                - 'metadata': (opcjonalnie) dodatkowe dane z modelu
            market_data: Opcjonalne dane rynkowe do analizy
            
        Returns:
            Dict: Wynik wykonania zlecenia
        """
        try:
            # Walidacja sygnału
            if not self._validate_signal(signal):
                logger.warning("Odrzucono nieprawidłowy sygnał AI")
                return {"status": "rejected", "reason": "Invalid signal format"}
            
            # Jeśli brak akcji lub pewność poniżej progu, użyj fallback
            if self._should_use_fallback(signal):
                if not self.fallback_strategy or not market_data:
                    logger.info("Brak akcji w sygnale AI i brak możliwości użycia strategii awaryjnej")
                    return {"status": "skipped", "reason": "No action, low confidence, fallback unavailable"}
                
                logger.info("Używam strategii awaryjnej zamiast sygnału AI")
                fallback_signal = self.fallback_strategy.generate_signal(market_data)
                
                # Konwertuj format sygnału fallback do naszego formatu
                signal = {
                    "action": fallback_signal["action"],
                    "confidence": fallback_signal["confidence"],
                    "symbol": signal.get("symbol", self.default_pair),
                    "price": fallback_signal["price"],
                    "stop_loss": fallback_signal.get("stop_loss"),
                    "take_profit": fallback_signal.get("take_profit"),
                    "metadata": {"source": "fallback_strategy", "reasons": fallback_signal.get("reasons", [])},
                }
            
            # Pobierz informacje o dostępnym kapitale
            capital = self._get_available_capital(signal.get("symbol", self.default_pair))
            
            # Jeśli nie mamy kapitału, zwróć błąd
            if capital <= 0:
                logger.warning(f"Niewystarczający kapitał dla sygnału {signal['action']}")
                return {"status": "rejected", "reason": "Insufficient capital"}
            
            # Przygotuj parametry zlecenia
            order_params = self._prepare_order_parameters(signal, capital)
            
            # Jeśli akcja to HOLD lub nie udało się przygotować parametrów, pomiń
            if not order_params:
                logger.info("Pomijam zlecenie (HOLD lub brak parametrów)")
                return {"status": "skipped", "reason": "HOLD or unable to prepare parameters"}
                
            # Logowanie przed wykonaniem
            logger.info(
                f"Wykonuję zlecenie: {order_params['side']} {order_params['symbol']} "
                f"ilość: {order_params['qty']} "
                f"typu: {order_params['order_type']}"
            )
            
            # Wyślij zlecenie na giełdę
            order_result = self.exchange.place_order(
                symbol=order_params["symbol"],
                side=order_params["side"],
                order_type=order_params["order_type"],
                qty=order_params["qty"],
                price=order_params.get("price"),
                time_in_force=order_params.get("time_in_force", "GTC"),
                reduce_only=order_params.get("reduce_only", False),
                tp_price=order_params.get("tp_price"),
                sl_price=order_params.get("sl_price"),
            )
            
            # Zapisz transakcję w historii
            trade_record = {
                "timestamp": datetime.now().isoformat(),
                "signal": signal,
                "order_params": order_params,
                "order_result": order_result,
                "success": "result" in order_result and "orderId" in order_result["result"],
            }
            self.trade_history.append(trade_record)
            
            # Logowanie po wykonaniu
            if trade_record["success"]:
                order_id = order_result["result"]["orderId"]
                logger.info(f"Zlecenie wysłane pomyślnie, ID: {order_id}")
                return {
                    "status": "success",
                    "order_id": order_id,
                    "symbol": order_params["symbol"],
                    "side": order_params["side"],
                    "quantity": order_params["qty"],
                    "order_type": order_params["order_type"],
                    "signal_source": signal.get("metadata", {}).get("source", "ai_model"),
                }
            else:
                error_code = order_result.get("ret_code")
                error_msg = order_result.get("ret_msg", "Unknown error")
                logger.error(f"Błąd zlecenia: {error_code} - {error_msg}")
                return {
                    "status": "failed",
                    "reason": f"API Error {error_code}: {error_msg}",
                    "symbol": order_params["symbol"],
                    "side": order_params["side"],
                }
                
        except Exception as e:
            logger.error(f"Wyjątek podczas wykonywania sygnału AI: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}

    def _validate_signal(self, signal: Dict) -> bool:
        """
        Sprawdza, czy sygnał ma poprawny format.
        
        Args:
            signal: Sygnał do walidacji
            
        Returns:
            bool: True jeśli sygnał jest poprawny, False w przeciwnym razie
        """
        # Sprawdź, czy mamy podstawowe pola
        if not isinstance(signal, dict):
            logger.warning("Sygnał nie jest słownikiem")
            return False
            
        # Sygnał może nie mieć akcji (fallback zostanie użyty)
        # lub mieć akcję HOLD (wtedy nie wykonujemy zlecenia)
        if "action" in signal and signal["action"] not in ["BUY", "SELL", "HOLD"]:
            logger.warning(f"Nieprawidłowa akcja: {signal.get('action')}")
            return False
            
        # Jeśli jest confidence, musi być liczbą między 0 a 1
        if "confidence" in signal:
            try:
                confidence = float(signal["confidence"])
                if not (0 <= confidence <= 1):
                    logger.warning(f"Nieprawidłowa wartość pewności: {confidence}")
                    return False
            except (ValueError, TypeError):
                logger.warning(f"Nie można skonwertować pewności do float: {signal.get('confidence')}")
                return False
                
        return True

    def _should_use_fallback(self, signal: Dict) -> bool:
        """
        Określa, czy powinniśmy użyć strategii awaryjnej zamiast sygnału AI.
        
        Args:
            signal: Sygnał AI
            
        Returns:
            bool: True jeśli należy użyć strategii awaryjnej
        """
        # Jeśli brak akcji lub akcja to HOLD
        if "action" not in signal or signal["action"] == "HOLD":
            return True
            
        # Jeśli pewność jest poniżej minimalnego progu
        if "confidence" in signal and float(signal["confidence"]) < self.min_confidence:
            return True
            
        return False

    def _get_available_capital(self, symbol: str) -> float:
        """
        Pobiera dostępny kapitał dla danej pary handlowej.
        
        Args:
            symbol: Para handlowa (np. "BTCUSDT")
            
        Returns:
            float: Dostępny kapitał w walucie kwotowanej
        """
        try:
            # Pobieramy informacje o portfelu
            wallet_info = self.exchange.get_wallet_balance()
            
            # Parsujemy symbol na walutę bazową i kwotowaną
            quote_currency = symbol[-4:] if symbol[-4:] in ["USDT", "BUSD"] else symbol[-3:]
            
            # Sprawdzamy, czy mamy tę walutę w portfelu
            if "result" not in wallet_info or "list" not in wallet_info["result"]:
                logger.error("Błąd podczas pobierania informacji o portfelu")
                return 0.0
                
            for account in wallet_info["result"]["list"]:
                for coin in account.get("coin", []):
                    if coin["coin"] == quote_currency:
                        available_balance = float(coin["availableToWithdraw"])
                        # Stosujemy alokację kapitału
                        usable_capital = available_balance * self.capital_allocation
                        logger.info(f"Dostępny kapitał: {usable_capital} {quote_currency}")
                        return usable_capital
                        
            logger.warning(f"Nie znaleziono waluty {quote_currency} w portfelu")
            return 0.0
            
        except Exception as e:
            logger.error(f"Błąd podczas pobierania dostępnego kapitału: {e}")
            return 0.0

    def _prepare_order_parameters(self, signal: Dict, capital: float) -> Optional[Dict]:
        """
        Przygotowuje parametry zlecenia na podstawie sygnału AI.
        
        Args:
            signal: Sygnał z modelu AI
            capital: Dostępny kapitał
            
        Returns:
            Dict or None: Parametry zlecenia lub None jeśli nie można przygotować
        """
        try:
            # Jeśli akcja to HOLD, nie wykonujemy zlecenia
            if signal.get("action") == "HOLD":
                return None
                
            # Pobierz podstawowe parametry
            action = signal.get("action")
            symbol = signal.get("symbol", self.default_pair)
            
            # Pobierz aktualną cenę rynkową, jeśli nie została podana
            current_price = signal.get("price")
            if current_price is None:
                ticker_info = self.exchange.get_ticker(symbol)
                if "result" in ticker_info and "list" in ticker_info["result"] and ticker_info["result"]["list"]:
                    current_price = float(ticker_info["result"]["list"][0]["lastPrice"])
                else:
                    logger.error(f"Nie można pobrać aktualnej ceny dla {symbol}")
                    return None
            else:
                current_price = float(current_price)
                
            # Określamy stronę zlecenia
            side = "Buy" if action == "BUY" else "Sell"
            
            # Określamy typ zlecenia (domyślnie Market)
            order_type = "Market"
            
            # Określamy poziomy TP i SL, jeśli nie podano
            stop_loss = signal.get("stop_loss")
            take_profit = signal.get("take_profit")
            
            # Obliczamy rozmiar pozycji na podstawie zarządzania ryzykiem
            risk_per_trade = self.risk_per_trade
            
            # Oblicz wielkość pozycji
            if stop_loss:
                # Oblicz odległość do stop loss
                stop_distance = abs(current_price - float(stop_loss))
                risk_amount = capital * risk_per_trade
                position_size = risk_amount / stop_distance if stop_distance > 0 else 0
            else:
                # Jeśli brak SL, użyj domyślnego ryzyka jako % ceny
                default_sl_percent = 0.02  # 2% 
                position_size = (capital * risk_per_trade) / (current_price * default_sl_percent)
                
                # Ustaw domyślny SL
                if action == "BUY":
                    stop_loss = current_price * (1 - default_sl_percent)
                else:
                    stop_loss = current_price * (1 + default_sl_percent)
                    
            # Ustaw domyślny TP jeśli nie podano (2x odległość do SL)
            if take_profit is None:
                sl_distance = abs(current_price - stop_loss)
                if action == "BUY":
                    take_profit = current_price + (2 * sl_distance)
                else:
                    take_profit = current_price - (2 * sl_distance)
                    
            # Limituj wielkość pozycji do dostępnego kapitału
            max_position_value = capital
            if position_size * current_price > max_position_value:
                position_size = max_position_value / current_price
                
            # Zaokrąglij ilość zgodnie z regułami giełdy 
            # To uproszczone, w praktyce powinieneś pobrać dokładne step size z API
            quantity = round(position_size, 6)
            
            # Sprawdź minimalną wartość zamówienia (zazwyczaj $10 na Bybit)
            min_order_value = 10.0
            if quantity * current_price < min_order_value:
                logger.warning(f"Wartość zlecenia poniżej minimum: ${quantity * current_price:.2f}")
                quantity = min_order_value / current_price
                quantity = round(quantity, 6)
                
            # Przygotuj parametry zlecenia
            order_params = {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "qty": quantity,
                "time_in_force": "GTC",
            }
            
            # Dla zleceń limitowych dodaj cenę
            if order_type == "Limit":
                order_params["price"] = current_price
                
            # Dodaj SL/TP jeśli podano
            if stop_loss:
                order_params["sl_price"] = float(stop_loss)
            if take_profit:
                order_params["tp_price"] = float(take_profit)
                
            logger.info(
                f"Przygotowano parametry zlecenia: {side} {symbol} "
                f"ilość: {quantity} @ {current_price}, "
                f"SL: {stop_loss}, TP: {take_profit}"
            )
            
            return order_params
            
        except Exception as e:
            logger.error(f"Błąd podczas przygotowywania parametrów zlecenia: {e}")
            return None

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """
        Zwraca historię wykonanych transakcji.
        
        Args:
            limit: Maksymalna liczba transakcji do zwrócenia
            
        Returns:
            List[Dict]: Historia transakcji
        """
        return self.trade_history[-limit:]

    def process_market_data(self, market_data: pd.DataFrame) -> Dict:
        """
        Przetwarza dane rynkowe i aktualizuje wewnętrzny stan.
        
        Args:
            market_data: DataFrame z danymi rynkowymi
            
        Returns:
            Dict: Wynik przetwarzania
        """
        try:
            # Przykładowe przetwarzanie danych rynkowych
            # W rzeczywistości możesz chcieć zaktualizować jakieś wewnętrzne metryki
            num_records = len(market_data)
            last_price = market_data["close"].iloc[-1] if "close" in market_data.columns else None
            
            logger.debug(f"Przetworzono {num_records} rekordów danych rynkowych. Ostatnia cena: {last_price}")
            
            return {
                "status": "processed",
                "records": num_records,
                "last_price": last_price,
            }
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania danych rynkowych: {e}")
            return {"status": "error", "reason": str(e)}

    def check_exchange_connectivity(self) -> bool:
        """
        Sprawdza połączenie z giełdą.
        
        Returns:
            bool: True jeśli połączenie działa, False w przeciwnym razie
        """
        return self.exchange.test_connectivity()


# Przykład użycia
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Ładujemy zmienne środowiskowe
    load_dotenv()
    
    # Pobieramy klucze API z .env
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    use_testnet = os.getenv("TEST_MODE", "true").lower() in ["true", "1", "t"]
    
    # Inicjalizujemy konektor
    bybit = BybitConnector(api_key, api_secret, use_testnet)
    
    # Inicjalizujemy AI Trade Bridge
    bridge = AITradeBridge(
        exchange_connector=bybit,
        min_confidence=0.6,
        risk_per_trade=0.02,
        default_pair="BTCUSDT"
    )
    
    # Sprawdzamy połączenie
    if bridge.check_exchange_connectivity():
        print("Połączenie z giełdą działa")
        
        # Przykładowy sygnał AI
        ai_signal = {
            "action": "BUY",
            "confidence": 0.85,
            "symbol": "BTCUSDT",
            "price": None,  # Użyje aktualnej ceny rynkowej
            "metadata": {
                "model_name": "trend_predictor_v1",
                "features": ["price_momentum", "volume_profile", "sentiment"],
                "source": "ai_model",
            }
        }
        
        # Wykonujemy sygnał
        result = bridge.execute_ai_signal(ai_signal)
        print(f"Wynik wykonania sygnału: {result}")
        
        # Przykład sygnału HOLD
        hold_signal = {
            "action": "HOLD",
            "confidence": 0.45,
            "symbol": "BTCUSDT",
        }
        
        # Ten sygnał nie powinien wygenerować zlecenia
        result = bridge.execute_ai_signal(hold_signal)
        print(f"Wynik sygnału HOLD: {result}")
        
    else:
        print("Nie można połączyć się z giełdą")
