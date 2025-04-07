
import logging
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Union, Tuple

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_trade_bridge.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_trade_bridge")

# Zależności
try:
    from config.config_loader import config_loader
    from data.execution.bybit_connector import BybitConnector
    from data.risk_management.risk_assessment import RiskAssessor
    from data.risk_management.position_sizing import fixed_fractional_position_size, dynamic_position_size
except ImportError as e:
    logger.error(f"Błąd importu: {e}")
    raise

class AITradeBridge:
    """
    Klasa łącząca predykcje modeli AI z wykonaniem zleceń handlowych
    """
    def __init__(self, exchange_connector=None, risk_assessor=None):
        """
        Inicjalizacja mostu AI-Trading
        
        Args:
            exchange_connector: Konektor do giełdy (domyślnie Bybit)
            risk_assessor: Komponent oceny ryzyka
        """
        # Inicjalizacja konektora giełdowego
        self.exchange = exchange_connector or BybitConnector()
        
        # Inicjalizacja oceny ryzyka
        self.risk_assessor = risk_assessor or RiskAssessor()
        
        # Ustawienia
        self.test_mode = config_loader.get_test_mode()
        self.max_position_size = float(config_loader.get_secret("MAX_POSITION_SIZE", "0.1"))
        self.enable_auto_trading = config_loader.get_secret("ENABLE_AUTO_TRADING", "false").lower() in ["true", "1", "yes"]
        
        # Metryki predykcji
        self.prediction_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
        
        # Callback dla integracji wyników
        self.on_trade_executed = None
        
        # Status i metryki
        self.last_signal = None
        self.last_prediction_time = None
        self.active_positions = {}
        self.trade_history = []
        
        # Wątek monitorujący
        self.monitoring_thread = None
        self.monitoring_active = False
        
        logger.info(f"AITradeBridge zainicjalizowany w trybie {'TESTOWYM' if self.test_mode else 'PRODUKCYJNYM'}")
        logger.info(f"Automatyczne wykonywanie zleceń: {'WŁĄCZONE' if self.enable_auto_trading else 'WYŁĄCZONE'}")
    
    def process_ai_prediction(self, prediction: Dict, symbol: str, confidence: float = None, 
                              additional_data: Dict = None) -> Dict:
        """
        Przetwarza predykcję AI i generuje sygnał handlowy
        
        Args:
            prediction (Dict): Predykcja AI, np. {'action': 'BUY', 'price': 50000.0}
            symbol (str): Symbol pary handlowej (np. 'BTCUSDT')
            confidence (float, optional): Poziom pewności predykcji (0.0-1.0)
            additional_data (Dict, optional): Dodatkowe dane dla kontekstu
            
        Returns:
            Dict: Informacja o przetworzonym sygnale
        """
        # Aktualizacja licznika predykcji
        self.prediction_metrics['total_predictions'] += 1
        self.last_prediction_time = datetime.now()
        
        # Domyślne wartości
        if confidence is None and 'confidence' in prediction:
            confidence = prediction['confidence']
        elif confidence is None:
            confidence = 0.5
            
        if additional_data is None:
            additional_data = {}
            
        # Przygotuj sygnał handlowy
        signal = {
            'action': prediction.get('action', 'HOLD'),  # BUY, SELL, HOLD
            'symbol': symbol,
            'confidence': confidence,
            'price': prediction.get('price'),
            'take_profit': prediction.get('take_profit'),
            'stop_loss': prediction.get('stop_loss'),
            'prediction_time': self.last_prediction_time,
            'metadata': additional_data,
            'processed': False,
            'execute_automatically': self.enable_auto_trading
        }
        
        # Zapisz ostatni sygnał
        self.last_signal = signal
        
        # Walidacja sygnału
        if signal['action'] not in ['BUY', 'SELL', 'HOLD']:
            logger.warning(f"Nieprawidłowa akcja: {signal['action']} dla {symbol}")
            signal['processed'] = True
            signal['status'] = 'INVALID_ACTION'
            return signal
            
        # Sprawdź minimalny poziom pewności
        min_confidence = float(config_loader.get_secret("MIN_TRADE_CONFIDENCE", "0.6"))
        if confidence < min_confidence and signal['action'] != 'HOLD':
            logger.info(f"Zbyt niski poziom pewności ({confidence} < {min_confidence}) dla {symbol} - pomijam")
            signal['processed'] = True
            signal['status'] = 'LOW_CONFIDENCE'
            return signal
            
        # Ocena ryzyka
        risk_assessment = self.risk_assessor.assess_trade_risk(
            symbol=symbol,
            action=signal['action'],
            current_price=signal['price'],
            stop_loss=signal['stop_loss']
        )
        
        signal['risk_assessment'] = risk_assessment
        
        # Sprawdź akceptowalny poziom ryzyka
        if risk_assessment['risk_level'] == 'high' and risk_assessment['risk_score'] > 0.8:
            logger.warning(f"Zbyt wysokie ryzyko ({risk_assessment['risk_score']}) dla {symbol} - pomijam")
            signal['processed'] = True
            signal['status'] = 'HIGH_RISK'
            return signal
            
        # Automatyczne wykonanie zlecenia, jeśli włączone
        if self.enable_auto_trading and signal['action'] != 'HOLD':
            execution_result = self.execute_trade(signal)
            signal.update(execution_result)
            
        return signal
        
    def execute_trade(self, signal: Dict) -> Dict:
        """
        Wykonuje transakcję na podstawie sygnału
        
        Args:
            signal (Dict): Sygnał handlowy
            
        Returns:
            Dict: Wynik wykonania transakcji
        """
        if not signal or signal['action'] == 'HOLD':
            return {'status': 'NO_ACTION', 'processed': True}
            
        symbol = signal['symbol']
        action = signal['action']
        
        try:
            # Pobierz aktualne dane rynkowe
            ticker_data = self.exchange.get_tickers(symbol)
            if not ticker_data['success']:
                logger.error(f"Nie udało się pobrać danych tickera dla {symbol}: {ticker_data['error_message']}")
                return {'status': 'MARKET_DATA_ERROR', 'processed': True}
                
            # Ustal aktualną cenę
            current_price = float(ticker_data['data']['list'][0]['lastPrice'])
            signal['market_price'] = current_price
            
            # Ustal wielkość pozycji
            account_data = self.exchange.get_wallet_balance()
            if not account_data['success']:
                logger.error(f"Nie udało się pobrać danych portfela: {account_data['error_message']}")
                return {'status': 'ACCOUNT_DATA_ERROR', 'processed': True}
                
            # Znajdź USDT lub quote asset
            balance = 0
            for asset in account_data['data']['list'][0]['coin']:
                if asset['coin'] == 'USDT':  # Można dostosować do innych walut bazowych
                    balance = float(asset['availableToWithdraw'])
                    break
                    
            if balance <= 0:
                logger.error(f"Brak dostępnego salda dla transakcji")
                return {'status': 'INSUFFICIENT_BALANCE', 'processed': True}
                
            # Oblicz wielkość zlecenia
            risk_per_trade = float(config_loader.get_secret("RISK_PER_TRADE", "0.02"))  # 2% domyślnie
            capital = balance * self.max_position_size  # Ogranicz do maksymalnej wielkości pozycji
            
            # Określ stop loss, jeśli nie podano
            stop_loss = signal.get('stop_loss')
            if stop_loss is None:
                if action == 'BUY':
                    stop_loss = current_price * 0.98  # Przykładowy stop loss -2%
                else:
                    stop_loss = current_price * 1.02  # Przykładowy stop loss +2%
                    
            # Oblicz odległość do stop loss
            stop_distance = abs(current_price - stop_loss)
            
            # Określ wielkość pozycji
            volatility = signal['metadata'].get('volatility', 0.01)  # Domyślnie 1%
            position_size = dynamic_position_size(
                capital=capital,
                risk_per_trade=risk_per_trade,
                stop_loss_distance=stop_distance,
                market_volatility=volatility
            )
            
            # Oblicz ilość kontraktów
            quantity = position_size / current_price
            
            # Zaokrąglij ilość zgodnie z regułami giełdy (przykład)
            quantity = round(quantity, 6)  # Dostosuj do wymagań giełdy
            
            # Określ TP, jeśli nie podano
            take_profit = signal.get('take_profit')
            if take_profit is None:
                if action == 'BUY':
                    take_profit = current_price * 1.03  # Przykładowy TP +3%
                else:
                    take_profit = current_price * 0.97  # Przykładowy TP -3%
            
            # Wykonaj zlecenie
            side = 'Buy' if action == 'BUY' else 'Sell'
            
            # Zapisz parametry zlecenia
            order_params = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'order_time': datetime.now()
            }
            
            # Wykonanie zlecenia na giełdzie
            if not self.test_mode:
                order_result = self.exchange.place_order(
                    symbol=symbol,
                    side=side, 
                    order_type='Market',
                    qty=quantity,
                    take_profit=take_profit,
                    stop_loss=stop_loss
                )
                
                if not order_result['success']:
                    logger.error(f"Błąd wykonania zlecenia: {order_result['error_message']}")
                    return {
                        'status': 'ORDER_ERROR',
                        'error': order_result['error_message'],
                        'processed': True
                    }
                    
                order_params['order_id'] = order_result['data']['orderId']
                order_params['status'] = 'EXECUTED'
                
            else:
                # W trybie testowym symuluj wykonanie
                logger.info(f"TRYB TESTOWY: Symulacja zlecenia {side} {quantity} {symbol} @ {current_price}")
                order_params['order_id'] = f"test-{int(time.time())}"
                order_params['status'] = 'TEST_EXECUTED'
            
            # Zapisz w historii transakcji
            self.trade_history.append(order_params)
            
            # Dodaj do aktywnych pozycji
            self.active_positions[symbol] = order_params
            
            # Wywołaj callback, jeśli istnieje
            if self.on_trade_executed:
                self.on_trade_executed(order_params)
                
            logger.info(f"Zlecenie {side} {quantity} {symbol} wykonane pomyślnie")
            
            return {
                'status': order_params['status'],
                'order_id': order_params['order_id'],
                'quantity': quantity,
                'price': current_price,
                'processed': True
            }
            
        except Exception as e:
            logger.error(f"Błąd podczas wykonywania transakcji: {e}")
            return {
                'status': 'EXECUTION_ERROR',
                'error': str(e),
                'processed': True
            }
    
    def start_position_monitoring(self, interval=60):
        """
        Uruchamia wątek monitorujący otwarte pozycje
        
        Args:
            interval (int): Interwał sprawdzania w sekundach
        """
        if self.monitoring_thread is not None and self.monitoring_active:
            logger.warning("Monitoring pozycji już uruchomiony")
            return
            
        self.monitoring_active = True
        
        def monitoring_job():
            while self.monitoring_active:
                try:
                    self.check_open_positions()
                except Exception as e:
                    logger.error(f"Błąd w monitoringu pozycji: {e}")
                    
                time.sleep(interval)
                
        self.monitoring_thread = threading.Thread(target=monitoring_job)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info(f"Monitoring pozycji uruchomiony (interwał: {interval}s)")
    
    def stop_position_monitoring(self):
        """Zatrzymuje wątek monitorujący"""
        self.monitoring_active = False
        if self.monitoring_thread is not None:
            self.monitoring_thread.join(timeout=1)
            self.monitoring_thread = None
            logger.info("Monitoring pozycji zatrzymany")
    
    def check_open_positions(self):
        """Sprawdza status otwartych pozycji i aktualizuje metryki"""
        if not self.active_positions:
            return
            
        for symbol, position in list(self.active_positions.items()):
            try:
                # Pobierz aktualną cenę
                ticker_data = self.exchange.get_tickers(symbol)
                if not ticker_data['success']:
                    continue
                    
                current_price = float(ticker_data['data']['list'][0]['lastPrice'])
                entry_price = position['price']
                side = position['side']
                
                # Oblicz zysk/stratę
                if side == 'Buy':
                    pnl_percent = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_percent = (entry_price - current_price) / entry_price * 100
                    
                # Aktualizuj pozycję
                position['current_price'] = current_price
                position['pnl_percent'] = pnl_percent
                position['last_checked'] = datetime.now()
                
                # Sprawdź, czy pozycja została zamknięta przez TP/SL
                if not self.test_mode:
                    # Sprawdź status zlecenia
                    order_status = self.exchange.get_active_orders(symbol=symbol, order_id=position['order_id'])
                    
                    if not order_status['success'] or not order_status['data']['list']:
                        # Zlecenie nie jest już aktywne, sprawdź historię
                        history = self.exchange.get_order_history(symbol=symbol)
                        
                        if history['success']:
                            for order in history['data']['list']:
                                if order['orderId'] == position['order_id']:
                                    if order['orderStatus'] == 'Filled':
                                        # Zamknięcie pozycji
                                        self._update_metrics(position, current_price)
                                        del self.active_positions[symbol]
                                        
                                        logger.info(f"Pozycja {symbol} zamknięta z wynikiem {pnl_percent:.2f}%")
                                        break
                
                # W trybie testowym symuluj zamknięcie pozycji po osiągnięciu TP/SL
                else:
                    take_profit = position.get('take_profit')
                    stop_loss = position.get('stop_loss')
                    
                    if (side == 'Buy' and (current_price >= take_profit or current_price <= stop_loss)) or \
                       (side == 'Sell' and (current_price <= take_profit or current_price >= stop_loss)):
                        # Symuluj zamknięcie pozycji
                        self._update_metrics(position, current_price)
                        del self.active_positions[symbol]
                        
                        logger.info(f"TRYB TESTOWY: Pozycja {symbol} zamknięta z wynikiem {pnl_percent:.2f}%")
                
            except Exception as e:
                logger.error(f"Błąd podczas sprawdzania pozycji {symbol}: {e}")
    
    def _update_metrics(self, position, close_price):
        """Aktualizuje metryki predykcji na podstawie zamkniętej pozycji"""
        entry_price = position['price']
        side = position['side']
        
        # Oblicz zysk/stratę
        if side == 'Buy':
            pnl_percent = (close_price - entry_price) / entry_price * 100
            was_profitable = close_price > entry_price
        else:
            pnl_percent = (entry_price - close_price) / entry_price * 100
            was_profitable = close_price < entry_price
            
        # Aktualizuj metryki
        if was_profitable:
            self.prediction_metrics['profitable_trades'] += 1
            self.prediction_metrics['correct_predictions'] += 1
            
            # Aktualizuj średni zysk
            current_total = self.prediction_metrics['avg_profit'] * (self.prediction_metrics['profitable_trades'] - 1)
            new_avg = (current_total + pnl_percent) / self.prediction_metrics['profitable_trades']
            self.prediction_metrics['avg_profit'] = new_avg
        else:
            self.prediction_metrics['losing_trades'] += 1
            
            # Aktualizuj średnią stratę
            current_total = self.prediction_metrics['avg_loss'] * (self.prediction_metrics['losing_trades'] - 1)
            new_avg = (current_total + abs(pnl_percent)) / self.prediction_metrics['losing_trades']
            self.prediction_metrics['avg_loss'] = new_avg
            
        # Aktualizuj win rate i accuracy
        total_trades = self.prediction_metrics['profitable_trades'] + self.prediction_metrics['losing_trades']
        self.prediction_metrics['win_rate'] = self.prediction_metrics['profitable_trades'] / total_trades if total_trades > 0 else 0
        self.prediction_metrics['accuracy'] = self.prediction_metrics['correct_predictions'] / self.prediction_metrics['total_predictions']
        
        # Oblicz profit factor
        total_profit = self.prediction_metrics['avg_profit'] * self.prediction_metrics['profitable_trades']
        total_loss = self.prediction_metrics['avg_loss'] * self.prediction_metrics['losing_trades']
        self.prediction_metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Dodaj zamkniętą pozycję do historii z ostatecznym wynikiem
        closed_position = position.copy()
        closed_position['close_price'] = close_price
        closed_position['pnl_percent'] = pnl_percent
        closed_position['close_time'] = datetime.now()
        closed_position['profitable'] = was_profitable
        
        self.trade_history.append(closed_position)
    
    def get_metrics(self):
        """Zwraca metryki predykcji i wykonania zleceń"""
        return {
            'prediction_metrics': self.prediction_metrics,
            'active_positions': len(self.active_positions),
            'total_trades': len(self.trade_history),
            'last_signal': self.last_signal,
            'last_prediction_time': self.last_prediction_time
        }
    
    def get_mock_prediction(self, symbol: str) -> Dict:
        """
        Generuje testową predykcję do celów integracji
        
        Args:
            symbol (str): Symbol pary handlowej
            
        Returns:
            Dict: Testowa predykcja
        """
        # Pobierz aktualną cenę
        ticker_data = self.exchange.get_tickers(symbol)
        if not ticker_data['success']:
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'price': 0.0,
                'take_profit': 0.0,
                'stop_loss': 0.0
            }
            
        current_price = float(ticker_data['data']['list'][0]['lastPrice'])
        
        # Losowa akcja (BUY, SELL, HOLD)
        actions = ['BUY', 'SELL', 'HOLD']
        action = actions[np.random.randint(0, len(actions))]
        
        # Losowy poziom pewności
        confidence = np.random.uniform(0.5, 0.95)
        
        # TP/SL
        take_profit = current_price * 1.03 if action == 'BUY' else current_price * 0.97
        stop_loss = current_price * 0.98 if action == 'BUY' else current_price * 1.02
        
        return {
            'action': action,
            'confidence': confidence,
            'price': current_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss
        }

# Główna integracja - przykład użycia
if __name__ == "__main__":
    # Inicjalizacja konfiguracji
    from config.config_loader import initialize_configuration
    config = initialize_configuration()
    
    # Utwórz konektor giełdowy (w trybie testowym)
    bybit = BybitConnector(test_mode=True)
    
    # Utwórz most AI-Trading
    bridge = AITradeBridge(exchange_connector=bybit)
    
    # Testowa predykcja
    symbol = "BTCUSDT"
    mock_prediction = bridge.get_mock_prediction(symbol)
    print(f"Testowa predykcja: {mock_prediction}")
    
    # Przetwórz predykcję
    result = bridge.process_ai_prediction(
        prediction=mock_prediction,
        symbol=symbol,
        additional_data={'volatility': 0.02, 'market_trend': 'bullish'}
    )
    
    print(f"Wynik przetwarzania: {result}")
    
    # Uruchom monitoring pozycji
    bridge.start_position_monitoring(interval=10)
    
    # Poczekaj na sprawdzenie pozycji
    time.sleep(30)
    
    # Zatrzymaj monitoring
    bridge.stop_position_monitoring()
    
    # Wyświetl metryki
    print(f"Metryki: {bridge.get_metrics()}")
