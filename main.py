"""
AI Trading System - Główny moduł aplikacji
-----------------------------------------
System integrujący modele AI z giełdami kryptowalut.
"""

import logging
import os
import sys
import time
from datetime import datetime
import traceback

# Obsługa zmiennych środowiskowych
try:
    from dotenv import load_dotenv
    # Konfiguracja środowiska przed importami
    load_dotenv()
    print("Zmienne środowiskowe załadowane z .env")
except ImportError:
    print("Moduł dotenv nie jest zainstalowany. Używam zmiennych systemowych.")

# Tworzenie katalogu logów
os.makedirs("logs", exist_ok=True)

# Konfiguracja systemu logowania
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_level_num = getattr(logging, log_level, logging.INFO)

# Konfiguracja formattera i handlerów
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')

# Handler do konsoli
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

# Handler do pliku
file_handler = logging.FileHandler("logs/app.log")
file_handler.setFormatter(log_formatter)

# Konfiguracja głównego loggera
root_logger = logging.getLogger()
root_logger.setLevel(log_level_num)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Logger dla modułu głównego
logger = logging.getLogger("main")
logger.info(f"Inicjalizacja aplikacji w trybie {os.getenv('APP_ENV', 'development')}")
logger.info(f"Poziom logowania: {log_level}")

try:
    # Importy modułów projektu - będą zaimportowane tylko jeśli środowisko jest poprawnie skonfigurowane
    try:
        from data.execution.bybit_connector import BybitConnector
        from data.utils.api_handler import APIHandler
        from data.data.market_data_fetcher import MarketDataFetcher
        from flask import Flask, jsonify, render_template
    except ImportError as e:
        logger.error(f"Błąd importu modułów: {e}")
        from flask import Flask, jsonify, render_template

    # Inicjalizacja Flask
    app = Flask(__name__, 
                template_folder="templates",
                static_folder="static")

    # Konfiguracja Bybit Connector w trybie symulacji
    logger.info("Inicjalizacja połączenia z Bybit...")

    # Sprawdzenie kluczy API - używanie trybu symulacji, jeśli brak kluczy
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    test_mode = os.getenv("TEST_MODE", "true").lower() in ["true", "1", "t"]

    # Jeśli brak kluczy lub to przykładowe klucze, włącz tryb symulacji
    simulation_mode = not api_key or api_key == "simulation_mode_key" or not api_secret

    if simulation_mode:
        logger.warning("Klucze API nie są skonfigurowane - uruchamianie w trybie symulacji")
    else:
        logger.info(f"Łączenie z Bybit w trybie {'testnet' if test_mode else 'mainnet'}")

    bybit = BybitConnector(
        api_key=api_key,
        api_secret=api_secret,
        use_testnet=test_mode,
        simulation_mode=simulation_mode
    )

    # Funkcja do testowania połączenia z Bybit
    @app.route('/api/test-bybit-connection')
    def test_bybit_connection():
        logger.info("Testowanie połączenia z Bybit...")
        try:
            start_time = time.time()
            connected = bybit.test_connectivity()
            response_time = time.time() - start_time

            if connected:
                # Pobierz ticker dla BTC/USDT
                btc_data = bybit.get_ticker("BTCUSDT")

                return jsonify({
                    'success': True,
                    'message': f'Połączenie z Bybit działa poprawnie! Czas odpowiedzi: {response_time:.2f}s',
                    'response_time': response_time,
                    'mode': 'testnet' if test_mode else 'mainnet',
                    'simulation': simulation_mode,
                    'btc_data': btc_data
                })
            else:
                logger.error("Test połączenia z Bybit nieudany")
                return jsonify({
                    'success': False,
                    'message': 'Nie można połączyć się z Bybit API',
                    'response_time': response_time,
                    'mode': 'testnet' if test_mode else 'mainnet',
                    'simulation': simulation_mode
                }), 500

        except Exception as e:
            logger.error(f"Błąd podczas testowania połączenia z Bybit: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'Błąd: {str(e)}',
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }), 500

    # Dashboard
    @app.route('/')
    def dashboard():
        """Główny dashboard aplikacji"""
        try:
            return render_template('dashboard.html')
        except Exception as e:
            logger.error(f"Błąd renderowania dashboard: {str(e)}")
            return f"Błąd renderowania dashboard: {str(e)}", 500

    # API endpoint dla danych portfela
    @app.route('/api/portfolio')
    def get_portfolio():
        """API endpoint zwracający dane portfela"""
        try:
            logger.info("Pobieranie danych portfela...")

            if simulation_mode:
                # Dane symulowane dla trybu demo
                return jsonify({
                    'success': True,
                    'balance': {
                        'USDT': 10000.0,
                        'BTC': 0.5,
                        'ETH': 5.0
                    },
                    'positions': [
                        {'symbol': 'BTCUSDT', 'amount': 0.1, 'entry_price': 65000.0},
                        {'symbol': 'ETHUSDT', 'amount': 2.0, 'entry_price': 3500.0}
                    ],
                    'simulation_mode': True
                })

            # Rzeczywiste dane z Bybit
            wallet_data = bybit.get_wallet_balance()
            logger.debug(f"Otrzymane dane portfela: {wallet_data}")

            # Sprawdź poprawność danych
            if 'result' not in wallet_data or 'list' not in wallet_data['result']:
                logger.error(f"Nieprawidłowy format danych portfela: {wallet_data}")
                return jsonify({
                    'success': False,
                    'message': 'Nieprawidłowy format danych z API',
                    'data': wallet_data
                }), 500

            return jsonify({
                'success': True,
                'data': wallet_data['result'],
                'simulation_mode': False
            })

        except Exception as e:
            error_details = {
                'success': False,
                'message': f'Błąd pobierania danych portfela: {str(e)}',
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
            logger.error(f"Błąd API /api/portfolio: {error_details['message']}")
            return jsonify(error_details), 500

    # API endpoint do pobierania danych rynkowych
    @app.route('/api/market-data/<symbol>')
    def get_market_data(symbol):
        """API endpoint zwracający dane rynkowe dla danego symbolu"""
        try:
            logger.info(f"Pobieranie danych rynkowych dla {symbol}...")

            # Pobierz ticker dla danego symbolu
            ticker_data = bybit.get_ticker(symbol)

            # Sprawdź poprawność danych
            if 'result' not in ticker_data or 'list' not in ticker_data['result']:
                logger.error(f"Nieprawidłowy format danych tickera: {ticker_data}")
                return jsonify({
                    'success': False,
                    'message': 'Nieprawidłowy format danych z API',
                    'data': ticker_data
                }), 500

            return jsonify({
                'success': True,
                'data': ticker_data['result'],
                'timestamp': datetime.now().isoformat(),
                'simulation_mode': simulation_mode
            })

        except Exception as e:
            error_details = {
                'success': False,
                'message': f'Błąd pobierania danych rynkowych: {str(e)}',
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
            logger.error(f"Błąd API /api/market-data/{symbol}: {error_details['message']}")
            return jsonify(error_details), 500

    # Uruchomienie aplikacji
    if __name__ == "__main__":
        try:
            logger.info("Uruchamianie serwera Flask na 0.0.0.0:5000")
            app.run(host='0.0.0.0', port=5000, debug=True)
        except Exception as e:
            logger.critical(f"Błąd uruchomienia serwera Flask: {str(e)}")
            sys.exit(1)

except Exception as e:
    logger.critical(f"Krytyczny błąd podczas inicjalizacji aplikacji: {str(e)}")
    logger.critical(traceback.format_exc())
    sys.exit(1)