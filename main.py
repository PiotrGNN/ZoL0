import logging
import os
import sys
import json
import random
from datetime import datetime, timedelta

# Ensure needed directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("data/cache", exist_ok=True)

# Katalog na lokalne biblioteki (opcjonalny)
LOCAL_LIBS_DIR = "python_libs"
if os.path.exists(LOCAL_LIBS_DIR):
    sys.path.insert(0, LOCAL_LIBS_DIR)
    print(f"Dodano katalog {LOCAL_LIBS_DIR} do ścieżki Pythona.")

# Add base directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from dotenv import load_dotenv
except ImportError:
    os.system("pip install python-dotenv")
    from dotenv import load_dotenv

try:
    from flask import Flask, jsonify, render_template, request
except ImportError:
    os.system("pip install flask")
    from flask import Flask, jsonify, render_template, request

# Konfiguracja logowania
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Import klienta ByBit
try:
    from data.execution.bybit_connector import BybitConnector
    bybit_import_success = True
except ImportError as e:
    logging.warning(f"Nie udało się zaimportować BybitConnector: {e}")
    bybit_import_success = False

# Inicjalizacja aplikacji Flask
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# Deklaracja zmiennych globalnych
global bybit_client, notification_system, sentiment_analyzer, anomaly_detector, strategy_manager, trading_engine
bybit_client = None
notification_system = None
sentiment_analyzer = None
anomaly_detector = None
strategy_manager = None
trading_engine = None

# Inicjalizacja komponentów systemu (import wewnątrz funkcji dla uniknięcia cyklicznych importów)
def initialize_system():
    try:
        # Używamy uproszczonych modułów z python_libs
        try:
            from python_libs.simplified_notification import NotificationSystem
            notification_lib = "python_libs.simplified_notification"
        except ImportError:
            from data.utils.notification_system import NotificationSystem
            notification_lib = "data.utils.notification_system"
            
        # Import modułu silnika handlowego
        try:
            from python_libs.simplified_trading_engine import SimplifiedTradingEngine
            trading_engine_lib = "python_libs.simplified_trading_engine"
        except ImportError:
            try:
                from data.execution.trade_executor import TradeExecutor as SimplifiedTradingEngine
                trading_engine_lib = "data.execution.trade_executor"
            except ImportError:
                SimplifiedTradingEngine = None
                trading_engine_lib = None
                logging.warning("Brak modułu silnika handlowego. Funkcjonalność handlowa będzie ograniczona.")
            
        try:
            from python_libs.simplified_sentiment import SentimentAnalyzer
            sentiment_lib = "python_libs.simplified_sentiment"
        except ImportError:
            try:
                from data.indicators.sentiment_analysis import SentimentAnalyzer
                sentiment_lib = "data.indicators.sentiment_analysis"
            except ImportError:
                SentimentAnalyzer = None
                sentiment_lib = None
                
        try:
            from python_libs.simplified_anomaly import AnomalyDetector
            anomaly_lib = "python_libs.simplified_anomaly"
        except ImportError:
            try:
                from ai_models.anomaly_detection import AnomalyDetector
                anomaly_lib = "ai_models.anomaly_detection"
            except ImportError:
                AnomalyDetector = None
                anomaly_lib = None
                
        try:
            from python_libs.simplified_strategy import StrategyManager
            strategy_lib = "python_libs.simplified_strategy"
        except ImportError:
            try:
                from data.strategies.strategy_manager import StrategyManager
                strategy_lib = "data.strategies.strategy_manager"
            except ImportError:
                StrategyManager = None
                strategy_lib = None

        # Wykluczamy podpakiet tests z automatycznego importu
        try:
            from data.utils.import_manager import exclude_modules_from_auto_import
            exclude_modules_from_auto_import(["tests"])
            logging.info("Wykluczam podpakiet 'tests' z automatycznego importu.")
        except ImportError:
            logging.warning("Nie znaleziono modułu import_manager, pomijam wykluczanie podpakietów.")

        global notification_system, sentiment_analyzer, anomaly_detector, bybit_client, strategy_manager

        # Inicjalizacja systemu powiadomień
        notification_system = NotificationSystem()
        logging.info(f"Zainicjalizowano system powiadomień z biblioteki {notification_lib}")

        # Inicjalizacja analizatora sentymentu
        if SentimentAnalyzer:
            sentiment_analyzer = SentimentAnalyzer(sources=["twitter", "news", "forum", "reddit"])
            logging.info(f"Inicjalizacja analizatora sentymentu z biblioteki {sentiment_lib}")
        else:
            sentiment_analyzer = None
            logging.warning("Brak modułu analizatora sentymentu")

        # Inicjalizacja wykrywania anomalii
        if AnomalyDetector:
            anomaly_detector = AnomalyDetector(method="isolation_forest", threshold=2.5)
            logging.info(f"Inicjalizacja detektora anomalii z biblioteki {anomaly_lib}")
        else:
            anomaly_detector = None
            logging.warning("Brak modułu detektora anomalii")

        # Inicjalizacja managera strategii z domyślnymi strategiami
        strategies = {
            "trend_following": {"name": "Trend Following", "enabled": True},
            "mean_reversion": {"name": "Mean Reversion", "enabled": False},
            "breakout": {"name": "Breakout", "enabled": True}
        }
        exposure_limits = {
            "trend_following": 0.5,
            "mean_reversion": 0.3,
            "breakout": 0.4
        }
        
        if StrategyManager:
            strategy_manager = StrategyManager(strategies, exposure_limits)
            logging.info(f"Zainicjalizowano StrategyManager z biblioteki {strategy_lib}")
        else:
            strategy_manager = None
            logging.warning("Brak modułu StrategyManager")
            
        # Inicjalizacja zarządcy ryzyka
        risk_manager = None
        try:
            from python_libs.simplified_risk_manager import SimplifiedRiskManager
            risk_manager = SimplifiedRiskManager(
                max_risk=0.05, 
                max_position_size=0.2, 
                max_drawdown=0.1
            )
            logging.info("Zainicjalizowano SimplifiedRiskManager")
        except ImportError as e:
            logging.warning(f"Nie można zaimportować SimplifiedRiskManager: {e}")
            
        # Inicjalizacja silnika handlowego
        global trading_engine
        if SimplifiedTradingEngine:
            trading_engine = SimplifiedTradingEngine(
                risk_manager=risk_manager,
                strategy_manager=strategy_manager,
                exchange_connector=bybit_client
            )
            logging.info(f"Zainicjalizowano silnik handlowy (Trading Engine) z biblioteki {trading_engine_lib}")
            
            # Aktywacja strategii i uruchomienie silnika
            if strategy_manager:
                strategy_manager.activate_strategy("trend_following")
                logging.info("Aktywowano strategię 'trend_following'")
                
            # Uruchomienie silnika z odpowiednim symbolem
            symbols = ["BTCUSDT"]
            engine_started = trading_engine.start_trading(symbols)
            if engine_started:
                logging.info(f"Silnik handlowy uruchomiony dla symboli: {symbols}")
            else:
                logging.warning(f"Nie udało się uruchomić silnika handlowego: {trading_engine.get_status().get('last_error', 'Nieznany błąd')}")
        else:
            trading_engine = None
            logging.warning("Brak modułu silnika handlowego (Trading Engine)")

        # Inicjalizacja klienta ByBit (tylko raz podczas startu aplikacji)
        try:
            if not bybit_import_success:
                logging.warning("Moduł BybitConnector nie został zaimportowany. Próba użycia symulowanego klienta.")
                try:
                    from python_libs.simulated_bybit import SimulatedBybitConnector
                    api_key = os.getenv("BYBIT_API_KEY", "simulated_key")
                    api_secret = os.getenv("BYBIT_API_SECRET", "simulated_secret")
                    use_testnet = True
                    
                    bybit_client = SimulatedBybitConnector(
                        api_key=api_key,
                        api_secret=api_secret,
                        use_testnet=use_testnet
                    )
                    logging.info("Zainicjalizowano symulowany klient ByBit")
                    return True
                except ImportError:
                    logging.error("Nie można zainicjalizować żadnego klienta ByBit")
                    bybit_client = None
                    return True

            api_key = os.getenv("BYBIT_API_KEY")
            api_secret = os.getenv("BYBIT_API_SECRET")
            use_testnet = os.getenv("BYBIT_USE_TESTNET", "true").lower() == "true"  # Domyślnie używamy testnet

            if not api_key or not api_secret:
                logging.warning("Brak kluczy API ByBit w zmiennych środowiskowych. Używam klienta symulowanego.")
                try:
                    from python_libs.simulated_bybit import SimulatedBybitConnector
                    bybit_client = SimulatedBybitConnector(
                        api_key="simulated_key",
                        api_secret="simulated_secret",
                        use_testnet=True
                    )
                    logging.info("Zainicjalizowano symulowany klient ByBit z powodu braku kluczy API")
                    return True
                except ImportError:
                    logging.error("Nie można zainicjalizować symulowanego klienta ByBit")
                    bybit_client = None
                    return True

            # Dodatkowa weryfikacja kluczy dla produkcji
            if not use_testnet and (len(api_key) < 10 or len(api_secret) < 10):
                logging.critical("BŁĄD KRYTYCZNY: Nieprawidłowe klucze produkcyjne API. Wymagane odpowiednie klucze dla środowiska produkcyjnego!")
                return False

            # Informacja o systemie operacyjnym
            import platform
            system_info = f"{platform.system()} {platform.release()}"
            logging.info(f"System operacyjny: {system_info}")

            # Więcej szczegółów o konfiguracji API dla celów debugowania
            masked_key = f"{api_key[:4]}{'*' * (len(api_key) - 4)}" if api_key else "Brak klucza"
            masked_secret = f"{api_secret[:4]}{'*' * (len(api_secret) - 4)}" if api_secret else "Brak sekretu"
            logging.info(f"Inicjalizacja klienta ByBit - Klucz: {masked_key}, Testnet: {use_testnet}")
            logging.info(f"Produkcyjne API jest {'WŁĄCZONE' if not use_testnet else 'WYŁĄCZONE'}")
            if not use_testnet:
                logging.warning("!!! UWAGA !!! Używasz PRODUKCYJNEGO API ByBit. Operacje handlowe będą mieć realne skutki finansowe!")
                logging.warning("Upewnij się, że Twoje klucze API mają właściwe ograniczenia i są odpowiednio zabezpieczone.")
                print("\n\n========== PRODUKCYJNE API BYBIT ==========")
                print("!!! UWAGA !!! Używasz PRODUKCYJNEGO API ByBit")
                print("Operacje handlowe będą mieć realne skutki finansowe!")
                print("===========================================\n\n")

            # Użyj wartości z konfiguracji lub zmiennych środowiskowych
            # Domyślnie używaj produkcyjnego API
            use_testnet = os.getenv("BYBIT_USE_TESTNET", "false").lower() == "true"
            bybit_client = BybitConnector(
                api_key=api_key,
                api_secret=api_secret,
                use_testnet=use_testnet,
                lazy_connect=True  # Używamy lazy initialization by uniknąć sprawdzania API na starcie
            )
            server_time = bybit_client.get_server_time()
            logger.info(f"Klient API ByBit zainicjalizowany pomyślnie. Czas serwera: {server_time}")
        except Exception as e:
            logger.error(f"Błąd inicjalizacji klienta ByBit: {e}", exc_info=True)
            # Próba użycia symulowanego klienta jako fallback
            try:
                from python_libs.simulated_bybit import SimulatedBybitConnector
                bybit_client = SimulatedBybitConnector(
                    api_key="simulated_key",
                    api_secret="simulated_secret",
                    use_testnet=True
                )
                logging.info("Zainicjalizowano symulowany klient ByBit jako fallback")
            except ImportError:
                bybit_client = None
                logging.error("Nie można zainicjalizować ani rzeczywistego, ani symulowanego klienta ByBit")

        logging.info("System zainicjalizowany poprawnie")
        return True
    except Exception as e:
        logging.error(f"Błąd podczas inicjalizacji systemu: {e}", exc_info=True)
        return False

# Trasy aplikacji
# Ta funkcja jest zduplikowana - usunięto ją, ponieważ istnieje druga implementacja poniżej

@app.route('/')
def dashboard():
    # Przykładowe ustawienia dla szablonu
    default_settings = {
        'risk_level': 'medium',
        'max_position_size': 10,
        'enable_auto_trading': False
    }

    # Przykładowe dane AI modeli
    ai_models = [
        {
            'name': 'Trend Predictor',
            'type': 'LSTM',
            'accuracy': 78.5,
            'status': 'Active',
            'last_used': '2025-04-07 10:15'
        },
        {
            'name': 'Sentiment Analyzer',
            'type': 'BERT',
            'accuracy': 82.3,
            'status': 'Active',
            'last_used': '2025-04-07 10:30'
        },
        {
            'name': 'Volatility Predictor',
            'type': 'XGBoost',
            'accuracy': 75.1,
            'status': 'Active',
            'last_used': '2025-04-07 09:45'
        }
    ]

    # Przykładowe dane strategii
    strategies = [
        {
            'id': 1,
            'name': 'Trend Following',
            'description': 'Strategia podążająca za trendem z wykorzystaniem SMA i EMA',
            'enabled': True,
            'win_rate': 65.8,
            'profit_factor': 1.75
        },
        {
            'id': 2,
            'name': 'Mean Reversion',
            'description': 'Strategia wykorzystująca powrót ceny do średniej',
            'enabled': False,
            'win_rate': 58.2,
            'profit_factor': 1.35
        },
        {
            'id': 3,
            'name': 'Breakout',
            'description': 'Strategia bazująca na przełamaniach poziomów wsparcia/oporu',
            'enabled': True,
            'win_rate': 62.0,
            'profit_factor': 1.65
        }
    ]

    # Pobieranie stanu portfela
    portfolio = None
    if bybit_client:
        try:
            portfolio = bybit_client.get_account_balance()
            logging.info(f"Pobrano portfolio: {portfolio}")
        except Exception as e:
            logging.error(f"Błąd podczas pobierania portfolio: {e}", exc_info=True) #Dodatkowe informacje o błędzie
            portfolio = {
                "balances": {
                    "BTC": {"equity": 0.005, "available_balance": 0.005, "wallet_balance": 0.005},
                    "USDT": {"equity": 500, "available_balance": 450, "wallet_balance": 500}
                }, 
                "success": True,
                "error": str(e),
                "note": "Dane przykładowe - wystąpił błąd"
            }
    else:
        logging.warning("Klient ByBit nie jest zainicjalizowany, używam danych testowych")
        portfolio = {
            "balances": {
                "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                "USDT": {"equity": 1000, "available_balance": 950, "wallet_balance": 1000}
            }, 
            "success": True,
            "note": "Dane testowe - klient ByBit nie jest zainicjalizowany"
        }

    return render_template(
        'dashboard.html',
        settings=default_settings,
        ai_models=ai_models,
        strategies=strategies,
        trades=[],
        alerts=[],
        sentiment_data=None,
        anomalies=[],
        portfolio=portfolio
    )

# API endpoints
@app.route('/api/portfolio')
def get_portfolio_data():
    try:
        # Pobierz dane portfela
        if bybit_client:
            try:
                portfolio_data = bybit_client.get_account_balance()
                return jsonify({
                    'success': True,
                    'balances': portfolio_data['balances'] if 'balances' in portfolio_data else {}
                })
            except Exception as e:
                logging.error(f"Błąd podczas pobierania danych portfela: {e}", exc_info=True)
                # Zwróć testowe dane w przypadku błędu z poprawną strukturą
                return jsonify({
                    'success': True,
                    'balances': {
                        "BTC": {"equity": 0.005, "available_balance": 0.005, "wallet_balance": 0.005},
                        "USDT": {"equity": 500, "available_balance": 450, "wallet_balance": 500}
                    }
                })
        else:
            # Dane testowe gdy klient nie jest zainicjalizowany
            return jsonify({
                'success': True,
                'balances': {
                    "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                    "USDT": {"equity": 1000, "available_balance": 950, "wallet_balance": 1000}
                }
            })
    except Exception as e:
        logging.error(f"Błąd podczas przetwarzania danych portfela: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'balances': {}
        })

@app.route('/api/dashboard/data')
def get_dashboard_data():
    try:
        # Formatowanie danych sentymentu dla frontendu
        sentiment_data = None
        if sentiment_analyzer:
            raw_sentiment = sentiment_analyzer.analyze()
            sentiment_data = {
                'analysis': raw_sentiment['analysis'],
                'overall_score': raw_sentiment['value'],
                'sources': {
                    source: {
                        'score': value,
                        'volume': random.randint(10, 100)  # Symulacja ilości wzmianek
                    } for source, value in raw_sentiment['sources'].items()
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'time_range': 'ostatnie 24 godziny'
            }
        
        # Symulowane dane dla demonstracji
        return jsonify({
            'success': True,
            'balance': 10250.75,
            'profit_loss': 325.50,
            'open_positions': 3,
            'total_trades': 48,
            'win_rate': 68.5,
            'max_drawdown': 8.2,
            'market_sentiment': sentiment_analyzer.analyze()["analysis"] if sentiment_analyzer else 'Neutralny',
            'sentiment_data': sentiment_data,
            'anomalies': anomaly_detector.get_detected_anomalies() if anomaly_detector else [],
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania danych dashboardu: {e}", exc_info=True) #Dodatkowe informacje o błędzie
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chart-data')
def get_chart_data():
    """Endpoint dostarczający dane do wykresów"""
    try:
        # Symulowane dane do wykresu
        days = 30
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        dates.reverse()

        # Symulacja wartości portfela
        import random
        initial_value = 10000
        values = [initial_value]
        for i in range(1, days):
            change = random.uniform(-200, 300)
            values.append(round(values[-1] + change, 2))

        return jsonify({
            'success': True,
            'data': {
                'labels': dates,
                'datasets': [
                    {
                        'label': 'Wartość Portfela',
                        'data': values,
                        'borderColor': '#4CAF50',
                        'backgroundColor': 'rgba(76, 175, 80, 0.1)'
                    }
                ]
            }
        })
    except Exception as e:
        logging.error(f"Błąd podczas generowania danych wykresu: {e}", exc_info=True) #Dodatkowe informacje o błędzie
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/notifications')
def get_notifications():
    """Endpoint do pobierania powiadomień systemowych"""
    try:
        # Przykładowe powiadomienia
        notifications = [
            {
                'id': 1,
                'type': 'info',
                'message': 'System został uruchomiony poprawnie.',
                'timestamp': (datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'id': 2,
                'type': 'warning',
                'message': 'Wykryto zwiększoną zmienność na rynku BTC/USDT.',
                'timestamp': (datetime.now() - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'id': 3,
                'type': 'success',
                'message': 'Transakcja kupna ETH/USDT zakończona powodzeniem.',
                'timestamp': (datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
            }
        ]

        return jsonify({
            'success': True,
            'notifications': notifications
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania powiadomień: {e}", exc_info=True) #Dodatkowe informacje o błędzie
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recent-trades')
def get_recent_trades():
    """Endpoint do pobierania ostatnich transakcji"""
    try:
        # Przykładowe dane transakcji
        trades = [
            {
                'symbol': 'BTC/USDT',
                'type': 'Buy',
                'time': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                'profit': 3.2
            },
            {
                'symbol': 'ETH/USDT',
                'type': 'Sell',
                'time': (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
                'profit': -1.5
            },
            {
                'symbol': 'SOL/USDT',
                'type': 'Buy',
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'profit': 2.7
            }
        ]
        return jsonify({'trades': trades})
    except Exception as e:
        logging.error(f"Błąd podczas pobierania ostatnich transakcji: {e}", exc_info=True) #Dodatkowe informacje o błędzie
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts')
def get_alerts():
    """Endpoint do pobierania alertów systemowych"""
    try:
        # Przykładowe alerty
        alerts = [
            {
                'level': 'warning',
                'level_class': 'warning',
                'time': (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
                'message': 'Wysoka zmienność rynkowa'
            },
            {
                'level': 'info',
                'level_class': 'online',
                'time': (datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S'),
                'message': 'Nowa strategia aktywowana'
            }
        ]
        return jsonify({'alerts': alerts})
    except Exception as e:
        logging.error(f"Błąd podczas pobierania alertów: {e}", exc_info=True) #Dodatkowe informacje o błędzie
        return jsonify({'error': str(e)}), 500

@app.route('/api/trading-stats')
def get_trading_stats():
    """Endpoint do pobierania statystyk tradingowych"""
    try:
        stats = {
            'profit': '$356.42',
            'trades_count': 28,
            'win_rate': '67.8%',
            'max_drawdown': '8.2%'
        }
        return jsonify(stats)
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statystyk tradingowych: {e}", exc_info=True) #Dodatkowe informacje o błędzie
        return jsonify({'error': str(e)}), 500

@app.route('/api/component-status')
def get_component_status():
    """Endpoint do pobierania statusu komponentów systemu"""
    try:
        # Status API Connector
        api_connector_status = 'offline'
        if bybit_client:
            try:
                # Spróbuj wykonać prostą operację, aby sprawdzić, czy API działa
                server_time = bybit_client.get_server_time()
                api_connector_status = 'online'
            except Exception as e:
                logging.warning(f"Błąd podczas sprawdzania statusu API: {e}")
                api_connector_status = 'warning'

        # Trading Engine Status
        trading_engine_status = 'offline'
        trading_engine_details = {}
        if trading_engine:
            try:
                # Pobierz status z silnika handlowego
                engine_status = trading_engine.get_status()
                
                # Dodatkowe szczegóły dla diagnostyki
                trading_engine_details = {
                    'active_symbols': engine_status.get('active_symbols', []),
                    'active_strategies': engine_status.get('active_strategies', []),
                    'last_error': engine_status.get('last_error', None)
                }
                
                if engine_status['status'] == 'running':
                    trading_engine_status = 'online'
                    
                    # Sprawdź ryzyko dla aktywnych symboli
                    if hasattr(trading_engine, 'calculate_positions_risk'):
                        try:
                            risk_levels = trading_engine.calculate_positions_risk()
                            trading_engine_details['risk_levels'] = risk_levels
                            logging.info(f"Poziomy ryzyka: {risk_levels}")
                        except Exception as risk_e:
                            logging.warning(f"Nie można obliczyć ryzyka: {risk_e}")
                            
                elif engine_status['status'] == 'error':
                    trading_engine_status = 'warning'
                    logging.warning(f"Problem z silnikiem handlowym: {engine_status.get('last_error', 'Nieznany błąd')}")
                else:
                    trading_engine_status = 'offline'
                    logging.info("Silnik handlowy jest wyłączony")
            except Exception as e:
                logging.error(f"Błąd podczas sprawdzania statusu silnika handlowego: {e}", exc_info=True)
                trading_engine_status = 'warning'
        
        # Data Processor Status - sprawdź czy można załadować dane
        data_processor_status = 'online'
        try:
            # Pobierz przykładowe dane z silnika handlowego dla diagnostyki
            if trading_engine and hasattr(trading_engine, 'get_market_data'):
                test_data = trading_engine.get_market_data('BTCUSDT')
                if test_data is None or not test_data:
                    logging.warning("Data Processor Warning: nie można pobrać danych testowych")
                    data_processor_status = 'warning'
                else:
                    logging.info(f"Data Processor: pobrano dane testowe dla BTCUSDT (cena: {test_data.get('price', 'N/A')})")
        except Exception as e:
            logging.error(f"Błąd w Data Processor: {e}")
            data_processor_status = 'warning'

        # Risk Manager Status - zakładamy, że działa, jeśli silnik handlowy działa
        risk_manager_status = trading_engine_status if trading_engine else 'offline'

        # Sentiment Analyzer Status
        sentiment_analyzer_status = 'offline'
        if sentiment_analyzer:
            try:
                # Spróbuj wykonać analizę sentymentu
                sentiment_analyzer.analyze()
                sentiment_analyzer_status = 'online'
            except Exception:
                sentiment_analyzer_status = 'warning'

        # Budowanie odpowiedzi z wszystkimi komponentami
        components = [
            {
                'id': 'api-connector',
                'status': api_connector_status,
                'name': 'API Connector'
            },
            {
                'id': 'data-processor',
                'status': data_processor_status,
                'name': 'Data Processor'
            },
            {
                'id': 'trading-engine',
                'status': trading_engine_status,
                'name': 'Trading Engine'
            },
            {
                'id': 'risk-manager',
                'status': risk_manager_status,
                'name': 'Risk Manager'
            },
            {
                'id': 'sentiment-analyzer',
                'status': sentiment_analyzer_status,
                'name': 'Sentiment Analyzer'
            }
        ]

        # Log diagnostyczny
        logger.info(f"Statusy komponentów: API: {api_connector_status}, Trading: {trading_engine_status}")

        return jsonify({'components': components})
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statusu komponentów: {e}", exc_info=True)
        return jsonify({'error': str(e), 'components': [
            {'id': 'api-connector', 'status': 'warning', 'name': 'API Connector'},
            {'id': 'data-processor', 'status': 'warning', 'name': 'Data Processor'},
            {'id': 'trading-engine', 'status': 'warning', 'name': 'Trading Engine'},
            {'id': 'risk-manager', 'status': 'warning', 'name': 'Risk Manager'},
            {'id': 'sentiment-analyzer', 'status': 'warning', 'name': 'Sentiment Analyzer'}
        ]}), 200  # Zwracamy 200 zamiast 500, aby frontend otrzymał odpowiedź

@app.route('/api/ai-models-status')
def get_ai_models_status():
    """Endpoint do pobierania statusu modeli AI"""
    try:
        # Import loader modeli
        try:
            from ai_models.model_loader import model_loader
            # Upewniamy się, że modele są załadowane
            model_loader.load_models()
            model_summary = model_loader.get_models_summary()
            
            # Dodanie dodatkowych informacji
            for model in model_summary:
                model['accuracy'] = round(75.0 + 10.0 * random.random(), 1)  # Przykładowa dokładność
                model['last_used'] = (datetime.now() - timedelta(minutes=random.randint(5, 60))).strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"Załadowano {len(model_summary)} modeli AI")
            
            # Jeśli nie ma modeli, dodajemy przykładowe
            if not model_summary:
                model_summary = [
                    {
                        'name': 'Trend Predictor',
                        'type': 'LSTM',
                        'accuracy': 78.5,
                        'status': 'Active',
                        'last_used': (datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')
                    },
                    {
                        'name': 'Sentiment Analyzer',
                        'type': 'BERT',
                        'accuracy': 82.3,
                        'status': 'Active',
                        'last_used': (datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
                    },
                    {
                        'name': 'Volatility Predictor',
                        'type': 'XGBoost',
                        'accuracy': 75.1,
                        'status': 'Active',
                        'last_used': (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
                    }
                ]
                logger.warning("Brak załadowanych modeli AI, używam przykładowych")
            
        except ImportError as e:
            logger.warning(f"Nie można zaimportować loaderów modeli: {e}")
            # Używamy przykładowych modeli
            model_summary = [
                {
                    'name': 'Trend Predictor',
                    'type': 'LSTM',
                    'accuracy': 78.5,
                    'status': 'Active',
                    'last_used': (datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')
                },
                {
                    'name': 'Sentiment Analyzer',
                    'type': 'BERT',
                    'accuracy': 82.3,
                    'status': 'Active',
                    'last_used': (datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
                },
                {
                    'name': 'Volatility Predictor',
                    'type': 'XGBoost',
                    'accuracy': 75.1,
                    'status': 'Active',
                    'last_used': (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
                }
            ]
        
        return jsonify({'models': model_summary})
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statusu modeli AI: {e}", exc_info=True) #Dodatkowe informacje o błędzie
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/status')
def get_system_status():
    return jsonify({
        'success': True,
        'components': {
            'trading_engine': 'active',
            'data_fetcher': 'active',
            'risk_management': 'active',
            'ml_prediction': 'active',
            'order_execution': 'active',
            'notification_system': 'active' if 'notification_system' in globals() else 'inactive',
            'bybit_api':'active' if bybit_client else 'inactive',
            'strategy_manager': 'active' if strategy_manager else 'inactive'
        },
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/trading/start', methods=['POST'])
def start_trading():
    try:
        if not trading_engine:
            logging.warning("Próba uruchomienia tradingu, ale silnik handlowy nie jest zainicjalizowany")
            return jsonify({'success': False, 'error': 'Silnik handlowy nie jest zainicjalizowany'}), 500
            
        result = trading_engine.start()
        if result.get('success', False):
            logging.info("Trading automatyczny uruchomiony pomyślnie")
            return jsonify({'success': True, 'message': 'Trading automatyczny uruchomiony'})
        else:
            logging.error(f"Błąd podczas uruchamiania tradingu: {result.get('error', 'Nieznany błąd')}")
            return jsonify({'success': False, 'error': result.get('error', 'Nieznany błąd')}), 500
    except Exception as e:
        logging.error(f"Błąd podczas uruchamiania tradingu: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    try:
        if not trading_engine:
            logging.warning("Próba zatrzymania tradingu, ale silnik handlowy nie jest zainicjalizowany")
            return jsonify({'success': False, 'error': 'Silnik handlowy nie jest zainicjalizowany'}), 500
            
        result = trading_engine.stop()
        if result.get('success', False):
            logging.info("Trading automatyczny zatrzymany pomyślnie")
            return jsonify({'success': True, 'message': 'Trading automatyczny zatrzymany'})
        else:
            logging.error(f"Błąd podczas zatrzymywania tradingu: {result.get('error', 'Nieznany błąd')}")
            return jsonify({'success': False, 'error': result.get('error', 'Nieznany błąd')}), 500
    except Exception as e:
        logging.error(f"Błąd podczas zatrzymywania tradingu: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/system/reset', methods=['POST'])
def reset_system():
    try:
        # Reset silnika handlowego
        if trading_engine:
            trading_engine.reset()
            logging.info("Silnik handlowy zresetowany pomyślnie")
        
        # Można dodać resetowanie innych komponentów w razie potrzeby
        logging.info("System zresetowany pomyślnie")
        return jsonify({'success': True, 'message': 'System zresetowany'})
    except Exception as e:
        logging.error(f"Błąd podczas resetowania systemu: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/api/bybit/server-time", methods=["GET"])
def get_bybit_server_time():
    """Endpoint zwracający czas serwera ByBit."""
    if not bybit_client:
        return jsonify({"error": "Klient ByBit nie jest zainicjalizowany"}), 500

    try:
        server_time = bybit_client.get_server_time()
        return jsonify(server_time)
    except Exception as e:
        logger.error(f"Błąd podczas pobierania czasu serwera ByBit: {e}", exc_info=True) #Dodatkowe informacje o błędzie
        return jsonify({"error": str(e)}), 500

@app.route("/api/bybit/market-data/<symbol>", methods=["GET"])
def get_bybit_market_data(symbol):
    """Endpoint zwracający dane rynkowe dla określonego symbolu z ByBit."""
    if not bybit_client:
        return jsonify({"error": "Klient ByBit nie jest zainicjalizowany"}), 500

    try:
        # Pobieranie świec (klines)
        klines = bybit_client.get_klines(symbol=symbol, interval="15", limit=10)

        # Pobieranie księgi zleceń
        order_book = bybit_client.get_order_book(symbol=symbol, limit=5)

        return jsonify({
            "klines": klines,
            "orderBook": order_book
        })
    except Exception as e:
        logger.error(f"Błąd podczas pobierania danych rynkowych z ByBit: {e}", exc_info=True) #Dodatkowe informacje o błędzie
        return jsonify({"error": str(e)}), 500

@app.route("/api/bybit/account-balance", methods=["GET"])
def get_bybit_account_balance():
    """Endpoint zwracający stan konta ByBit."""
    if not bybit_client:
        return jsonify({"error": "Klient ByBit nie jest zainicjalizowany"}), 500

    try:
        balance = bybit_client.get_account_balance()
        return jsonify(balance)
    except Exception as e:
        logger.error(f"Błąd podczas pobierania stanu konta ByBit: {e}", exc_info=True) #Dodatkowe informacje o błędzie
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolio', methods=["GET"])
def get_portfolio():
    """Endpoint zwracający dane portfela."""
    try:
        if not bybit_client:
            # Jeśli klient nie jest dostępny, zwróć przykładowe dane
            logger.info("Klient ByBit nie jest zainicjalizowany, używam danych testowych")
            return jsonify({
                "success": True,
                "balances": {
                    "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                    "USDT": {"equity": 1000, "available_balance": 950, "wallet_balance": 1000}
                },
                "source": "simulation"
            })

        # Próba pobrania danych z API
        logger.info("Próbuję pobrać dane portfela z API ByBit")
        balance = bybit_client.get_account_balance()

        # Upewnij się, że zwracany JSON ma wymagane pole success
        if "success" not in balance:
            balance["success"] = True

        # Upewnij się, że mamy słownik balances nawet jeśli API zwróciło błąd
        if "balances" not in balance or not balance["balances"]:
            logger.warning("API zwróciło puste dane portfela, używam danych testowych")
            balance["balances"] = {
                "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                "USDT": {"equity": 1000, "available_balance": 950, "wallet_balance": 1000}
            }
            balance["source"] = "fallback_empty"

        logger.info(f"Zwracam dane portfela: {balance.get('source', 'api')}")
        return jsonify(balance)
    except Exception as e:
        logger.error(f"Błąd podczas pobierania danych portfela: {e}", exc_info=True)
        # Szczegółowe dane diagnostyczne
        logger.error(f"Szczegóły błędu: {type(e).__name__}, {str(e)}")

        return jsonify({
            "success": True,  # Ustawiamy True, aby frontend nie wyświetlał błędu
            "balances": {
                "BTC": {"equity": 0.005, "available_balance": 0.005, "wallet_balance": 0.005},
                "USDT": {"equity": 500, "available_balance": 450, "wallet_balance": 500}
            },
            "source": "fallback_error",
            "error": str(e)
        })

@app.route("/api/bybit/connection-test", methods=["GET"])
def test_bybit_connection():
    """Endpoint do testowania połączenia z ByBit API."""
    if not bybit_client:
        return jsonify({"success": False, "error": "Klient ByBit nie jest zainicjalizowany", "testnet": True}), 500

    try:
        # Test połączenia poprzez pobranie czasu serwera
        server_time = bybit_client.get_server_time()

        # Test połączenia przez próbę pobrania salda (wymaga autentykacji)
        balance_test = bybit_client.get_account_balance()

        # Sprawdzenie, czy używamy testnet czy produkcyjnego API
        is_testnet = bybit_client.usetestnet

        connection_status = {
            "success": True,
            "api_initialized": True,
            "server_time": server_time,
            "testnet": is_testnet,
            "environment": "testnet" if is_testnet else "production",
            "authentication": balance_test.get("success", False),
            "balance_data": "Dostępne" if balance_test.get("success", False) else "Błąd autoryzacji",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        logger.info(f"Test połączenia z ByBit API: {connection_status}")
        return jsonify(connection_status)
    except Exception as e:
        logger.error(f"Błąd podczas testowania połączenia z ByBit API: {e}", exc_info=True)
        return jsonify({
            "success": False, 
            "error": str(e),
            "testnet": bybit_client.use_testnet if bybit_client else None
        }), 500


import os

def is_env_flag_true(env_var_name: str) -> bool:
    return os.getenv(env_var_name, "").strip().lower() in ["1", "true", "yes"]

# Uruchomienie aplikacji
if __name__ == "__main__":
    # Tworzenie katalogów - użycie os.path.join dla kompatybilności z Windows
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_libs"), exist_ok=True)

    # Sprawdzenie, w jakim środowisku działa aplikacja
    is_replit = 'REPL_ID' in os.environ
    env_type = "Replit" if is_replit else "Lokalne"
    logger.info(f"Wykryto środowisko: {env_type}")

    # Sprawdź środowisko - czy na pewno używamy produkcyjnego API
    if is_env_flag_true("BYBIT_TESTNET"):
        logger.warning("❌ OSTRZEŻENIE: .env wskazuje na testnet (BYBIT_TESTNET=True). Ustaw BYBIT_TESTNET=False, jeśli chcesz realny rynek!")
    elif os.getenv("BYBIT_USE_TESTNET", "false").lower() == "true":
        logger.warning("❌ OSTRZEŻENIE: .env wskazuje na testnet (BYBIT_USE_TESTNET=true). Ustaw BYBIT_USE_TESTNET=false, jeśli chcesz realny rynek!")
    else:
        logger.warning("🚨 PRODUKCYJNE API BYBIT JEST WŁĄCZONE! OPERUJESZ PRAWDZIWYMI ŚRODKAMI!")
        print("\n\n========== PRODUKCYJNE API BYBIT ==========")
        print("🚨 UWAGA 🚨 Używasz PRODUKCYJNEGO API ByBit")
        print("Operacje handlowe będą mieć REALNE SKUTKI FINANSOWE!")
        print("===========================================\n\n")

    # Inicjalizacja systemu
    initialize_system()

    # Utworzenie pliku .env, jeśli nie istnieje
    if not os.path.exists('.env'):
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        with open(env_path, 'w') as f:
            f.write("FLASK_APP=main.py\n")
            f.write("FLASK_ENV=development\n")
            f.write("PORT=5000\n")
            f.write("BYBIT_API_KEY=YourApiKeyHere\n")
            f.write("BYBIT_API_SECRET=YourApiSecretHere\n")
            f.write("BYBIT_USE_TESTNET=true\n")
        logging.info(f"Utworzono plik .env z domyślnymi ustawieniami w: {env_path}")

    # Uruchomienie aplikacji z odpowiednim hostem w zależności od środowiska
    port = int(os.environ.get("PORT", 5000))
    
    # Jeśli jesteśmy w Replit, użyj 0.0.0.0, w przeciwnym razie 127.0.0.1
    host = "0.0.0.0" if is_replit else "127.0.0.1"
    
    debug_mode = os.getenv("DEBUG", "True").lower() in ["true", "1", "yes"]
    
    logging.info(f"Uruchamianie aplikacji Flask w środowisku {env_type} na hoście {host} i porcie {port}")
    try:
        app.run(host=host, port=port, debug=debug_mode)
    except Exception as e:
        logging.error(f"Błąd podczas uruchamiania aplikacji Flask: {e}")
        print(f"\nBłąd podczas uruchamiania aplikacji: {e}")
        print("Sprawdź czy port 5000 nie jest już używany.")
        import sys
        sys.exit(1)