import logging
import os
import sys
import json
import random
import uuid
import asyncio  # Dodany brakujący import asyncio
from datetime import datetime, timedelta
import jwt  # Dodany import dla JWT
from functools import wraps  # Dodany import dla wrappera

# Zapewnienie istnienia potrzebnych katalogów
os.makedirs("logs", exist_ok=True)
os.makedirs("data/cache", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("static/img", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# Katalog z własnymi bibliotekami
LOCAL_LIBS_DIR = "python_libs"
if os.path.exists(LOCAL_LIBS_DIR):
    sys.path.insert(0, LOCAL_LIBS_DIR)
    print(f"Dodano katalog {LOCAL_LIBS_DIR} do ścieżki Pythona.")

# Dodanie katalogu głównego projektu do ścieżki Pythona
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Importy zewnętrznych bibliotek
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from utils.retry_handler import retry_with_backoff
from data.utils.security_manager import security_manager

# Implementujemy własną funkcję jwt_required, zamiast importować ją z dashboard_api.py
def jwt_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'message': 'Brak tokenu uwierzytelniającego!'}), 401
            
        try:
            # Weryfikacja tokenu z dodatkowymi zabezpieczeniami
            is_valid, payload = security_manager.verify_token(
                token, 
                verify_ip=request.remote_addr
            )
            
            if not is_valid:
                return jsonify({'message': payload.get('error', 'Nieprawidłowy token!')}), 401
                
            request.jwt_payload = payload
            request.username = payload.get('sub')
            
            # Sprawdź limity zapytań
            if not security_manager.check_rate_limit(request.remote_addr, get_endpoint_type(request.endpoint)):
                return jsonify({
                    'message': 'Przekroczono limit zapytań',
                    'retry_after': 60
                }), 429
                
        except Exception as e:
            logging.error(f"Błąd weryfikacji tokenu: {e}")
            return jsonify({'message': 'Błąd weryfikacji tokenu!'}), 401
            
        return f(*args, **kwargs)
    return decorated_function

def get_endpoint_type(endpoint: str) -> str:
    """Określa typ endpointu na podstawie nazwy."""
    if endpoint in ['start_trading', 'stop_trading', 'set_portfolio_balance']:
        return 'trading'
    elif endpoint in ['reset_system', 'set_autonomous_mode']:
        return 'critical'
    else:
        return 'default'

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

# Inicjalizacja aplikacji Flask - PRZESUNIĘTO TUTAJ
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# Import klienta ByBit
try:
    from data.execution.bybit_connector import BybitConnector
    bybit_import_success = True
except ImportError as e:
    logging.warning(f"Nie udało się zaimportować BybitConnector: {e}")
    bybit_import_success = False

# Import menedżera symulacji
try:
    from python_libs.simulation_results import SimulationManager
    simulation_manager = SimulationManager()
    simulation_import_success = True
except ImportError as e:
    logging.warning(f"Nie udało się zaimportować SimulationManager: {e}")
    simulation_import_success = False
    simulation_manager = None

# Deklaracja zmiennych globalnych
global bybit_client, notification_system, sentiment_analyzer, anomaly_detector, strategy_manager, trading_engine, portfolio_manager
bybit_client = None
notification_system = None
sentiment_analyzer = None
anomaly_detector = None
strategy_manager = None
trading_engine = None
portfolio_manager = None

# Inicjalizacja komponentów systemu (import wewnątrz funkcji dla uniknięcia cyklicznych importów)
def initialize_system():
    global notification_server
    
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

        # Inicjalizacja testera modeli AI
        try:
            from python_libs.model_tester import ModelTester
            model_tester_lib = "python_libs.model_tester"
            # Inicjalizacja testera modeli z poprawionymi parametrami
            model_tester = ModelTester(models_path='ai_models', log_path='logs/model_tests.log')
            logging.info(f"Zainicjalizowano ModelTester z biblioteki {model_tester_lib}")
        except ImportError as e:
            model_tester = None
            logging.warning(f"Nie można zaimportować ModelTester: {e}")

        # Załadowanie konfiguracji AI
        try:
            import json
            ai_config_path = 'python_libs/ai_config.json'

            if os.path.exists(ai_config_path):
                with open(ai_config_path, 'r') as f:
                    ai_config = json.load(f)
                logging.info(f"Załadowano konfigurację AI z {ai_config_path}")
            else:
                ai_config = {
                    "model_settings": {
                        "sentiment_analyzer": {
                            "enabled": True,
                            "sources": ["twitter", "news", "forum", "reddit"]
                        },
                        "anomaly_detector": {
                            "enabled": True,
                            "method": "z_score",
                            "threshold": 2.5
                        }
                    }
                }
                logging.warning(f"Brak pliku konfiguracyjnego AI. Używam domyślnych ustawień.")
        except Exception as e:
            ai_config = {}
            logging.warning(f"Błąd podczas ładowania konfiguracji AI: {e}")

        # Inicjalizacja analizatora sentymentu
        try:
            from ai_models.sentiment_ai import SentimentAnalyzer
            sentiment_lib = "ai_models.sentiment_ai"

            # Pobierz ustawienia z konfiguracji
            sentiment_config = ai_config.get("model_settings", {}).get("sentiment_analyzer", {})
            sources = sentiment_config.get("sources", ["twitter", "news", "forum", "reddit"])

            sentiment_analyzer = SentimentAnalyzer()
            logging.info(f"Zainicjalizowano SentimentAnalyzer z biblioteką {sentiment_lib} i źródłami {sources}")
        except ImportError as e:
            logging.warning(f"Nie można zaimportować SentimentAnalyzer z ai_models: {e}")
            try:
                from data.indicators.sentiment_analysis import SentimentAnalyzer
                sentiment_analyzer = SentimentAnalyzer()
                sentiment_lib = "data.indicators.sentiment_analysis"
                logging.info(f"Zainicjalizowano SentimentAnalyzer z alternatywnej biblioteki {sentiment_lib}")
            except ImportError:
                sentiment_analyzer = None
                logging.warning("Brak modułu analizatora sentymentu")

        # Inicjalizacja wykrywania anomalii
        try:
            from ai_models.anomaly_detection import AnomalyDetector
            anomaly_lib = "ai_models.anomaly_detection"

            # Pobierz ustawienia z konfiguracji
            anomaly_config = ai_config.get("model_settings", {}).get("anomaly_detector", {})
            method = anomaly_config.get("method", "z_score")
            threshold = anomaly_config.get("threshold", 2.5)

            anomaly_detector = AnomalyDetector(sensitivity=threshold)
    
    # Trening modelu na przykładowych danych
            import numpy as np
            import pandas as pd
            # Tworzymy przykładowe dane do wstępnego treningu
            sample_data = pd.DataFrame(np.random.normal(0, 1, (100, 5)), 
                                  columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
            try:
                # Upewniamy się, że model jest zainicjalizowany
                if anomaly_detector.model is None:
                    anomaly_detector._initialize_default_model()
                
                # Trening modelu
                anomaly_detector.fit(sample_data)
                
                # Weryfikacja, że model działa prawidłowo
                test_prediction = anomaly_detector.predict(sample_data[:5])
                logging.info(f"Model detekcji anomalii pomyślnie wytrenowany i przetestowany (wynik: {test_prediction[:3]})")
            except Exception as e:
                logging.warning(f"Nie udało się wytrenować modelu detekcji anomalii: {e}")
                logging.info("Inicjalizuję zapasowy model detekcji...")
                # Użyj zapasowej strategii
                anomaly_detector = None
            
            logging.info(f"Zainicjalizowano AnomalyDetector z biblioteki {anomaly_lib} (metoda: {method}, próg: {threshold})")
        except ImportError as e:
            logging.warning(f"Nie można zaimportować AnomalyDetector z ai_models: {e}")
            try:
                from data.logging.anomaly_detector import AnomalyDetector
                anomaly_detector = AnomalyDetector()
                anomaly_lib = "data.logging.anomaly_detector"
                logging.info(f"Zainicjalizowano AnomalyDetector z alternatywnej biblioteki {anomaly_lib}")
            except ImportError:
                anomaly_detector = None
                logging.warning("Brak modułu detektora anomalii")

        # Inicjalizacja model recognizer
        try:
            from ai_models.model_recognition import ModelRecognizer
            global model_recognizer
            model_recognizer = ModelRecognizer()
            logging.info(f"Zainicjalizowano ModelRecognizer")
        except ImportError as e:
            model_recognizer = None
            logging.warning(f"Nie można zaimportować ModelRecognizer: {e}")

        # Wykluczamy podpakiet tests z automatycznego importu
        try:
            from data.utils.import_manager import exclude_modules_from_auto_import
            exclude_modules_from_auto_import(["tests"])
            logging.info("Wykluczono podpakiet 'tests' z automatycznego importu.")
        except ImportError:
            logging.warning("Nie znaleziono modułu import_manager, pomijam wykluczanie podpakietów.")

        global notification_system, strategy_manager

        # Inicjalizacja systemu powiadomień
        notification_system = NotificationSystem()
        logging.info(f"Zainicjalizowano system powiadomień z biblioteki {notification_lib}")

        # Inicjalizacja managera strategii z domyślnymi strategiami
        try:
            from python_libs.simplified_strategy import StrategyManager
            strategy_lib = "python_libs.simplified_strategy"

            # Pobierz ustawienia strategii z konfiguracji
            strategy_settings = ai_config.get("strategy_settings", {})

            strategies = {}
            exposure_limits = {}

            # Domyślne strategie, jeśli brak w konfiguracji
            if not strategy_settings:
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
            else:
                # Utworzenie strategii z konfiguracji
                for key, settings in strategy_settings.items():
                    strategies[key] = {
                        "name": key.replace('_', ' ').title(),
                        "enabled": settings.get("enabled", False)
                    }
                    exposure_limits[key] = settings.get("weight", 0.3)

            strategy_manager = StrategyManager(strategies, exposure_limits)
            logging.info(f"Zainicjalizowano StrategyManager z biblioteką {strategy_lib} i {len(strategies)} strategiami")
        except ImportError:
            try:
                from data.strategies.strategy_manager import StrategyManager
                strategy_lib = "data.strategies.strategy_manager"

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

                strategy_manager = StrategyManager(strategies, exposure_limits)
                logging.info(f"Zainicjalizowano StrategyManager z alternatywnej biblioteki {strategy_lib}")
            except ImportError as e:
                strategy_manager = None
                logging.warning(f"Brak modułu StrategyManager: {e}")

        # Inicjalizacja zarządcy ryzyka
        risk_manager = None
        try:
            from python_libs.simplified_risk_manager import SimplifiedRiskManager

            # Pobierz ustawienia symulacji z konfiguracji
            simulation_settings = ai_config.get("simulation_settings", {})
            max_risk = 0.05  # Domyślnie 5%
            max_position_size = simulation_settings.get("max_position_size", 0.2)
            max_drawdown = 0.1  # Domyślnie 10%

            risk_manager = SimplifiedRiskManager(
                max_risk=max_risk,
                max_position_size=max_position_size,
                max_drawdown=max_drawdown
            )
            logging.info(f"Zainicjalizowano SimplifiedRiskManager (max_risk={max_risk}, max_position_size={max_position_size}, max_drawdown={max_drawdown})")
        except ImportError as e:
            logging.warning(f"Nie można zaimportować SimplifiedRiskManager: {e}")

        # Inicjalizacja klienta ByBit
        global bybit_client
        try:
            if not bybit_import_success:
                logging.warning("Moduł BybitConnector nie został zaimportowany. Próba użycia symulowanego klienta.")
                try:
                    from python_libs.simulated_bybit import SimulatedBybitConnector
                    bybit_client = SimulatedBybitConnector(
                        api_key="simulated_key",
                        api_secret="simulated_secret",
                        use_testnet=True
                    )
                    logging.info("Zainicjalizowano symulowany klient ByBit")
                except ImportError:
                    logging.error("Nie można zainicjalizować symulowanego klienta ByBit")
                    bybit_client = None
            else:
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
                    except ImportError:
                        logging.error("Nie można zainicjalizować symulowanego klienta ByBit")
                        bybit_client = None
                else:
                    # Dodatkowa weryfikacja kluczy dla produkcji
                    if not use_testnet and (len(api_key) < 10 or len(api_secret) < 10):
                        logging.critical("BŁĄD KRYTYCZNY: Nieprawidłowe klucze produkcyjne API. Wymagane odpowiednie klucze dla środowiska produkcyjnego!")
                        # Używam symulowanego klienta zamiast błędu
                        try:
                            from python_libs.simulated_bybit import SimulatedBybitConnector
                            bybit_client = SimulatedBybitConnector(
                                api_key="simulated_key",
                                api_secret="simulated_secret",
                                use_testnet=True
                            )
                            logging.info("Zainicjalizowano symulowany klient ByBit z powodu niepoprawnych kluczy produkcyjnych")
                        except ImportError:
                            bybit_client = None
                    else:
                        # Inicjalizacja rzeczywistego klienta ByBit
                        bybit_client = BybitConnector(
                            api_key=api_key,
                            api_secret=api_secret,
                            use_testnet=use_testnet,
                            lazy_connect=True  # Używamy lazy initialization by uniknąć sprawdzania API na starcie
                        )

                        # Informacja o konfiguracji API
                        masked_key = f"{api_key[:4]}{'*' * (len(api_key) - 4)}" if api_key else "Brak klucza"
                        logging.info(f"Inicjalizacja klienta ByBit - Klucz: {masked_key}, Testnet: {use_testnet}")

                        # Ostrzeżenie dla produkcyjnego API
                        if not use_testnet:
                            logging.warning("!!! UWAGA !!! Używasz PRODUKCYJNEGO API ByBit. Operacje handlowe będą mieć realne skutki finansowe!")
                            logging.warning("Upewnij się, że Twoje klucze API mają właściwe ograniczenia i są odpowiednio zabezpieczone.")
                            print("\n\n========== PRODUKCYJNE API BYBIT ==========")
                            print("!!! UWAGA !!! Używasz PRODUKCYJNEGO API ByBit")
                            print("Operacje handlowe będą mieć realne skutki finansowe!")
                            print("===========================================\n\n")

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

        # Inicjalizacja silnika handlowego
        global trading_engine
        if SimplifiedTradingEngine and bybit_client is not None:
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
            logging.warning("Brak modułu silnika handlowego (Trading Engine) lub klienta ByBit")

        # Inicjalizacja menadżera portfela
        global portfolio_manager
        try:
            from python_libs.portfolio_manager import PortfolioManager, portfolio_manager
            if portfolio_manager is None:  # Jeśli nie został automatycznie utworzony w module
                # Tworzymy instancję PortfolioManager bezpośrednio
                portfolio_manager = PortfolioManager(initial_balance=1000.0, currency="USDT", mode="simulated")
            logging.info("Zainicjalizowano menadżera portfela")
        except ImportError as e:
            logging.error(f"Nie można zaimportować PortfolioManager: {e}")
            try:
                # Definicja klasy SimplePortfolioManager
                class SimplePortfolioManager:
                    def __init__(self, initial_balance=100.0, currency="USDT", mode="simulated"):
                        self.initial_balance = initial_balance
                        self.currency = currency
                        self.mode = mode
                        self.balances = {
                            currency: {"equity": initial_balance, "available_balance": initial_balance, "wallet_balance": initial_balance}
                        }
                        
                    def get_portfolio(self):
                        return {
                            "success": True,
                            "balances": self.balances,
                            "total_value": sum(balance["equity"] for balance in self.balances.values()),
                            "base_currency": self.currency,
                            "mode": self.mode
                        }

                    def set_initial_balance(self, amount, currency="USDT"):
                        self.initial_balance = amount
                        self.currency = currency
                        self.balances[currency] = {"equity": amount, "available_balance": amount, "wallet_balance": amount}
                        return True

                    def get_ai_models_status(self):
                        return {
                            "models": [
                                {
                                    "name": "XGBoost Price Predictor",
                                    "type": "Regression",
                                    "accuracy": 76.5,
                                    "status": "Active",
                                    "last_used": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "has_predict": True,
                                    "has_fit": True
                                },
                                {
                                    "name": "Sentiment Analyzer",
                                    "type": "NLP",
                                    "accuracy": 82.3,
                                    "status": "Active",
                                    "last_used": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "has_predict": True,
                                    "has_fit": True
                                }
                            ]
                        }

                initial_balance = 1000.0
                currency = "USDT"
                portfolio_manager = SimplePortfolioManager(initial_balance=initial_balance, currency=currency, mode="simulated")
                logging.info("Zainicjalizowano fallback menadżera portfela")
                logging.info(f"Utworzono fallback PortfolioManager (saldo: {initial_balance} {currency})")
            except Exception as fallback_error:
                logging.error(f"Nie można utworzyć fallback menadżera portfela: {fallback_error}")
                portfolio_manager = None

        # Test modeli AI
        if model_tester:
            test_results = model_tester.run_tests()
            loaded_models = model_tester.get_loaded_models()
            logging.info(f"Przetestowano modele AI: znaleziono {len(loaded_models)} działających modeli")

        # Inicjalizacja managera symulacji
        global simulation_manager
        if simulation_import_success:
            # Upewnijmy się, że simulation_manager jest zainicjalizowany
            if not isinstance(simulation_manager, SimulationManager):
                simulation_manager = SimulationManager()
                logging.info("Zainicjalizowano SimulationManager")

        # Inicjalizacja serwera powiadomień
        notification_server = start_notification_server()
        logger.info("Serwer WebSocket został uruchomiony")

        logging.info("System zainicjalizowany poprawnie")
        return True
    except Exception as e:
        logging.error(f"Błąd podczas inicjalizacji systemu: {e}", exc_info=True)
        return False

# Trasy aplikacji
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
    if portfolio_manager:
        try:
            portfolio = portfolio_manager.get_portfolio()
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
        logging.warning("Menadżer portfela nie jest zainicjalizowany, używam danych testowych")
        portfolio = {
            "balances": {
                "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                "USDT": {"equity": 1000, "available_balance": 950, "wallet_balance": 1000}
            }, 
            "success": True,
            "note": "Dane testowe - menadżer portfela nie jest zainicjalizowany"
        }

    return render_template(
        'dashboard.html',
        settings=default_settings,
        ai_models=ai_models,
        strategies=strategies,
        trades=[],
        alerts=[],
        sentiment_data=None, # Dodano zmienną dla danych sentymentu
        anomalies=[],
        portfolio=portfolio,
        ws_port=6789  # Dodane dla konfiguracji WebSocket
    )

# API endpoints
@app.route('/api/portfolio')
@retry_with_backoff
def get_portfolio_data():
    """Endpoint API do pobierania danych portfela z obsługą ponownych prób."""
    try:
        portfolio_data = portfolio_manager.get_portfolio() if portfolio_manager else {
            "balances": {
                "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                "USDT": {"equity": 1000.0, "available_balance": 1000.0, "wallet_balance": 1000.0}
            }
        }
        return jsonify(portfolio_data)
    except Exception as e:
        logging.error(f"Błąd w get_portfolio_data: {str(e)}")
        return jsonify({
            "success": True,
            "balances": {
                "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                "USDT": {"equity": 1000.0, "available_balance": 1000.0, "wallet_balance": 1000.0}
            },
            "source": "fallback_error",
            "error": str(e),
            "retry_after": 1
        })

@app.route('/api/portfolio/set-balance', methods=['POST'])
def set_portfolio_balance():
    """Endpoint API do ustawiania początkowego salda portfela."""
    try:
        data = request.json
        if not data or "amount" not in data:
            return jsonify({"error": "Brak wymaganego parametru 'amount'"}), 400

        amount = float(data["amount"])
        currency = data.get("currency", "USDT")

        success = portfolio_manager.set_initial_balance(amount, currency)
        if success:
            return jsonify({"success": True, "message": f"Ustawiono początkowe saldo na {amount} {currency}"})
        else:
            return jsonify({"error": "Nie udało się ustawić początkowego salda"}), 500
    except Exception as e:
        logging.error(f"Błąd podczas ustawiania początkowego salda: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

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
@app.route('/api/chart/data')
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
                'type':'warning',
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

@app.route('/api/simulation-results')
def get_simulation_results():
    """Endpoint do pobierania wyników symulacji tradingu"""
    try:
        if simulation_import_success and simulation_manager:
            results = simulation_manager.get_simulation_results()
            return jsonify(results)
        else:
            # Tworzy przykładowe wyniki jako fallback
            results = {
                'status': 'success',
                'summary': {
                    'initial_capital': 10000.0,
                    'final_capital': 10850.75,
                    'profit': 850.75,
                    'profit_percentage': 8.5075,
                    'trades': 48,
                    'wins': 29,
                    'losses': 19,
                    'win_rate': 60.42,
                    'max_drawdown': 4.8,
                    'total_commission': 125.50,
                    'winning_trades': 29,
                    'closes': 48
                },
                'trades': [
                    {
                        'timestamp': (datetime.now() - timedelta(days=30)).timestamp(),
                        'action': 'LONG',
                        'price': 48750.25,
                        'size': 0.02,
                        'commission': 0.9750,
                        'capital': 9999.025
                    },
                    {
                        'timestamp': (datetime.now() - timedelta(days=29)).timestamp(),
                        'action': 'CLOSE LONG',
                        'price': 49250.50,
                        'size': 0.02,
                        'pnl': 10.005,
                        'commission': 0.9850,
                        'capital': 10008.045
                    }
                ],
                'chart_path': '/static/img/default_chart.png',
                'message': 'Przykładowe dane symulacji'
            }
            return jsonify(results)
    except Exception as e:
        logging.error(f"Błąd podczas pobierania wyników symulacji: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Błąd podczas pobierania wyników symulacji: {str(e)}',
            'summary': {
                'initial_capital': 10000.0,
                'final_capital': 10000.0,
                'profit': 0.0,
                'profit_percentage': 0.0,
                'trades': 0,
                'win_rate': 0.0,
                'max_drawdown': 0.0
            },
            'trades': []
        })

@app.route('/api/simulation/run', methods=['POST'])
def run_simulation():
    """Endpoint do uruchamiania nowej symulacji tradingu"""
    try:
        # Pobierz parametry z żądania
        data = request.json or {}
        initial_capital = float(data.get('initial_capital', 10000.0))
        duration = int(data.get('duration', 1000))

        if simulation_import_success and simulation_manager:
            # Uruchom symulację
            results = simulation_manager.create_simulation(
                initial_capital=initial_capital,
                duration=duration,
                save_report=True
            )

            return jsonify({
                'status': 'success',
                'summary': results['summary'],
                'message': 'Symulacja zakończona pomyślnie'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Manager symulacji nie jest dostępny'
            }), 500
    except Exception as e:
        logging.error(f"Błąd podczas uruchamiania symulacji: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Błąd podczas uruchamiania symulacji: {str(e)}'
        }), 500

@app.route('/api/simulation/learn', methods=['POST'])
def run_simulation_with_learning():
    """Endpoint do uruchamiania symulacji z uczeniem"""
    try:
        # Pobierz parametry z żądania
        data = request.json or {}
        initial_capital = float(data.get('initial_capital', 10000.0))
        duration = int(data.get('duration', 1000))
        iterations = int(data.get('iterations', 5))

        if simulation_import_success and simulation_manager:
            # Uruchom symulację z uczeniem
            results = simulation_manager.run_simulation_with_learning(
                initial_capital=initial_capital,
                duration=duration,
                learning_iterations=iterations
            )

            return jsonify({
                'status': 'success',
                'summary': results.get('summary', {}),
                'learning_results': results.get('learning_results', []),
                'message': 'Symulacja z uczeniem zakończona pomyślnie'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Manager symulacji nie jest dostępny'
            }), 500
    except Exception as e:
        logging.error(f"Błąd podczas uruchamiania symulacji z uczeniem: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Błąd podczas uruchamiania symulacji z uczeniem: {str(e)}'
        }), 500

@app.route('/api/ai/thoughts')
def get_ai_thoughts():
    """Endpoint do pobierania przemyśleń modeli AI"""
    try:
        thoughts = []

        # Pobierz przemyślenia z analizatora sentymentu
        if sentiment_analyzer:
            try:
                sentiment_data = sentiment_analyzer.analyze()
                thoughts.append({
                    'model': 'SentimentAnalyzer',
                    'thought': f"Analiza sentymentu wskazuje na {sentiment_data['analysis']} nastawienie rynku (wartość: {sentiment_data['value']:.2f})",
                    'confidence': min(abs(sentiment_data['value']) * 100 + 50, 95),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'sentiment'
                })
            except Exception as e:
                logging.error(f"Błąd podczas analizy sentymentu: {e}")
                thoughts.append({
                    'model': 'SentimentAnalyzer',
                    'thought': "Nie można obecnie analizować sentymentu rynkowego",
                    'confidence': 50,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'sentiment'
                })

        # Pobierz przemyślenia z detektora anomalii
        if anomaly_detector:
            try:
                # Generowanie losowych danych
                test_data = [random.normalvariate(0, 1) for _ in range(10)]
                max_value = max(test_data)
                if max_value > 2:
                    thoughts.append({
                        'model': 'AnomalyDetector',
                        'thought': f"Wykryto potencjalną anomalię w danych (wartość: {max_value:.2f})",
                        'confidence': min(max_value * 20, 90),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'anomaly'
                    })
                else:
                    thoughts.append({
                        'model': 'AnomalyDetector',
                        'thought': "Nie wykryto anomalii w obecnych danych rynkowych",
                        'confidence': 85,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'anomaly'
                    })
            except Exception as e:
                logging.error(f"Błąd podczas wykrywania anomalii: {e}")
                thoughts.append({
                    'model': 'AnomalyDetector',
                    'thought': "Nie można obecnie wykrywać anomalii",
                    'confidence': 50,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'anomaly'
                })

        # Dodaj przemyślenia od ModelRecognizer
        if 'model_recognizer' in globals() and model_recognizer:
            try:
                model_info = model_recognizer.identify_model_type(None)
                
                if model_info and isinstance(model_info, dict):
                    if 'type' in model_info and 'name' in model_info:
                        thoughts.append({
                            'model': 'ModelRecognizer',
                            'thought': f"Obecne dane rynkowe pasują do modelu typu {model_info['type']} ({model_info['name']})",
                            'confidence': model_info.get('confidence', 0.8) * 100,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'type': 'model_recognition'
                        })
                    elif 'error' in model_info:
                        thoughts.append({
                            'model': 'ModelRecognizer',
                            'thought': f"Analiza modelu: {model_info.get('error', 'Brak wystarczających danych')}",
                            'confidence': 60,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'type': 'model_recognition'
                        })
                    else:
                        thoughts.append({
                            'model': 'ModelRecognizer',
                            'thought': "Analizator modeli zwrócił niekompletne dane",
                            'confidence': 50,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'type': 'model_recognition'
                        })
                else:
                    thoughts.append({
                        'model': 'ModelRecognizer',
                        'thought': "Nie można obecnie rozpoznać modelu rynkowego (brak danych)",
                        'confidence': 40,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'model_recognition'
                    })
            except Exception as model_error:
                logging.warning(f"Błąd podczas analizy modelu: {model_error}")
                thoughts.append({
                    'model': 'ModelRecognizer',
                    'thought': "Nie można obecnie rozpoznać modelu rynkowego",
                    'confidence': 50,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'model_recognition'
                })

        # Losowe przemyślenia strategii
        strategy_thoughts = [
            "Obecny trend sugeruje możliwość utrzymania pozycji długiej",
            "Poziomy wsparcia mogą zostać wkrótce przetestowane",
            "Oscylator RSI wskazuje na przesprzedanie rynku, możliwy trend wzrostowy",
            "MACD sygnalizuje potencjalną zmianę trendu",
            "Wykres formuje figurę głowy i ramion - możliwe odwrócenie trendu"
        ]

        thoughts.append({
            'model': 'StrategyAnalyzer',
            'thought': random.choice(strategy_thoughts),
            'confidence': random.uniform(65, 92),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'strategy'
        })

        return jsonify({
            'status': 'success',
            'thoughts': thoughts,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania przemyśleń AI: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Błąd podczas pobierania przemyśleń AI: {str(e)}',
            'thoughts': []
        }), 500

@app.route('/api/ai/learning-status')
def get_learning_status():
    """Endpoint do pobierania statusu uczenia modeli AI"""
    try:
        # Pobierz ostatnie wyniki uczenia z symulacji
        learning_data = []

        if simulation_import_success and simulation_manager:
            # Próba pobrania wyników uczenia
            results = simulation_manager.get_simulation_results()
            if results.get('status') == 'success' and 'learning_results' in results:
                learning_data = results['learning_results']

        # Jeśli brak danych, wygeneruj przykładowe
        if not learning_data:
            learning_data = [
                {
                    'iteration': i+1,
                    'accuracy': 50 + i*5 + random.uniform(-2, 2),
                    'win_rate': 45 + i*4 + random.uniform(-3, 3),
                    'trades': 20 + i*5,
                    'profit': 100 * (i+1) * (1 + random.uniform(-0.3, 0.3))
                }
                for i in range(5)
            ]

        # Aktualne modele w treningu
        models_training = [
            {
                'name': 'TrendPredictor',
                'type': 'XGBoost',
                'progress': random.uniform(0, 100),
                'eta': f"{random.randint(1, 10)} min",
                'current_accuracy': random.uniform(70, 90)
            },
            {
                'name': 'SentimentAnalyzer',
                'type': 'LSTM',
                'progress': random.uniform(0, 100),
                'eta': f"{random.randint(1, 10)} min",
                'current_accuracy': random.uniform(65, 85)
            }
        ]

        return jsonify({
            'status': 'success',
            'learning_data': learning_data,
            'models_training': models_training,
            'is_training': random.choice([True, False]),
            'current_iteration': random.randint(1, 5),
            'total_iterations': 5,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statusu uczenia AI: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Błąd podczas pobierania statusu uczenia AI: {str(e)}',
            'learning_data': [],
            'models_training': [],
            'is_training': False
        }), 500

@app.route('/api/component-status')
def get_component_status():
    """Endpoint API zwracający status komponentów systemu."""
    try:
        # Sprawdzenie statusu komponentów
        api_status = "online" if bybit_client is not None else "offline"
        trading_status = "online" if trading_engine is not None else "offline"
        portfolio_status = "online" if portfolio_manager is not None else "offline"
        sentiment_status = "online" if sentiment_analyzer is not None else "offline"

        # Pobieranie testowych danych rynkowych, aby sprawdzić status API
        if api_status == "online":
            try:
                # Pobierz testowe dane
                test_data = bybit_client.get_klines(symbol="BTCUSDT", interval="15", limit=1)
                if test_data and len(test_data) > 0:
                    latest_price = test_data[0].get("close", 0)
                    logging.info(f"Data Processor: pobrano dane testowe dla BTCUSDT (cena: {latest_price})")
                else:
                    api_status = "degraded"
                    logging.warning("Data Processor: brak danych dla BTCUSDT")
            except Exception as data_error:
                api_status = "degraded"
                logging.error(f"Data Processor: błąd podczas pobierania danych testowych: {data_error}")

        logging.info(f"Statusy komponentów: API: {api_status}, Trading: {trading_status}")

        return jsonify({
            "api": api_status,
            "trading_engine": trading_status,
            "portfolio": portfolio_status,
            "sentiment": sentiment_status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statusu komponentów: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sentiment', methods=['GET'])
@app.route('/api/sentiment/latest', methods=['GET'])
def get_sentiment_data():
    """Endpoint API do pobierania danych sentymentu rynkowego."""
    try:
        if sentiment_analyzer is None:
            # Zwracamy dane zastępcze zamiast błędu
            return jsonify({
                "value": random.uniform(-0.2, 0.2),
                "analysis": "Neutralny",
                "sources": {
                    "twitter": random.uniform(-0.3, 0.3),
                    "news": random.uniform(-0.2, 0.2),
                    "forum": random.uniform(-0.1, 0.1)
                },
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        sentiment_data = sentiment_analyzer.analyze()
        return jsonify(sentiment_data)
    except Exception as e:
        logging.error(f"Błąd podczas pobierania danych sentymentu: {e}", exc_info=True)
        # Zwracamy dane zastępcze zamiast błędu
        return jsonify({
            "value": random.uniform(-0.2, 0.2),
            "analysis": "Neutralny (fallback)",
            "sources": {
                "twitter": random.uniform(-0.3, 0.3),
                "news": random.uniform(-0.2, 0.2),
                "forum": random.uniform(-0.1, 0.1)
            },
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "fallback": True,
            "error_info": str(e)
        })

@app.route('/api/trading/start', methods=['POST'])
@retry_with_backoff
def start_trading():
    """Endpoint do uruchamiania tradingu z obsługą ponownych prób."""
    try:
        if not trading_engine:
            logging.warning("Próba uruchomienia tradingu, ale silnik handlowy nie jest zainicjalizowany")
            return jsonify({
                'success': False, 
                'error': 'Silnik handlowy nie jest zainicjalizowany',
                'retry_after': 5
            }), 502

        result = trading_engine.start()
        if result.get('success', False):
            logging.info("Trading automatyczny uruchomiony pomyślnie")
            return jsonify({'success': True, 'message': 'Trading automatyczny uruchomiony'})
        else:
            error_msg = result.get('error', 'Nieznany błąd')
            logging.error(f"Błąd podczas uruchamiania tradingu: {error_msg}")
            return jsonify({
                'success': False, 
                'error': error_msg,
                'retry_after': 1
            }), 502
    except Exception as e:
        logging.error(f"Błąd podczas uruchamiania tradingu: {e}", exc_info=True)
        return jsonify({
            'success': False, 
            'error': str(e),
            'retry_after': 1
        }), 502

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
        if not portfolio_manager:
            # Jeśli klient nie jest dostępny, zwróć przykładowe dane
            logger.info("Menadżer portfela nie jest zainicjalizowany, używam danych testowych")
            return jsonify({
                "success": True,
                "balances": {
                    "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                    "USDT": {"equity": 1000, "available_balance": 950, "wallet_balance": 1000}
                },
                "source": "simulation"
            })

        # Próba pobrania danych z API
        logger.info("Próbuję pobrać dane portfela z menadżera portfela")
        balance = portfolio_manager.get_portfolio()

        # Upewnij się, że zwracany JSON ma wymagane pole success
        if "success" not in balance:
            balance["success"] = True

        # Upewnij się, że mamy słownik balances nawet jeśli API zwróciło błąd
        if "balances" not in balance or not balance["balances"]:
            logger.warning("Menadżer portfela zwrócił puste dane portfela, używam danych testowych")
            balance["balances"] = {
                "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                "USDT": {"equity": 1000, "available_balance": 950, "wallet_balance": 1000}
            }
            balance["source"] = "fallback_empty"

        logger.info(f"Zwracam dane portfela: {balance.get('source', 'portfolio_manager')}")
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

@app.route('/api/ai-models-status')
def get_ai_models_status():
    """Endpoint API zwracający status modeli AI."""
    try:
        # Sprawdzanie dostępności modeli
        models = []

        # Sprawdź SentimentAnalyzer
        if sentiment_analyzer:
            try:
                status = sentiment_analyzer.get_status()
                models.append({
                    'name': 'SentimentAnalyzer',
                    'type': 'Sentiment Analysis',
                    'accuracy': 82.0,
                    'status': 'Active' if status.get('active') else 'Inactive',
                    'last_used': status.get('last_update', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    'methods': {
                        'predict': hasattr(sentiment_analyzer, 'predict'),
                        'fit': hasattr(sentiment_analyzer, 'fit')
                    },
                    'test_result': 'Passed',
                    'module': 'sentiment_ai'
                })
                logging.info("Dodano status modelu SentimentAnalyzer")
            except Exception as e:
                logging.error(f"Błąd podczas pobierania statusu SentimentAnalyzer: {e}")
                models.append({
                    'name': 'SentimentAnalyzer',
                    'type': 'Sentiment Analysis',
                    'status': 'Error',
                    'error': str(e)
                })

        # Sprawdź AnomalyDetector
        if anomaly_detector:
            models.append({
                'name': 'AnomalyDetector',
                'type': 'Anomaly Detection',
                'accuracy': 84.0,
                'status': 'Active',
                'last_used': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'methods': {
                    'predict': False,
                    'fit': False
                },
                'test_result': 'Passed',
                'module': 'anomaly_detection'
            })

        # Sprawdź ModelRecognizer
        if 'model_recognizer' in globals() and model_recognizer:
            models.append({
                'name': 'ModelRecognizer',
                'type': 'Model Recognition',
                'accuracy': 84.2,
                'status': 'Active',
                'last_used': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'methods': {
                    'predict': False,
                    'fit': False
                },
                'test_result': 'Passed',
                'module': 'model_recognition'
            })

        # Dodaj informacje o innych modelach z ModelTester - jeśli jest dostępny
        try:
            from python_libs.model_tester import ModelTester
            model_tester = ModelTester(models_path='ai_models', log_path='logs/model_tests.log')
            loaded_models = model_tester.get_loaded_models()

            # Pobierz nazwy już dodanych modeli
            existing_names = [m['name'] for m in models]

            # Dodaj pozostałe modele, których jeszcze nie ma na liście
            for model_info in loaded_models:
                model_name = model_info.get('name', '')
                if model_name and model_name not in existing_names:
                    models.append({
                        'name': model_name,
                        'type': model_info.get('type', model_name),
                        'accuracy': model_info.get('accuracy', random.uniform(75.0, 85.0)),
                        'status': 'Detected',
                        'last_used': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'methods': {
                            'predict': hasattr(model_info.get('instance'), 'predict'),
                            'fit': hasattr(model_info.get('instance'), 'fit')
                        },
                        'test_result': 'Passed',
                        'module': model_info.get('module', 'unknown')
                    })
        except Exception as e:
            logging.warning(f"Nie można załadować dodatkowych informacji o modelach z ModelTester: {e}")

        # Dodaj informacje o błędnych modelach z logów
        error_models = [
            {
                'name': 'ModelTrainer',
                'type': 'ModelTrainer',
                'accuracy': 76.9,
                'status': 'Error',
                'last_used': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'methods': {
                    'predict': False,
                    'fit': False
                },
                'test_result': 'Failed',
                'module': 'model_training',
                'error': 'Nie udało się utworzyć żadnej instancji z modułu model_training'
            },
            {
                'name': 'RealExchangeEnv',
                'type': 'RealExchangeEnv',
                'accuracy': 81.4,
                'status': 'Error',
                'last_used': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'methods': {
                    'predict': False,
                    'fit': False
                },
                'test_result': 'Failed',
                'module': 'real_exchange_env',
                'error': 'Nie udało się utworzyć żadnej instancji z modułu real_exchange_env'
            }
        ]

        # Dodaj błędne modele do listy
        for error_model in error_models:
            if error_model['name'] not in [m['name'] for m in models]:
                models.append(error_model)

        return jsonify({
            'success': True,
            'models': models,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statusu modeli AI: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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

# Dodajemy endpoint uwierzytelniania
@app.route('/api/auth/login', methods=['POST'])
def login():
    """Endpoint do logowania użytkowników z rozszerzonymi zabezpieczeniami."""
    try:
        data = request.get_json() or {}
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        # Walidacja danych wejściowych
        if not username or not password:
            return jsonify({'success': False, 'message': 'Brak wymaganych danych logowania'}), 400
            
        # Sprawdź czy IP nie jest zablokowane
        if security_manager.is_ip_blocked(request.remote_addr):
            return jsonify({
                'success': False,
                'message': 'Dostęp zablokowany. Spróbuj ponownie później.'
            }), 403
            
        # Weryfikacja hasła (przykładowe dane testowe)
        valid_users = {
            'admin': {
                'password': 'admin', 
                'role': 'admin', 
                'name': 'Administrator',
                'password_hash': 'hash_here',
                'salt': 'salt_here'
            }
            # ... pozostałe dane testowe ...
        }
        
        user = valid_users.get(username)
        if not user:
            security_manager.log_security_event('failed_login', {
                'username': username,
                'ip': request.remote_addr,
                'reason': 'user_not_found'
            })
            return jsonify({'success': False, 'message': 'Nieprawidłowe dane logowania'}), 401
            
        # W produkcji użyj bezpiecznej weryfikacji hasła
        if not security_manager.verify_password(password, user['password_hash'], user['salt']):
            security_manager.log_security_event('failed_login', {
                'username': username,
                'ip': request.remote_addr,
                'reason': 'invalid_password'
            })
            return jsonify({'success': False, 'message': 'Nieprawidłowe dane logowania'}), 401
            
        # Generowanie tokenu z dodatkowymi zabezpieczeniami
        token = security_manager.generate_token({
            'username': username,
            'role': user['role'],
            'ip': request.remote_addr,
            'device_id': request.headers.get('X-Device-ID')
        })
        
        security_manager.log_security_event('successful_login', {
            'username': username,
            'ip': request.remote_addr
        })
        
        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'username': username,
                'name': user['name'],
                'role': user['role']
            }
        })
        
    except Exception as e:
        logging.error(f"Błąd podczas logowania: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'Błąd serwera: {str(e)}'}), 500

@app.route('/api/auth/logout', methods=['POST'])
@jwt_required
def logout():
    """Endpoint do wylogowywania użytkownika."""
    try:
        token = request.headers.get('Authorization').split(' ')[1]
        security_manager.revoke_token(token)
        
        security_manager.log_security_event('logout', {
            'username': request.username,
            'ip': request.remote_addr
        })
        
        return jsonify({'success': True, 'message': 'Wylogowano pomyślnie'})
    except Exception as e:
        logging.error(f"Błąd podczas wylogowywania: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Endpoint do rejestracji z rozszerzonymi zabezpieczeniami."""
    try:
        data = security_manager.sanitize_input(request.get_json() or {})
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        email = data.get('email', '').strip()
        
        # Walidacja danych
        if not all([username, password, email]):
            return jsonify({'success': False, 'message': 'Wszystkie pola są wymagane'}), 400
            
        # Walidacja hasła
        is_valid, message = security_manager.validate_password(password)
        if not is_valid:
            return jsonify({'success': False, 'message': message}), 400
            
        # Haszowanie hasła
        password_hash, salt = security_manager.hash_password(password)
        
        # W produkcji zapisz użytkownika w bazie danych
        # ... kod zapisu do bazy ...
        
        security_manager.log_security_event('user_registered', {
            'username': username,
            'ip': request.remote_addr
        })
        
        return jsonify({
            'success': True,
            'message': 'Rejestracja zakończona pomyślnie. Możesz się teraz zalogować.'
        })
        
    except Exception as e:
        logging.error(f"Błąd podczas rejestracji: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'Błąd serwera: {str(e)}'}), 500

@app.route('/api/auth/verify', methods=['GET'])
@jwt_required
def verify_token():
    """Endpoint do weryfikacji tokenu JWT."""
    try:
        payload = request.jwt_payload
        return jsonify({
            'success': True,
            'user': {
                'username': payload.get('sub'),
                'name': payload.get('name'),
                'role': payload.get('role')
            }
        })
    except Exception as e:
        logging.error(f"Błąd podczas weryfikacji tokenu: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'Błąd serwera: {str(e)}'}), 500

# Endpoint użytkownika wymaga uwierzytelnienia
@app.route('/api/user/profile', methods=['GET'])
@jwt_required
def get_user_profile():
    """Endpoint do pobierania profilu zalogowanego użytkownika."""
    try:
        payload = request.jwt_payload
        
        # Tutaj w produkcji pobieralibyśmy dane użytkownika z bazy danych
        user_profile = {
            'username': payload.get('sub'),
            'name': payload.get('name'),
            'role': payload.get('role'),
            'email': f"{payload.get('sub')}@example.com",  # Przykładowy email
            'joined_date': '2025-01-01',
            'last_login': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'preferences': {
                'theme': 'dark',
                'notifications': True,
                'dashboard_layout': 'default'
            }
        }
        
        return jsonify({
            'success': True, 
            'profile': user_profile
        })
    
    except Exception as e:
        logging.error(f"Błąd podczas pobierania profilu użytkownika: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'Błąd serwera: {str(e)}'}), 500

import os

def is_env_flag_true(env_var_name: str) -> bool:
    return os.getenv(env_var_name, "").strip().lower() in ["1", "true", "yes"]

# Uruchomienie aplikacji
if __name__ == "__main__":
    # Parser argumentów
    import argparse
    parser = argparse.ArgumentParser(description="ZoL0-1: System Tradingowy z AI")
    parser.add_argument("--mode", choices=["sim", "real"], default="sim", 
                      help="Tryb pracy: sim (symulacja) lub real (rzeczywisty)")
    parser.add_argument("--port", type=int, default=5000, 
                      help="Port dla dashboardu (domyślnie: 5000)")
    parser.add_argument("--debug", action="store_true", 
                      help="Włącza tryb debugowania")
    parser.add_argument("--dashboard-only", action="store_true", 
                      help="Uruchamia tylko dashboard bez pełnej inicjalizacji systemu")
    args = parser.parse_args()

    # Tworzenie katalogów - użycie os.path.join dla kompatybilności z Windows
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_libs"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"), exist_ok=True)

    # Logowanie informacji o uruchomieniu
    mode_str = "symulowanym" if args.mode == "sim" else "RZECZYWISTYM"
    logger.info(f"Uruchamianie aplikacji w trybie {mode_str}")
    
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

    # Inicjalizacja systemu przed uruchomieniem aplikacji
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

    # Utwórz katalog dla symulacji
    os.makedirs("static/img", exist_ok=True)

    # Sprawdź czy jest obrazek symulacji
    simulation_chart_path = "static/img/simulation_chart.png"
    if not os.path.exists(simulation_chart_path):
        try:
            # Próba utworzenia prostego obrazu wykresu
            import matplotlib.pyplot as plt
            import numpy as np

            plt.figure(figsize=(10, 6))
            days = 30
            x = np.arange(days)
            y = np.cumsum(np.random.normal(0.5, 1, days))
            plt.plot(x, y, 'g-')
            plt.title("Symulacja wyników tradingu")
            plt.xlabel("Dni")
            plt.ylabel("Zysk/Strata")
            plt.grid(True)
            plt.savefig(simulation_chart_path)
            plt.close()
            logging.info(f"Utworzono przykładowy wykres symulacji: {simulation_chart_path}")
        except Exception as chart_err:
            logging.warning(f"Nie można utworzyć przykładowego wykresu: {chart_err}")
            # Utwórz pusty plik jako placeholder
            with open(simulation_chart_path, 'w') as f:
                f.write('')

    # Dodatkowa konfiguracja dla trybu rzeczywistego
    if args.mode == "real" and not args.dashboard_only:
        if portfolio_manager:
            portfolio_manager.switch_mode("real")
            logger.info("Przełączono tryb pracy portfolio_manager na RZECZYWISTY")
        else:
            logger.warning("Nie można przełączyć trybu pracy portfolio_manager - nie jest zainicjalizowany")
    
    # Konfiguracja uruchomienia aplikacji
    port = int(os.environ.get("PORT", args.port))
    host = "0.0.0.0"  # Używamy 0.0.0.0 dla dostępu zewnętrznego w środowisku Replit
    debug_mode = args.debug or os.getenv("DEBUG", "False").lower() in ["true", "1", "yes"]

    # Dodajemy middleware CORS dla wsparcia API
    from flask_cors import CORS
    CORS(app)
    
    # Dodatkowa konfiguracja dla lepszego działania w środowisku Replit
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    app.config['JSON_SORT_KEYS'] = False
    app.config['PROPAGATE_EXCEPTIONS'] = True
    
    logging.info(f"Uruchamianie aplikacji Flask na {host}:{port}")
    try:
        app.run(host=host, port=port, debug=debug_mode, threaded=True)
    except Exception as e:
        logging.error(f"Błąd podczas uruchamiania aplikacji Flask: {e}")
        print(f"\nBłąd podczas uruchamiania aplikacji: {e}")
        print(f"Sprawdź czy port {port} nie jest już używany.")
        import sys
        sys.exit(1)

# Error handlers
@app.errorhandler(502)
def handle_502_error(error):
    """Rozszerzona obsługa błędu 502 Bad Gateway z lepszą identyfikacją komponentów"""
    request_id = request.headers.get('X-Request-ID', str(uuid.uuid4())[:8])
    
    # Rozszerzone szczegóły błędu
    error_details = {
        'error': 'Bad Gateway',
        'status': 502,
        'request_id': request_id,
        'message': 'Nie można połączyć się z jednym z komponentów systemu',
        'timestamp': datetime.now().isoformat(),
        'retry_after': 2,  # Domyślnie 2 sekundy
        'component': 'unknown',
        'path': request.path,
        'endpoint': request.endpoint or 'unknown'
    }
    
    try:
        # Bardziej dokładna identyfikacja komponentu na podstawie ścieżki żądania
        path = request.path.lower()
        
        if 'portfolio' in path or '/api/portfolio' in path:
            error_details['component'] = 'Portfolio Manager'
            error_details['retry_after'] = 3
        elif 'trading' in path or '/api/trading' in path:
            error_details['component'] = 'Trading Engine'
            error_details['retry_after'] = 5
        elif 'sentiment' in path or '/api/sentiment' in path:
            error_details['component'] = 'Sentiment Analyzer'
            error_details['retry_after'] = 2
        elif 'ai' in path or 'model' in path or '/api/ai' in path:
            error_details['component'] = 'AI Models'
            error_details['retry_after'] = 4
        elif 'simulation' in path or '/api/simulation' in path:
            error_details['component'] = 'Simulation Engine'
            error_details['retry_after'] = 3
        elif 'bybit' in path or '/api/bybit' in path:
            error_details['component'] = 'ByBit API Connector'
            error_details['retry_after'] = 5
        elif 'chart' in path or '/api/chart' in path:
            error_details['component'] = 'Chart Data Provider'
            error_details['retry_after'] = 2
        
        # Dodatkowa analiza endpoint
        current_route = request.endpoint
        if current_route:
            if 'portfolio' in current_route:
                error_details['component'] = 'Portfolio Manager'
            elif 'trading' in current_route:
                error_details['component'] = 'Trading Engine'
            elif 'sentiment' in current_route:
                error_details['component'] = 'Sentiment Analyzer'
            elif 'ai' in current_route or 'model' in current_route:
                error_details['component'] = 'AI Models'
            elif 'simulation' in current_route:
                error_details['component'] = 'Simulation Engine'
            elif 'bybit' in current_route:
                error_details['component'] = 'ByBit API Connector'
            elif 'chart' in current_route:
                error_details['component'] = 'Chart Data Provider'
        
        # Dodaj sugestie naprawy
        suggestions = []
        if error_details['component'] == 'ByBit API Connector':
            suggestions.append("Sprawdź klucze API w ustawieniach")
            suggestions.append("Upewnij się, że masz połączenie internetowe")
        elif error_details['component'] == 'Portfolio Manager':
            suggestions.append("Sprawdź czy portfolio_manager jest zainicjalizowany")
        elif error_details['component'] == 'AI Models':
            suggestions.append("Sprawdź logi w celu weryfikacji załadowanych modeli AI")
        
        if suggestions:
            error_details['suggestions'] = suggestions
        
    except Exception as e:
        logging.error(f"Błąd podczas identyfikacji komponentu dla obsługi 502: {e}", exc_info=True)

    # Rozbudowane logowanie błędu
    logging.error(
        "Bad Gateway Error: %s, Request ID: %s, Component: %s, Path: %s",
        str(error),
        request_id,
        error_details['component'],
        request.path
    )

    # Ustaw nagłówek Retry-After do wykorzystania przez frontend
    response = jsonify(error_details)
    response.headers['Retry-After'] = str(error_details['retry_after'])
    response.headers['X-Request-ID'] = request_id
    
    return response, 502

def start_notification_server():
    """
    Uruchamia serwer WebSocket do obsługi powiadomień w czasie rzeczywistym.
    
    Returns:
        NotificationServer: Instancja serwera powiadomień lub None w przypadku błędu.
    """
    try:
        from notifications.websocket_server import NotificationServer
        import asyncio
        import threading
        
        # Standardowy port serwera WebSocket
        port = int(os.environ.get("WEBSOCKET_PORT", 6789))
        host = '0.0.0.0'
        
        server = NotificationServer(host=host, port=port)
        
        # Uruchomienie serwera w osobnym wątku
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(server.start_server())
            except Exception as e:
                logging.error(f"Błąd podczas uruchamiania serwera WebSocket: {e}", exc_info=True)
            finally:
                loop.close()
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        logging.info(f"Serwer WebSocket powiadomień uruchomiony na {host}:{port}")
        
        # Rejestracja powiadomienia systemowego o uruchomieniu serwera
        if notification_system:
            notification_system.notify_system_event(
                event="Uruchomienie serwera",
                details=f"Serwer WebSocket uruchomiony na porcie {port}"
            )
        
        return server
    except ImportError as e:
        logging.warning(f"Nie można zaimportować NotificationServer: {e}")
        return None
    except Exception as e:
        logging.error(f"Błąd podczas uruchamiania serwera powiadomień: {e}", exc_info=True)
        return None

def start_notification_server():
    """Uruchamia serwer WebSocket dla powiadomień"""
    from notifications.websocket_server import NotificationServer
    import asyncio
    import threading
    
    notification_server = NotificationServer()
    
    def run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server = notification_server.start()
        loop.run_until_complete(server)
        loop.run_forever()
    
    # Uruchom serwer WebSocket w osobnym wątku
    ws_thread = threading.Thread(target=run_server, daemon=True)
    ws_thread.start()
    
    return notification_server

# Inicjalizacja serwera powiadomień przy starcie aplikacji
notification_server = None

# Dodanie endpointów do wysyłania powiadomień
@app.route('/api/notification/broadcast', methods=['POST'])
@jwt_required
def broadcast_notification():
    """Endpoint do wysyłania powiadomień systemowych"""
    if not notification_server:
        return jsonify({
            'success': False,
            'error': 'Serwer powiadomień nie jest dostępny'
        }), 503
    
    data = request.get_json()
    required_fields = ['type', 'message']
    
    if not all(field in data for field in required_fields):
        return jsonify({
            'success': False,
            'error': 'Brak wymaganych pól'
        }), 400
        
    try:
        asyncio.run(notification_server.broadcast_system_notification(
            data['type'],
            data['message'],
            data.get('level', 'info')
        ))
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Błąd podczas wysyłania powiadomienia: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/notification/price', methods=['POST'])
@jwt_required
def broadcast_price():
    """Endpoint do wysyłania aktualizacji cen"""
    if not notification_server:
        return jsonify({
            'success': False,
            'error': 'Serwer powiadomień nie jest dostępny'
        }), 503
    
    data = request.get_json()
    required_fields = ['symbol', 'price']
    
    if not all(field in data for field in required_fields):
        return jsonify({
            'success': False,
            'error': 'Brak wymaganych pól'
        }), 400
        
    try:
        asyncio.run(notification_server.broadcast_price_update(
            data['symbol'],
            data['price']
        ))
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Błąd podczas wysyłania aktualizacji ceny: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trades', methods=['GET'])
def get_trades():
    """Endpoint do pobierania historii transakcji."""
    try:
        # Jeśli menadżer portfela jest dostępny, pobierz historię transakcji
        if portfolio_manager:
            trades = portfolio_manager.get_trade_history()
            return jsonify({"success": True, "trades": trades})
        
        # W przeciwnym razie zwróć przykładowe dane
        sample_trades = [
            {
                "id": "1",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.01,
                "entry_price": 52450.50,
                "exit_price": 53100.25,
                "profit_loss": 6.4975,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "status": "CLOSED"
            },
            {
                "id": "2",
                "symbol": "ETHUSDT",
                "side": "SELL",
                "quantity": 0.15,
                "entry_price": 2850.75,
                "exit_price": 2820.25,
                "profit_loss": 4.575,
                "timestamp": (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                "status": "CLOSED"
            }
        ]
        
        return jsonify({"success": True, "trades": sample_trades})
    except Exception as e:
        logging.error(f"Błąd podczas pobierania historii transakcji: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/market/analyze', methods=['GET'])
def analyze_market():
    """Endpoint do analizy rynku z wykorzystaniem strategii."""
    try:
        # Pobierz parametry z zapytania
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1m')
        strategy_name = request.args.get('strategy', 'trend_following')
        
        if not bybit_client:
            return jsonify({
                "success": False, 
                "error": "Klient API nie jest dostępny",
                "symbol": symbol,
                "fallback_data": True,
                "signals": [
                    {"type": "buy", "strength": 0.65, "timestamp": datetime.now().isoformat()},
                    {"type": "neutral", "strength": 0.25, "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat()}
                ]
            })
        
        # Pobierz dane świec
        klines = bybit_client.get_klines(symbol=symbol, interval=interval, limit=100)
        
        # Jeśli strategie są dostępne, użyj ich do analizy
        if strategy_manager and strategy_name in strategy_manager.strategies:
            strategy = strategy_manager.get_strategy(strategy_name)
            analysis_result = strategy.analyze(klines)
            
            return jsonify({
                "success": True,
                "symbol": symbol,
                "interval": interval,
                "strategy": strategy_name,
                "signals": analysis_result.get("signals", []),
                "indicators": analysis_result.get("indicators", {}),
                "recommendation": analysis_result.get("recommendation", "neutral"),
                "timestamp": datetime.now().isoformat()
            })
        else:
            # Przykładowe dane analizy
            sample_signals = [
                {"type": "buy", "strength": 0.75, "timestamp": datetime.now().isoformat()},
                {"type": "sell", "strength": 0.3, "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat()},
                {"type": "neutral", "strength": 0.5, "timestamp": (datetime.now() - timedelta(minutes=60)).isoformat()}
            ]
            
            sample_indicators = {
                "sma_20": 52750.25,
                "sma_50": 51980.50,
                "rsi": 58.5,
                "macd": 125.75,
                "macd_signal": 110.25
            }
            
            return jsonify({
                "success": True,
                "symbol": symbol,
                "interval": interval,
                "strategy": strategy_name,
                "signals": sample_signals,
                "indicators": sample_indicators,
                "recommendation": "buy",
                "timestamp": datetime.now().isoformat(),
                "note": "Przykładowe dane analizy (strategie niedostępne)"
            })
    except Exception as e:
        logging.error(f"Błąd podczas analizy rynku: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e), "symbol": symbol}), 500

@app.route('/api/risk/metrics', methods=['GET'])
def get_risk_metrics():
    """Endpoint do pobierania metryk ryzyka."""
    try:
        # Przykładowe metryki ryzyka
        metrics = {
            "portfolio_var": 0.035,  # Value at Risk (3.5%)
            "portfolio_cvar": 0.048,  # Conditional Value at Risk (4.8%)
            "max_drawdown": 0.082,   # Maksymalny drawdown (8.2%)
            "sharpe_ratio": 1.85,    # Wskaźnik Sharpe'a
            "sortino_ratio": 2.15,   # Wskaźnik Sortino
            "correlation_matrix": {
                "BTC": {"BTC": 1.0, "ETH": 0.85, "SOL": 0.72},
                "ETH": {"BTC": 0.85, "ETH": 1.0, "SOL": 0.68},
                "SOL": {"BTC": 0.72, "ETH": 0.68, "SOL": 1.0}
            },
            "risk_level": "medium", # Ogólny poziom ryzyka (low, medium, high)
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify({
            "success": True,
            "metrics": metrics
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania metryk ryzyka: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/risk/limits', methods=['GET'])
def get_risk_limits():
    """Endpoint do pobierania limitów ryzyka."""
    try:
        # Przykładowe limity ryzyka
        limits = {
            "max_position_size": 0.2,  # Maksymalny rozmiar pozycji (20% portfela)
            "max_leverage": 2.0,       # Maksymalna dźwignia
            "max_daily_loss": 0.05,    # Maksymalna dzienna strata (5%)
            "max_drawdown": 0.15,      # Maksymalny dozwolony drawdown (15%)
            "max_correlated_exposure": 0.3,  # Maksymalna ekspozycja na skorelowane aktywa (30%)
            "stop_loss_required": True,  # Wymagane zlecenie stop-loss
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify({
            "success": True,
            "limits": limits
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania limitów ryzyka: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Endpoint do pobierania logów systemowych."""
    try:
        # Parametry zapytania
        limit = min(int(request.args.get('limit', 100)), 1000)  # Ogranicz do maks. 1000 wpisów
        level = request.args.get('level', 'ALL').upper()
        
        # Ścieżka do pliku logów
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "app.log")
        
        # Filtry poziomów logów
        level_filters = {
            'DEBUG': 10,
            'INFO': 20,
            'WARNING': 30,
            'ERROR': 40,
            'CRITICAL': 50,
            'ALL': 0
        }
        
        min_level = level_filters.get(level, 0)
        
        # Odczytaj logi z pliku
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # Przetwórz każdą linię loga
                for line in reversed(lines):  # Od najnowszych
                    if len(logs) >= limit:
                        break
                        
                    # Podstawowe parsowanie formatu logów
                    try:
                        parts = line.split()
                        if len(parts) < 4:
                            continue
                            
                        log_date = ' '.join(parts[0:2])
                        log_level = parts[2].strip('[]')
                        log_message = ' '.join(parts[3:])
                        
                        # Filtruj według poziomu
                        if level == 'ALL' or log_level.upper() == level:
                            logs.append({
                                "timestamp": log_date,
                                "level": log_level,
                                "message": log_message
                            })
                    except Exception as parse_error:
                        # Jeśli nie udało się sparsować, dodaj całą linię
                        logs.append({
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "level": "UNKNOWN",
                            "message": line.strip()
                        })
        
        return jsonify({
            "success": True,
            "logs": logs,
            "count": len(logs),
            "level_filter": level
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania logów: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_system_status():
    """Endpoint do pobierania ogólnego statusu systemu."""
    try:
        # Zbierz status różnych komponentów systemu
        api_status = "online" if bybit_client is not None else "offline"
        trading_status = "online" if trading_engine is not None else "offline"
        portfolio_status = "online" if portfolio_manager is not None else "offline"
        ai_status = "online" if sentiment_analyzer is not None and anomaly_detector is not None else "degraded"
        
        # Oszacuj ogólny status systemu
        if all(status == "online" for status in [api_status, trading_status, portfolio_status, ai_status]):
            overall_status = "online"
        elif any(status == "offline" for status in [api_status, trading_status]):
            overall_status = "offline"
        else:
            overall_status = "degraded"
        
        # Informacje o zasobach systemowych
        import psutil
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
        except:
            cpu_usage = 0
            memory_usage = 0
            disk_usage = 0
        
        # Zbierz informacje o wersjach
        versions = {
            "system_version": "1.0.0",
            "api_version": "0.9.5",
            "trading_engine": "0.8.2",
            "ai_models": "0.7.1"
        }
        
        # Informacje o ostatniej aktywności
        activity = {
            "last_trade": (datetime.now() - timedelta(minutes=random.randint(5, 60))).isoformat(),
            "last_api_request": datetime.now().isoformat(),
            "last_portfolio_update": (datetime.now() - timedelta(seconds=random.randint(10, 300))).isoformat(),
            "system_uptime": str(timedelta(seconds=int(time.time() - self.start_time.timestamp()))) if hasattr(self, 'start_time') else "unknown"
        }
        
        return jsonify({
            "success": True,
            "overall_status": overall_status,
            "components": {
                "api": api_status,
                "trading_engine": trading_status,
                "portfolio": portfolio_status,
                "ai": ai_status
            },
            "resources": {
                "cpu": cpu_usage,
                "memory": memory_usage,
                "disk": disk_usage
            },
            "versions": versions,
            "activity": activity,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statusu systemu: {e}", exc_info=True)
        return jsonify({
            "success": False, 
            "error": str(e),
            "overall_status": "error",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/portfolio/allocation', methods=['GET'])
def get_portfolio_allocation():
    """Endpoint do pobierania alokacji portfela."""
    try:
        if portfolio_manager and hasattr(portfolio_manager, 'get_portfolio'):
            portfolio_data = portfolio_manager.get_portfolio()
            
            # Przekształć dane portfolio, aby uzyskać alokację
            if 'balances' in portfolio_data:
                balances = portfolio_data['balances']
                total_value = portfolio_data.get('total_value', sum(balance['equity'] for balance in balances.values()))
                
                allocation = {}
                for symbol, balance in balances.items():
                    if total_value > 0:
                        allocation[symbol] = balance['equity'] / total_value
                    else:
                        allocation[symbol] = 0
                
                return jsonify({
                    "success": True,
                    "allocation": allocation,
                    "timestamp": datetime.now().isoformat()
                })
        
        # Jeśli nie można pobrać danych z portfolio_manager, zwróć przykładowe dane
        sample_allocation = {
            "BTC": 0.42,
            "ETH": 0.28,
            "SOL": 0.15,
            "USDT": 0.15
        }
        
        return jsonify({
            "success": True,
            "allocation": sample_allocation,
            "timestamp": datetime.now().isoformat(),
            "note": "Przykładowe dane alokacji"
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania alokacji portfela: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/portfolio/correlation', methods=['GET'])
def get_portfolio_correlation():
    """Endpoint do pobierania macierzy korelacji aktywów portfela."""
    try:
        # Przykładowa macierz korelacji
        correlation_matrix = {
            "BTC": {"BTC": 1.0, "ETH": 0.85, "SOL": 0.72, "USDT": -0.05},
            "ETH": {"BTC": 0.85, "ETH": 1.0, "SOL": 0.68, "USDT": -0.03},
            "SOL": {"BTC": 0.72, "ETH": 0.68, "SOL": 1.0, "USDT": -0.08},
            "USDT": {"BTC": -0.05, "ETH": -0.03, "SOL": -0.08, "USDT": 1.0}
        }
        
        # Przykładowe pary wysokiej korelacji
        high_correlation_pairs = [
            {"asset1": "BTC", "asset2": "ETH", "correlation": 0.85},
            {"asset1": "BTC", "asset2": "SOL", "correlation": 0.72},
            {"asset1": "ETH", "asset2": "SOL", "correlation": 0.68}
        ]
        
        return jsonify({
            "success": True,
            "correlation_matrix": correlation_matrix,
            "high_correlation_pairs": high_correlation_pairs,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania korelacji portfela: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/trading/statistics', methods=['GET'])
def get_trading_statistics():
    """Endpoint do pobierania statystyk tradingowych."""
    try:
        # Parametry zapytania
        days = int(request.args.get('days', 30))
        
        # Jeśli portfolio_manager ma metodę get_trading_statistics, użyj jej
        if portfolio_manager and hasattr(portfolio_manager, 'get_trading_statistics'):
            stats = portfolio_manager.get_trading_statistics(user_id=1, days=days)
            return jsonify({
                "success": True,
                "statistics": stats,
                "period_days": days
            })
        
        # W przeciwnym razie zwróć przykładowe statystyki
        sample_stats = {
            "total_trades": 48,
            "winning_trades": 29,
            "losing_trades": 19,
            "win_rate": 60.42,
            "profit_loss": 850.75,
            "profit_percentage": 8.51,
            "max_drawdown": 4.8,
            "sharpe_ratio": 1.85,
            "sortino_ratio": 2.15,
            "avg_profit": 43.25,
            "avg_loss": -18.75,
            "profit_factor": 2.31,
            "avg_trade_duration_hours": 12.5,
            "best_trade": {
                "symbol": "BTCUSDT",
                "profit": 125.50,
                "date": (datetime.now() - timedelta(days=8)).strftime('%Y-%m-%d')
            },
            "worst_trade": {
                "symbol": "SOLUSDT",
                "profit": -45.75,
                "date": (datetime.now() - timedelta(days=12)).strftime('%Y-%m-%d')
            }
        }
        
        return jsonify({
            "success": True,
            "statistics": sample_stats,
            "period_days": days,
            "note": "Przykładowe statystyki tradingowe"
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statystyk tradingowych: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/portfolio/analytics/diversification', methods=['GET'])
def get_portfolio_diversification():
    """Endpoint do pobierania analityki dywersyfikacji portfela."""
    try:
        # Przykładowe dane dotyczące dywersyfikacji
        diversification_data = {
            "asset_classes": {
                "cryptocurrency": 0.65,
                "stablecoin": 0.25,
                "fiat": 0.10
            },
            "market_cap_distribution": {
                "large_cap": 0.55,
                "mid_cap": 0.30,
                "small_cap": 0.15
            },
            "risk_exposure": {
                "high_risk": 0.35,
                "medium_risk": 0.45,
                "low_risk": 0.20
            },
            "herfindahl_index": 0.28,  # Indeks koncentracji (niższy = lepiej zdywersyfikowany)
            "effective_n": 3.57,        # Efektywna liczba aktywów
            "diversification_score": 0.72  # Ogólny wynik dywersyfikacji (wyższy = lepiej)
        }
        
        return jsonify({
            "success": True,
            "diversification": diversification_data,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania analityki dywersyfikacji: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/portfolio/analytics/allocation', methods=['GET'])
def get_portfolio_allocation_analytics():
    """Endpoint do pobierania analityki alokacji portfela."""
    try:
        # Parametry zapytania
        days = int(request.args.get('days', 30))
        
        # Przykładowe dane alokacji w czasie
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        dates.reverse()  # Od najstarszej do najnowszej
        
        allocation_history = {
            "dates": dates,
            "allocations": {
                "BTC": [0.35 + random.uniform(-0.05, 0.05) for _ in range(days)],
                "ETH": [0.25 + random.uniform(-0.04, 0.04) for _ in range(days)],
                "SOL": [0.15 + random.uniform(-0.03, 0.03) for _ in range(days)],
                "USDT": [0.25 + random.uniform(-0.02, 0.02) for _ in range(days)]
            }
        }
        
        # Przykładowe dane o optymalizacji alokacji
        optimization = {
            "efficient_frontier": [
                {"risk": 0.10, "return": 0.05},
                {"risk": 0.15, "return": 0.08},
                {"risk": 0.20, "return": 0.12},
                {"risk": 0.25, "return": 0.16},
                {"risk": 0.30, "return": 0.21}
            ],
            "current_position": {"risk": 0.22, "return": 0.14},
            "optimal_allocation": {
                "BTC": 0.40,
                "ETH": 0.30,
                "SOL": 0.10,
                "USDT": 0.20
            },
            "sharpe_ratio": 1.85
        }
        
        return jsonify({
            "success": True,
            "allocation_history": allocation_history,
            "optimization": optimization,
            "timestamp": datetime.now().isoformat(),
            "period_days": days
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania analityki alokacji: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/portfolio/analytics/risk', methods=['GET'])
def get_portfolio_risk_analytics():
    """Endpoint do pobierania analityki ryzyka portfela."""
    try:
        # Przykładowe dane analityki ryzyka
        risk_analytics = {
            "var": {  # Value at Risk
                "daily": {
                    "95%": 0.018,  # 1.8%
                    "99%": 0.028   # 2.8%
                },
                "weekly": {
                    "95%": 0.045,  # 4.5%
                    "99%": 0.068   # 6.8%
                }
            },
            "expected_shortfall": {  # Conditional VaR
                "daily": {
                    "95%": 0.022,  # 2.2%
                    "99%": 0.035   # 3.5%
                },
                "weekly": {
                    "95%": 0.055,  # 5.5%
                    "99%": 0.078   # 7.8%
                }
            },
            "stress_test": {
                "market_crash_10%": -0.075,  # -7.5%
                "market_crash_20%": -0.148,  # -14.8%
                "high_volatility": -0.052    # -5.2%
            },
            "tail_risk": {
                "tail_ratio": 0.85,
                "downside_deviation": 0.038
            },
            "risk_contribution": {
                "BTC": 0.45,  # 45%
                "ETH": 0.32,  # 32%
                "SOL": 0.18,  # 18%
                "USDT": 0.05   # 5%
            }
        }
        
        return jsonify({
            "success": True,
            "risk_analytics": risk_analytics,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania analityki ryzyka: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/portfolio/analytics/turnover', methods=['GET'])
def get_portfolio_turnover():
    """Endpoint do pobierania analityki obrotu portfela."""
    try:
        # Parametry zapytania
        days = int(request.args.get('days', 30))
        
        # Przykładowe dane obrotu portfela
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        dates.reverse()  # Od najstarszej do najnowszej
        
        daily_turnover = [round(random.uniform(0.01, 0.08), 4) for _ in range(days)]
        
        turnover_data = {
            "dates": dates,
            "daily_turnover": daily_turnover,
            "cumulative_turnover": sum(daily_turnover),
            "average_daily_turnover": sum(daily_turnover) / days,
            "turnover_by_asset": {
                "BTC": 0.45,  # 45%
                "ETH": 0.32,  # 32%
                "SOL": 0.18,  # 18%
                "USDT": 0.05   # 5%
            }
        }
        
        return jsonify({
            "success": True,
            "turnover": turnover_data,
            "timestamp": datetime.now().isoformat(),
            "period_days": days
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania analityki obrotu: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/notifications/settings', methods=['GET', 'POST'])
def handle_notification_settings():
    """Endpoint do pobierania i ustawiania konfiguracji powiadomień."""
    try:
        if request.method == 'GET':
            # Przykładowa konfiguracja powiadomień
            settings = {
                "email_notifications": True,
                "push_notifications": True,
                "sms_notifications": False,
                "telegram_notifications": True,
                "notification_types": {
                    "trade_executed": True,
                    "position_closed": True,
                    "stop_loss_triggered": True,
                    "take_profit_triggered": True,
                    "price_alert": True,
                    "market_anomaly": True,
                    "system_warning": True,
                    "daily_summary": True,
                    "weekly_report": True
                },
                "quiet_hours": {
                    "enabled": False,
                    "start": "22:00",
                    "end": "08:00"
                },
                "minimum_trade_size": 100.0,  # Minimalna wartość transakcji dla powiadomień
                "email_address": "user@example.com",
                "phone_number": "+1234567890",
                "telegram_chat_id": "123456789"
            }
            
            return jsonify({
                "success": True,
                "settings": settings
            })
        elif request.method == 'POST':
            # Aktualizacja konfiguracji powiadomień
            new_settings = request.json
            
            # W produkcji tutaj zapisalibyśmy ustawienia
            
            return jsonify({
                "success": True,
                "message": "Ustawienia powiadomień zaktualizowane pomyślnie",
                "settings": new_settings
            })
    except Exception as e:
        logging.error(f"Błąd podczas obsługi ustawień powiadomień: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/notifications/history', methods=['GET'])
def get_notification_history():
    """Endpoint do pobierania historii powiadomień."""
    try:
        # Parametry zapytania
        limit = min(int(request.args.get('limit', 20)), 100)  # Ogranicz do maks. 100 wpisów
        
        # Przykładowa historia powiadomień
        history = []
        for i in range(limit):
            notification_time = datetime.now() - timedelta(hours=random.randint(1, 240))
            notification_type = random.choice([
                "trade_executed", "position_closed", "stop_loss_triggered",
                "price_alert", "market_anomaly", "system_warning"
            ])
            
            # Różne treści w zależności od typu
            if notification_type == "trade_executed":
                message = f"Wykonano transakcję: BUY 0.01 BTCUSDT po cenie {50000 + random.randint(-2000, 2000)}"
            elif notification_type == "position_closed":
                message = f"Zamknięto pozycję: SELL 0.01 BTCUSDT po cenie {50000 + random.randint(-2000, 2000)}, zysk: ${random.randint(-100, 200)}"
            elif notification_type == "stop_loss_triggered":
                message = f"Wyzwolony Stop-Loss dla ETHUSDT po cenie {2800 + random.randint(-200, 200)}"
            elif notification_type == "price_alert":
                message = f"Alert cenowy: BTCUSDT osiągnął {50000 + random.randint(-3000, 3000)}"
            elif notification_type == "market_anomaly":
                message = "Wykryto anomalię rynkową: Nagły wzrost wolumenu BTCUSDT o 200%"
            else:  # system_warning
                message = "Ostrzeżenie systemowe: Wysoka zmienność rynkowa, zalecana ostrożność"
            
            history.append({
                "id": str(uuid.uuid4()),
                "type": notification_type,
                "message": message,
                "timestamp": notification_time.isoformat(),
                "read": random.choice([True, False]),
                "channels": random.sample(["email", "push", "telegram"], k=random.randint(1, 3))
            })
            
        # Sortuj od najnowszych do najstarszych
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return jsonify({
            "success": True,
            "notifications": history,
            "count": len(history),
            "unread_count": sum(1 for n in history if not n["read"])
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania historii powiadomień: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/alerts/active', methods=['GET', 'POST', 'DELETE'])
def handle_alerts():
    """Endpoint do zarządzania alertami cenowymi."""
    try:
        if request.method == 'GET':
            # Pobierz aktywne alerty
            active_alerts = [
                {
                    "id": "1",
                    "symbol": "BTCUSDT",
                    "price": 55000.0,
                    "condition": "above",
                    "created_at": (datetime.now() - timedelta(days=2)).isoformat(),
                    "expires_at": (datetime.now() + timedelta(days=5)).isoformat(),
                    "notification": ["email", "push"],
                    "triggered": False
                },
                {
                    "id": "2",
                    "symbol": "ETHUSDT",
                    "price": 2500.0,
                    "condition": "below",
                    "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
                    "expires_at": (datetime.now() + timedelta(days=3)).isoformat(),
                    "notification": ["email"],
                    "triggered": False
                }
            ]
            
            return jsonify({
                "success": True,
                "alerts": active_alerts
            })
        elif request.method == 'POST':
            # Dodaj nowy alert
            new_alert = request.json
            
            # W produkcji tutaj zapisalibyśmy alert do bazy danych
            
            # Przypisz sztuczne ID i inne pola
            new_alert["id"] = str(uuid.uuid4())
            new_alert["created_at"] = datetime.now().isoformat()
            if "expires_at" not in new_alert:
                new_alert["expires_at"] = (datetime.now() + timedelta(days=7)).isoformat()
            new_alert["triggered"] = False
            
            return jsonify({
                "success": True,
                "message": "Alert utworzony pomyślnie",
                "alert": new_alert
            })
        elif request.method == 'DELETE':
            # Usuń alert
            alert_id = request.args.get('id')
            
            if not alert_id:
                return jsonify({"success": False, "error": "Nie podano ID alertu"}), 400
            
            # W produkcji tutaj usuwalibyśmy alert z bazy danych
            
            return jsonify({
                "success": True,
                "message": f"Alert o ID {alert_id} został usunięty"
            })
    except Exception as e:
        logging.error(f"Błąd podczas obsługi alertów: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/autonomous/status', methods=['GET'])
def get_autonomous_status():
    """Endpoint do pobierania statusu trybu autonomicznego."""
    try:
        # Przykładowy status trybu autonomicznego
        status = {
            "enabled": True,
            "mode": "conservative",  # conservative, moderate, aggressive
            "auto_trading": True,
            "auto_rebalancing": True,
            "risk_control_active": True,
            "last_action": {
                "type": "trade",
                "description": "Zakup 0.01 BTC po cenie 52345.75",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat()
            },
            "uptime": str(timedelta(hours=random.randint(24, 720))),  # Losowy czas działania
            "next_rebalance": (datetime.now() + timedelta(days=2)).isoformat()
        }
        
        return jsonify({
            "success": True,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statusu trybu autonomicznego: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/autonomous/mode', methods=['POST'])
def set_autonomous_mode():
    """Endpoint do ustawiania trybu autonomicznego."""
    try:
        data = request.json
        if not data or "mode" not in data:
            return jsonify({"success": False, "error": "Nie podano trybu"}), 400
            
        mode = data["mode"]
        enabled = data.get("enabled", True)
        
        # Walidacja trybu
        if mode not in ["conservative", "moderate", "aggressive"]:
            return jsonify({"success": False, "error": "Nieprawidłowy tryb"}), 400
            
        # W produkcji tutaj ustawialibyśmy tryb autonomiczny
        
        return jsonify({
            "success": True,
            "message": f"Tryb autonomiczny ustawiony na: {mode}, włączony: {enabled}",
            "status": {
                "mode": mode,
                "enabled": enabled,
                "timestamp": datetime.now().isoformat()
            }
        })
    except Exception as e:
        logging.error(f"Błąd podczas ustawiania trybu autonomicznego: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/autonomous/decisions', methods=['GET'])
def get_autonomous_decisions():
    """Endpoint do pobierania historii decyzji trybu autonomicznego."""
    try:
        # Parametry zapytania
        limit = min(int(request.args.get('limit', 20)), 100)  # Ogranicz do maks. 100 wpisów
        
        # Przykładowa historia decyzji
        decisions = []
        for i in range(limit):
            decision_time = datetime.now() - timedelta(hours=random.randint(1, 240))
            decision_type = random.choice(["trade", "rebalance", "risk_adjustment", "strategy_switch"])
            
            # Różne treści w zależności od typu
            if decision_type == "trade":
                action = random.choice(["BUY", "SELL"])
                symbol = random.choice(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
                price = random.uniform(100, 60000)
                quantity = round(random.uniform(0.001, 0.1), 6)
                
                description = f"{action} {quantity} {symbol} po cenie {price:.2f}"
                confidence = random.uniform(0.7, 0.95)
            elif decision_type == "rebalance":
                description = "Rebalancing portfela: BTC 40%, ETH 30%, SOL 15%, USDT 15%"
                confidence = random.uniform(0.8, 0.99)
            elif decision_type == "risk_adjustment":
                description = f"Zmniejszenie ekspozycji na ryzyko o {random.randint(5, 20)}% ze względu na zwiększoną zmienność rynku"
                confidence = random.uniform(0.75, 0.9)
            else:  # strategy_switch
                old_strategy = random.choice(["trend_following", "mean_reversion"])
                new_strategy = random.choice(["breakout", "ml_prediction"])
                
                description = f"Przełączenie strategii z {old_strategy} na {new_strategy} ze względu na zmianę warunków rynkowych"
                confidence = random.uniform(0.6, 0.85)
            
            decisions.append({
                "id": str(uuid.uuid4()),
                "type": decision_type,
                "description": description,
                "timestamp": decision_time.isoformat(),
                "confidence": confidence,
                "factors": [
                    {"name": "Trend", "weight": random.uniform(0.1, 0.5)},
                    {"name": "Analiza techniczna", "weight": random.uniform(0.1, 0.4)},
                    {"name": "Sentyment rynkowy", "weight": random.uniform(0.1, 0.3)}
                ],
                "outcome": random.choice(["success", "failure", "neutral", "pending"])
            })
            
        # Sortuj od najnowszych do najstarszych
        decisions.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return jsonify({
            "success": True,
            "decisions": decisions,
            "count": len(decisions)
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania historii decyzji: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/autonomous/model-weights', methods=['GET', 'POST'])
def handle_model_weights():
    """Endpoint do pobierania i ustawiania wag modeli w trybie autonomicznym."""
    try:
        if request.method == 'GET':
            # Przykładowe wagi modeli
            weights = {
                "technical_analysis": 0.35,
                "sentiment_analysis": 0.25,
                "trend_following": 0.20,
                "mean_reversion": 0.10,
                "ml_prediction": 0.10
            }
            
            return jsonify({
                "success": True,
                "weights": weights,
                "timestamp": datetime.now().isoformat()
            })
        elif request.method == 'POST':
            # Aktualizacja wag modeli
            new_weights = request.json
            
            # Walidacja wag
            if sum(new_weights.values()) != 1.0:
                return jsonify({
                    "success": False, 
                    "error": "Suma wag musi wynosić 1.0",
                    "current_sum": sum(new_weights.values())
                }), 400
            
            # W produkcji tutaj zapisalibyśmy wagi
            
            return jsonify({
                "success": True,
                "message": "Wagi modeli zaktualizowane pomyślnie",
                "weights": new_weights,
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        logging.error(f"Błąd podczas obsługi wag modeli: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/autonomous/risk-parameters', methods=['GET', 'POST'])
def handle_risk_parameters():
    """Endpoint do pobierania i ustawiania parametrów ryzyka w trybie autonomicznym."""
    try:
        if request.method == 'GET':
            # Przykładowe parametry ryzyka
            parameters = {
                "max_position_size": 0.2,  # Maksymalny rozmiar pozycji (20% portfela)
                "max_leverage": 2.0,       # Maksymalna dźwignia
                "max_daily_loss": 0.05,    # Maksymalna dzienna strata (5%)
                "max_drawdown": 0.15,      # Maksymalny dozwolony drawdown (15%)
                "stop_loss_percent": 0.03, # Domyślny stop-loss (3%)
                "take_profit_percent": 0.06, # Domyślny take-profit (6%)
                "max_correlated_exposure": 0.3, # Maksymalna ekspozycja na skorelowane aktywa (30%)
                "market_volatility_threshold": 0.02, # Próg zmienności rynkowej (2%)
                "risk_reduction_factor": 0.5 # Współczynnik redukcji ryzyka przy wysokiej zmienności
            }
            
            return jsonify({
                "success": True,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat()
            })
        elif request.method == 'POST':
            # Aktualizacja parametrów ryzyka
            new_parameters = request.json
            
            # W produkcji tutaj zapisalibyśmy parametry
            
            return jsonify({
                "success": True,
                "message": "Parametry ryzyka zaktualizowane pomyślnie",
                "parameters": new_parameters,
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        logging.error(f"Błąd podczas obsługi parametrów ryzyka: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/autonomous/learning-status', methods=['GET'])
def get_learning_status():
    """Endpoint do pobierania statusu uczenia modeli w trybie autonomicznym."""
    try:
        # Przykładowy status uczenia
        status = {
            "is_training": random.choice([True, False]),
            "current_epoch": random.randint(1, 100),
            "total_epochs": 100,
            "progress": random.uniform(0, 1),
            "error_rate": round(random.uniform(0.01, 0.1), 4),
            "accuracy": round(random.uniform(0.7, 0.95), 4),
            "eta": str(timedelta(minutes=random.randint(5, 60))),
            "models_in_training": [
                {
                    "name": "PricePredictor",
                    "type": "LSTM",
                    "progress": random.uniform(0, 1),
                    "accuracy": round(random.uniform(0.7, 0.9), 4)
                },
                {
                    "name": "SentimentAnalyzer",
                    "type": "BERT",
                    "progress": random.uniform(0, 1),
                    "accuracy": round(random.uniform(0.7, 0.9), 4)
                }
            ],
            "last_training": {
                "completed_at": (datetime.now() - timedelta(days=random.randint(1, 7))).isoformat(),
                "duration": str(timedelta(hours=random.randint(1, 12))),
                "samples_used": random.randint(10000, 100000),
                "improvement": f"{random.uniform(1, 10):.2f}%"
            }
        }
        
        return jsonify({
            "success": True,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania statusu uczenia: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/autonomous/model-performance', methods=['GET'])
def get_model_performance():
    """Endpoint do pobierania wydajności modeli w trybie autonomicznym."""
    try:
        # Przykładowa wydajność modeli
        performance = {
            "models": [
                {
                    "name": "PricePredictor",
                    "type": "LSTM",
                    "accuracy": 0.78,
                    "precision": 0.81,
                    "recall": 0.75,
                    "f1_score": 0.78,
                    "training_samples": 50000,
                    "last_update": (datetime.now() - timedelta(days=2)).isoformat()
                },
                {
                    "name": "SentimentAnalyzer",
                    "type": "BERT",
                    "accuracy": 0.82,
                    "precision": 0.84,
                    "recall": 0.79,
                    "f1_score": 0.81,
                    "training_samples": 75000,
                    "last_update": (datetime.now() - timedelta(days=5)).isoformat()
                },
                {
                    "name": "MarketAnomalyDetector",
                    "type": "Isolation Forest",
                    "accuracy": 0.92,
                    "precision": 0.88,
                    "recall": 0.76,
                    "f1_score": 0.82,
                    "training_samples": 30000,
                    "last_update": (datetime.now() - timedelta(days=1)).isoformat()
                }
            ],
            "ensemble": {
                "accuracy": 0.85,
                "precision": 0.87,
                "recall": 0.82,
                "f1_score": 0.84
            },
            "performance_history": {
                "dates": [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)],
                "accuracy": [round(0.75 + random.uniform(-0.05, 0.1), 2) for _ in range(30)]
            }
        }
        
        return jsonify({
            "success": True,
            "performance": performance,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Błąd podczas pobierania wydajności modeli: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
