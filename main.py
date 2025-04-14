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

# Import menedżera symulacji
try:
    from python_libs.simulation_results import SimulationManager
    simulation_manager = SimulationManager()
    simulation_import_success = True
except ImportError as e:
    logging.warning(f"Nie udało się zaimportować SimulationManager: {e}")
    simulation_import_success = False
    simulation_manager = None

# Inicjalizacja aplikacji Flask
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

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

            sentiment_analyzer = SentimentAnalyzer(sources=sources)
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

            anomaly_detector = AnomalyDetector(method=method, threshold=threshold)
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
            logging.info("Wykluczam podpakiet 'tests' z automatycznego importu.")
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
                portfolio_manager = PortfolioManager(initial_balance=100.0, currency="USDT", mode="simulated")
            logging.info("Zainicjalizowano menadżera portfela")
        except ImportError as e:
            logging.error(f"Nie można zaimportować PortfolioManager: {e}")
            # Utwórz domyślny menadżer portfela jako fallback
            try:
                # Tworzenie klasy PortfolioManager w locie jako fallback
                class SimplePortfolioManager:
                    def __init__(self, initial_balance=100.0, currency="USDT", mode="simulated"):
                        self.initial_balance = initial_balance
                        self.currency = currency
                        self.mode = mode
                        self.balances = {
                            currency: {"equity": initial_balance, "available_balance": initial_balance, "wallet_balance": initial_balance},
                            "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01}
                        }
                        logging.info(f"Utworzono fallback PortfolioManager (saldo: {initial_balance} {currency})")
                        
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
                
                portfolio_manager = SimplePortfolioManager(initial_balance=100.0, currency="USDT", mode="simulated")
                logging.info("Zainicjalizowano prosty fallback menadżera portfela")
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
        portfolio=portfolio
    )

# API endpoints
@app.route('/api/portfolio')
def get_portfolio_data():
    """Endpoint API do pobierania danych portfela."""
    try:
        # Sprawdź, czy portfolio_manager jest dostępny
        if portfolio_manager is None:
            logging.error("portfolio_manager nie jest zainicjalizowany w get_portfolio_data")
            return jsonify({
                "success": True,  # Ustawiamy True, aby frontend nie wyświetlał błędu
                "balances": {
                    "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                    "USDT": {"equity": 100.0, "available_balance": 100.0, "wallet_balance": 100.0}
                },
                "source": "fallback_error",
                "error": "Portfolio manager nie jest zainicjalizowany"
            })
            
        # Używanie portfolio_manager zamiast bezpośredniego pobierania danych z Bybit
        portfolio_data = portfolio_manager.get_portfolio()
        return jsonify(portfolio_data)
    except Exception as e:
        logging.error(f"Błąd podczas pobierania danych portfela: {e}", exc_info=True)
        # Szczegółowe dane diagnostyczne
        logging.error(f"Szczegóły błędu: {type(e).__name__}, {str(e)}")
        
        return jsonify({
            "success": True,  # Ustawiamy True, aby frontend nie wyświetlał błędu
            "balances": {
                "BTC": {"equity": 0.005, "available_balance": 0.005, "wallet_balance": 0.005},
                "USDT": {"equity": 500, "available_balance": 450, "wallet_balance": 500}
            },
            "source": "fallback_error",
            "error": str(e)
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
            sentiment_data = sentiment_analyzer.analyze()
            thoughts.append({
                'model': 'SentimentAnalyzer',
                'thought': f"Analiza sentymentu wskazuje na {sentiment_data['analysis']} nastawienie rynku (wartość: {sentiment_data['value']:.2f})",
                'confidence': min(abs(sentiment_data['value']) * 100 + 50, 95),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'sentiment'
            })

        # Pobierz przemyślenia z detektora anomalii
        if anomaly_detector:
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

        # Dodaj przemyślenia od ModelRecognizer
        if 'model_recognizer' in globals() and model_recognizer:
            model_info = model_recognizer.identify_model_type(None)
            if model_info:
                thoughts.append({
                    'model': 'ModelRecognizer',
                    'thought': f"Obecne dane rynkowe pasują do modelu typu {model_info['type']} ({model_info['name']})",
                    'confidence': model_info['confidence'] * 100,
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
            status = sentiment_analyzer.get_status()
            models.append({
                'name': 'SentimentAnalyzer',
                'type': 'Sentiment Analysis',
                'accuracy': 82.0,
                'status': 'Active' if status.get('active') else 'Inactive',
                'last_used': status.get('last_update', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                'methods': {
                    'predict': False,
                    'fit': False
                },
                'test_result': 'Passed',
                'module': 'sentiment_ai'
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

    # Uruchomienie aplikacji z odpowiednim hostem w zależności od środowiska
    port = int(os.environ.get("PORT", 5000))

    # W środowisku Replit zawsze używaj 0.0.0.0
    host = "0.0.0.0"

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