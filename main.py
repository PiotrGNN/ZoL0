"""Trading System Entry Point

This module initializes and runs the trading system with proper lifecycle management.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from flask import Flask, jsonify, request
from datetime import datetime
import atexit

# Add project root to path
project_root = str(Path(__file__).parent.absolute())
if (project_root not in sys.path):
    sys.path.insert(0, project_root)

# Import core components
from config import config, get_logger, check_configuration
from core.monitoring.metrics import get_metrics_collector, MetricsCollector
from core.notifications import (
    get_notification_manager,
    Notification,
    NotificationPriority,
)
from core.components import (
    initialize_trading_engine,
    initialize_risk_manager,
    initialize_strategy_manager,
    initialize_portfolio_manager,
    initialize_market_data,
)
from core.database import DatabaseManager
from core.exchange import ExchangeConnector
from core.trading import TradingEngine
from core.market_data import MarketDataManager


# --- STUBS for missing modules ---
class MetricsCollector:
    def get_recent_metrics(self):
        return {"metrics": "not implemented"}


def get_metrics_collector():
    return MetricsCollector()


class Notification:
    def __init__(self, title, message, priority=None):
        self.title = title
        self.message = message
        self.priority = priority


class NotificationPriority:
    LOW = "low"
    CRITICAL = "critical"


def get_notification_manager():
    class NotificationManager:
        def send_notification(self, notification):
            pass

    return NotificationManager()


# --- END STUBS ---

# Initialize core services
logger = get_logger()
metrics = get_metrics_collector()
notifications = get_notification_manager()


class TradingSystem:
    """Trading system singleton managing component lifecycle."""

    def __init__(self, config_dict: dict):
        self.config = config_dict
        self.db_manager: Optional[DatabaseManager] = None
        self.exchange: Optional[ExchangeConnector] = None
        self.portfolio_manager = None
        self.risk_manager = None
        self.strategy_manager = None
        self.market_data: Optional[MarketDataManager] = None
        self.trading_engine: Optional[TradingEngine] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize all system components."""
        logger.info("[TradingSystem] Inicjalizacja systemu...")
        try:
            logger.info(f"[TradingSystem] config: {self.config}")
            logger.info(f"[TradingSystem] get_logger: {get_logger}")
            logger.info(f"[TradingSystem] check_configuration: {check_configuration}")
        except Exception as e:
            logger.error(f"[TradingSystem] Błąd podczas logowania configu: {e}")
        try:
            # Verify configuration first
            if not check_configuration(exit_on_error=False):
                return False

            config_dict = self.config

            # Initialize database
            db_path = config_dict.get("database.path", "data/trading.db")
            if not os.path.isabs(db_path):
                db_path = os.path.join(project_root, db_path)
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            self.db_manager = DatabaseManager(db_path)
            logger.info(f"Database initialized at {db_path}")

            # Initialize exchange connection
            self.exchange = ExchangeConnector(
                api_key=os.getenv("BYBIT_API_KEY"),
                api_secret=os.getenv("BYBIT_API_SECRET"),
                testnet=config_dict.get("exchange.use_testnet", True),
            )
            logger.info("Exchange connector initialized")

            # Initialize core components
            self.strategy_manager = initialize_strategy_manager(config_dict)
            self.risk_manager = initialize_risk_manager(config_dict)
            self.portfolio_manager = initialize_portfolio_manager(
                self.db_manager, config_dict
            )
            self.market_data = initialize_market_data(self.exchange, config_dict)

            # Initialize trading engine last
            self.trading_engine = initialize_trading_engine(
                exchange=self.exchange,
                strategy_manager=self.strategy_manager,
                risk_manager=self.risk_manager,
                portfolio_manager=self.portfolio_manager,
                market_data=self.market_data,
                config=config_dict,
            )

            self._initialized = True
            logger.info("System initialization completed")
            notifications.send_notification(
                Notification(
                    "System Initialized",
                    "Trading system initialization completed successfully",
                    priority=NotificationPriority.LOW,
                )
            )
            return True

        except Exception as e:
            logger.error(f"Error during system initialization: {str(e)}")
            notifications.send_notification(
                Notification(
                    "Initialization Failed",
                    f"System initialization failed: {str(e)}",
                    priority=NotificationPriority.CRITICAL,
                )
            )
            return False

    def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        try:
            if self.trading_engine:
                self.trading_engine.stop()
                logger.info("Trading engine stopped")

            if self.exchange:
                self.exchange.close()
                logger.info("Exchange connection closed")

            if self.db_manager:
                self.db_manager.close()
                logger.info("Database connection closed")

            logger.info("System shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "initialized": self._initialized,
            "trading_engine": {
                "status": (
                    "active"
                    if self.trading_engine and self.trading_engine.is_running()
                    else "inactive"
                ),
                "mode": self.trading_engine.mode if self.trading_engine else None,
            },
            "exchange": {
                "status": (
                    "connected"
                    if self.exchange and self.exchange.is_connected()
                    else "disconnected"
                ),
                "testnet": self.exchange.is_testnet if self.exchange else None,
            },
            "components": {
                "database": self.db_manager is not None,
                "portfolio": self.portfolio_manager is not None,
                "risk_manager": self.risk_manager is not None,
                "strategy_manager": self.strategy_manager is not None,
                "market_data": self.market_data is not None,
            },
        }


# Create Flask application
app = Flask(__name__)
config_dict = config.config
app.config.update(
    ENV="development" if config_dict.get("environment.debug", True) else "production",
    DEBUG=config_dict.get("environment.debug", True),
    TESTING=False,
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY", os.urandom(24)),
    JSON_SORT_KEYS=False,
    JSONIFY_PRETTYPRINT_REGULAR=True,
)

# Initialize trading system singleton
trading_system = TradingSystem(config.config)
trading_system.initialize()  # Ensure initialization always runs, even under Gunicorn

# Register shutdown handler
atexit.register(trading_system.shutdown)


@app.route("/")
def index():
    """API documentation endpoint."""
    return jsonify(
        {
            "status": "Trading system running",
            "version": "1.0.0",
            "endpoints": {
                "GET /": "API documentation",
                "GET /health": "System health check",
                "GET /api/status": "System status and metrics",
                "GET /api/portfolio": "Current portfolio state",
                "GET /api/trades": "Trading history",
                "GET /api/market/analyze": "Market analysis",
                "GET /api/component-status": "Component health status",
            },
        }
    )


@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": trading_system.get_status()["components"],
        }
    )


@app.route("/api/status")
def get_status():
    """System status endpoint."""
    if not trading_system._initialized:
        return jsonify({"error": "System not initialized"}), 503

    return jsonify(
        {
            "status": trading_system.get_status(),
            "metrics": metrics.get_recent_metrics(),
            "portfolio": (
                trading_system.portfolio_manager.get_summary()
                if trading_system.portfolio_manager
                else None
            ),
        }
    )


@app.route("/api/portfolio")
def get_portfolio():
    """Portfolio status endpoint."""
    if not trading_system.portfolio_manager:
        return jsonify({"error": "Portfolio manager not initialized"}), 503

    return jsonify(trading_system.portfolio_manager.get_detailed_status())


@app.route("/api/trades")
def get_trades():
    """Trading history endpoint."""
    if not trading_system.db_manager:
        return jsonify({"error": "Database not initialized"}), 503

    limit = request.args.get("limit", default=50, type=int)
    offset = request.args.get("offset", default=0, type=int)
    return jsonify(trading_system.db_manager.get_recent_trades(limit, offset))


@app.route("/api/market/analyze")
def analyze_market():
    """Market analysis endpoint."""
    if not trading_system.trading_engine:
        return jsonify({"error": "Trading engine not initialized"}), 503

    symbol = request.args.get("symbol", default="BTCUSDT")
    timeframe = request.args.get("timeframe", default="1h")

    analysis = trading_system.trading_engine.analyze_market(symbol, timeframe)
    return jsonify(analysis)


@app.route("/api/component-status")
def get_component_status():
    """Component health status endpoint."""
    return jsonify(trading_system.get_status())


if __name__ == "__main__":
    # Initialization already done above
    logger.info("Starting API server")
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=config_dict.get("environment.debug", True))
