"""Main application module."""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config_loader import ConfigLoader
from utils.monitoring import get_metrics_collector
from utils.notification_system import get_notification_manager, Notification, NotificationPriority
from data.logging.system_logger import get_logger

# Initialize core systems
logger = get_logger()
metrics = get_metrics_collector()
notifications = get_notification_manager()
config = ConfigLoader().load()

def initialize_system():
    """Initialize trading system components."""
    try:
        # Import core components
        from python_libs.portfolio_manager import PortfolioManager
        from python_libs.model_initializer import model_initializer
        from data.utils.database_manager import DatabaseManager

        # Initialize database
        db_path = config.get("database", {}).get("path", "trading.db")
        db_manager = DatabaseManager(db_path)
        logger.log_info("Database initialized")

        # Initialize portfolio manager
        global portfolio_manager
        portfolio_manager = PortfolioManager(db_manager=db_manager, config=config)
        logger.log_info("Portfolio manager initialized")

        # Initialize AI models
        logger.log_info("Initializing AI models...")
        model_initializer.initialize_models()

        # Initialize trading components
        _initialize_trading_components()

        logger.log_info("System initialization completed")
        notifications.send_notification(
            Notification(
                "System Initialized",
                "Trading system initialization completed successfully",
                priority=NotificationPriority.LOW
            )
        )
        return True

    except Exception as e:
        logger.log_error(f"Error during system initialization: {e}", error_type="InitializationError")
        notifications.send_notification(
            Notification(
                "Initialization Failed",
                f"System initialization failed: {str(e)}",
                priority=NotificationPriority.CRITICAL
            )
        )
        return False

def _initialize_trading_components():
    """Initialize trading-specific components."""
    try:
        # Initialize strategy manager
        from python_libs.simplified_strategy import StrategyManager
        
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
        
        global strategy_manager
        strategy_manager = StrategyManager(strategies, exposure_limits)
        logger.log_info(f"Strategy manager initialized with {len(strategies)} strategies")

        # Initialize risk manager
        from python_libs.simplified_risk_manager import SimplifiedRiskManager
        global risk_manager
        risk_manager = SimplifiedRiskManager(
            max_risk=config.get("trading", {}).get("max_risk", 0.05),
            max_position_size=config.get("trading", {}).get("max_position_size", 0.2),
            max_drawdown=config.get("trading", {}).get("max_drawdown", 0.1)
        )
        logger.log_info("Risk manager initialized")

        # Initialize exchange connector
        from python_libs.simulated_bybit import SimulatedBybitConnector
        global exchange
        exchange = SimulatedBybitConnector(
            api_key=config.get("exchange", {}).get("api_key", "simulated_key"),
            api_secret=config.get("exchange", {}).get("api_secret", "simulated_secret"),
            use_testnet=config.get("exchange", {}).get("use_testnet", True)
        )
        logger.log_info("Exchange connector initialized")

        # Initialize trading engine
        if all([strategy_manager, risk_manager, exchange]):
            from python_libs.simplified_trading_engine import SimplifiedTradingEngine
            global trading_engine
            trading_engine = SimplifiedTradingEngine(
                risk_manager=risk_manager,
                strategy_manager=strategy_manager,
                exchange_connector=exchange
            )
            
            # Start trading engine with default symbols
            symbols = config.get("trading", {}).get("symbols", ["BTCUSDT"])
            if trading_engine.start_trading(symbols):
                logger.log_info(f"Trading engine started with symbols: {symbols}")
                metrics.record_trade()  # Record successful trading start
            else:
                logger.log_error("Failed to start trading engine")
                metrics.record_error("trading_engine_start_failed")
        else:
            logger.log_error("Could not initialize trading engine - missing dependencies")
            metrics.record_error("trading_engine_missing_dependencies")

    except Exception as e:
        logger.log_error(f"Error initializing trading components: {e}", error_type="InitializationError")
        metrics.record_error("trading_components_initialization_failed")
        raise

if __name__ == "__main__":
    if initialize_system():
        logger.log_info("System started successfully")
    else:
        logger.log_error("System startup failed")
