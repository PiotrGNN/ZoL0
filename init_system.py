#!/usr/bin/env python3
"""System initialization and setup script."""

import os
import sys
import logging
from pathlib import Path
import yaml
from dotenv import load_dotenv
from typing import Dict, Any

from python_libs.db_manager import DatabaseManager
from python_libs.bybit_connector import BybitConnector
from python_libs.risk_manager import RiskManager
from python_libs.trade_executor import TradeExecutor
from python_libs.strategy_manager import StrategyManager
from python_libs.market_data_manager import MarketDataManager
from python_libs.portfolio_manager import PortfolioManager
from python_libs.monitoring import MonitoringSystem
from python_libs.notification_system import NotificationSystem
from python_libs.system_state import SystemState
from python_libs.performance_tracker import PerformanceTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/init.log'),
        logging.StreamHandler()
    ]
)

def create_directories() -> None:
    """Create required directory structure."""
    directories = [
        'logs',
        'data/cache',
        'saved_models',
        'config',
        'static/img',
        'python_libs/__pycache__',
        'ai_models/__pycache__'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def load_configuration() -> Dict[str, Any]:
    """Load system configuration.
    
    Returns:
        Configuration dictionary
    """
    # Load environment variables
    load_dotenv()
    
    # Load main configuration
    config_path = 'config/config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)
        
    return config

def initialize_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize all system components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing component instances
    """
    try:
        # Initialize database
        db_manager = DatabaseManager(config.get('database', {}).get('path', 'data/trading.db'))
        
        # Initialize exchange connector
        exchange = BybitConnector(
            api_key=os.getenv('BYBIT_API_KEY'),
            api_secret=os.getenv('BYBIT_API_SECRET'),
            testnet=os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
        )
        
        # Initialize core components
        risk_manager = RiskManager(config)
        strategy_manager = StrategyManager(config)
        market_data = MarketDataManager(exchange, db_manager)
        portfolio_manager = PortfolioManager(db_manager, config)
        trade_executor = TradeExecutor(exchange, risk_manager)
        
        # Initialize monitoring and notification
        monitoring = MonitoringSystem(db_manager, config)
        notification = NotificationSystem(config)
        
        # Initialize performance tracking
        performance_tracker = PerformanceTracker(db_manager, config)
        
        # Initialize system state manager
        system_state = SystemState(config, db_manager, exchange)
        
        components = {
            'db_manager': db_manager,
            'exchange': exchange,
            'risk_manager': risk_manager,
            'strategy_manager': strategy_manager,
            'market_data': market_data,
            'portfolio_manager': portfolio_manager,
            'trade_executor': trade_executor,
            'monitoring': monitoring,
            'notification': notification,
            'performance_tracker': performance_tracker,
            'system_state': system_state
        }
        
        logging.info("All components initialized successfully")
        return components
        
    except Exception as e:
        logging.error(f"Failed to initialize components: {e}")
        sys.exit(1)

async def verify_system(components: Dict[str, Any]) -> bool:
    """Verify system functionality.
    
    Args:
        components: Dictionary of system components
        
    Returns:
        Boolean indicating success
    """
    try:
        # Verify database connection
        components['db_manager'].init_db()
        
        # Verify exchange connection
        wallet = await components['exchange'].get_wallet_balance()
        if not wallet.get('success'):
            raise Exception("Failed to connect to exchange")
            
        # Initialize system state
        success = await components['system_state'].start_system()
        if not success:
            raise Exception("Failed to start system state")
            
        # Send test notification
        await components['notification'].send_notification(
            "System initialization complete",
            level="INFO",
            data={"status": "ready"}
        )
        
        logging.info("System verification completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"System verification failed: {e}")
        return False

async def initialize_system() -> bool:
    """Initialize complete trading system.
    
    Returns:
        Boolean indicating success
    """
    try:
        # Create directory structure
        create_directories()
        
        # Load configuration
        config = load_configuration()
        
        # Initialize components
        components = initialize_components(config)
        
        # Verify system
        if not await verify_system(components):
            return False
            
        logging.info("Trading system initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"System initialization failed: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(initialize_system())