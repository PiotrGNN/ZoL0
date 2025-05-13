"""Core trading system components initialization."""

import os
from typing import Dict, Any
from ..database import DatabaseManager
from ..exchange import ExchangeConnector
from ..market_data import MarketDataManager
from ..portfolio import PortfolioManager
from ..risk import RiskManager
from ..strategies import StrategyManager
from ..trading import TradingEngine
from config import get_logger

logger = get_logger()


def initialize_market_data(
    exchange: ExchangeConnector, config: Dict[str, Any]
) -> MarketDataManager:
    """Initialize market data manager."""
    market_data_config = config.get("market_data", {})
    market_data = MarketDataManager(
        exchange=exchange,
        config=market_data_config
    )
    logger.info("Market data manager initialized")
    return market_data


def initialize_risk_manager(config: Dict[str, Any]) -> RiskManager:
    """Initialize risk management system."""
    risk_config = config.get("trading", {})

    # Przekazujemy cały słownik konfiguracyjny, nie rozpakowujemy argumentów
    risk_manager = RiskManager(risk_config)
    logger.info("Risk manager initialized")
    return risk_manager


def initialize_strategy_manager(config: Dict[str, Any]) -> StrategyManager:
    """Initialize trading strategy manager."""
    strategies_config = config.get("strategies", {})
    strategy_manager = StrategyManager(strategies_config)
    logger.info(
        f"Strategy manager initialized with {len(strategy_manager.strategies)} strategies"
    )
    return strategy_manager


def initialize_portfolio_manager(
    db_manager: DatabaseManager, config: Dict[str, Any]
) -> PortfolioManager:
    """Initialize portfolio management system."""
    portfolio_manager = PortfolioManager(
        initial_balance=config.get("trading.initial_balance", 10000.0),
        base_currency=config.get("trading.base_currency", "USDT"),
        db_manager=db_manager,
    )
    logger.info("Portfolio manager initialized")
    return portfolio_manager


def initialize_trading_engine(
    exchange: ExchangeConnector,
    strategy_manager: StrategyManager,
    risk_manager: RiskManager,
    portfolio_manager: PortfolioManager,
    market_data: MarketDataManager,  # zachowujemy dla kompatybilności, ale nie przekazujemy dalej
    config: Dict[str, Any],
) -> TradingEngine:
    """Initialize main trading engine."""
    trading_engine = TradingEngine(
        exchange=exchange,
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        strategy_manager=strategy_manager,
        config=config,
    )
    logger.info("Trading engine initialized")
    return trading_engine
