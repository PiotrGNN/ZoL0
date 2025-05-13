"""Strategy management system."""

import importlib
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from config import get_logger
from ..market_data import MarketDataManager
from .base import Strategy
from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy

logger = get_logger()
logger.setLevel(logging.DEBUG)


class StrategyManager:
    """Manages trading strategies and signal generation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy manager.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.strategies: Dict[str, Strategy] = {}
        self._initialize_strategies()

    def _initialize_strategies(self) -> None:
        """Initialize configured strategies."""
        strategy_configs = self.config
        builtin_strategies = {
            "trend_following": TrendFollowingStrategy,
            "mean_reversion": MeanReversionStrategy,
            "breakout": BreakoutStrategy,
        }
        for name, cfg in strategy_configs.items():
            if not cfg.get("enabled", False):
                continue
            try:
                logger.debug(f"[StrategyManager] Initializing strategy: {name} with config: {cfg}")
                if name == "trend_following":
                    parameters = {k: v for k, v in cfg.items() if k not in ("enabled", "timeframes", "indicators")}
                    logger.debug(f"[StrategyManager] trend_following parameters: {parameters}")
                    strategy = TrendFollowingStrategy(
                        timeframes=cfg.get("timeframes", ["1h"]),
                        indicators=cfg.get("indicators", []),
                        parameters=parameters
                    )
                elif name == "breakout":
                    logger.debug(f"[StrategyManager] breakout args: symbol=BTCUSDT, lookback={cfg.get('lookback', 20)}, breakout_threshold={cfg.get('breakout_threshold', 1.0)}")
                    strategy = BreakoutStrategy(
                        symbol="BTCUSDT",
                        lookback=cfg.get("lookback", 20),
                        breakout_threshold=cfg.get("breakout_threshold", 1.0)
                    )
                elif name == "mean_reversion":
                    logger.debug(f"[StrategyManager] mean_reversion args: symbol=BTCUSDT, lookback={cfg.get('lookback', 20)}, entry_threshold={cfg.get('entry_threshold', 2.0)}, exit_threshold={cfg.get('exit_threshold', 0.5)}")
                    strategy = MeanReversionStrategy(
                        symbol="BTCUSDT",
                        lookback=cfg.get("lookback", 20),
                        entry_threshold=cfg.get("entry_threshold", 2.0),
                        exit_threshold=cfg.get("exit_threshold", 0.5)
                    )
                else:
                    module = importlib.import_module(f".{name}", "core.strategies")
                    strategy_class = getattr(module, f"{name.title()}Strategy")
                    filtered_cfg = {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool))}
                    logger.debug(f"[StrategyManager] custom strategy {name} args: {filtered_cfg}")
                    strategy = strategy_class(**filtered_cfg)
                self.strategies[name] = strategy
                logger.info(f"Initialized strategy: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize strategy {name}: {e}")

    def analyze_pair(
        self, market_data: MarketDataManager, symbol: str
    ) -> Dict[str, Any]:
        """Run analysis using all active strategies.

        Args:
            market_data: Market data manager
            symbol: Trading pair symbol

        Returns:
            Analysis results from all strategies
        """
        results = {}

        for name, strategy in self.strategies.items():
            try:
                # Get analysis for each timeframe
                timeframe_results = {}
                for timeframe in strategy.timeframes:
                    # Get required data
                    klines = market_data.get_klines(
                        symbol=symbol, timeframe=timeframe, limit=strategy.min_periods
                    )

                    if len(klines) < strategy.min_periods:
                        continue

                    # Calculate indicators
                    indicators = market_data.calculate_indicators(
                        symbol=symbol,
                        timeframe=timeframe,
                        indicator_names=strategy.indicators,
                    )

                    # Run strategy analysis
                    analysis = strategy.analyze(
                        symbol=symbol,
                        klines=klines,
                        indicators=indicators,
                        timeframe=timeframe,
                    )

                    timeframe_results[timeframe] = analysis

                results[name] = timeframe_results

            except Exception as e:
                logger.error(f"Error in strategy {name} for {symbol}: {e}")

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get strategy manager status.

        Returns:
            Status information
        """
        return {
            "active_strategies": list(self.strategies.keys()),
            "strategies": {
                name: {
                    "timeframes": strategy.timeframes,
                    "indicators": strategy.indicators,
                    "min_periods": strategy.min_periods,
                }
                for name, strategy in self.strategies.items()
            },
        }

    def validate_strategy(self, name: str) -> List[str]:
        """Validate a strategy configuration.

        Args:
            name: Strategy name

        Returns:
            List of validation errors
        """
        errors = []
        strategy = self.strategies.get(name)

        if not strategy:
            return [f"Strategy {name} not found"]

        # Validate required methods
        required_methods = ["analyze", "validate"]
        for method in required_methods:
            if not hasattr(strategy, method):
                errors.append(f"Strategy {name} missing required method: {method}")

        # Validate configuration
        if not strategy.timeframes:
            errors.append(f"Strategy {name} has no timeframes configured")

        if not strategy.indicators:
            errors.append(f"Strategy {name} has no indicators configured")

        # Run strategy's own validation
        if hasattr(strategy, "validate"):
            strategy_errors = strategy.validate()
            if strategy_errors:
                errors.extend(strategy_errors)

        return errors

    def backtest_strategy(
        self,
        name: str,
        market_data: MarketDataManager,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Run strategy backtest.

        Args:
            name: Strategy name
            market_data: Market data manager
            symbol: Trading pair symbol
            start_time: Backtest start time
            end_time: Backtest end time (optional)

        Returns:
            Backtest results
        """
        strategy = self.strategies.get(name)
        if not strategy:
            raise ValueError(f"Strategy {name} not found")

        results = {"trades": [], "metrics": {}, "equity_curve": pd.DataFrame()}

        try:
            # Get historical data
            for timeframe in strategy.timeframes:
                klines = market_data.get_historical_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                )

                # Calculate indicators
                indicators = market_data.calculate_indicators(
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator_names=strategy.indicators,
                    klines=klines,
                )

                # Run strategy backtest
                if hasattr(strategy, "backtest"):
                    timeframe_results = strategy.backtest(
                        symbol=symbol,
                        klines=klines,
                        indicators=indicators,
                        timeframe=timeframe,
                    )

                    # Merge results
                    results["trades"].extend(timeframe_results.get("trades", []))
                    results["metrics"].update(timeframe_results.get("metrics", {}))

                    if "equity_curve" in timeframe_results:
                        if results["equity_curve"].empty:
                            results["equity_curve"] = timeframe_results["equity_curve"]
                        else:
                            results["equity_curve"] = pd.concat(
                                [
                                    results["equity_curve"],
                                    timeframe_results["equity_curve"],
                                ]
                            ).sort_index()

        except Exception as e:
            logger.error(f"Backtest failed for strategy {name}: {e}")
            results["error"] = str(e)

        return results
