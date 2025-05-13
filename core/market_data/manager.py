"""Market data management system."""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config import get_logger
from ..exchange import ExchangeConnector
from .indicators import calculate_indicator
from .cache import DataCache

logger = get_logger()


class MarketDataManager:
    """Manages market data retrieval, caching, and analysis."""

    def __init__(self, exchange: ExchangeConnector, config: Dict[str, Any]):
        """Initialize market data manager.

        Args:
            exchange: Exchange connector
            config: Configuration dictionary
        """
        logger.debug(f"[MarketDataManager] Init with config: {config}")
        self.exchange = exchange
        self.config = config
        self.cache = DataCache(config.get("cache.path", "data/cache"))

        # Default timeframes if not configured
        self.timeframes = config.get(
            "market_data.timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"]
        )

        # Data fetch intervals (in seconds)
        self.update_intervals = {
            "1m": 30,  # Update every 30 seconds
            "5m": 60,  # Update every minute
            "15m": 180,  # Update every 3 minutes
            "1h": 600,  # Update every 10 minutes
            "4h": 1800,  # Update every 30 minutes
            "1d": 3600,  # Update every hour
        }

        # Track last update time for each symbol/timeframe
        self._last_update: Dict[str, Dict[str, float]] = {}

    def get_klines(
        self, symbol: str, timeframe: str, limit: int = 100, use_cache: bool = True
    ) -> pd.DataFrame:
        """Get candlestick data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            limit: Number of candlesticks
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Initialize tracking dict for symbol
            if symbol not in self._last_update:
                self._last_update[symbol] = {}

            now = time.time()
            last_update = self._last_update[symbol].get(timeframe, 0)
            update_interval = self.update_intervals.get(timeframe, 300)

            # Check if update needed
            needs_update = now - last_update > update_interval or not use_cache

            if needs_update:
                # Fetch new data from exchange
                klines = self.exchange.get_klines(
                    symbol=symbol, timeframe=timeframe, limit=limit
                )

                # Update cache
                if use_cache:
                    self.cache.save_klines(
                        symbol=symbol, timeframe=timeframe, data=klines
                    )

                self._last_update[symbol][timeframe] = now

            else:
                # Load from cache
                klines = self.cache.load_klines(
                    symbol=symbol, timeframe=timeframe, limit=limit
                )

            return klines

        except Exception as e:
            logger.error(f"Error getting klines for {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def get_historical_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Get historical candlestick data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            start_time: Start time
            end_time: End time (optional)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Try loading from cache first
            klines = self.cache.load_historical_klines(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
            )

            if klines is None:
                # Fetch from exchange if not in cache
                klines = self.exchange.get_historical_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                )

                # Save to cache
                self.cache.save_historical_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    data=klines,
                    start_time=start_time,
                    end_time=end_time,
                )

            return klines

        except Exception as e:
            logger.error(
                f"Error getting historical klines for {symbol} {timeframe}: {e}"
            )
            return pd.DataFrame()

    def get_orderbook(
        self, symbol: str, limit: int = 20
    ) -> Dict[str, List[List[float]]]:
        """Get current orderbook.

        Args:
            symbol: Trading pair symbol
            limit: Orderbook depth

        Returns:
            Dictionary with bids and asks
        """
        try:
            return self.exchange.get_orderbook(symbol, limit)
        except Exception as e:
            logger.error(f"Error getting orderbook for {symbol}: {e}")
            return {"bids": [], "asks": []}

    def calculate_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicator_names: Optional[List[str]] = None,
        klines: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.Series]:
        """Calculate technical indicators.

        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe
            indicator_names: List of indicators to calculate
            klines: Optional pre-loaded klines data

        Returns:
            Dictionary of indicator Series
        """
        try:
            # Get klines if not provided
            if klines is None:
                klines = self.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=500,  # Enough data for most indicators
                )

            if klines.empty:
                return {}

            # Calculate all configured indicators if none specified
            if not indicator_names:
                indicator_names = self.config.get(
                    "indicators",
                    [
                        "sma",
                        "ema",
                        "rsi",
                        "macd",
                        "bollinger_bands",
                        "stochastic",
                        "adx",
                        "atr",
                    ],
                )

            results = {}

            # Calculate each indicator
            for name in indicator_names:
                try:
                    result = calculate_indicator(
                        name=name,
                        klines=klines,
                        params=self.config.get(f"indicators.{name}", {}),
                    )

                    if isinstance(result, tuple):
                        # Handle multiple return values (e.g. MACD)
                        for i, value in enumerate(result):
                            results[f"{name}_{i+1}"] = value
                    else:
                        results[name] = result

                except Exception as e:
                    logger.error(f"Error calculating {name}: {e}")

            return results

        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol} {timeframe}: {e}")
            return {}

    def get_status(self) -> Dict[str, Any]:
        """Get manager status.

        Returns:
            Status information
        """
        return {
            "cache_size": self.cache.get_size(),
            "timeframes": self.timeframes,
            "update_intervals": self.update_intervals,
            "last_updates": {
                symbol: {
                    tf: datetime.fromtimestamp(ts).isoformat()
                    for tf, ts in timeframes.items()
                }
                for symbol, timeframes in self._last_update.items()
            },
        }
