"""Base class for trading strategies."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from config import get_logger

logger = get_logger()


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(
        self, timeframes: List[str], indicators: List[str], parameters: Dict[str, Any]
    ):
        """Initialize strategy.

        Args:
            timeframes: List of timeframes to analyze
            indicators: Required technical indicators
            parameters: Strategy parameters
        """
        self.timeframes = timeframes
        self.indicators = indicators
        self.parameters = parameters
        self.min_periods = parameters.get("min_periods", 100)
        self.name = self.__class__.__name__.replace("Strategy", "")

    @abstractmethod
    def analyze(
        self,
        symbol: str,
        klines: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        timeframe: str,
    ) -> Dict[str, Any]:
        """
        Analyze market data and indicators to generate trading signals.
        Returns a dictionary with signal details (e.g., 'buy', 'sell', 'hold').
        """
        # Example: Simple moving average crossover
        close = klines["close"] if "close" in klines else None
        if close is not None and "sma_fast" in indicators and "sma_slow" in indicators:
            if indicators["sma_fast"].iloc[-1] > indicators["sma_slow"].iloc[-1]:
                return {"signal": "buy", "confidence": 0.8}
            elif indicators["sma_fast"].iloc[-1] < indicators["sma_slow"].iloc[-1]:
                return {"signal": "sell", "confidence": 0.8}
            else:
                return {"signal": "hold", "confidence": 0.5}
        return {"signal": "hold", "confidence": 0.5}

    def validate(self) -> List[str]:
        """Validate strategy configuration.

        Returns:
            List of validation errors
        """
        errors = []

        # Validate timeframes
        if not self.timeframes:
            errors.append(f"{self.name}: No timeframes configured")

        # Validate indicators
        if not self.indicators:
            errors.append(f"{self.name}: No indicators configured")

        # Validate min_periods
        if self.min_periods < 20:
            errors.append(f"{self.name}: min_periods must be at least 20")

        return errors

    def backtest(
        self,
        symbol: str,
        klines: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        timeframe: str,
    ) -> Dict[str, Any]:
        """Run strategy backtest.

        Args:
            symbol: Trading pair symbol
            klines: OHLCV candlestick data
            indicators: Technical indicators
            timeframe: Current timeframe

        Returns:
            Backtest results
        """
        results = {"trades": [], "metrics": {}, "equity_curve": pd.DataFrame()}

        try:
            initial_balance = 10000  # Default starting balance
            current_balance = initial_balance
            position = None
            equity_curve = []

            # Analyze each candle
            for i in range(self.min_periods, len(klines)):
                # Get current window of data
                window_klines = klines.iloc[: i + 1]
                window_indicators = {
                    name: series.iloc[: i + 1] for name, series in indicators.items()
                }

                # Get analysis for this point
                analysis = self.analyze(
                    symbol=symbol,
                    klines=window_klines,
                    indicators=window_indicators,
                    timeframe=timeframe,
                )

                # Process signals
                for signal in analysis.get("signals", []):
                    # Handle entry signals
                    if signal["type"] == "entry" and position is None:
                        entry_price = klines.iloc[i]["close"]
                        size = (
                            current_balance * 0.02
                        ) / entry_price  # 2% risk per trade
                        position = {
                            "side": signal["side"],
                            "entry_price": entry_price,
                            "size": size,
                            "entry_time": klines.index[i],
                        }

                    # Handle exit signals
                    elif signal["type"] == "exit" and position is not None:
                        exit_price = klines.iloc[i]["close"]
                        pnl = (
                            (exit_price - position["entry_price"]) * position["size"]
                            if position["side"] == "buy"
                            else (position["entry_price"] - exit_price)
                            * position["size"]
                        )

                        # Record trade
                        results["trades"].append(
                            {
                                "entry_time": position["entry_time"],
                                "exit_time": klines.index[i],
                                "symbol": symbol,
                                "side": position["side"],
                                "entry_price": position["entry_price"],
                                "exit_price": exit_price,
                                "size": position["size"],
                                "pnl": pnl,
                                "balance": current_balance + pnl,
                            }
                        )

                        current_balance += pnl
                        position = None

                # Record equity point
                equity_curve.append(
                    {
                        "timestamp": klines.index[i],
                        "balance": current_balance,
                        "unrealized_pnl": (
                            (
                                (klines.iloc[i]["close"] - position["entry_price"])
                                * position["size"]
                            )
                            if position and position["side"] == "buy"
                            else (
                                (
                                    (position["entry_price"] - klines.iloc[i]["close"])
                                    * position["size"]
                                )
                                if position
                                else 0
                            )
                        ),
                    }
                )

            # Convert equity curve to DataFrame
            results["equity_curve"] = pd.DataFrame(equity_curve).set_index("timestamp")

            # Calculate metrics
            if results["trades"]:
                trades_df = pd.DataFrame(results["trades"])
                results["metrics"] = {
                    "total_trades": len(trades_df),
                    "winning_trades": len(trades_df[trades_df["pnl"] > 0]),
                    "total_pnl": trades_df["pnl"].sum(),
                    "win_rate": len(trades_df[trades_df["pnl"] > 0]) / len(trades_df),
                    "average_trade": trades_df["pnl"].mean(),
                    "max_drawdown": self._calculate_max_drawdown(
                        results["equity_curve"]["balance"]
                    ),
                    "sharpe_ratio": self._calculate_sharpe_ratio(
                        results["equity_curve"]["balance"]
                    ),
                }

        except Exception as e:
            logger.error(f"Backtest failed for {self.name}: {e}")
            results["error"] = str(e)

        return results

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve.

        Args:
            equity: Equity curve series

        Returns:
            Maximum drawdown as a percentage
        """
        peak = equity.expanding(min_periods=1).max()
        drawdown = (equity - peak) / peak
        return abs(drawdown.min())

    def _calculate_sharpe_ratio(self, equity: pd.Series) -> float:
        """Calculate Sharpe ratio from equity curve.

        Args:
            equity: Equity curve series

        Returns:
            Sharpe ratio
        """
        returns = equity.pct_change().dropna()
        if len(returns) < 2:
            return 0
        return returns.mean() / returns.std() * (252**0.5)  # Annualized
