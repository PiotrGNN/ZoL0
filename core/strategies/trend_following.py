"""Trend following strategy implementation."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Strategy
from config import get_logger

logger = get_logger()


class TrendFollowingStrategy(Strategy):
    """Multi-timeframe trend following strategy.

    Uses multiple indicators to identify trends:
    - EMA crossovers (fast/slow)
    - RSI for trend confirmation
    - ADX for trend strength
    - ATR for volatility-based position sizing
    """

    def __init__(
        self, timeframes: List[str], indicators: List[str], parameters: Dict[str, Any]
    ):
        """Initialize strategy with default parameters if not provided."""
        default_params = {
            "min_periods": 100,
            "ema_fast": 12,
            "ema_slow": 26,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "adx_period": 14,
            "adx_threshold": 25,
            "atr_period": 14,
            "risk_per_trade": 0.02,  # 2% risk per trade
            "profit_target_atr": 3.0,  # Take profit at 3x ATR
            "stop_loss_atr": 2.0,  # Stop loss at 2x ATR
        }

        # Merge provided parameters with defaults
        parameters = {**default_params, **parameters}

        # Ensure required indicators
        required_indicators = ["ema_fast", "ema_slow", "rsi", "adx", "atr"]
        indicators = list(set(indicators + required_indicators))

        super().__init__(timeframes, indicators, parameters)

    def analyze(
        self,
        symbol: str,
        klines: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        timeframe: str,
    ) -> Dict[str, Any]:
        """Analyze market data and generate trading signals.

        Args:
            symbol: Trading pair symbol
            klines: OHLCV candlestick data
            indicators: Technical indicators
            timeframe: Current timeframe

        Returns:
            Analysis results including signals
        """
        results = {"signals": [], "metrics": {}, "analysis": {}}

        try:
            # Get current indicator values
            current = {name: series.iloc[-1] for name, series in indicators.items()}

            # Get previous indicator values
            previous = {name: series.iloc[-2] for name, series in indicators.items()}

            # Calculate trend metrics
            trend_strength = self._calculate_trend_strength(current, previous)
            risk_metrics = self._calculate_risk_metrics(current, klines.iloc[-1])

            # Generate signals based on trend analysis
            signals = self._generate_signals(
                trend_strength=trend_strength,
                risk_metrics=risk_metrics,
                current=current,
                previous=previous,
            )

            results["signals"] = signals
            results["metrics"] = {
                "trend_strength": trend_strength,
                "risk_score": risk_metrics["risk_score"],
                "volatility": risk_metrics["volatility"],
            }
            results["analysis"] = {
                "trend": {
                    "direction": trend_strength["direction"],
                    "strength": trend_strength["strength"],
                    "momentum": trend_strength["momentum"],
                },
                "risk": risk_metrics,
            }

        except Exception as e:
            logger.error(f"Error analyzing {symbol} on {timeframe}: {e}")

        return results

    def validate(self) -> List[str]:
        """Validate strategy configuration."""
        errors = super().validate()

        # Validate indicator periods
        if self.parameters["ema_fast"] >= self.parameters["ema_slow"]:
            errors.append(f"{self.name}: Fast EMA period must be less than slow EMA")

        # Validate risk parameters
        if not 0 < self.parameters["risk_per_trade"] <= 0.05:
            errors.append(f"{self.name}: risk_per_trade must be between 0 and 0.05")

        return errors

    def _calculate_trend_strength(
        self, current: Dict[str, float], previous: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate trend strength metrics."""
        # Determine trend direction from EMA crossover
        ema_trend = (
            1
            if current["ema_fast"] > current["ema_slow"]
            else -1 if current["ema_fast"] < current["ema_slow"] else 0
        )

        # Calculate momentum from RSI
        rsi = current["rsi"]
        rsi_momentum = (
            1
            if rsi > 50 and rsi < self.parameters["rsi_overbought"]
            else -1 if rsi < 50 and rsi > self.parameters["rsi_oversold"] else 0
        )

        # Get trend strength from ADX
        adx = current["adx"]
        trend_strength = "strong" if adx > self.parameters["adx_threshold"] else "weak"

        return {
            "direction": ema_trend,
            "momentum": rsi_momentum,
            "strength": trend_strength,
            "score": (
                abs(ema_trend)
                + abs(rsi_momentum)
                + (1 if trend_strength == "strong" else 0)
            ),
        }

    def _calculate_risk_metrics(
        self, current: Dict[str, float], candle: pd.Series
    ) -> Dict[str, Any]:
        """Calculate risk and volatility metrics."""
        # Get ATR for volatility measurement
        atr = current["atr"]

        # Calculate stop and target prices
        stop_distance = atr * self.parameters["stop_loss_atr"]
        target_distance = atr * self.parameters["profit_target_atr"]

        return {
            "volatility": atr / candle["close"],  # Normalized ATR
            "risk_score": stop_distance / target_distance,
            "stop_distance": stop_distance,
            "target_distance": target_distance,
        }

    def _generate_signals(
        self,
        trend_strength: Dict[str, Any],
        risk_metrics: Dict[str, Any],
        current: Dict[str, float],
        previous: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on analysis."""
        signals = []

        # Check for entry conditions
        if (
            # Strong trend in either direction
            trend_strength["score"] >= 2
            and
            # Acceptable volatility
            risk_metrics["volatility"] < 0.05
            and
            # Good risk/reward
            risk_metrics["risk_score"] < 0.7
        ):
            # Long entry
            if (
                trend_strength["direction"] > 0
                and trend_strength["momentum"] > 0
                and previous["ema_fast"] <= previous["ema_slow"]
                and current["ema_fast"] > current["ema_slow"]
            ):
                signals.append(
                    {
                        "type": "entry",
                        "side": "buy",
                        "strength": trend_strength["score"],
                        "stop_loss": risk_metrics["stop_distance"],
                        "take_profit": risk_metrics["target_distance"],
                    }
                )

            # Short entry
            elif (
                trend_strength["direction"] < 0
                and trend_strength["momentum"] < 0
                and previous["ema_fast"] >= previous["ema_slow"]
                and current["ema_fast"] < current["ema_slow"]
            ):
                signals.append(
                    {
                        "type": "entry",
                        "side": "sell",
                        "strength": trend_strength["score"],
                        "stop_loss": risk_metrics["stop_distance"],
                        "take_profit": risk_metrics["target_distance"],
                    }
                )

        # Check for exit conditions
        elif (
            # Trend reversal
            (
                trend_strength["direction"] > 0
                and current["rsi"] > self.parameters["rsi_overbought"]
            )
            or (
                trend_strength["direction"] < 0
                and current["rsi"] < self.parameters["rsi_oversold"]
            )
            or
            # Trend weakening
            (trend_strength["strength"] == "weak" and trend_strength["score"] < 2)
        ):
            signals.append({"type": "exit", "reason": "trend_reversal"})

        return signals
