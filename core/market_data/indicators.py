"""Technical indicator calculations."""

from typing import Dict, Any, Union, Tuple
import pandas as pd
import numpy as np
from config import get_logger

logger = get_logger()


def calculate_indicator(
    name: str, klines: pd.DataFrame, params: Dict[str, Any]
) -> Union[pd.Series, Tuple[pd.Series, ...]]:
    """Calculate technical indicator.

    Args:
        name: Indicator name
        klines: OHLCV candlestick data
        params: Calculation parameters

    Returns:
        Calculated indicator(s)
    """
    indicators = {
        "sma": simple_moving_average,
        "ema": exponential_moving_average,
        "rsi": relative_strength_index,
        "macd": moving_average_convergence_divergence,
        "bollinger_bands": bollinger_bands,
        "stochastic": stochastic_oscillator,
        "adx": average_directional_index,
        "atr": average_true_range,
    }

    if name not in indicators:
        raise ValueError(f"Unknown indicator: {name}")

    return indicators[name](klines, params)


def simple_moving_average(klines: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Calculate Simple Moving Average."""
    period = params.get("period", 20)
    return klines["close"].rolling(window=period).mean()


def exponential_moving_average(
    klines: pd.DataFrame, params: Dict[str, Any]
) -> pd.Series:
    """Calculate Exponential Moving Average."""
    period = params.get("period", 20)
    return klines["close"].ewm(span=period, adjust=False).mean()


def relative_strength_index(klines: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Calculate Relative Strength Index."""
    period = params.get("period", 14)

    # Calculate price changes
    delta = klines["close"].diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Calculate RS and RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def moving_average_convergence_divergence(
    klines: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal, and Histogram."""
    fast_period = params.get("fast_period", 12)
    slow_period = params.get("slow_period", 26)
    signal_period = params.get("signal_period", 9)

    # Calculate EMAs
    fast_ema = exponential_moving_average(klines, {"period": fast_period})
    slow_ema = exponential_moving_average(klines, {"period": slow_period})

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Calculate histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(
    klines: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    period = params.get("period", 20)
    std_dev = params.get("std_dev", 2)

    # Calculate middle band (SMA)
    middle_band = simple_moving_average(klines, {"period": period})

    # Calculate standard deviation
    rolling_std = klines["close"].rolling(window=period).std()

    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)

    return upper_band, middle_band, lower_band


def stochastic_oscillator(
    klines: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator."""
    k_period = params.get("k_period", 14)
    d_period = params.get("d_period", 3)

    # Calculate %K
    low_min = klines["low"].rolling(window=k_period).min()
    high_max = klines["high"].rolling(window=k_period).max()

    k = 100 * ((klines["close"] - low_min) / (high_max - low_min))

    # Calculate %D
    d = k.rolling(window=d_period).mean()

    return k, d


def average_directional_index(
    klines: pd.DataFrame, params: Dict[str, Any]
) -> pd.Series:
    """Calculate Average Directional Index."""
    period = params.get("period", 14)

    # Calculate True Range
    high_low = klines["high"] - klines["low"]
    high_close = abs(klines["high"] - klines["close"].shift())
    low_close = abs(klines["low"] - klines["close"].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate Directional Movement
    pos_dm = klines["high"].diff()
    neg_dm = -klines["low"].diff()

    pos_dm = pos_dm.where((pos_dm > neg_dm) & (pos_dm > 0), 0.0)
    neg_dm = neg_dm.where((neg_dm > pos_dm) & (neg_dm > 0), 0.0)

    # Calculate smoothed averages
    tr_avg = tr.ewm(alpha=1 / period, adjust=False).mean()
    pos_dm_avg = pos_dm.ewm(alpha=1 / period, adjust=False).mean()
    neg_dm_avg = neg_dm.ewm(alpha=1 / period, adjust=False).mean()

    # Calculate Directional Indicators
    pdi = 100 * (pos_dm_avg / tr_avg)
    ndi = 100 * (neg_dm_avg / tr_avg)

    # Calculate Directional Index
    dx = 100 * abs(pdi - ndi) / (pdi + ndi)

    # Calculate ADX
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx


def average_true_range(klines: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Calculate Average True Range."""
    period = params.get("period", 14)

    # Calculate True Range
    high_low = klines["high"] - klines["low"]
    high_close = abs(klines["high"] - klines["close"].shift())
    low_close = abs(klines["low"] - klines["close"].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate ATR
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    return atr
