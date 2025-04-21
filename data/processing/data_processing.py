"""
data_processing.py
-----------------
Moduł do przetwarzania danych finansowych.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Union

def winsorize_series(series: pd.Series, limits: Tuple[float, float] = (0.05, 0.05)) -> pd.Series:
    """
    Winsoryzuje serię danych, zastępując wartości ekstremalne wartościami z zadanych kwantyli.

    Parameters:
        series (pd.Series): Seria danych do winsoryzacji
        limits (Tuple[float, float]): Dolny i górny limit (jako ułamki), domyślnie (0.05, 0.05) dla 5%

    Returns:
        pd.Series: Winsoryzowana seria danych
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input musi być pandas Series")
    
    if not all(0 <= x <= 1 for x in limits):
        raise ValueError("Limity muszą być w przedziale [0, 1]")

    lower_percentile = series.quantile(limits[0])
    upper_percentile = series.quantile(1 - limits[1])

    winsorized = series.copy()
    winsorized[series < lower_percentile] = lower_percentile
    winsorized[series > upper_percentile] = upper_percentile

    logging.info(
        f"Winsoryzacja: zakres [{lower_percentile:.2f}, {upper_percentile:.2f}], "
        f"zastąpiono {(series < lower_percentile).sum()} dolnych i "
        f"{(series > upper_percentile).sum()} górnych wartości"
    )

    return winsorized