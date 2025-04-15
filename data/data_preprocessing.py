
"""
data_preprocessing.py
---------------------
Moduł zawierający funkcje do przygotowania danych dla modeli AI/ML.
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, Dict, List, Any

def prepare_data_for_model(data: Union[Dict, List, np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Konwertuje różne formaty danych do formatu odpowiedniego dla modeli ML/AI.

    Args:
        data: Dane wejściowe w różnych formatach (słownik, lista, DataFrame, array)

    Returns:
        np.ndarray: Dane w formacie numpy.ndarray gotowe do użycia w modelach
    """
    logger = logging.getLogger(__name__)

    try:
        # Jeśli dane są None, zwróć pustą tablicę
        if data is None:
            logger.warning("Otrzymano None jako dane wejściowe")
            return np.array([])

        # Jeśli dane są już typu ndarray, zwróć je bezpośrednio
        if isinstance(data, np.ndarray):
            return data

        # Jeśli dane są w formacie DataFrame, konwertuj na ndarray
        if isinstance(data, pd.DataFrame):
            return data.values

        # Jeśli dane są listą, konwertuj na ndarray
        if isinstance(data, list):
            return np.array(data)

        # Jeśli dane są w formacie dict z kluczami OHLCV
        if isinstance(data, dict):
            if all(k in data for k in ['open', 'high', 'low', 'close', 'volume']):
                # Utwórz DataFrame z danych OHLCV
                df = pd.DataFrame({
                    'open': data['open'],
                    'high': data['high'],
                    'low': data['low'],
                    'close': data['close'],
                    'volume': data['volume']
                })
                return df.values
            elif 'close' in data:
                # Jeśli mamy przynajmniej dane zamknięcia
                return np.array(data['close']).reshape(-1, 1)
            elif len(data) > 0:
                # Próba użycia pierwszej dostępnej serii danych
                first_key = list(data.keys())[0]
                if isinstance(data[first_key], (list, np.ndarray)):
                    logger.info(f"Używam danych z klucza '{first_key}' jako wejścia")
                    return np.array(data[first_key]).reshape(-1, 1)

        # Jeśli format nie jest rozpoznany, zgłoś błąd
        logger.error(f"Nierozpoznany format danych: {type(data)}")
        return np.array([])  # Zwróć pustą tablicę zamiast zgłaszania błędu

    except Exception as e:
        logger.error(f"Błąd podczas przygotowywania danych: {e}")
        return np.array([])  # Zwróć pustą tablicę w przypadku błędu
