
#!/usr/bin/env python3
"""
test_data_conversion.py - Skrypt testujący funkcję konwersji danych dla modeli.

Ten skrypt testuje funkcję prepare_data_for_model z modułu ai_models.model_training,
aby upewnić się, że prawidłowo konwertuje dane w różnych formatach do formatu 
odpowiedniego dla modeli.
"""

import os
import sys
import numpy as np
import pandas as pd
import logging

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("test_data_conversion")

# Upewnij się, że mamy dostęp do modułów projektu
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from ai_models.model_training import prepare_data_for_model
    logger.info("Zaimportowano funkcję prepare_data_for_model")
except ImportError as e:
    logger.error(f"Nie można zaimportować prepare_data_for_model: {e}")
    sys.exit(1)

def test_dict_conversion():
    """Testuje konwersję danych ze słownika."""
    logger.info("Test konwersji danych ze słownika")
    
    # Przykładowe dane OHLCV w formacie słownika
    data_dict = {
        'open': [19610.0, 19403.9, 20174.3, 20167.4, 20754.3],
        'high': [20328.3, 19963.0, 20496.7, 21348.0, 20414.3],
        'low': [19719.6, 19585.8, 19752.9, 18636.9, 19995.5],
        'close': [20039.7, 19687.8, 20396.3, 20679.5, 19513.8],
        'volume': [845.3, 328.4, 903.4, 320.0, 975.8],
        'timestamp': [1744718931.6, 1744715331.6, 1744711731.6, 1744708131.6, 1744704531.6]
    }
    
    try:
        result = prepare_data_for_model(data_dict)
        logger.info(f"Wynik konwersji: Tablica o kształcie {result.shape}")
        logger.info(f"Typ wyniku: {type(result)}")
        logger.info("Konwersja ze słownika zakończona sukcesem")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas konwersji ze słownika: {e}")
        return False

def test_dataframe_conversion():
    """Testuje konwersję danych z DataFrame."""
    logger.info("Test konwersji danych z DataFrame")
    
    # Przykładowe dane OHLCV w formacie DataFrame
    data_dict = {
        'open': [19610.0, 19403.9, 20174.3, 20167.4, 20754.3],
        'high': [20328.3, 19963.0, 20496.7, 21348.0, 20414.3],
        'low': [19719.6, 19585.8, 19752.9, 18636.9, 19995.5],
        'close': [20039.7, 19687.8, 20396.3, 20679.5, 19513.8],
        'volume': [845.3, 328.4, 903.4, 320.0, 975.8],
        'timestamp': [1744718931.6, 1744715331.6, 1744711731.6, 1744708131.6, 1744704531.6]
    }
    df = pd.DataFrame(data_dict)
    
    try:
        result = prepare_data_for_model(df)
        logger.info(f"Wynik konwersji: Tablica o kształcie {result.shape}")
        logger.info(f"Typ wyniku: {type(result)}")
        logger.info("Konwersja z DataFrame zakończona sukcesem")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas konwersji z DataFrame: {e}")
        return False

def test_numpy_array_conversion():
    """Testuje konwersję danych z tablicy NumPy."""
    logger.info("Test konwersji danych z tablicy NumPy")
    
    # Przykładowe dane w formacie tablicy NumPy
    data_array = np.random.rand(10, 5)  # 10 próbek, 5 cech
    
    try:
        result = prepare_data_for_model(data_array)
        logger.info(f"Wynik konwersji: Tablica o kształcie {result.shape}")
        logger.info(f"Typ wyniku: {type(result)}")
        logger.info("Konwersja z tablicy NumPy zakończona sukcesem")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas konwersji z tablicy NumPy: {e}")
        return False

def main():
    """Główna funkcja testowa."""
    logger.info("Rozpoczynam testy konwersji danych")
    
    tests = [
        ("Konwersja ze słownika", test_dict_conversion),
        ("Konwersja z DataFrame", test_dataframe_conversion),
        ("Konwersja z tablicy NumPy", test_numpy_array_conversion),
    ]
    
    success_count = 0
    
    for test_name, test_func in tests:
        logger.info(f"Uruchamiam test: {test_name}")
        if test_func():
            success_count += 1
            logger.info(f"Test {test_name} zakończony sukcesem")
        else:
            logger.error(f"Test {test_name} zakończony niepowodzeniem")
            
    logger.info(f"Testy zakończone. Udane: {success_count}/{len(tests)}")
    
    if success_count == len(tests):
        logger.info("Wszystkie testy zakończone sukcesem!")
        return 0
    else:
        logger.warning("Niektóre testy zakończyły się niepowodzeniem.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
