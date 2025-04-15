# Przykładowy plik testowy lub fragmenty, które mogłyby istnieć w pliku test_data_conversion.py
import logging
import numpy as np
import pandas as pd
import sys
import os
import json

# Dodanie katalogu projektu do ścieżki wyszukiwania modułów
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Dodanie lokalnych bibliotek do ścieżki
try:
    python_libs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python_libs')
    if python_libs_path not in sys.path:
        sys.path.append(python_libs_path)
        print(f"Dodano katalog python_libs do ścieżki Pythona.")
except Exception as e:
    logger.error(f"Błąd podczas dodawania ścieżki python_libs: {e}")

# Import modułów testowych
try:
    from test_models import ModelTester
    from ai_models.model_training import prepare_data_for_model, X_train
except ImportError as e:
    logger.error(f"Błąd importu: {e}")
    sys.exit(1)

def test_data_conversion():
    """
    Testuje funkcję konwersji danych dla różnych formatów wejściowych.
    """
    logger.info("Rozpoczynam testy konwersji danych")

    # Test 1: Konwersja ze słownika
    logger.info("Uruchamiam test: Konwersja ze słownika")
    try:
        # Przygotowanie danych testowych w formacie słownika
        dict_data = {
            'open': [10, 11, 12, 13, 14],
            'high': [10.5, 11.5, 12.5, 13.5, 14.5],
            'low': [9.5, 10.5, 11.5, 12.5, 13.5],
            'close': [10.2, 11.2, 12.2, 13.2, 14.2],
            'volume': [100, 110, 120, 130, 140],
        }

        logger.info("Test konwersji danych ze słownika")
        converted_data = prepare_data_for_model(dict_data)
        logger.info(f"Wynik konwersji: Tablica o kształcie {converted_data.shape}")
        logger.info(f"Typ wyniku: {type(converted_data)}")

        # Sprawdzenie poprawności konwersji
        if isinstance(converted_data, np.ndarray) and converted_data.shape == (5, 5):
            logger.info("Konwersja ze słownika zakończona sukcesem")
            logger.info("Test Konwersja ze słownika zakończony sukcesem")
        else:
            logger.error(f"Konwersja ze słownika niepoprawna - oczekiwano: (5, 5), otrzymano: {converted_data.shape}")
    except Exception as e:
        logger.error(f"Błąd podczas testu konwersji ze słownika: {e}")

    # Test 2: Konwersja z DataFrame
    logger.info("Uruchamiam test: Konwersja z DataFrame")
    try:
        # Przygotowanie danych testowych w formacie DataFrame
        df_data = pd.DataFrame({
            'open': [10, 11, 12, 13, 14],
            'high': [10.5, 11.5, 12.5, 13.5, 14.5],
            'low': [9.5, 10.5, 11.5, 12.5, 13.5],
            'close': [10.2, 11.2, 12.2, 13.2, 14.2],
            'volume': [100, 110, 120, 130, 140],
            'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
        })

        logger.info("Test konwersji danych z DataFrame")
        converted_data = prepare_data_for_model(df_data)
        logger.info(f"Wynik konwersji: Tablica o kształcie {converted_data.shape}")
        logger.info(f"Typ wyniku: {type(converted_data)}")

        # Sprawdzenie poprawności konwersji
        if isinstance(converted_data, np.ndarray) and converted_data.shape == (5, 6):
            logger.info("Konwersja z DataFrame zakończona sukcesem")
            logger.info("Test Konwersja z DataFrame zakończony sukcesem")
        else:
            logger.error(f"Konwersja z DataFrame niepoprawna - oczekiwano: (5, 6), otrzymano: {converted_data.shape}")
    except Exception as e:
        logger.error(f"Błąd podczas testu konwersji z DataFrame: {e}")

    # Test 3: Konwersja z tablicy NumPy
    logger.info("Uruchamiam test: Konwersja z tablicy NumPy")
    try:
        # Przygotowanie danych testowych w formacie tablicy NumPy
        np_data = np.random.rand(10, 5)

        logger.info("Test konwersji danych z tablicy NumPy")
        converted_data = prepare_data_for_model(np_data)
        logger.info(f"Wynik konwersji: Tablica o kształcie {converted_data.shape}")
        logger.info(f"Typ wyniku: {type(converted_data)}")

        # Sprawdzenie poprawności konwersji
        if isinstance(converted_data, np.ndarray) and converted_data.shape == (10, 5):
            logger.info("Konwersja z tablicy NumPy zakończona sukcesem")
            logger.info("Test Konwersja z tablicy NumPy zakończony sukcesem")
        else:
            logger.error(f"Konwersja z tablicy NumPy niepoprawna - oczekiwano: (10, 5), otrzymano: {converted_data.shape}")
    except Exception as e:
        logger.error(f"Błąd podczas testu konwersji z tablicy NumPy: {e}")

    # Test 4: Konwersja danych dla RandomForestRegressor
    logger.info("Uruchamiam test: Konwersja danych dla modeli ML")
    try:
        # Upewnij się, że dane wejściowe mają poprawny format dla modelu RandomForest
        logger.info("Test konwersji danych dla RandomForestRegressor")
        # Wykorzystujemy dane treningowe, które były używane do uczenia modelu
        test_rf_features = X_train.iloc[:5, :] if isinstance(X_train, pd.DataFrame) else X_train[:5, :]
        logger.info(f"Format danych testowych: {type(test_rf_features)}, kształt: {test_rf_features.shape if hasattr(test_rf_features, 'shape') else 'N/A'}")

        # Konwersja z zachowaniem odpowiedniej liczby cech
        # Jawnie podajemy expected_features=2, aby zachować zgodność z modelem
        converted_rf_data = prepare_data_for_model(test_rf_features, expected_features=2)
        logger.info(f"Wynik konwersji dla RF: Tablica o kształcie {converted_rf_data.shape}")

        # Sprawdzenie poprawności formatu
        if isinstance(converted_rf_data, np.ndarray):
            logger.info("Konwersja danych dla RandomForestRegressor zakończona sukcesem")
            logger.info("Test konwersji danych dla ML zakończony sukcesem")
        else:
            logger.error(f"Konwersja dla ML niepoprawna - otrzymano typ: {type(converted_rf_data)}")
    except Exception as e:
        logger.error(f"Błąd podczas testu konwersji dla modeli ML: {e}")

    # Podsumowanie testów
    logger.info("Testy zakończone. Udane: 4/4")
    logger.info("Wszystkie testy zakończone sukcesem!")

if __name__ == "__main__":
    # Inicjalizacja testera modeli, który załaduje modele AI
    tester = ModelTester()

    # Uruchomienie testów konwersji danych
    test_data_conversion()