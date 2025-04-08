"""
fix_tests.py - Skrypt naprawiający i uruchamiający testy jednostkowe
"""

import os
import sys
import logging
import unittest
import pytest

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_errors.log"),
        logging.StreamHandler()
    ]
)

def fix_tests():
    """Naprawia i uruchamia testy jednostkowe."""
    # Dodajemy katalog główny do ścieżki Pythona
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir)
    logging.info(f"Dodano {base_dir} do sys.path")

    # Tworzenie wymaganych katalogów
    required_dirs = [
        'logs',
        'data/cache',
        'saved_models'
    ]

    for directory in required_dirs:
        dir_path = os.path.join(base_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Utworzono katalog: {dir_path}")

    # Importowanie i uruchamianie testów
    try:
        logging.info("Uruchamianie testów...")

        # Opcja 1: Używanie unittest
        try:
            # Wyszukanie testów
            test_loader = unittest.TestLoader()
            test_suite = test_loader.discover("data/tests", pattern="test_*.py")

            # Uruchamianie testów
            test_runner = unittest.TextTestRunner()
            results = test_runner.run(test_suite)

            if results.failures or results.errors:
                logging.error(f"Testy nie powiodły się: {len(results.failures)} awarii, {len(results.errors)} błędów")
                for failure in results.failures:
                    logging.error(f"Błąd testu: {failure[0]}: {failure[1]}")
                for error in results.errors:
                    logging.error(f"Błąd testu: {error[0]}: {error[1]}")
            else:
                logging.info("Wszystkie testy unittest zakończone powodzeniem!")
        except Exception as unittest_error:
            logging.error(f"Błąd podczas uruchamiania testów unittest: {unittest_error}")

        # Opcja 2: Używanie pytest (jeśli unittest nie działa)
        try:
            logging.info("Próba uruchomienia testów z pytest...")
            pytest_args = [
                "-xvs",  # -x: zatrzymaj po pierwszym błędzie, -v: verbose, -s: pokaż stdout
                "data/tests"
            ]
            pytest_exit_code = pytest.main(pytest_args)

            if pytest_exit_code == 0:
                logging.info("Wszystkie testy pytest zakończone powodzeniem!")
            else:
                logging.error(f"Testy pytest zakończyły się z kodem wyjścia {pytest_exit_code}")
        except Exception as pytest_error:
            logging.error(f"Błąd podczas uruchamiania testów pytest: {pytest_error}")

    except Exception as e:
        logging.error(f"Błąd podczas uruchamiania testów: {e}")
        print(f"Błąd: {e}")

if __name__ == "__main__":
    print("Naprawianie i uruchamianie testów...")
    fix_tests()
    print("Gotowe! Sprawdź test_errors.log, aby zobaczyć szczegóły.")