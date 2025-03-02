"""
ropmer_temp.py
--------------
Moduł do prototypowania i testowania eksperymentalnych podejść, wskaźników oraz modeli.
Zawiera funkcje i klasy umożliwiające:
- Przeprowadzanie eksperymentów z czytelnym logowaniem i dokumentacją (docstringi).
- Tagowanie eksperymentów (np. nazwa, data, hiperparametry) oraz zapisywanie wyników w folderze `saved_models` lub bazie danych.
- Szybkie włączanie/wyłączanie konkretnych funkcji testowych poprzez parametry konfiguracyjne.
- Obsługę wyjątków i automatyczne raportowanie błędów, aby eksperymenty nie przerywały pracy systemu.
"""

import json
import logging
import os
import random
from datetime import datetime

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("ropmer_temp.log"), logging.StreamHandler()],
)


class ExperimentManager:
    """
    Klasa zarządzająca eksperymentami, umożliwiająca tagowanie, uruchamianie, logowanie i zapisywanie wyników.
    """

    def __init__(self, save_dir="saved_models", enable_experiments=True):
        """
        Inicjalizacja ExperimentManager.

        Parameters:
            save_dir (str): Folder do zapisywania wyników eksperymentów.
            enable_experiments (bool): Flaga umożliwiająca włączanie/wyłączanie eksperymentów.
        """
        self.save_dir = save_dir
        self.enable_experiments = enable_experiments
        os.makedirs(self.save_dir, exist_ok=True)
        logging.info(
            "ExperimentManager zainicjalizowany. save_dir: '%s', enable_experiments: %s",
            self.save_dir,
            self.enable_experiments,
        )

    def tag_experiment(self, name, hyperparams=None):
        """
        Generuje unikalny tag dla eksperymentu na podstawie nazwy, daty oraz hiperparametrów.

        Parameters:
            name (str): Nazwa eksperymentu.
            hyperparams (dict, optional): Słownik hiperparametrów eksperymentu.

        Returns:
            str: Unikalny tag eksperymentu.
        """
        tag = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if hyperparams:
            hyper_str = "_".join(f"{k}-{v}" for k, v in hyperparams.items())
            tag = f"{tag}_{hyper_str}"
        return tag

    def run_experiment(self, name, experiment_func, *args, **kwargs):
        """
        Uruchamia podaną funkcję eksperymentalną, loguje przebieg, taguje eksperyment oraz zapisuje wynik.
        Automatycznie obsługuje wyjątki, aby eksperymenty nie przerywały pracy systemu.

        Parameters:
            name (str): Nazwa eksperymentu.
            experiment_func (callable): Funkcja, która przeprowadza eksperyment.
            *args, **kwargs: Argumenty przekazywane do funkcji eksperymentalnej.

        Returns:
            Any: Wynik zwrócony przez funkcję eksperymentalną, lub None w przypadku błędu.
        """
        if not self.enable_experiments:
            logging.info("Eksperymenty są wyłączone. Pomijanie eksperymentu: %s", name)
            return None

        tag = self.tag_experiment(name, kwargs.get("hyperparams"))
        logging.info("Uruchamianie eksperymentu: %s", tag)
        try:
            result = experiment_func(*args, **kwargs)
            self.save_result(tag, result)
            logging.info("Eksperyment '%s' zakończony pomyślnie.", tag)
            return result
        except Exception as e:
            logging.error("Błąd w eksperymencie '%s': %s", tag, e, exc_info=True)
            return None

    def save_result(self, tag, result):
        """
        Zapisuje wynik eksperymentu w formacie JSON do folderu `saved_models`.

        Parameters:
            tag (str): Tag eksperymentu, użyty jako nazwa pliku.
            result (Any): Wynik eksperymentu, który będzie zapisany (konwertowany na JSON).
        """
        try:
            filename = os.path.join(self.save_dir, f"{tag}.json")
            with open(filename, "w") as f:
                json.dump(result, f, indent=4, default=str)
            logging.info("Wynik eksperymentu zapisany w: %s", filename)
        except Exception as e:
            logging.error("Błąd przy zapisie wyniku eksperymentu '%s': %s", tag, e)


# Przykładowa funkcja eksperymentalna
def experimental_indicator(data, param=1):
    """
    Przykładowa funkcja obliczająca eksperymentalny wskaźnik na podstawie danych.

    Parameters:
        data (list of float): Lista wartości numerycznych.
        param (int, optional): Parametr wpływający na skalę wskaźnika (domyślnie 1).

    Returns:
        dict: Słownik zawierający obliczony wskaźnik oraz użyty parametr.
    """
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Dane muszą być niepustą listą liczb.")
    indicator_value = sum(data) * param / len(data)
    return {"indicator_value": indicator_value, "param_used": param}


# Przykładowa konfiguracja eksperymentalna
if __name__ == "__main__":
    # Inicjalizacja managera eksperymentów
    exp_manager = ExperimentManager(save_dir="saved_models", enable_experiments=True)

    # Przygotowanie przykładowych danych
    sample_data = [random.uniform(0, 100) for _ in range(50)]

    # Uruchomienie eksperymentu z tagowaniem hiperparametrów
    result = exp_manager.run_experiment(
        name="sample_indicator",
        experiment_func=experimental_indicator,
        data=sample_data,
        param=2,
        hyperparams={"param": 2},
    )

    logging.info("Wynik eksperymentu: %s", result)
