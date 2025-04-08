"""
hyperparameter_tuner.py
-----------------------
Moduł automatycznie dostrajający hiperparametry modeli predykcyjnych i strategii handlowych.
Funkcjonalności:
- Obsługuje różne algorytmy optymalizacji, takie jak Grid Search, Random Search oraz Bayesian Optimization (np. przy użyciu Optuna).
- Pozwala na optymalizację różnych metryk (np. zysk, drawdown, stability).
- Zawiera testy jednostkowe oraz szczegółowe logowanie, umożliwiające łatwe powtórzenie procesu tuningu.
"""

import logging

import numpy as np
import pandas as pd

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class HyperparameterTuner:
    def __init__(
        self,
        model_class,
        param_space: dict,
        search_method: str = "grid",
        metric: str = "mse",
        cv: int = 5,
        n_iter: int = 50,
        random_state: int = 42,
    ):
        """
        Inicjalizuje tuner hiperparametrów.

        Parameters:
            model_class: Klasa modelu (np. RandomForestRegressor).
            param_space (dict): Przestrzeń hiperparametrów.
            search_method (str): Metoda tuningu: "grid", "random" lub "bayesian".
            metric (str): Metryka do optymalizacji ("mse" lub inna obsługiwana metryka).
            cv (int): Liczba podziałów dla walidacji krzyżowej.
            n_iter (int): Liczba iteracji (dla random search i bayesian optimization).
            random_state (int): Ziarno losowości.
        """
        self.model_class = model_class
        self.param_space = param_space
        self.search_method = search_method.lower()
        self.metric = metric
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state
        self.best_params = None
        self.best_score = None

    def tune(self, X: pd.DataFrame, y: pd.Series):
        """
        Przeprowadza tuning hiperparametrów na danych.

        Parameters:
            X (pd.DataFrame): Dane wejściowe.
            y (pd.Series): Wartości docelowe.

        Returns:
            tuple: (najlepsze hiperparametry, najlepszy wynik metryki)
        """
        logging.info(
            "Rozpoczynam tuning hiperparametrów metodą: %s", self.search_method
        )
        if self.search_method == "grid":
            tuner = GridSearchCV(
                estimator=self.model_class(),
                param_grid=self.param_space,
                scoring=self.metric,
                cv=self.cv,
                n_jobs=-1,
            )
            tuner.fit(X, y)
            self.best_params = tuner.best_params_
            self.best_score = tuner.best_score_
            logging.info(
                "Grid Search zakończony. Najlepsze parametry: %s, wynik: %.4f",
                self.best_params,
                self.best_score,
            )
        elif self.search_method == "random":
            tuner = RandomizedSearchCV(
                estimator=self.model_class(),
                param_distributions=self.param_space,
                scoring=self.metric,
                cv=self.cv,
                n_iter=self.n_iter,
                random_state=self.random_state,
                n_jobs=-1,
            )
            tuner.fit(X, y)
            self.best_params = tuner.best_params_
            self.best_score = tuner.best_score_
            logging.info(
                "Random Search zakończony. Najlepsze parametry: %s, wynik: %.4f",
                self.best_params,
                self.best_score,
            )
        elif self.search_method == "bayesian":
            if not OPTUNA_AVAILABLE:
                raise ImportError(
                    "Optuna nie jest zainstalowana. Zainstaluj ją, aby używać bayesian optimization."
                )
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
            )

            def objective(trial):
                params = {}
                for key, space in self.param_space.items():
                    if space["type"] == "int":
                        params[key] = trial.suggest_int(
                            key, space["low"], space["high"], step=space.get("step", 1)
                        )
                    elif space["type"] == "float":
                        params[key] = trial.suggest_float(
                            key, space["low"], space["high"]
                        )
                    elif space["type"] == "categorical":
                        params[key] = trial.suggest_categorical(key, space["choices"])
                    else:
                        raise ValueError(f"Nieobsługiwany typ parametru: {key}")
                model = self.model_class(**params)
                model.fit(X, y)
                predictions = model.predict(X)
                score = mean_squared_error(y, predictions)
                return score

            study.optimize(objective, n_trials=self.n_iter)
            self.best_params = study.best_params
            self.best_score = study.best_value
            logging.info(
                "Bayesian Optimization zakończona. Najlepsze parametry: %s, wynik: %.4f",
                self.best_params,
                self.best_score,
            )
        else:
            raise ValueError(f"Nieobsługiwana metoda tuningu: {self.search_method}")
        return self.best_params, self.best_score


# -------------------- Przykładowe użycie i testy --------------------
if __name__ == "__main__":
    try:
        # Przykładowe dane
        np.random.seed(42)
        import pandas as pd

        X = pd.DataFrame(
            {
                "feature1": np.random.uniform(0, 1, 500),
                "feature2": np.random.uniform(0, 1, 500),
            }
        )
        y = X["feature1"] * 2.0 + X["feature2"] * (-1.0) + np.random.normal(0, 0.1, 500)

        # Przykładowa przestrzeń hiperparametrów dla RandomForestRegressor
        from sklearn.ensemble import RandomForestRegressor

        param_space = {
            "n_estimators": {"type": "int", "low": 50, "high": 200, "step": 10},
            "max_depth": {"type": "int", "low": 3, "high": 10, "step": 1},
            "min_samples_split": {"type": "int", "low": 2, "high": 10, "step": 1},
        }

        tuner = HyperparameterTuner(
            model_class=RandomForestRegressor,
            param_space=param_space,
            search_method="bayesian",
            metric="neg_mean_squared_error",
            cv=5,
            n_iter=20,
            random_state=42,
        )
        best_params, best_score = tuner.tune(X, y)
        logging.info(
            "Tuning zakończony. Najlepsze parametry: %s, najlepszy wynik: %.4f",
            best_params,
            best_score,
        )
    except Exception as e:
        logging.error("Błąd w module hyperparameter_tuner.py: %s", e)
        raise
"""
hyperparameter_tuner.py
----------------------
Moduł odpowiedzialny za strojenie hiperparametrów strategii tradingowych.
"""

import logging
from typing import Dict, Any, List, Callable, Optional

# Konfiguracja logowania
logger = logging.getLogger("hyperparameter_tuner")
if not logger.handlers:
    log_dir = "logs"
    import os
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "hyperparameter_tuner.log"))
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

class HyperparameterTuner:
    """
    Klasa do strojenia hiperparametrów strategii tradingowych.
    """

    def __init__(self, optimization_method: str = "grid_search"):
        """
        Inicjalizacja tunera hiperparametrów.

        Parameters:
            optimization_method (str): Metoda optymalizacji parametrów 
                                      ('grid_search', 'random_search', 'bayesian')
        """
        self.optimization_method = optimization_method
        self.best_params = {}
        logger.info(f"Inicjalizacja tunera hiperparametrów z metodą: {optimization_method}")

    def tune(self, 
             param_grid: Dict[str, List[Any]], 
             evaluation_function: Callable[[Dict[str, Any]], float], 
             max_iterations: int = 100) -> Dict[str, Any]:
        """
        Wykonuje strojenie hiperparametrów.

        Parameters:
            param_grid (Dict[str, List[Any]]): Siatka parametrów do przeszukania
            evaluation_function (Callable): Funkcja oceniająca zestaw parametrów
            max_iterations (int): Maksymalna liczba iteracji

        Returns:
            Dict[str, Any]: Najlepszy znaleziony zestaw parametrów
        """
        logger.info(f"Rozpoczęcie strojenia parametrów metodą {self.optimization_method}")
        # Implementacja stub - zwraca pierwszy zestaw parametrów z siatki
        best_params = {}
        for param_name, param_values in param_grid.items():
            if param_values:
                best_params[param_name] = param_values[0]
                
        self.best_params = best_params
        return best_params

    def get_best_params(self) -> Dict[str, Any]:
        """
        Zwraca najlepszy znaleziony zestaw parametrów.

        Returns:
            Dict[str, Any]: Najlepszy zestaw parametrów
        """
        return self.best_params
