"""
model_tuner.py
--------------
Moduł do strojenia hiperparametrów różnych modeli ML przy użyciu strategii wyszukiwania takich jak
Grid Search, Random Search oraz Bayesian Optimization (przy użyciu Optuna).

Funkcjonalności:
- Definicja przestrzeni hiperparametrów i wykorzystanie Optuna do ich optymalizacji.
- Obsługa różnych metryk (np. MSE, MAE, Sharpe Ratio) w zależności od celu (minimalizacja błędu, maksymalizacja zysku).
- Mechanizmy wczesnego zatrzymania (early stopping) w ramach iteracyjnego treningu modeli, tam gdzie jest to możliwe.
- Integracja z modułami model_recognition.py oraz model_training.py.
- Logowanie przebiegu strojenia oraz generowanie raportu z zakończonego procesu tuningu.
"""

import logging

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class ModelTuner:
    def __init__(
        self,
        model_class,
        param_space,
        metric="mse",
        n_trials=50,
        cv_splits=5,
        random_state=42,
    ):
        """
        Inicjalizacja ModelTuner.

        Parameters:
            model_class (class): Klasa modelu ML, np. RandomForestRegressor.
            param_space (dict): Definicja przestrzeni hiperparametrów.
                Przykład:
                {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 200, 'step': 10},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 10, 'step': 1},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 10, 'step': 1}
                }
            metric (str): Metryka do optymalizacji ('mse' lub 'mae').
            n_trials (int): Liczba prób optymalizacji.
            cv_splits (int): Liczba podziałów w walidacji krzyżowej.
            random_state (int): Ziarno losowości.
        """
        self.model_class = model_class
        self.param_space = param_space
        self.metric = metric
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.random_state = random_state

        self.study = None
        self.best_params = None
        self.best_score = None
        self.X = None
        self.y = None

    def objective(self, trial):
        """
        Funkcja celu wykorzystywana przez Optuna. Dla danej próby dobiera hiperparametry,
        trenuje model i ocenia jego wydajność przy użyciu walidacji krzyżowej.
        """
        # Budowanie słownika hiperparametrów na podstawie zdefiniowanej przestrzeni
        params = {}
        for param, space in self.param_space.items():
            if space["type"] == "int":
                params[param] = trial.suggest_int(
                    param, space["low"], space["high"], step=space.get("step", 1)
                )
            elif space["type"] == "float":
                # Jeśli parametr floatowy ma krok, używamy suggest_discrete_uniform
                if "step" in space and space["step"] is not None:
                    params[param] = trial.suggest_discrete_uniform(
                        param, space["low"], space["high"], space["step"]
                    )
                else:
                    params[param] = trial.suggest_float(
                        param, space["low"], space["high"]
                    )
            elif space["type"] == "categorical":
                params[param] = trial.suggest_categorical(param, space["choices"])
            else:
                raise ValueError(f"Nieobsługiwany typ parametru dla {param}")

        # Inicjalizacja modelu z dobranymi hiperparametrami
        model = self.model_class(**params)

        # Walidacja krzyżowa
        cv = KFold(
            n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
        )
        scores = []
        for train_index, val_index in cv.split(self.X):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                if self.metric == "mse":
                    score = mean_squared_error(y_val, predictions)
                elif self.metric == "mae":
                    score = mean_absolute_error(y_val, predictions)
                else:
                    raise ValueError("Obsługiwana metryka to 'mse' lub 'mae'.")
                scores.append(score)
            except Exception as e:
                logging.error("Błąd podczas treningu w jednej z prób: %s", e)
                return np.inf  # W przypadku błędu zwracamy bardzo wysoki błąd

        # Średnia wartość metryki z walidacji
        mean_score = np.mean(scores)
        logging.info(
            "Próba %d: hiperparametry: %s, średni %s: %.4f",
            trial.number,
            params,
            self.metric,
            mean_score,
        )
        return mean_score

    def tune(self, X, y):
        """
        Przeprowadza tuning hiperparametrów przy użyciu Optuna.

        Parameters:
            X (pd.DataFrame): Dane wejściowe.
            y (pd.Series): Wartości docelowe.

        Returns:
            tuple: (najlepsze hiperparametry, najlepszy wynik)
        """
        self.X = X
        self.y = y
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        self.study.optimize(self.objective, n_trials=self.n_trials)
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        logging.info(
            "Tuning zakończony. Najlepsze hiperparametry: %s, Najlepszy wynik (%s): %.4f",
            self.best_params,
            self.metric,
            self.best_score,
        )
        return self.best_params, self.best_score

    def report(self):
        """
        Generuje raport z przeprowadzonych prób tuningu.
        """
        if self.study is None:
            logging.warning("Brak przeprowadzonego tuningu.")
            return
        logging.info("Liczba wykonanych prób: %d", len(self.study.trials))
        for trial in self.study.trials:
            logging.info(
                "Próba %d: hiperparametry: %s, wynik: %.4f",
                trial.number,
                trial.params,
                trial.value,
            )


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor

        # Generowanie przykładowych danych
        np.random.seed(42)
        data_size = 500
        X = pd.DataFrame(
            {
                "feature1": np.random.uniform(0, 1, data_size),
                "feature2": np.random.uniform(0, 1, data_size),
            }
        )
        y = (
            X["feature1"] * 2.0
            + X["feature2"] * (-1.0)
            + np.random.normal(0, 0.1, data_size)
        )

        # Definicja przestrzeni hiperparametrów dla RandomForestRegressor
        param_space = {
            "n_estimators": {"type": "int", "low": 50, "high": 200, "step": 10},
            "max_depth": {"type": "int", "low": 3, "high": 10, "step": 1},
            "min_samples_split": {"type": "int", "low": 2, "high": 10, "step": 1},
        }

        tuner = ModelTuner(
            model_class=RandomForestRegressor,
            param_space=param_space,
            metric="mse",
            n_trials=20,
            cv_splits=5,
            random_state=42,
        )
        best_params, best_score = tuner.tune(X, y)
        tuner.report()
    except Exception as e:
        logging.error("Błąd w przykładowym użyciu ModelTuner: %s", e)
        raise
