"""
AI_strategy_generator.py
------------------------
Moduł do generowania strategii handlowych opartych na algorytmach AI, takich jak sieci neuronowe, modele gradient boosting czy meta-learning.
Funkcjonalności:
- Automatyczne wyszukiwanie i testowanie zestawów hiperparametrów dla modeli AI.
- Mechanizmy selekcji cech (feature selection) oraz budowanie ensemble (stacking, blending) w celu zwiększenia skuteczności strategii.
- Integracja z modułem reinforcement_learning.py, umożliwiająca łączenie podejścia AI z uczeniem ze wzmocnieniem.
- Logowanie wyników, automatyczne testy oraz możliwość przeprowadzenia backtestu i testów real-time.
- System jest skalowalny i obsługuje strategie dla różnych rozmiarów kapitału w dłuższej perspektywie.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor

# Opcjonalnie, integracja z reinforcement learning (przykładowo importujemy DQNAgent)
# from reinforcement_learning import DQNAgent

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AIStrategyGenerator:
    def __init__(self, data: pd.DataFrame, target: str):
        """
        Inicjalizuje generator strategii AI.

        Parameters:
            data (pd.DataFrame): Dane wejściowe zawierające cechy i zmienną docelową.
            target (str): Nazwa kolumny ze zmienną docelową.
        """
        self.data = data
        self.target = target
        self.features = data.drop(columns=[target])
        self.labels = data[target]
        self.best_model = None
        self.best_params = None

    def feature_selection(self, k: int = 10):
        """
        Przeprowadza selekcję cech, wybierając najlepsze k cech na podstawie testu F.

        Parameters:
            k (int): Liczba cech do wybrania.

        Returns:
            pd.DataFrame: DataFrame z wybranymi cechami.
        """
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(self.features, self.labels)
        cols = self.features.columns[selector.get_support()]
        logging.info("Wybrane cechy: %s", list(cols))
        self.features = self.features[cols]
        return self.features

    def hyperparameter_tuning(self, model, param_grid: dict, cv: int = 5):
        """
        Wykonuje tuning hiperparametrów dla danego modelu za pomocą Grid Search.

        Parameters:
            model: Instancja modelu ML (np. GradientBoostingRegressor, MLPRegressor).
            param_grid (dict): Słownik z przestrzenią hiperparametrów.
            cv (int): Liczba podziałów walidacji krzyżowej.

        Returns:
            model: Model z najlepszymi hiperparametrami.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        self.best_params = grid.best_params_
        logging.info("Najlepsze hiperparametry: %s", self.best_params)
        self.best_model = grid.best_estimator_
        return self.best_model

    def build_ensemble(self):
        """
        Buduje ensemble modeli poprzez stacking/blending.
        W tym przykładzie wykorzystujemy VotingRegressor do łączenia predykcji.

        Returns:
            VotingRegressor: Ensemble modeli.
        """
        # Przykładowe modele
        model1 = GradientBoostingRegressor(random_state=42)
        model2 = MLPRegressor(random_state=42, max_iter=500)

        # Parametry można dostroić osobno przed zbudowaniem ensemble
        ensemble = VotingRegressor(estimators=[("gbr", model1), ("mlp", model2)])
        logging.info("Zbudowano ensemble modeli: GradientBoostingRegressor i MLPRegressor.")
        self.best_model = ensemble
        return ensemble

    def evaluate_strategy(self):
        """
        Ocena wygenerowanej strategii na podstawie podziału danych.
        Zwraca metryki, np. MSE, oraz loguje wyniki.

        Returns:
            dict: Wyniki oceny strategii.
        """
        from sklearn.metrics import mean_squared_error

        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)
        self.best_model.fit(X_train, y_train)
        predictions = self.best_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        logging.info("Ocena strategii - MSE: %.4f", mse)
        return {"MSE": mse}

    def generate_strategy(self):
        """
        Kompleksowy pipeline generowania strategii:
        - Selekcja cech.
        - Tuning hiperparametrów dla wybranego modelu.
        - Budowa ensemble.
        - Ocena strategii.

        Returns:
            dict: Raport zawierający wyniki oceny strategii oraz najlepsze hiperparametry.
        """
        self.feature_selection(k=min(10, self.features.shape[1]))

        # Przykładowy tuning dla GradientBoostingRegressor
        from sklearn.ensemble import GradientBoostingRegressor

        param_grid = {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5],
        }
        tuned_model = self.hyperparameter_tuning(GradientBoostingRegressor(random_state=42), param_grid)

        # Budowanie ensemble (można łączyć różne modele, w tym wyniki RL)
        self.build_ensemble()

        # Ewaluacja strategii
        evaluation_results = self.evaluate_strategy()

        report = {
            "best_hyperparameters": self.best_params,
            "evaluation": evaluation_results,
        }
        logging.info("Strategia AI wygenerowana pomyślnie. Raport: %s", report)
        return report


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Przykładowe dane: symulacja historycznych danych rynkowych
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=500, freq="B")
        data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 500),
                "feature2": np.random.normal(5, 2, 500),
                "feature3": np.random.normal(10, 3, 500),
                "target": np.random.normal(0, 1, 500),
            },
            index=dates,
        )

        ai_generator = AIStrategyGenerator(data, target="target")
        report = ai_generator.generate_strategy()
        logging.info("Raport z generowania strategii AI: %s", report)
    except Exception as e:
        logging.error("Błąd w module AI_strategy_generator.py: %s", e)
        raise
