"""
AI_strategy_generator.py
----------------------
Moduł do generowania strategii tradingowych przy użyciu sztucznej inteligencji.
"""

import logging
import os
import random
import time
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Konfiguracja logowania
logger = logging.getLogger("ai_strategy_generator")
if not logger.handlers:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "ai_strategy_generator.log"))
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

class AIStrategyGenerator:
    """
    Klasa do generowania strategii tradingowych przy użyciu AI.
    """

    def __init__(self, data: pd.DataFrame = None, model_type: str = "xgboost", target: str = None):
        """
        Inicjalizuje generator strategii AI.

        Parameters:
            data (pd.DataFrame, optional): DataFrame z danymi treningowymi
            model_type (str): Typ modelu ('xgboost', 'lstm', 'random_forest')
            target (str, optional): Nazwa kolumny zawierającej zmienną docelową
        """
        self.model_type = model_type
        self.best_model = None
        self.models = []
        self.training_history = []
        self.data = data
        self.target = target
        self.best_params = {}

        if data is not None and target is not None:
            self.X = data.drop(columns=[target])
            self.y = data[target]
        else:
            self.X = None
            self.y = None

        logger.info(f"Zainicjalizowano generator strategii AI z modelem: {model_type}")
        if data is not None:
            logger.info(f"Załadowano dane o wymiarach: {data.shape}")
        if target is not None:
            logger.info(f"Ustawiono zmienną docelową: {target}")

    def load_data(self, data: Any) -> bool:
        """
        Ładuje dane treningowe.

        Parameters:
            data (Any): Dane treningowe.

        Returns:
            bool: Czy operacja się powiodła.
        """
        try:
            # Implementacja ładowania danych
            # W szablonie po prostu logujemy informację
            logger.info(f"Załadowano dane treningowe: {type(data)}")
            self.data = data
            return True
        except Exception as e:
            logger.error(f"Błąd podczas ładowania danych: {e}")
            return False

    def preprocess_data(self) -> Tuple[Any, Any, Any, Any]:
        """
        Przetwarza dane treningowe.

        Returns:
            Tuple[Any, Any, Any, Any]: Krotka (X_train, X_test, y_train, y_test).
        """
        # Implementacja przetwarzania danych
        # W szablonie zwracamy dummy dane
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        logger.info("Przetworzono dane treningowe")
        return X_train, X_test, y_train, y_test

    def train_model(self) -> bool:
        """
        Trenuje model.

        Returns:
            bool: Czy operacja się powiodła.
        """
        try:
            # Przygotowanie danych
            X_train, X_test, y_train, y_test = self.preprocess_data()

            # Implementacja trenowania modelu
            # W szablonie po prostu logujemy informację
            logger.info(f"Trenowanie modelu {self.model_type}...")
            time.sleep(1)  # Symulacja trenowania

            # Tworzenie dummy modelu
            self.best_model = {"type": self.model_type, "trained": True}

            logger.info(f"Model {self.model_type} wytrenowany pomyślnie")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas trenowania modelu: {e}")
            return False

    def predict(self, data: Any) -> Dict[str, Any]:
        """
        Generuje predykcje na podstawie danych.

        Parameters:
            data (Any): Dane do predykcji.

        Returns:
            Dict[str, Any]: Wyniki predykcji.
        """
        try:
            if self.best_model is None:
                raise ValueError("Nie ustawiono najlepszego modelu przed predykcją.")

            # Implementacja predykcji
            # W szablonie zwracamy dummy dane
            prediction = {
                "signal": random.choice(["buy", "sell", "hold"]),
                "confidence": random.uniform(0.5, 1.0),
                "prediction_time": time.time()
            }

            logger.info(f"Wygenerowano predykcję: {prediction}")
            return prediction
        except Exception as e:
            logger.error(f"Błąd podczas generowania predykcji: {e}")
            return {"signal": "hold", "confidence": 0.0, "error": str(e)}

    def generate_strategy(self) -> Dict[str, Any]:
        """
        Generuje kompletną strategię tradingową.

        Returns:
            Dict[str, Any]: Skonfigurowana strategia
        """
        try:
            if self.best_model is None:
                # Jeśli nie ma modelu, najpierw zbuduj ensemble
                self.build_ensemble()

            # Przeprowadź ewaluację
            evaluation = self.evaluate_strategy()

            strategy = {
                "name": f"AI_{self.model_type}_Strategy",
                "type": self.model_type,
                "parameters": {
                    "window": 20,
                    "threshold": 0.5,
                    "stop_loss": 0.02,
                    "take_profit": 0.04
                },
                "best_hyperparameters": self.best_params,
                "evaluation": evaluation,
                "generated_at": time.time()
            }

            logger.info(f"Wygenerowano strategię: {strategy}")
            return strategy

        except Exception as e:
            logger.error(f"Błąd podczas generowania strategii: {e}")
            return {"name": "Fallback_Strategy", "error": str(e)}

    def save_model(self, path: str) -> bool:
        """
        Zapisuje model do pliku.

        Parameters:
            path (str): Ścieżka do zapisu modelu.

        Returns:
            bool: Czy operacja się powiodła.
        """
        try:
            if self.best_model is None:
                raise ValueError("Nie ustawiono najlepszego modelu przed zapisem.")

            # Implementacja zapisywania modelu
            # W szablonie po prostu logujemy informację
            logger.info(f"Zapisano model do: {path}")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania modelu: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """
        Ładuje model z pliku.

        Parameters:
            path (str): Ścieżka do modelu.

        Returns:
            bool: Czy operacja się powiodła.
        """
        try:
            # Implementacja ładowania modelu
            # W szablonie po prostu logujemy informację
            logger.info(f"Załadowano model z: {path}")
            self.best_model = {"type": self.model_type, "trained": True, "loaded": True}
            return True
        except Exception as e:
            logger.error(f"Błąd podczas ładowania modelu: {e}")
            return False

    def evaluate_model(self) -> Dict[str, Any]:
        """
        Ewaluuje model.

        Returns:
            Dict[str, Any]: Wyniki ewaluacji.
        """
        try:
            if self.best_model is None:
                raise ValueError("Nie ustawiono najlepszego modelu przed ewaluacją.")

            # Implementacja ewaluacji modelu
            # W szablonie zwracamy dummy dane
            evaluation = {
                "accuracy": random.uniform(0.6, 0.9),
                "precision": random.uniform(0.6, 0.9),
                "recall": random.uniform(0.6, 0.9),
                "f1_score": random.uniform(0.6, 0.9),
                "evaluated_at": time.time()
            }

            logger.info(f"Ewaluacja modelu: {evaluation}")
            return evaluation
        except Exception as e:
            logger.error(f"Błąd podczas ewaluacji modelu: {e}")
            return {"accuracy": 0.0, "error": str(e)}

    def feature_selection(self, k: int = 2) -> pd.DataFrame:
        """
        Wybiera k najlepszych cech na podstawie korelacji ze zmienną docelową.

        Parameters:
            k (int): Liczba cech do wybrania

        Returns:
            pd.DataFrame: DataFrame zawierający wybrane cechy
        """
        if self.X is None or self.y is None:
            raise ValueError("Brak danych do selekcji cech")

        try:
            # Oblicz korelacje między cechami a zmienną docelową
            correlations = pd.Series()
            for column in self.X.columns:
                correlations[column] = abs(self.X[column].corr(self.y))

            # Wybierz k najlepszych cech
            top_features = correlations.nlargest(k).index
            selected_features = self.X[top_features]

            logger.info(f"Wybrano {k} najlepszych cech: {list(top_features)}")
            return selected_features

        except Exception as e:
            logger.error(f"Błąd podczas selekcji cech: {e}")
            raise

    def hyperparameter_tuning(self, model: Any, param_grid: Dict[str, List[Any]], cv: int = 3) -> Any:
        """
        Przeprowadza tuning hiperparametrów modelu.

        Parameters:
            model: Model do dostrojenia
            param_grid: Siatka parametrów do przeszukania
            cv: Liczba foldów w cross-validation

        Returns:
            Any: Najlepszy model
        """
        if self.X is None or self.y is None:
            raise ValueError("Brak danych do tuningu hiperparametrów")

        try:
            # Przeprowadź grid search
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error')
            grid_search.fit(self.X, self.y)

            # Zapisz najlepsze parametry
            self.best_params = grid_search.best_params_
            self.best_model = grid_search.best_estimator_

            logger.info(f"Najlepsze parametry: {self.best_params}")
            return self.best_model

        except Exception as e:
            logger.error(f"Błąd podczas tuningu hiperparametrów: {e}")
            raise

    def build_ensemble(self, n_models: int = 3) -> Any:
        """
        Buduje ensemble modeli.

        Parameters:
            n_models (int): Liczba modeli w ensemble

        Returns:
            Any: Wytrenowany ensemble
        """
        if self.X is None or self.y is None:
            raise ValueError("Brak danych do budowy ensemble")

        try:
            # Utwórz podstawowe modele
            models = [
                ('rf', RandomForestRegressor(n_estimators=100)),
                ('gb', GradientBoostingRegressor(n_estimators=100))
            ][:n_models]

            # Utwórz i wytrenuj ensemble
            ensemble = VotingRegressor(estimators=models)
            ensemble.fit(self.X, self.y)

            self.best_model = ensemble
            logger.info(f"Zbudowano ensemble z {n_models} modeli")
            return ensemble

        except Exception as e:
            logger.error(f"Błąd podczas budowy ensemble: {e}")
            raise

    def evaluate_strategy(self) -> Dict[str, float]:
        """
        Ocenia skuteczność strategii.

        Returns:
            Dict[str, float]: Słownik z metrykami oceny
        """
        if self.best_model is None:
            raise ValueError("Brak modelu do ewaluacji")

        try:
            # Oblicz predykcje
            predictions = self.best_model.predict(self.X)

            # Oblicz metryki
            mse = mean_squared_error(self.y, predictions)
            r2 = r2_score(self.y, predictions)
            cv_scores = cross_val_score(self.best_model, self.X, self.y, cv=3)

            evaluation = {
                'MSE': mse,
                'R2': r2,
                'CV_mean': cv_scores.mean(),
                'CV_std': cv_scores.std()
            }

            logger.info(f"Wyniki ewaluacji strategii: {evaluation}")
            return evaluation

        except Exception as e:
            logger.error(f"Błąd podczas ewaluacji strategii: {e}")
            raise

# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        import numpy as np
        import pandas as pd
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

        ai_generator = AIStrategyGenerator(model_type="xgboost") # Example model type
        ai_generator.load_data(data) # Load data
        if ai_generator.train_model(): # Train Model
            strategy = ai_generator.generate_strategy()
            print(f"Wygenerowana strategia: {strategy}")
            evaluation = ai_generator.evaluate_model()
            print(f"Ewaluacja modelu: {evaluation}")
        else:
            print("Trenowanie modelu nie powiodło się.")


    except Exception as e:
        logger.error("Błąd w module AI_strategy_generator.py: %s", e)
        raise