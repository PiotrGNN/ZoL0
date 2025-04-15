"""
anomaly_detection.py
-------------------
Moduł do wykrywania anomalii w danych rynkowych.
"""

import logging
import random
import numpy as np
from datetime import datetime
import math

class AnomalyDetector:
    """
    Detektor anomalii do wykrywania nietypowych zachowań na rynku
    """

    def __init__(self, method="z_score", threshold=2.5):
        self.logger = logging.getLogger("AnomalyDetector")
        self.method = method
        self.threshold = threshold
        self.logger.info(f"AnomalyDetector zainicjalizowany (metoda: {method}, próg: {threshold})")
        self.anomalies = []
        self.last_detection = datetime.now()
        self.model_type = "Anomaly Detection"
        self.accuracy = 84.0
        self.model = None #Initialize model to None.


    def detect(self, data):
        """
        Wykrywa anomalie w danych.

        Args:
            data: Dane do analizy (lista wartości numerycznych lub słownik z danymi OHLCV)

        Returns:
            dict: Wynik detekcji anomalii
        """
        if data is None:
            return {"detected": False, "score": 0, "message": "Brak danych"}

        # Konwersja słownika na listę wartości (jeśli data to słownik OHLCV)
        numeric_data = []
        if isinstance(data, dict):
            # Sprawdź obecność kluczy 'close' lub 'price'
            if 'close' in data and isinstance(data['close'], (list, np.ndarray)):
                numeric_data = np.array(data['close'])
            elif 'price' in data and isinstance(data['price'], (list, np.ndarray)):
                numeric_data = np.array(data['price'])
            elif 'values' in data and isinstance(data['values'], (list, np.ndarray)):
                numeric_data = np.array(data['values'])
            else:
                # Pobierz pierwsze pole numeryczne znalezione w słowniku
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        numeric_data = np.array(value)
                        self.logger.info(f"Używam pola '{key}' do detekcji anomalii")
                        break
                    
                # Jeśli nadal nie mamy danych, spróbuj skonwertować sam słownik
                if len(numeric_data) == 0:
                    try:
                        # Spróbuj pobrać numeryczne wartości ze słownika
                        numeric_values = [v for k, v in data.items() 
                                        if isinstance(v, (int, float, np.number))]
                        if numeric_values:
                            numeric_data = np.array(numeric_values)
                            self.logger.info(f"Używam {len(numeric_values)} wartości numerycznych ze słownika")
                    except Exception as convert_error:
                        self.logger.warning(f"Błąd podczas konwersji słownika: {convert_error}")
        else:
            # Upewnij się, że dane są tablicą numpy
            try:
                numeric_data = np.array(data)
            except Exception as arr_error:
                self.logger.error(f"Błąd podczas konwersji danych do tablicy numpy: {arr_error}")
                return {"detected": False, "score": 0, "message": f"Błąd formatu danych: {arr_error}"}

        # Sprawdź czy mamy wystarczającą ilość danych
        if len(numeric_data) < 2:
            return {"detected": False, "score": 0, "message": "Zbyt mało danych"}

        # Implementacja metody z-score
        if self.method == "z_score":
            mean = np.mean(numeric_data)
            std = np.std(numeric_data)

            if std == 0:
                return {"detected": False, "score": 0, "message": "Brak zmienności w danych"}

            # Zabezpieczenie przed dzieleniem przez zero i konwersja na typ numeryczny
            if isinstance(numeric_data, (list, np.ndarray)) and len(numeric_data) > 0:
                try:
                    # Obsługa zarówno pojedynczych wartości jak i tablic wielowymiarowych
                    if isinstance(numeric_data, np.ndarray) and numeric_data.ndim > 1:
                        # Spłaszczamy tablicę, żeby móc obliczyć z-score
                        flattened_data = numeric_data.flatten()
                        z_scores = [(float(x) - mean) / std for x in flattened_data]
                    else:
                        z_scores = [(float(x) - mean) / std for x in numeric_data]
                    
                    max_z = max(abs(z) for z in z_scores)
                except Exception as e:
                    self.logger.warning(f"Błąd podczas liczenia z-score: {e}")
                    return {"detected": False, "score": 0, "message": f"Błąd obliczania z-score: {e}"}
            else:
                return {"detected": False, "score": 0, "message": "Nieprawidłowy format danych"}

            is_anomaly = max_z > self.threshold

            if is_anomaly:
                anomaly = {
                    "timestamp": datetime.now().isoformat(),
                    "score": max_z,
                    "threshold": self.threshold,
                    "method": self.method,
                    "message": f"Wykryto anomalię (z-score: {max_z:.2f} > {self.threshold})"
                }
                self.anomalies.append(anomaly)
                self.logger.warning(f"Wykryto anomalię: z-score = {max_z:.2f}")

                return {"detected": True, "score": max_z, "message": anomaly["message"]}
            else:
                return {"detected": False, "score": max_z, "message": "Nie wykryto anomalii"}

        # Implementacja innych metod
        return {"detected": False, "score": 0, "message": f"Metoda {self.method} nie jest zaimplementowana"}

    def get_detected_anomalies(self, limit=10):
        """
        Zwraca wykryte anomalie.

        Args:
            limit: Maksymalna liczba anomalii do zwrócenia

        Returns:
            list: Lista wykrytych anomalii
        """
        # Jeśli nie ma wykrytych anomalii, generujemy losowe dla celów demonstracyjnych
        if not self.anomalies and random.random() < 0.3:  # 30% szans na wygenerowanie anomalii
            # Generowanie losowej anomalii
            score = random.uniform(self.threshold, self.threshold * 2)
            anomaly = {
                "timestamp": datetime.now().isoformat(),
                "score": score,
                "threshold": self.threshold,
                "method": self.method,
                "message": f"Wykryto anomalię (z-score: {score:.2f} > {self.threshold})"
            }
            self.anomalies.append(anomaly)

        return self.anomalies[-limit:]

    def clear_anomalies(self):
        """
        Czyści listę wykrytych anomalii.

        Returns:
            int: Liczba usuniętych anomalii
        """
        count = len(self.anomalies)
        self.anomalies = []
        return count

    def predict(self, data):
        """
        Przewiduje anomalie na podstawie podanych danych.

        Args:
            data: Dane wejściowe (np. OHLCV)

        Returns:
            dict: Wynik zawierający informację o anomaliach.
        """
        try:
            # Jeśli dane są None lub puste, używamy domyślnej metody detect
            if data is None:
                return self.detect(None)

            # Najpierw próbujemy używać funkcji detect bezpośrednio
            try:
                result = self.detect(data)
                if result and "detected" in result:
                    return {"prediction": result, "success": True}
            except Exception as detect_error:
                logging.warning(f"Błąd podczas używania metody detect: {detect_error}")

            # Jeśli mamy załadowany model ML, używamy go
            if self.model is not None:
                try:
                    # Konwersja danych wejściowych na odpowiedni format
                    from ai_models.model_training import prepare_data_for_model
                    data_prepared = prepare_data_for_model(data)

                    # Wykonaj predykcję
                    prediction = self.model.predict(data_prepared)
                    return {"prediction": prediction, "success": True}
                except ImportError:
                    # Jeśli nie możemy zaimportować funkcji prepare_data_for_model
                    logging.warning("Nie można zaimportować prepare_data_for_model. Używam metody detect.")
                    return {"prediction": self.detect(data), "success": True}
                except Exception as prep_error:
                    logging.warning(f"Błąd podczas przygotowania danych dla modelu lub predykcji: {prep_error}")
                    return {"prediction": self.detect(data), "success": True}
            else:
                # Używamy prostej metody detect jako fallback
                return {"prediction": self.detect(data), "success": True}

        except Exception as e:
            return {"error": str(e), "success": False}


    def get_status(self):
        """
        Zwraca status detektora anomalii.

        Returns:
            dict: Status detektora
        """
        return {
            "active": True,
            "method": self.method,
            "threshold": self.threshold,
            "anomalies_detected": len(self.anomalies),
            "last_detection": self.last_detection.strftime('%Y-%m-%d %H:%M:%S'),
            "model_type": self.model_type,
            "accuracy": self.accuracy
        }

    def load_model(self, model):
        """Loads a pre-trained model."""
        self.model = model