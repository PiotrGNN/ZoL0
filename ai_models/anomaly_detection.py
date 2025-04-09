"""
anomaly_detection.py - Lekka implementacja wykrywania anomalii
"""
import random
import logging
import numpy as np
from datetime import datetime, timedelta

class AnomalyDetector:
    """
    Uproszczona implementacja wykrywania anomalii bazująca na podstawowych
    algorytmach statystycznych zamiast ciężkich bibliotek ML.
    """

    def __init__(self, method="z_score", threshold=2.5):
        """
        Inicjalizacja detektora anomalii.

        Args:
            method (str): Metoda wykrywania anomalii ('z_score', 'isolation_forest', 'mad')
            threshold (float): Próg dla wykrywania anomalii
        """
        self.method = method
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Zainicjalizowano AnomalyDetector z metodą {method} i progiem {threshold}")

        self.data_buffer = []
        self.buffer_size = 100
        self.detected_anomalies = []
        self.last_detection = datetime.now()

    def detect(self, data_point=None):
        """
        Wykrywa anomalie w danym punkcie danych lub generuje symulowane wyniki.

        Args:
            data_point (float, optional): Punkt danych do analizy. Jeśli None, generuje losowe dane.

        Returns:
            dict: Wynik detekcji anomalii
        """
        # Generowanie losowego punktu danych, jeśli nie podano
        if data_point is None:
            # Generuje losową wartość z dodatkową szansą na anomalię
            data_point = random.normalvariate(0, 1)
            if random.random() < 0.05:  # 5% szansa na anomalię
                data_point *= 3  # Wartość odstająca

        # Dodanie danych do bufora
        self.data_buffer.append(data_point)
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)

        # Wykrycie anomalii z użyciem odpowiedniej metody
        is_anomaly = False
        score = 0

        if self.method == "z_score" and len(self.data_buffer) >= 10:
            # Metoda z-score
            mean = np.mean(self.data_buffer)
            std = np.std(self.data_buffer) or 1  # Unikanie dzielenia przez zero
            z_score = abs((data_point - mean) / std)
            score = z_score
            is_anomaly = z_score > self.threshold

        elif self.method == "mad" and len(self.data_buffer) >= 10:
            # Metoda MAD (Median Absolute Deviation)
            median = np.median(self.data_buffer)
            mad = np.median([abs(x - median) for x in self.data_buffer])
            if mad == 0:
                mad = 1  # Unikanie dzielenia przez zero
            score = abs(data_point - median) / mad
            is_anomaly = score > self.threshold

        elif self.method == "isolation_forest":
            # Uproszczona symulacja Isolation Forest
            # W rzeczywistej implementacji użylibyśmy scikit-learn
            if len(self.data_buffer) >= 20:
                # Im bardziej odbiega od średniej, tym większe prawdopodobieństwo anomalii
                mean = np.mean(self.data_buffer)
                std = np.std(self.data_buffer) or 1
                z_score = abs((data_point - mean) / std)

                # Symulacja wyniku Isolation Forest
                # Wartości blisko -1 wskazują na anomalie, blisko 1 na dane normalne
                # Przekształcamy z-score do skali Isolation Forest
                score = -0.5 - 0.5 * min(1, z_score / self.threshold)
                is_anomaly = score < -0.6  # Próg dla Isolation Forest
            else:
                # Za mało danych
                is_anomaly = False
                score = 0

        # Zapisanie wyniku detekcji
        detection_result = {
            "timestamp": datetime.now(),
            "value": data_point,
            "score": score,
            "is_anomaly": is_anomaly,
            "method": self.method,
            "threshold": self.threshold
        }

        # Zapisanie anomalii, jeśli została wykryta
        if is_anomaly:
            self.detected_anomalies.append(detection_result)
            # Ograniczenie liczby przechowywanych anomalii
            if len(self.detected_anomalies) > 50:
                self.detected_anomalies.pop(0)

            self.logger.info(f"Wykryto anomalię: wartość={data_point}, wynik={score}")

        self.last_detection = datetime.now()

        return detection_result

    def get_detected_anomalies(self):
        """
        Zwraca listę wykrytych anomalii.

        Returns:
            list: Lista wykrytych anomalii
        """
        # Czyszczenie starych anomalii (starszych niż 24h)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.detected_anomalies = [a for a in self.detected_anomalies 
                                 if a["timestamp"] > cutoff_time]

        # Format danych zrozumiały dla frontendu
        formatted_anomalies = []
        for anomaly in self.detected_anomalies:
            formatted_anomalies.append({
                "timestamp": anomaly["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "value": anomaly["value"],
                "score": anomaly["score"],
                "method": anomaly["method"],
                "description": f"Anomalia wykryta metodą {anomaly['method']} z wynikiem {anomaly['score']:.2f}"
            })

        return formatted_anomalies

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
            "buffer_size": len(self.data_buffer),
            "anomalies_count": len(self.detected_anomalies),
            "last_detection": self.last_detection.strftime("%Y-%m-%d %H:%M:%S")
        }

if __name__ == "__main__":
    # Przykładowe użycie
    detector = AnomalyDetector(method="z_score", threshold=2.5)

    # Symulacja danych i detekcji
    for _ in range(100):
        # Normalne dane
        result = detector.detect(random.normalvariate(0, 1))
        if result["is_anomaly"]:
            print(f"Wykryto anomalię: {result}")

    # Wprowadzenie anomalii
    result = detector.detect(10.0)  # Wyraźna anomalia
    print(f"Test anomalii: {result}")

    # Pobranie listy anomalii
    anomalies = detector.get_detected_anomalies()
    print(f"Liczba wykrytych anomalii: {len(anomalies)}")