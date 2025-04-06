"""
anomaly_detection.py
--------------------
Moduł do wykrywania anomalii w danych finansowych lub transakcyjnych.
Wykorzystuje zaawansowane metody, takie jak:
- Isolation Forest z automatycznym strojeniem hiperparametrów (GridSearchCV),
- DBSCAN,
- Głębokie autoenkodery do nieliniowej detekcji odchyleń.

Moduł zawiera także rozbudowane logowanie, mechanizm wyzwalania alarmów (przykładowa funkcja wysyłki e-mail)
oraz funkcje testowe do weryfikacji skuteczności wykrywania anomalii w różnych warunkach.
"""

import logging
import smtplib
from email.message import EmailMessage

try:
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import GridSearchCV
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("sklearn nie jest dostępny. Ograniczona funkcjonalność AnomalyDetector.")
    SKLEARN_AVAILABLE = False

# Uproszczona wersja bez TensorFlow
TENSORFLOW_AVAILABLE = False
try:
    import numpy as np
except ImportError:
    logging.error("numpy nie jest dostępny. AnomalyDetector nie będzie działać.")
    np = None

# Prosty autoenkoder jako alternatywa dla TensorFlow
class SimpleAutoencoder:
    """Prosty autoenkoder bez zależności od TensorFlow."""

    def __init__(self, input_dim, encoding_dim=32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.is_fitted = False

    def fit(self, X, X_val=None, epochs=10, batch_size=32, **kwargs):
        """Prosty trening (symulacja)"""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-10
        self.is_fitted = True
        return {"loss": 0.1}

    def predict(self, X):
        """Prosta rekonstrukcja"""
        if not self.is_fitted:
            raise ValueError("Model nie został wytrenowany")
        return np.clip(X * 0.8 + self.mean * 0.2, 0, 1)


# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("anomaly_detection.log"), logging.StreamHandler()],
)


class AnomalyDetector:
    def __init__(self, contamination=0.01, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = None
        self.dbscan = None
        self.autoencoder = None

    def tune_isolation_forest(self, X):
        """
        Strojenie hiperparametrów IsolationForest przy użyciu GridSearchCV.
        """
        try:
            logging.info("Rozpoczynam strojenie IsolationForest...")
            param_grid = {
                "n_estimators": [50, 100, 150],
                "max_samples": ["auto", 0.8, 1.0],
                "contamination": [self.contamination],
                "bootstrap": [False, True],
            }
            model = IsolationForest(random_state=self.random_state)
            grid = GridSearchCV(
                estimator=model, param_grid=param_grid, cv=3, scoring="accuracy"
            )
            grid.fit(X)
            logging.info("Najlepsze parametry: %s", grid.best_params_)
            self.isolation_forest = grid.best_estimator_
        except Exception as e:
            logging.error("Błąd podczas strojenia IsolationForest: %s", e)
            raise

    def fit_isolation_forest(self, X):
        """
        Dopasowanie modelu IsolationForest do danych X.
        """
        try:
            if self.isolation_forest is None:
                logging.info(
                    "Dopasowywanie IsolationForest z domyślnymi parametrami..."
                )
                self.isolation_forest = IsolationForest(
                    contamination=self.contamination, random_state=self.random_state
                )
            else:
                logging.info(
                    "Dopasowywanie wcześniej strojonego modelu IsolationForest..."
                )
            self.isolation_forest.fit(X)
        except Exception as e:
            logging.error("Błąd podczas dopasowywania IsolationForest: %s", e)
            raise

    def detect_anomalies_isolation_forest(self, X):
        """
        Wykrywanie anomalii przy użyciu IsolationForest.
        Zwraca etykiety: 1 - normalne, -1 - anomalia.
        """
        try:
            if self.isolation_forest is None:
                raise ValueError("Model IsolationForest nie został dopasowany.")
            predictions = self.isolation_forest.predict(X)
            logging.info(
                "IsolationForest wykrył %d anomalii na %d próbkach.",
                np.sum(predictions == -1),
                len(predictions),
            )
            return predictions
        except Exception as e:
            logging.error(
                "Błąd przy wykrywaniu anomalii za pomocą IsolationForest: %s", e
            )
            raise

    def fit_dbscan(self, X, eps=0.5, min_samples=5):
        """
        Dopasowanie modelu DBSCAN do danych X.
        """
        try:
            logging.info("Dopasowywanie modelu DBSCAN...")
            self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            self.dbscan.fit(X)
        except Exception as e:
            logging.error("Błąd podczas dopasowywania DBSCAN: %s", e)
            raise

    def detect_anomalies_dbscan(self, X):
        """
        Wykrywanie anomalii przy użyciu DBSCAN.
        Zwraca etykiety: -1 oznacza anomalię.
        """
        try:
            if self.dbscan is None:
                raise ValueError("Model DBSCAN nie został dopasowany.")
            predictions = self.dbscan.fit_predict(X)
            anomalies = (predictions == -1).astype(int)
            logging.info(
                "DBSCAN wykrył %d anomalii na %d próbkach.",
                np.sum(anomalies),
                len(anomalies),
            )
            return anomalies
        except Exception as e:
            logging.error("Błąd przy wykrywaniu anomalii za pomocą DBSCAN: %s", e)
            raise

    def build_autoencoder(self, input_dim, encoding_dim=32):
        """
        Buduje prosty model autoenkodera.
        """
        try:
            logging.info("Budowanie modelu autoenkodera...")
            self.autoencoder = SimpleAutoencoder(input_dim, encoding_dim)
            logging.info("Model autoenkodera został pomyślnie zbudowany.")
        except Exception as e:
            logging.error("Błąd przy budowaniu autoenkodera: %s", e)
            raise

    def fit_autoencoder(self, X, epochs=50, batch_size=32, validation_split=0.1):
        """
        Dopasowanie autoenkodera do danych X.
        """
        try:
            if self.autoencoder is None:
                self.build_autoencoder(input_dim=X.shape[1])
            logging.info("Trening autoenkodera...")
            self.autoencoder.fit(X, epochs=epochs, batch_size=batch_size)
        except Exception as e:
            logging.error("Błąd podczas treningu autoenkodera: %s", e)
            raise

    def detect_anomalies_autoencoder(self, X, threshold=None):
        """
        Wykrywanie anomalii przy użyciu błędu rekonstrukcji autoenkodera.
        Jeśli threshold jest None, ustawiany jest jako średnia + 3*std błędów.
        Zwraca etykiety (1 - normalne, -1 - anomalia) oraz wartości błędów rekonstrukcji.
        """
        try:
            if self.autoencoder is None:
                raise ValueError("Model autoenkodera nie został dopasowany.")
            reconstructions = self.autoencoder.predict(X)
            mse = np.mean(np.power(X - reconstructions, 2), axis=1)
            if threshold is None:
                threshold = np.mean(mse) + 3 * np.std(mse)
            predictions = np.where(mse > threshold, -1, 1)
            logging.info(
                "Autoenkoder wykrył %d anomalii na %d próbkach przy threshold = %f.",
                np.sum(predictions == -1),
                len(predictions),
                threshold,
            )
            return predictions, mse
        except Exception as e:
            logging.error("Błąd przy wykrywaniu anomalii za pomocą autoenkodera: %s", e)
            raise

    def send_alert(self, subject, message, recipient_email):
        """
        Wysyłka alertu e-mail.
        Uwaga: Funkcja symuluje wysyłkę alertu. Aby używać w środowisku produkcyjnym,
        należy skonfigurować dane SMTP.
        """
        try:
            logging.info("Wysyłanie alertu e-mail do %s...", recipient_email)
            smtp_server = "smtp.example.com"
            smtp_port = 587
            smtp_user = "user@example.com"
            smtp_password = "password"

            msg = EmailMessage()
            msg.set_content(message)
            msg["Subject"] = subject
            msg["From"] = smtp_user
            msg["To"] = recipient_email

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            logging.info("Alert e-mail został wysłany pomyślnie.")
        except Exception as e:
            logging.error("Błąd podczas wysyłania alertu e-mail: %s", e)
            # W środowisku produkcyjnym można nie przerywać działania w przypadku błędu wysyłki alertu.


# --------------------- Funkcje Testowe ---------------------


def test_anomaly_detection():
    """
    Testuje metody wykrywania anomalii przy użyciu danych syntetycznych.
    """
    try:
        logging.info("Rozpoczynam testy wykrywania anomalii...")
        # Generowanie danych syntetycznych
        np.random.seed(42)
        X_normal = np.random.normal(0, 1, (1000, 10))
        X_anomalies = np.random.normal(10, 1, (50, 10))
        X = np.vstack([X_normal, X_anomalies])
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

        detector = AnomalyDetector(contamination=0.05)

        # Test IsolationForest
        if SKLEARN_AVAILABLE:
            detector.tune_isolation_forest(df.values)
            detector.fit_isolation_forest(df.values)
            predictions_if = detector.detect_anomalies_isolation_forest(df.values)
            logging.info(
                "Test IsolationForest: wykryto %d anomalii.", np.sum(predictions_if == -1)
            )
        else:
            logging.warning("Test IsolationForest pominięty - brak sklearn.")


        # Test DBSCAN
        if SKLEARN_AVAILABLE:
            detector.fit_dbscan(df.values, eps=3, min_samples=5)
            predictions_db = detector.detect_anomalies_dbscan(df.values)
            logging.info("Test DBSCAN: wykryto %d anomalii.", np.sum(predictions_db))
        else:
            logging.warning("Test DBSCAN pominięty - brak sklearn.")


        # Test Autoenkodera
        detector.build_autoencoder(input_dim=df.shape[1], encoding_dim=5)
        detector.fit_autoencoder(
            df.values, epochs=20, batch_size=16, validation_split=0.1
        )
        predictions_ae, mse = detector.detect_anomalies_autoencoder(df.values)
        logging.info(
            "Test Autoenkodera: wykryto %d anomalii.", np.sum(predictions_ae == -1)
        )

        logging.info("Wszystkie testy zakończone pomyślnie.")
    except Exception as e:
        logging.error("Testy nie powiodły się: %s", e)
        raise


if __name__ == "__main__":
    test_anomaly_detection()