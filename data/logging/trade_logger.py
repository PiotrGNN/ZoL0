"""
trade_logger.py
----------------
Moduł do logowania wszystkich wykonanych transakcji (kupno, sprzedaż, stop-loss, itp.).

Funkcjonalności:
- Zapisuje informacje o czasie, cenie, wielkości pozycji, strategii oraz ID zlecenia giełdy.
- Integruje się z modułem HEAD (plik HEAD w folderze logs) umożliwiając szybkie odtwarzanie historii transakcji w razie audytu.
- Umożliwia generowanie raportów transakcyjnych w formacie CSV i JSON.
- Zapewnia skalowalność i odporność na awarie poprzez buforowanie logów oraz mechanizmy ponownego zapisu.
"""

import csv
import json
import logging
import os
import time
from datetime import datetime

# Konfiguracja logowania
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "trade_logger.log")),
        logging.StreamHandler(),
    ],
)

# Ścieżka do pliku HEAD (dla szybkiego audytu)
HEAD_FILE = os.path.join(LOG_DIR, "HEAD")


class TradeLogger:
    def __init__(self, log_buffer_size: int = 100, report_dir: str = "./reports"):
        """
        Inicjalizuje moduł TradeLogger.

        Parameters:
            log_buffer_size (int): Maksymalna liczba transakcji w buforze przed zapisem do pliku.
            report_dir (str): Folder, w którym będą zapisywane raporty transakcyjne.
        """
        self.log_buffer = []
        self.log_buffer_size = log_buffer_size
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        logging.info(
            "TradeLogger zainicjalizowany z buforem o rozmiarze %d.",
            self.log_buffer_size,
        )

    def log_trade(self, trade_info: dict):
        """
        Dodaje rekord transakcji do bufora i zapisuje do pliku, jeśli bufor osiągnie określony rozmiar.

        Parameters:
            trade_info (dict): Słownik zawierający informacje o transakcji, np.:
                {
                    "timestamp": "2023-01-01 12:00:00",
                    "symbol": "BTCUSDT",
                    "action": "BUY",
                    "quantity": 0.001,
                    "price": 30000,
                    "strategy": "breakout",
                    "order_id": 123456
                }
        """
        try:
            # Dodaj czas jeśli nie został podany
            if "timestamp" not in trade_info:
                trade_info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_buffer.append(trade_info)
            logging.info("Zarejestrowano transakcję: %s", trade_info)
            # Aktualizacja pliku HEAD
            self._update_head(trade_info["timestamp"], trade_info.get("order_id", "N/A"))
            if len(self.log_buffer) >= self.log_buffer_size:
                self.flush_logs()
        except Exception as e:
            logging.error("Błąd podczas logowania transakcji: %s", e)
            raise

    def _update_head(self, timestamp: str, order_id):
        """
        Aktualizuje plik HEAD z informacjami o ostatniej transakcji.
        """
        try:
            head_content = f"HEAD: {timestamp} - OrderID: {order_id}"
            with open(HEAD_FILE, "w") as f:
                f.write(head_content)
            logging.info("Plik HEAD zaktualizowany: %s", head_content)
        except Exception as e:
            logging.error("Błąd przy aktualizacji pliku HEAD: %s", e)
            raise

    def flush_logs(self):
        """
        Zapisuje zawartość bufora logów do pliku transakcyjnego w formacie CSV.
        W przypadku błędu podczas zapisu, bufor nie jest czyszczony.
        """
        try:
            if not self.log_buffer:
                return
            report_file = os.path.join(self.report_dir, f"trade_report_{int(time.time())}.csv")
            fieldnames = [
                "timestamp",
                "symbol",
                "action",
                "quantity",
                "price",
                "strategy",
                "order_id",
            ]
            with open(report_file, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in self.log_buffer:
                    writer.writerow(record)
            logging.info("Zapisano raport transakcji w pliku: %s", report_file)
            self.log_buffer.clear()
        except Exception as e:
            logging.error("Błąd podczas zapisu raportu transakcji: %s", e)
            # W przypadku błędu, nie czyszczymy bufora, aby móc spróbować ponownie później.
            raise

    def generate_json_report(self, output_file: str):
        """
        Generuje raport transakcyjny w formacie JSON z zapisanych transakcji w buforze.

        Parameters:
            output_file (str): Ścieżka do pliku JSON, w którym zapisany zostanie raport.
        """
        try:
            with open(output_file, "w") as f:
                json.dump(self.log_buffer, f, indent=4)
            logging.info("Raport transakcji zapisany w formacie JSON: %s", output_file)
        except Exception as e:
            logging.error("Błąd przy generowaniu JSON report: %s", e)
            raise


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        trade_logger = TradeLogger(log_buffer_size=5)

        # Przykładowe transakcje
        sample_trades = [
            {
                "symbol": "BTCUSDT",
                "action": "BUY",
                "quantity": 0.001,
                "price": 30000,
                "strategy": "breakout",
                "order_id": 101,
            },
            {
                "symbol": "ETHUSDT",
                "action": "SELL",
                "quantity": 0.01,
                "price": 2000,
                "strategy": "mean_reversion",
                "order_id": 102,
            },
            {
                "symbol": "BNBUSDT",
                "action": "BUY",
                "quantity": 0.5,
                "price": 250,
                "strategy": "trend_following",
                "order_id": 103,
            },
            {
                "symbol": "BTCUSDT",
                "action": "SELL",
                "quantity": 0.001,
                "price": 31000,
                "strategy": "stop_loss",
                "order_id": 104,
            },
            {
                "symbol": "ETHUSDT",
                "action": "BUY",
                "quantity": 0.02,
                "price": 2100,
                "strategy": "breakout",
                "order_id": 105,
            },
        ]

        for trade in sample_trades:
            trade_logger.log_trade(trade)

        # Wygenerowanie raportu JSON
        json_report_file = "./reports/trade_report.json"
        trade_logger.generate_json_report(json_report_file)
    except Exception as e:
        logging.error("Błąd w module trade_logger.py: %s", e)
        raise
