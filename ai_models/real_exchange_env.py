"""
real_exchange_env.py
--------------------
Moduł odzwierciedlający rzeczywiste warunki giełdowe.

Funkcjonalności:
- Pobieranie danych z API giełdy w czasie rzeczywistym.
- Zawieranie transakcji: kupno, sprzedaż, obsługa pozycji.
- Obsługa prowizji, poślizgu cenowego, margin call oraz dźwigni.
- Zapis transakcji do systemu logów dla audytu.
- Skalowalność dla dużych wolumenów handlowych.
"""

import logging
import requests
import time
import random
from datetime import datetime

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler("real_exchange_env.log"),
                              logging.StreamHandler()])

class RealExchangeEnv:
    def __init__(self, api_url, api_key, initial_capital=10000, leverage=1.0, margin_call_threshold=0.2,
                 commission_rate=0.001, slippage_range=0.05):
        """
        Inicjalizacja środowiska rzeczywistego.

        Parameters:
            api_url (str): URL API giełdy.
            api_key (str): Klucz API do autoryzacji.
            initial_capital (float): Początkowy kapitał.
            leverage (float): Maksymalna dźwignia.
            margin_call_threshold (float): Procentowa strata, po której następuje margin call.
            commission_rate (float): Prowizja transakcyjna.
            slippage_range (float): Maksymalny procent poślizgu cenowego.
        """
        self.api_url = api_url
        self.api_key = api_key
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.leverage = leverage
        self.margin_call_threshold = margin_call_threshold
        self.commission_rate = commission_rate
        self.slippage_range = slippage_range

        self.current_price = None
        self.position = 0  # 1 = long, -1 = short, 0 = brak pozycji
        self.entry_price = None
        self.transaction_log_file = "transactions.log"
        self.load_initial_price()

    def load_initial_price(self):
        """Pobiera początkową cenę z API lub ustawia wartość domyślną."""
        try:
            self.current_price = self.get_real_time_price()
        except Exception as e:
            logging.error("Nie udało się pobrać ceny przy starcie: %s", e)
            self.current_price = 100.0  # Domyślna cena

    def get_real_time_price(self):
        """
        Pobiera cenę w czasie rzeczywistym z API giełdowego.
        
        Returns:
            float: Aktualna cena.
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(self.api_url, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            price = float(data.get("price"))
            return price
        except Exception as e:
            logging.error("Błąd pobierania ceny z API: %s", e)
            fallback_price = self.current_price or 100.0
            return fallback_price * (1 + random.uniform(-0.01, 0.01))  # Symulowany ruch ceny

    def reset(self):
        """Resetuje środowisko do stanu początkowego."""
        self.current_capital = self.initial_capital
        self.position = 0
        self.entry_price = None
        self.current_price = self.get_real_time_price()
        logging.info("Reset środowiska: Kapitał=%.2f, Cena=%.2f", self.current_capital, self.current_price)
        return self._get_state()

    def _get_state(self):
        """Zwraca bieżący stan środowiska."""
        return {
            "price": self.current_price,
            "capital": self.current_capital,
            "position": self.position
        }

    def step(self, action):
        """
        Wykonuje krok w środowisku na podstawie podanej akcji.

        Actions:
            'buy'  - kupno (long),
            'sell' - sprzedaż (short),
            'hold' - brak zmiany pozycji.

        Returns:
            tuple: (next_state, reward, done, info)
        """
        try:
            price = self.get_real_time_price()
            self.current_price = price

            # Symulacja poślizgu cenowego
            slippage = price * random.uniform(-self.slippage_range, self.slippage_range)
            effective_price = price + slippage

            reward = 0.0

            if action == "buy":
                if self.position == 0:
                    self.position = 1
                    self.entry_price = effective_price
                    logging.info("BUY: Otwarta pozycja long @ %.2f", effective_price)
                elif self.position == -1:
                    reward = (self.entry_price - effective_price) * self.leverage - (effective_price * self.commission_rate)
                    self.current_capital += reward
                    logging.info("BUY: Zamknięcie short, otwarcie long, reward=%.2f", reward)
                    self.position = 1
                    self.entry_price = effective_price
            elif action == "sell":
                if self.position == 0:
                    self.position = -1
                    self.entry_price = effective_price
                    logging.info("SELL: Otwarta pozycja short @ %.2f", effective_price)
                elif self.position == 1:
                    reward = (effective_price - self.entry_price) * self.leverage - (effective_price * self.commission_rate)
                    self.current_capital += reward
                    logging.info("SELL: Zamknięcie long, otwarcie short, reward=%.2f", reward)
                    self.position = -1
                    self.entry_price = effective_price
            elif action == "hold":
                if self.position == 1:
                    reward = (effective_price - self.entry_price) * self.leverage
                elif self.position == -1:
                    reward = (self.entry_price - effective_price) * self.leverage
                logging.info("HOLD: Pozycja utrzymana, reward=%.2f", reward)
            else:
                raise ValueError(f"Nieznana akcja: {action}")

            # Sprawdzenie margin call
            if self.position != 0:
                unrealized_loss = (self.entry_price - effective_price) * self.leverage if self.position == 1 else (effective_price - self.entry_price) * self.leverage
                if (unrealized_loss / self.initial_capital) > self.margin_call_threshold:
                    logging.warning("Margin Call! Zamknięcie pozycji.")
                    reward -= unrealized_loss
                    self.current_capital -= unrealized_loss
                    self.position = 0
                    self.entry_price = None

            self.log_transaction(action, effective_price, reward)

            done = self.current_capital <= 0
            return self._get_state(), reward, done, {"action": action, "effective_price": effective_price, "capital": self.current_capital}

        except Exception as e:
            logging.error("Błąd w metodzie step: %s", e)
            raise

    def log_transaction(self, action, price, reward):
        """Zapisuje transakcję do logów."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} | {action.upper()} | Price: {price:.2f} | Reward: {reward:.2f} | Capital: {self.current_capital:.2f}\n"
            with open(self.transaction_log_file, "a") as f:
                f.write(log_entry)
            logging.info("Transakcja zapisana: %s", log_entry.strip())
        except Exception as e:
            logging.error("Błąd zapisu transakcji: %s", e)

# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    env = RealExchangeEnv(api_url="https://api.exchange.example.com/price", api_key="your_api_key_here")
    state = env.reset()
    for _ in range(50):
        action = random.choice(["buy", "sell", "hold"])
        state, reward, done, info = env.step(action)
        print(f"Akcja: {action}, Reward: {reward:.2f}, Kapitał: {info['capital']:.2f}")
        if done:
            break
        time.sleep(1)
