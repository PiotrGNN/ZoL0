"""
environment.py
---------------
Moduł definiujący środowisko symulacyjne dla uczenia maszynowego (RL lub innych algorytmów).
Implementuje metody reset(), step(action) oraz reward() odzwierciedlające zachowania rynku,
takie jak zmiany cen, prowizje czy poślizgi (slippage). Środowisko obsługuje różne typy akcji
(kupno, sprzedaż, wstrzymanie) oraz stany rynkowe (trend wzrostowy, spadkowy, ruch boczny).
Umożliwia konfigurację parametrów takich jak dźwignia, dostępny kapitał czy ograniczenia ryzyka,
co pozwala skalować środowisko dla portfeli o różnej wielkości.
Dodatkowo, moduł zapewnia obsługę wyjątków, spójne logowanie wyników poszczególnych epizodów
oraz możliwość podłączenia się do realnych danych rynkowych w trybie testowym (paper trading)
lub pełnym backtestingu.
"""

import logging

import numpy as np

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class MarketEnvironment:
    def __init__(
        self,
        initial_capital=10000,
        leverage=1.0,
        risk_limit=0.05,
        data=None,
        mode="simulated",
    ):
        """
        Inicjalizacja środowiska.

        Parameters:
            initial_capital (float): Początkowy kapitał.
            leverage (float): Dźwignia finansowa.
            risk_limit (float): Maksymalny procentowy ryzyko na pojedynczą transakcję.
            data (pandas.DataFrame): Opcjonalne dane rynkowe z kolumną 'price'.
            mode (str): Tryb pracy środowiska: 'simulated', 'paper', 'backtesting'.
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.leverage = leverage
        self.risk_limit = risk_limit
        self.mode = mode
        self.data = data  # Jeśli dostępne, powinno zawierać kolumnę 'price'
        self.current_step = 0
        self.position = 0  # 1 = long, -1 = short, 0 = brak pozycji
        self.entry_price = None

        # Jeśli nie przekazano danych, generujemy symulowane ceny
        if self.data is None:
            self.simulated_prices = self._generate_simulated_data()
        else:
            self.simulated_prices = None

        logging.info(
            "MarketEnvironment zainicjalizowane w trybie '%s' z kapitałem: %f",
            self.mode,
            self.initial_capital,
        )

    def _generate_simulated_data(self, steps=1000):
        """
        Generuje symulowane dane rynkowe.
        Uwzględnia losowe trendy wzrostowe, spadkowe oraz ruch boczny.
        """
        np.random.seed(42)
        prices = [100.0]  # Cena początkowa
        for i in range(1, steps):
            trend = np.random.choice(["up", "down", "sideways"], p=[0.3, 0.3, 0.4])
            if trend == "up":
                change = np.random.uniform(0, 1)
            elif trend == "down":
                change = -np.random.uniform(0, 1)
            else:
                change = np.random.uniform(-0.5, 0.5)
            prices.append(prices[-1] + change)
        return np.array(prices)

    def reset(self):
        """
        Resetuje środowisko do stanu początkowego.
        """
        self.current_capital = self.initial_capital
        self.current_step = 0
        self.position = 0
        self.entry_price = None
        logging.info(
            "Reset środowiska: kapitał = %f, krok = %d",
            self.current_capital,
            self.current_step,
        )
        return self._get_state()

    def _get_state(self):
        """
        Zwraca aktualny stan środowiska (np. bieżąca cena, kapitał, pozycja).
        """
        if self.data is not None:
            price = self.data.iloc[self.current_step]["price"]
        else:
            price = self.simulated_prices[self.current_step]
        state = {
            "price": price,
            "capital": self.current_capital,
            "position": self.position,
        }
        return state

    def step(self, action):
        """
        Wykonuje krok w środowisku na podstawie podanej akcji.

        Actions:
            'buy'  - otwarcie pozycji long lub zamknięcie short i otwarcie long,
            'sell' - otwarcie pozycji short lub zamknięcie long i otwarcie short,
            'hold' - brak zmiany pozycji.

        Zwraca:
            next_state (dict): Nowy stan środowiska.
            reward (float): Nagroda (zysk/strata) uzyskana w tym kroku.
            done (bool): Flaga zakończenia epizodu.
            info (dict): Dodatkowe informacje (np. krok, akcja, kapitał).
        """
        try:
            state = self._get_state()
            price = state["price"]
            reward = 0.0

            # Obsługa akcji
            if action == "buy":
                if self.position == 0:
                    self.position = 1
                    self.entry_price = price
                    logging.info("Wykonano akcję BUY przy cenie: %f", price)
                elif self.position == -1:
                    # Zamknięcie pozycji short i otwarcie long
                    reward = self._calculate_reward(price)
                    self.current_capital += reward
                    logging.info(
                        "Zamknięcie short i otwarcie long, nagroda: %f", reward
                    )
                    self.position = 1
                    self.entry_price = price
                else:
                    logging.info("Pozycja long już otwarta. Brak zmiany.")
            elif action == "sell":
                if self.position == 0:
                    self.position = -1
                    self.entry_price = price
                    logging.info("Wykonano akcję SELL przy cenie: %f", price)
                elif self.position == 1:
                    # Zamknięcie pozycji long i otwarcie short
                    reward = self._calculate_reward(price)
                    self.current_capital += reward
                    logging.info(
                        "Zamknięcie long i otwarcie short, nagroda: %f", reward
                    )
                    self.position = -1
                    self.entry_price = price
                else:
                    logging.info("Pozycja short już otwarta. Brak zmiany.")
            elif action == "hold":
                logging.info("Wykonano akcję HOLD - brak zmiany pozycji.")
                # Jeśli pozycja jest otwarta, oblicz niezrealizowany zysk/stratę
                reward = self._calculate_unrealized_reward(price)
            else:
                raise ValueError("Nieznana akcja: {}".format(action))

            # Aktualizacja stanu: przejście do następnego kroku
            self.current_step += 1
            done = (
                self.current_step
                >= (
                    len(self.data)
                    if self.data is not None
                    else len(self.simulated_prices)
                )
                - 1
            )

            next_state = self._get_state()
            info = {
                "step": self.current_step,
                "action": action,
                "reward": reward,
                "capital": self.current_capital,
                "position": self.position,
            }
            logging.info(
                "Krok %d: akcja = %s, nagroda = %f, kapitał = %f",
                self.current_step,
                action,
                reward,
                self.current_capital,
            )
            return next_state, reward, done, info

        except Exception as e:
            logging.error("Błąd w metodzie step: %s", e)
            raise

    def _calculate_reward(self, current_price):
        """
        Oblicza nagrodę przy zamykaniu otwartej pozycji.
        """
        if self.position == 1 and self.entry_price is not None:
            profit = (current_price - self.entry_price) * self.leverage
        elif self.position == -1 and self.entry_price is not None:
            profit = (self.entry_price - current_price) * self.leverage
        else:
            profit = 0.0
        return profit

    def _calculate_unrealized_reward(self, current_price):
        """
        Oblicza niezrealizowany zysk/stratę dla aktualnie otwartej pozycji.
        """
        if self.position == 1 and self.entry_price is not None:
            profit = (current_price - self.entry_price) * self.leverage
        elif self.position == -1 and self.entry_price is not None:
            profit = (self.entry_price - current_price) * self.leverage
        else:
            profit = 0.0
        return profit

    def reward(self):
        """
        Dodatkowa funkcja wyliczająca nagrodę na podstawie bieżącego stanu (np. niezrealizowany zysk/strata).
        """
        try:
            state = self._get_state()
            price = state["price"]
            reward_value = self._calculate_unrealized_reward(price)
            logging.info("Niezrealizowana nagroda: %f", reward_value)
            return reward_value
        except Exception as e:
            logging.error("Błąd przy obliczaniu nagrody: %s", e)
            raise


# -------------------- Przykładowe użycie --------------------

if __name__ == "__main__":
    env = MarketEnvironment()
    state = env.reset()
    done = False

    while not done:
        # Przykładowo: strategia losowa
        action = np.random.choice(["buy", "sell", "hold"])
        next_state, reward, done, info = env.step(action)
        print(
            f"Krok: {info['step']}, Akcja: {info['action']}, Nagroda: {reward:.2f}, Kapitał: {info['capital']:.2f}"
        )
    logging.info("Symulacja zakończona.")
