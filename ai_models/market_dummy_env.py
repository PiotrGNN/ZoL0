"""
market_dummy_env.py
-------------------
Moduł oferujący uproszczone środowisko rynkowe do szybkiego prototypowania i testowania algorytmów.
Symulowane są losowe wahania cen z uwzględnieniem spreadu, prowizji oraz potencjalnych poślizgów cenowych.
Środowisko umożliwia regulację zmienności oraz płynności rynku, co pozwala testować strategie na różnych poziomach kapitału.
Implementuje metody: reset(), step(action) i reward() z prostą obsługą logowania oraz wyjątków.
"""

import logging
import numpy as np

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

class MarketDummyEnv:
    def __init__(self, initial_capital=10000, volatility=1.0, liquidity=1000, spread=0.05, commission=0.001, slippage=0.1):
        """
        Inicjalizacja środowiska.
        
        Parameters:
            initial_capital (float): Początkowy kapitał inwestora.
            volatility (float): Parametr określający zmienność cen.
            liquidity (float): Parametr wpływający na szybkość zmian cen (niższa wartość -> mniejsza płynność).
            spread (float): Różnica między ceną kupna a sprzedaży.
            commission (float): Prowizja od transakcji (procentowo, np. 0.001 oznacza 0.1%).
            slippage (float): Potencjalny poślizg cenowy (dodatkowa korekta ceny przy realizacji transakcji).
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.volatility = volatility
        self.liquidity = liquidity
        self.spread = spread
        self.commission = commission
        self.slippage = slippage
        
        # Początkowy stan
        self.current_price = 100.0  # cena początkowa
        self.current_step = 0
        self.position = 0          # 0 - brak pozycji, 1 - long, -1 - short
        self.entry_price = None

    def reset(self):
        """
        Resetuje środowisko do stanu początkowego.
        
        Returns:
            dict: Aktualny stan środowiska.
        """
        try:
            self.current_capital = self.initial_capital
            self.current_price = 100.0
            self.current_step = 0
            self.position = 0
            self.entry_price = None
            logging.info("Reset środowiska: kapitał = %f, cena = %f, krok = %d", self.current_capital, self.current_price, self.current_step)
            return self._get_state()
        except Exception as e:
            logging.error("Błąd przy resetowaniu środowiska: %s", e)
            raise

    def _get_state(self):
        """
        Zwraca aktualny stan środowiska.
        
        Returns:
            dict: Stan zawierający bieżącą cenę, kapitał i pozycję.
        """
        return {
            'price': self.current_price,
            'capital': self.current_capital,
            'position': self.position
        }

    def step(self, action):
        """
        Wykonuje krok w środowisku na podstawie podanej akcji.
        
        Actions:
            'buy'  - otwarcie pozycji long lub zamknięcie pozycji short i otwarcie long,
            'sell' - otwarcie pozycji short lub zamknięcie pozycji long i otwarcie short,
            'hold' - brak zmiany pozycji.
        
        Proces:
            - Symulacja zmiany ceny na podstawie losowego ruchu (zależnego od zmienności i płynności).
            - Uwzględnienie spreadu, prowizji i poślizgu cenowego przy realizacji transakcji.
            - Aktualizacja stanu kapitału oraz pozycji.
        
        Returns:
            tuple: (next_state (dict), reward (float), done (bool), info (dict))
        """
        try:
            # Symulacja losowej zmiany ceny
            price_change = np.random.normal(loc=0, scale=self.volatility / self.liquidity)
            # Poślizg cenowy i spread
            slippage_effect = np.random.uniform(-self.slippage, self.slippage)
            self.current_price += price_change + slippage_effect

            reward = 0.0

            # Obsługa akcji
            if action == 'buy':
                if self.position == 0:
                    # Otwarcie pozycji long: cena zakupu z dodanym spreadem
                    effective_price = self.current_price * (1 + self.spread)
                    self.position = 1
                    self.entry_price = effective_price
                    logging.info("BUY: Otwarcie pozycji long przy cenie: %f", effective_price)
                elif self.position == -1:
                    # Zamknięcie pozycji short i otwarcie long
                    effective_price = self.current_price * (1 - self.spread)
                    reward = self.entry_price - effective_price
                    reward -= effective_price * self.commission
                    self.current_capital += reward
                    logging.info("BUY: Zamknięcie pozycji short, reward: %f, nowy kapitał: %f", reward, self.current_capital)
                    self.position = 1
                    self.entry_price = effective_price
                else:
                    logging.info("BUY: Pozycja long już otwarta. Brak działania.")
            elif action == 'sell':
                if self.position == 0:
                    # Otwarcie pozycji short: cena sprzedaży z odjętym spreadem
                    effective_price = self.current_price * (1 - self.spread)
                    self.position = -1
                    self.entry_price = effective_price
                    logging.info("SELL: Otwarcie pozycji short przy cenie: %f", effective_price)
                elif self.position == 1:
                    # Zamknięcie pozycji long i otwarcie short
                    effective_price = self.current_price * (1 + self.spread)
                    reward = effective_price - self.entry_price
                    reward -= effective_price * self.commission
                    self.current_capital += reward
                    logging.info("SELL: Zamknięcie pozycji long, reward: %f, nowy kapitał: %f", reward, self.current_capital)
                    self.position = -1
                    self.entry_price = effective_price
                else:
                    logging.info("SELL: Pozycja short już otwarta. Brak działania.")
            elif action == 'hold':
                # Brak zmiany pozycji – obliczamy niezrealizowany zysk/stratę
                if self.position == 1:
                    effective_price = self.current_price * (1 + self.spread)
                    reward = effective_price - self.entry_price
                elif self.position == -1:
                    effective_price = self.current_price * (1 - self.spread)
                    reward = self.entry_price - effective_price
                logging.info("HOLD: Pozycja utrzymywana, niezrealizowany reward: %f", reward)
            else:
                raise ValueError("Nieznana akcja: {}".format(action))

            # Aktualizacja kroku
            self.current_step += 1
            done = self.current_step >= 1000  # Przykładowy warunek zakończenia epizodu

            next_state = self._get_state()
            info = {
                'step': self.current_step,
                'action': action,
                'reward': reward,
                'capital': self.current_capital,
                'position': self.position
            }
            logging.info("Krok %d: akcja = %s, reward = %f, kapitał = %f", self.current_step, action, reward, self.current_capital)
            return next_state, reward, done, info

        except Exception as e:
            logging.error("Błąd w metodzie step: %s", e)
            raise

    def reward(self):
        """
        Oblicza niezrealizowany zysk/stratę na podstawie aktualnej pozycji.
        
        Returns:
            float: Niezrealizowany reward.
        """
        try:
            if self.position == 1:
                effective_price = self.current_price * (1 + self.spread)
                return effective_price - self.entry_price if self.entry_price else 0.0
            elif self.position == -1:
                effective_price = self.current_price * (1 - self.spread)
                return self.entry_price - effective_price if self.entry_price else 0.0
            else:
                return 0.0
        except Exception as e:
            logging.error("Błąd przy obliczaniu reward: %s", e)
            raise

# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        env = MarketDummyEnv(initial_capital=5000, volatility=2.0, liquidity=500, spread=0.02, commission=0.002, slippage=0.05)
        state = env.reset()
        done = False

        while not done:
            # Prosty przykład strategii: losowy wybór akcji
            action = np.random.choice(['buy', 'sell', 'hold'])
            state, reward, done, info = env.step(action)
            print(f"Krok: {info['step']}, Akcja: {info['action']}, Reward: {reward:.2f}, Kapitał: {info['capital']:.2f}")
    except Exception as e:
        logging.error("Błąd w symulacji środowiska: %s", e)
        raise
