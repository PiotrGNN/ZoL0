"""
reinforcement_learning.py
-------------------------
Moduł implementujący algorytmy uczenia ze wzmocnieniem (RL) dla tradingu.
Obsługuje Deep Q-Network (DQN) z technikami:
- Epsilon-greedy exploration i dynamicznym zanikiem epsilon.
- Stabilizacją uczenia przez target network i experience replay.
- Obsługą wielu GPU (jeśli dostępne) poprzez tf.distribute.
- Integracją ze środowiskiem symulowanym (environment.py) i rzeczywistym (real_exchange_env.py).
"""

import logging
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import os

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Sprawdzenie dostępności GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    logging.info(f"Wykryto {len(gpus)} GPU, włączam strategię MirroredStrategy")
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = None


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        discount_factor=0.99,
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=100000,
        target_update_freq=5,
    ):
        """
        Inicjalizacja agenta DQN.

        Parameters:
            state_size (int): Rozmiar wektora stanu.
            action_size (int): Liczba możliwych akcji.
            discount_factor (float): Współczynnik dyskontowy (gamma).
            learning_rate (float): Wskaźnik uczenia.
            epsilon (float): Początkowy współczynnik eksploracji.
            epsilon_min (float): Minimalny epsilon (granica eksploracji).
            epsilon_decay (float): Współczynnik zanikania epsilon.
            batch_size (int): Rozmiar mini-batch dla experience replay.
            memory_size (int): Maksymalna wielkość pamięci replay.
            target_update_freq (int): Co ile epizodów aktualizować target network.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.target_update_freq = target_update_freq
        self.train_step = 0

        # Budowanie modeli
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        """
        Buduje sieć neuronową reprezentującą funkcję Q.

        Returns:
            tf.keras.Model: Model sieci.
        """
        try:
            with strategy.scope() if strategy else tf.device("/CPU:0"):
                model = Sequential()
                model.add(Input(shape=(self.state_size,)))
                model.add(Dense(128, activation="relu"))
                model.add(Dense(128, activation="relu"))
                model.add(Dense(self.action_size, activation="linear"))
                model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")
        except Exception as e:
            logging.error(f"Błąd podczas budowania modelu Sequential: {e}")
            # Fallback do prostszego modelu w razie błędu
            model = Sequential()
            model.add(Dense(64, input_shape=(self.state_size,), activation="relu"))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(self.action_size, activation="linear"))
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")

        logging.info("Model sieci Q został zbudowany.")
        return model

    def update_target_model(self):
        """Aktualizuje wagi modelu docelowego."""
        # Upewnienie się, że model źródłowy jest skompilowany
        if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
            self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")
            logging.info("Skompilowano model źródłowy")

        self.target_model.set_weights(self.model.get_weights())

        # Upewnienie się, że model docelowy jest skompilowany
        if not hasattr(self.target_model, 'optimizer') or self.target_model.optimizer is None:
            self.target_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")
            logging.info("Skompilowano model docelowy")

        logging.info("Wagi modelu docelowego zaktualizowane.")

    def remember(self, state, action, reward, next_state, done):
        """Dodaje doświadczenie do pamięci replay."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Wybiera akcję na podstawie epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            logging.debug("Losowa akcja: %d", action)
            return action
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        action = np.argmax(q_values)
        logging.debug("Wybrana akcja: %d", action)
        return action

    def replay(self):
        """Trenuje model na podstawie doświadczeń z pamięci replay."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.array([state]), verbose=0)[0]
            if done:
                target[action] = reward
            else:
                t = self.target_model.predict(np.array([next_state]), verbose=0)[0]
                target[action] = reward + self.discount_factor * np.amax(t)
            states.append(state)
            targets.append(target)

        self.model.train_on_batch(np.array(states), np.array(targets))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_model()

    def save(self, name):
        """Zapisuje model do pliku."""
        self.model.save(name)
        logging.info("Model zapisany do %s", name)

    def load(self, name):
        """Ładuje model z pliku."""
        self.model = tf.keras.models.load_model(name)
        self.update_target_model()
        logging.info("Model załadowany z %s", name)


# -------------------- Trening --------------------


def train_dqn(agent, env, episodes=1000, max_steps=500):
    """
    Trenuje agenta DQN w danym środowisku.

    Parameters:
        agent (DQNAgent): Instancja agenta.
        env: Środowisko (symulowane lub realne) z metodami reset() i step(action).
        episodes (int): Liczba epizodów.
        max_steps (int): Maksymalna liczba kroków w epizodzie.
    """
    for e in range(episodes):
        state = env.reset()
        state = np.array(list(state.values()))
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.array(list(next_state.values()))

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()

            if done:
                break

        logging.info(
            "Epizod %d: Reward: %.2f, Epsilon: %.4f", e + 1, total_reward, agent.epsilon
        )

    # Sprawdź, czy model ma warstwy przed zapisem
    if len(agent.model.layers) == 0:
        logging.warning("Model nie ma warstw, dodaję podstawowe warstwy przed zapisem")
        # Dodajemy minimalne warstwy, by model był poprawny
        agent.model.add(Dense(64, input_shape=(agent.state_size,), activation="relu"))
        agent.model.add(Dense(64, activation="relu"))
        agent.model.add(Dense(agent.action_size, activation="linear"))
        agent.model.compile(optimizer=Adam(learning_rate=agent.learning_rate), loss="mse")

    agent.save("dqn_trained_model.h5")


# -------------------- Przykładowe użycie --------------------

if __name__ == "__main__":
    try:

        class DummyEnv:
            def reset(self):
                return {"price": 100.0, "capital": 10000, "position": 0}

            def step(self, action):
                reward = np.random.uniform(-1, 1)
                return (
                    {"price": 100.0, "capital": 10000, "position": 0},
                    reward,
                    False,
                    {},
                )

        env = DummyEnv()
        agent = DQNAgent(state_size=3, action_size=3)
        train_dqn(agent, env, episodes=50, max_steps=100)
    except Exception as e:
        logging.error("Błąd: %s", e)
        raise

class ReinforcementLearner:
    """
    Klasa implementująca uczenie przez wzmacnianie z użyciem sieci neuronowej.
    """

    def __init__(self, state_size=10, action_size=3, learning_rate=0.001, gamma=0.95, epsilon=1.0):
        """
        Inicjalizacja modelu uczenia przez wzmacnianie.

        Args:
            state_size: Wymiar przestrzeni stanów
            action_size: Liczba możliwych akcji
            learning_rate: Współczynnik uczenia
            gamma: Współczynnik dyskontowania nagród
            epsilon: Prawdopodobieństwo eksploracji
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = []
        self.model = self._build_model()
        self.model_path = "models/reinforcement_learner_model.pkl"

        # Spróbuj załadować zapisany model, jeśli istnieje
        self._try_load_model()

    def _build_model(self):
        """
        Buduje model sieci neuronowej.

        Returns:
            Model Sequential
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.optimizers import Adam

            model = Sequential()
            model.add(Dense(24, input_dim=self.state_size, activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))

            # Automatyczna kompilacja modelu
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['accuracy']
            )
            logging.info("Model Sequential zbudowany i skompilowany")
            return model
        except ImportError:
            print("TensorFlow nie jest zainstalowany. Używanie pustego modelu.")
            class DummyModel:
                def predict(self, state):
                    import numpy as np
                    return np.random.random((1, 3))
                def fit(self, states, target, epochs=1, verbose=0):
                    pass
            return DummyModel()

    def _try_load_model(self):
        """
        Próbuje załadować wcześniej zapisany model.
        """
        import os
        import pickle
        import logging

        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)

                if 'model' in model_data and 'metadata' in model_data:
                    # Jeśli model ma zdefiniowane warstwy, użyj go
                    if hasattr(model_data['model'], 'layers') and len(model_data['model'].layers) > 0:
                        self.model = model_data['model']
                        metadata = model_data['metadata']

                        # Aktualizuj parametry na podstawie metadanych
                        if 'epsilon' in metadata:
                            self.epsilon = metadata['epsilon']
                        if 'gamma' in metadata:
                            self.gamma = metadata['gamma']
                        if 'learning_rate' in metadata:
                            self.learning_rate = metadata['learning_rate']

                        logging.info(f"Załadowano zapisany model ReinforcementLearner z {self.model_path}")
                    else:
                        logging.warning("Zapisany model nie ma warstw, tworzę nowy model")
            except Exception as e:
                logging.error(f"Błąd podczas ładowania modelu ReinforcementLearner: {e}")

    def save_model(self, force=False):
        """
        Zapisuje model wraz z metadanymi.

        Args:
            force: Czy wymusić zapisanie modelu nawet jeśli nie ma warstw
        """
        import os
        import pickle
        import logging
        import datetime

        try:
            # Sprawdź czy model ma warstwy (czy jest to model Sequential)
            if hasattr(self.model, 'layers') and (len(self.model.layers) > 0 or force):
                # Upewnij się, że katalog istnieje
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

                # Przygotuj metadane
                metadata = {
                    'state_size': self.state_size,
                    'action_size': self.action_size,
                    'epsilon': self.epsilon,
                    'gamma': self.gamma,
                    'learning_rate': self.learning_rate,
                    'save_date': datetime.datetime.now().isoformat(),
                    'model_type': 'ReinforcementLearner'
                }

                # Przygotuj dane do zapisu
                model_data = {
                    'model': self.model,
                    'metadata': metadata
                }

                # Zapisz model z metadanymi
                with open(self.model_path, 'wb') as f:
                    pickle.dump(model_data, f)

                logging.info(f"Model ReinforcementLearner zapisany do {self.model_path}")
                return True
            else:
                logging.warning("Model nie ma warstw, nie można zapisać")
                return False
        except Exception as e:
            logging.error(f"Błąd podczas zapisywania modelu ReinforcementLearner: {e}")
            return False

    def remember(self, state, action, reward, next_state, done):
        """
        Zapisuje doświadczenie w pamięci.

        Args:
            state: Obecny stan
            action: Wykonana akcja
            reward: Otrzymana nagroda
            next_state: Następny stan
            done: Czy epizod się zakończył
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Wybiera akcję na podstawie stanu.

        Args:
            state: Obecny stan

        Returns:
            int: Wybrana akcja
        """
        import numpy as np

        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)

        act_values = self.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size=32):
        """
        Trenuje model na podstawie doświadczeń.

        Args:
            batch_size: Rozmiar partii danych

        Returns:
            float: Wartość funkcji straty
        """
        import numpy as np
        import logging

        if len(self.memory) < batch_size:
            return 0

        minibatch = np.random.choice(self.memory, batch_size, replace=False)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.predict(next_states), axis=1)) * (1 - dones)
        targets_full = self.predict(states)

        ind = np.array([i for i in range(batch_size)])
        targets_full[ind, actions] = targets

        history = self.fit(states, targets_full, epochs=1, verbose=0)
        loss = history.history['loss'][0] if hasattr(history, 'history') else 0

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Po każdym treningu, zapisz model
        self.save_model()

        return loss

    def predict(self, state):
        """
        Przewiduje wartości Q dla danego stanu.

        Args:
            state: Obecny stan

        Returns:
            numpy.ndarray: Przewidywane wartości Q
        """
        import numpy as np
        import logging

        # Rozszerz wymiar, jeśli potrzeba
        if len(np.array(state).shape) == 1:
            state = np.array(state).reshape(1, -1)

        # Sprawdź czy model jest inicjalizowany
        if not hasattr(self.model, 'predict'):
            logging.error("Model nie ma metody predict, tworzę nowy model")
            self.model = self._build_model()

        # Sprawdź czy model ma warstwy (dla Sequential)
        if hasattr(self.model, 'layers') and not self.model.layers:
            logging.error("Model Sequential nie ma warstw, tworzę nowy model")
            self.model = self._build_model()

        # Sprawdź czy model jest skompilowany (dla Sequential)
        if hasattr(self.model, '_is_compiled') and not self.model._is_compiled:
            logging.warning("Model Sequential nie jest skompilowany, kompiluję model")
            try:
                from tensorflow.keras.optimizers import Adam
                self.model.compile(
                    optimizer=Adam(learning_rate=self.learning_rate), 
                    loss='mse',
                    metrics=['accuracy']
                )
                logging.info("Model Sequential został pomyślnie skompilowany")
            except Exception as e:
                logging.error(f"Błąd podczas kompilacji modelu: {e}")

        # Dopasuj wymiary danych wejściowych
        if hasattr(self.model, 'input_shape') and state.shape[1] != self.model.input_shape[1]:
            logging.warning(f"Wymiary danych wejściowych nie pasują: {state.shape[1]} != {self.model.input_shape[1]}")
            if state.shape[1] < self.model.input_shape[1]:
                # Dodaj brakujące kolumny z zerami
                padding = np.zeros((state.shape[0], self.model.input_shape[1] - state.shape[1]))
                state = np.hstack((state, padding))
            else:
                # Ogranicz liczbę kolumn
                state = state[:, :self.model.input_shape[1]]

        return self.model.predict(state)

    def fit(self, X, y, epochs=1, verbose=0):
        """
        Trenuje model na danych.

        Args:
            X: Dane wejściowe
            y: Dane wyjściowe
            epochs: Liczba epok
            verbose: Poziom informacji o treningu

        Returns:
            history: Historia treningu
        """
        import logging

        # Sprawdź czy model jest skompilowany (dla Sequential)
        if hasattr(self.model, '_is_compiled') and not self.model._is_compiled:
            logging.warning("Model Sequential nie jest skompilowany, kompiluję model")
            try:
                from tensorflow.keras.optimizers import Adam
                self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            except Exception as e:
                logging.error(f"Błąd podczas kompilacji modelu: {e}")

        history = self.model.fit(X, y, epochs=epochs, verbose=verbose)

        # Po treningu zapisz model
        self.save_model()

        return history