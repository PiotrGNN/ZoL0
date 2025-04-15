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
# # # # # # import tensorflow as tf  # Zakomentowano - opcjonalny pakiet  # Zakomentowano - opcjonalny pakiet  # Zakomentowano - opcjonalny pakiet  # Zakomentowano - opcjonalny pakiet
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

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
        self.target_model.set_weights(self.model.get_weights())
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
ReinforcementLearner = DQNAgent
