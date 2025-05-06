"""
Reinforcement Learning implementation using DQN with experience replay
and target networks for trading.
"""

import tensorflow as tf
import numpy as np
from collections import deque
import random
import logging
import os
from typing import Any

# Use tf.keras instead of standalone imports
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
BatchNormalization = tf.keras.layers.BatchNormalization
Adam = tf.keras.optimizers.Adam
clone_model = tf.keras.models.clone_model

class DQNAgent:
    """
    DQN Agent with experience replay and target network for trading.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        update_target_every: int = 100
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.target_update_counter = 0
        
        # Create memory buffer for experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Create main model and target model
        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        
        self.logger = logging.getLogger(__name__)

    def _build_model(self) -> tf.keras.Sequential:
        """Builds a neural network model for Q-learning"""
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model

    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """Stores experience in memory for replay"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Selects action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
            
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size: int) -> float:
        """Trains on a batch of experiences"""
        if len(self.memory) < batch_size:
            return 0.0

        # Sample batch of experiences
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare training data
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        # Predict Q-values
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Update Q-values for actions taken
        for i in range(batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + \
                    self.gamma * np.amax(next_q_values[i])

        # Train the model
        history = self.model.fit(
            states, current_q_values, 
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )

        # Update epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network if needed
        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            self.logger.info("Target network updated")

        return history.history['loss'][0]

    def save(self, filepath: str) -> bool:
        """Saves the model and training state"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save models
            self.model.save(f"{filepath}_main")
            self.target_model.save(f"{filepath}_target")
            
            # Save training state
            training_state = {
                'epsilon': self.epsilon,
                'target_update_counter': self.target_update_counter
            }
            np.save(f"{filepath}_state.npy", training_state)
            
            self.logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False

    def load(self, filepath: str) -> bool:
        """Loads the model and training state"""
        try:
            # Load models
            self.model = tf.keras.models.load_model(f"{filepath}_main")
            self.target_model = tf.keras.models.load_model(f"{filepath}_target")
            
            # Load training state
            training_state = np.load(f"{filepath}_state.npy", allow_pickle=True).item()
            self.epsilon = training_state['epsilon']
            self.target_update_counter = training_state['target_update_counter']
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def get_stats(self) -> dict:
        """Returns current training statistics"""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'model_summary': self.model.summary(),
            'target_update_counter': self.target_update_counter
        }

# Example usage
if __name__ == "__main__":
    # Initialize agent
    state_size = 4  # [price, position, capital, unrealized_pnl]
    action_size = 3  # [hold, buy, sell]
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Training loop example
    episodes = 100
    max_steps = 1000
    
    for episode in range(episodes):
        state = np.random.random(state_size)  # Example initial state
        total_reward = 0
        
        for step in range(max_steps):
            # Get action
            action = agent.act(state)
            
            # Example environment step (replace with actual environment)
            next_state = np.random.random(state_size)
            reward = np.random.random() - 0.5
            done = False
            
            # Store experience and train
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay(agent.batch_size)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
