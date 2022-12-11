# Source: https://github.com/marload/DeepRL-TensorFlow2/blob/master/DRQN/DRQN_Discrete.py

import random

import gym
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam

import wandb
from memory import Memory


class Agent:
    def __init__(self, env):
        self.env = env

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.states = np.zeros([8, self.state_dim])

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.gamma = 0.95

        self.optimizer = Adam(0.001)
        self.compute_loss = keras.losses.MeanSquaredError()

        self.model = keras.Sequential()
        self.model.add(Input((8, self.state_dim)))
        self.model.add(LSTM(32, activation='tanh'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(self.action_dim))

        self.memory = Memory(1000000)
        self.model.compile(self.optimizer, self.compute_loss)
        self.batch_size = 32

        self.model.set_weights(self.model.get_weights())

    def calculate_td_error(self, reward, next_state, state):
        return reward + self.gamma * np.argmax(self.model.predict(next_state)[0]) - np.argmax(
            self.model.predict(state)[0])

    def reshape(self, state):
        return np.reshape(state, [1, 8, self.state_dim])

    def memorize(self, state, action, reward, next_state, done):
        state = self.reshape(state)
        next_state = self.reshape(next_state)

        self.memory.add(self.calculate_td_error(reward, next_state, state), (state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return random.randrange(0, self.action_dim)
        else:
            state = self.reshape(state)
            q_value = self.model.predict(state)[0]
            return np.argmax(q_value)

    def calculate_target_f(self, batch):
        state, action, reward, next_state, done = batch
        target = reward + (self.gamma * np.amax(self.model.predict(next_state)[0]) if not done else 0)
        target_f = self.model.predict(state)
        target_f[0][action] = target
        return target_f

    def replay(self):
        batch, s_weight = self.memory.sample(self.batch_size)
        for i in range(len(batch)):
            state, action, reward, next_state, done = batch[i]
            self.model.fit(state, self.calculate_target_f(batch[i]), epochs=10, verbose=0, sample_weight=np.array([s_weight[i]]))

        self.update_epsilon()

    def update_states(self, next_state):
        self.states = np.roll(self.states, -1, axis=0)
        self.states[-1] = next_state

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * 0.995, 0.01)

    def train(self):
        for ep in range(300):
            self.states = np.zeros([8, self.state_dim])
            self.update_states(self.env.reset())

            done, total_reward = False, 0
            while True:
                action = self.get_action(self.states)
                next_state, reward, done, _ = self.env.step(action)

                prev_states = self.states
                self.update_states(next_state)
                self.memorize(prev_states, action, reward, self.states, done)
                total_reward += reward

                if done:
                    break

            if self.memory.tree.n_entries > 100:
                self.replay()

            self.update_epsilon()

            self.model.set_weights(self.model.get_weights())

            wandb.log({'Reward': total_reward})


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = Agent(env)
    keras.backend.set_floatx('float64')
    wandb.init(name='test-per', project="dqn-lstm-per")
    agent.train()
