import numpy as np
import torch
import random
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import ReplayBuffer
from model.res_fcn import ResFCN


class DDQN_agent:
    def __init__(self, observation_space, action_space, batch_size=64, gamma=0.99,
                 exploration_rate=1.0, replacement_frequency=15, learning_rate=1e-4):
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.replacement_frequency = replacement_frequency
        self.learning_rate = learning_rate
        self.resfcn = ResFCN

        # Initialize the Behavior Network and the Target Network
        self.model = DQN("MlpPolicy", (32,), action_space, verbose=1,
                         learning_rate=learning_rate, buffer_size=10000, gamma=gamma)
        self.target_model = DQN("MlpPolicy", (32,), action_space, verbose=0,
                                learning_rate=learning_rate, buffer_size=10000, gamma=gamma)
        self.target_model.load_state_dict(self.model.state_dict())
        self.replay_buffer = ReplayBuffer(self.model.replay_buffer_size)

    def select_action(self, state):
        processed_state = self.resfcn(state)
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_space.n)
        else:
            action_probs = self.model.predict(processed_state)
            return np.argmax(action_probs[0])

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store((state, action, reward, next_state, done))

    def sample_batch(self):
        minibatch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = minibatch
        return states, actions, rewards, next_states, dones

    def compute_target(self, minibatch):
        states, actions, rewards, next_states, dones = minibatch
        next_state_actions = self.target_model.predict(next_states)
        next_state_max_q_values = np.max(next_state_actions, axis=1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_state_max_q_values
        return target_q_values

    def train(self, iteration_times):
        for iteration in range(iteration_times):
            state = np.zeros(self.observation_space.shape)
            for j in range(self.action_space.n):
                action = self.select_action(state)
                next_state, reward, done, _ = self.model.env.step(action)
                self.store_transition(state, action, reward, next_state, done)
                state = next_state

                if done:
                    break

            minibatch = self.sample_batch()
            states, actions, rewards, next_states, dones = minibatch
            target_outputs = self.compute_target(minibatch)    # TODO:not used
            self.model.learn(states, actions, rewards, next_states, dones)

            if iteration % self.replacement_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # Update exploration rate
            self.exploration_rate = max(0.1, 1 - iteration / 1000)