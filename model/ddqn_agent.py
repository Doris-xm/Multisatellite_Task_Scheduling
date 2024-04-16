import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
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
        self.resfcn = ResFCN(observation_space.shape[0])
        env = gym.make('CartPole-v1')
        self.env = DummyVecEnv([lambda: env])

        # Initialize the Behavior Network and the Target Network
        self.model = DQN("MlpPolicy", (32,), action_space, verbose=1,
                         learning_rate=learning_rate, buffer_size=10000, gamma=gamma)

    def train(self, iteration_times):
        self.model.learn(total_timesteps=iteration_times)

        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=10, render=False)
        self.env.close()
        return mean_reward, std_reward