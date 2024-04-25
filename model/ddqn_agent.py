import numpy as np
import torch
from torch.nn import functional as F
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from model.res_fcn import ResFCN
from simulator.Env import SatelliteTaskSchedulingEnv


class DDQN_agent:
    def __init__(self,  batch_size=64, gamma=0.99,
                 exploration_rate=1.0, replacement_frequency=15, learning_rate=1e-4, window_size=20):

        self.env = SatelliteTaskSchedulingEnv()
        self.env = DummyVecEnv([lambda: self.env])
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.replacement_frequency = replacement_frequency
        self.learning_rate = learning_rate
        self.resfcn = ResFCN(self.observation_space.shape[0])
        self.window_size = window_size

        # Initialize the Behavior Network and the Target Network
        self.model = DQN(policy="MlpPolicy", env=self.env, verbose=1,
                         learning_rate=learning_rate, buffer_size=10000, gamma=gamma)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.model.learn(total_timesteps=gradient_steps, log_interval=4)
        self.model.save("dqn_satellite_task_scheduling")

    def receding_horizon_condition(self):
        # return self.env.current_time - self.env.last_scheduling_time >= self.window_size
        pass

    def execute_receding_horizon_optimization(self):
        pass

    def event_triggered_condition(self):
        pass

    def execute_event_triggered_optimization(self):
        pass