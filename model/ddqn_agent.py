import numpy as np
import torch
from torch.nn import functional as F
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from model.res_fcn import ResFCN


class DDQN_agent:
    def __init__(self, env, batch_size=64, gamma=0.99,
                 exploration_rate=1.0, replacement_frequency=15, learning_rate=1e-4, window_size=20):

        self.env = env
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
        # Switch to train mode (this affects batch norm / dropout)
        self.model.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self.model._update_learning_rate(self.model.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):

            # Sample replay buffer
            replay_data = self.model.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # Check Receding Horizon Condition (公式22)
            if self.receding_horizon_condition():
                self.execute_receding_horizon_optimization()

            # Check Event-Triggered Condition (公式23)
            elif self.event_triggered_condition():
                self.execute_event_triggered_optimization()

            # Otherwise, continue with Algorithm 1
            else:
                with torch.no_grad():
                    # Compute the next Q-values using the target network
                    next_q_values = self.model.q_net_target(replay_data.next_observations)
                    # Follow greedy policy: use the one with the highest value
                    next_q_values, _ = next_q_values.max(dim=1)
                    # Avoid potential broadcast issue
                    next_q_values = next_q_values.reshape(-1, 1)
                    # 1-step TD target
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
    
                # Get current Q-values estimates
                current_q_values = self.model.q_net(replay_data.observations)
    
                # Retrieve the q-values for the actions from the replay buffer
                current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())
    
                # Compute Huber loss (less sensitive to outliers)
                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                losses.append(loss.item())
    
                # Optimize the policy
                self.model.policy.optimizer.zero_grad()
                loss.backward()
                # Clip gradient norm
                torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.model.max_grad_norm)
                self.model.policy.optimizer.step()

        # Increase update counter
        self.model._n_updates += gradient_steps

        self.model.logger.record("train/n_updates", self.model._n_updates, exclude="tensorboard")
        self.model.logger.record("train/loss", np.mean(losses))

    def receding_horizon_condition(self):
        # return self.env.current_time - self.env.last_scheduling_time >= self.window_size
        pass

    def execute_receding_horizon_optimization(self):
        pass

    def event_triggered_condition(self):
        pass

    def execute_event_triggered_optimization(self):
        pass