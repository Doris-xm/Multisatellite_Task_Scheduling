import argparse

from stable_baselines3 import DQN

from model.ddqn_agent import DDQN_agent
from simulator.Env import SatelliteTaskSchedulingEnv


def train_model(total_timesteps):
    # 创建 DQN 模型并训练
    model = DDQN_agent()
    model.train(total_timesteps)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate RL model.")
    parser.add_argument("--total_timesteps", type=int, default=200000, help="Total number of training timesteps")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--model_path", type=str, default="dqn_satellite_task_scheduling",
                        help="Path to save the trained model")

    args = parser.parse_args()
    # if args.train:
    train_model(args.total_timesteps)
    # evaluate_model(args.model_path)
