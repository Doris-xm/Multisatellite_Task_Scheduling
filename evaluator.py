import argparse

from stable_baselines3 import DQN

from simulator.Env import SatelliteTaskSchedulingEnv


def evaluate_model(model_path):  # success rate, priority sum,
    model = DQN.load(model_path)
    env = SatelliteTaskSchedulingEnv()

    obs, info = env.reset()
    total_step = 0
    total_reward = 0.0
    total_success = 0
    total_priority = 0
    total_switch = 0
    total_success_priority = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        # print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_step += 1
        total_reward += reward
        total_success += info['is_success']
        total_priority += info['priority']
        total_switch += info['if_switch']
        total_success_priority += info['priority'] * info['is_success']

        if terminated or truncated:
            print("Total step: ", total_step)
            print("Average reward: ", total_reward / total_step)
            print("Success rate: ", total_success / total_step)
            print("Success priority: ", total_success_priority, '/', total_priority)
            print("Total switch: ", total_switch)
            env.show_info()
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate RL model.")
    parser.add_argument("--model_path", type=str, default="dqn_satellite_task_scheduling",
                        help="Path to save the trained model")

    args = parser.parse_args()
    evaluate_model(args.model_path)