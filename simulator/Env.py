import gymnasium as gym
import numpy as np

from simulator.ResourceSatellite import ResourceSatellite


class SatelliteTaskSchedulingEnv(gym.Env):
    def __init__(self, task_satellites=None, resource_satellites=None, tasks=None, stop_time=1000):

        self.global_time = 0
        self.stop_time = stop_time
        self.min_E = 40

        self.task_satellites = task_satellites  # 暂时没有用到task_satellites,直接输入tasks
        self.resources_num = 20
        self.resources = resource_satellites
        self.tasks = tasks
        self.max_beam = max([len(res) for res in self.resources.resource_satellites])
        # 定义状态和动作空间
        # actions: [0, max_num_of(resource_satellites)] 表示选择第i个资源卫星, 一次选择一个
        self.action_space = gym.spaces.Discrete(self.resources_num)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.resources_num,
                                                                                 self.max_beam), dtype=np.float32)  # 假设状态是一个n_observations维的向量
        self.reward = 0
        self.observation = np.zeros((self.resources_num, self.max_beam))

    def step(self, action):
        # 将动作限制在有效范围内
        action = np.clip(action, 1, len(self.resources.resource_satellites))

        self.global_time += 1

        #检查约束
        beam = self.resources.schedule_task(self.tasks[0], action)
        if beam == -1:
            return self.observation, -1, False, {}

        # 更新状态
        self.observation[action][beam] += self.tasks[0].time

        priority = self.tasks[0].priority
        success_num = 1
        beam_switch_num = 1  # what's the difference between success_num and beam_switch_num?
        self.reward += 1.2 * priority + 2 * success_num - beam_switch_num

        self.tasks.pop(0)


        # 判断是否终止
        done = self.global_time >= self.stop_time or len(self.tasks) == 0

        return self.observation, self.reward, done, {}

    # def reset(self):
    #     self.global_time = 0
    #     return self.observation
    def render(self, mode='human'):
        pass

    def close(self):
        pass


