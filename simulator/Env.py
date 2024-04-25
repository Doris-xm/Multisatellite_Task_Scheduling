from __future__ import annotations

from random import random
from typing import Tuple, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

from config.default import Config
from simulator.Task import create_tasks


class SatelliteTaskSchedulingEnv(gym.Env):
    def __init__(self, config: Config = None):
        if config is None:
            config = Config
        self.config = config

        self.horizon_start = 0
        self.stop_time = self.config.TOTAL_TIME
        self.d_grids = self.config.D_GRIDS

        self.ts_num = self.config.TS_NUM  # number of task satellites
        self.beam_num = self.config.BEAM_NUM
        self.rs_num = self.config.RS_NUM  # number of resource satellites
        self.rs_list = []
        self.tasks = []
        # self.rs_list = self.create_resource_satellites()
        # self.tasks = create_tasks(self.ts_num, self.config.MAX_PRIORITY, self.config.MAX_TASK_TIME,
        #                           self.config.MAX_TASK_E, self.stop_time)
        self.min_E = self.config.E_MIN

        # 定义状态和动作空间
        # actions: [0, max_num_of(resource_satellites)] 表示选择第i个资源卫星, 一次选择一个
        self.action_space = gym.spaces.Discrete(self.rs_num + 1)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.rs_num, self.beam_num, self.d_grids),
                                                dtype=np.float32)

        self.state = np.zeros((self.rs_num, self.beam_num, self.d_grids))
        self.cur_task = 0

    def step(self, action):
        success = False
        task = self.tasks[self.cur_task]
        self.cur_task += 1
        if_switch = False
        if 0 < action <= self.rs_num:   # 如果action合法
            for i in range(self.beam_num):     # 遍历beam
                # 检查是否有足够的能量
                if self.rs_list[action - 1][i]['E_left'] - task.energy < self.min_E:
                    continue

                # 检查是否有足够的时间窗口, 遍历可执行的时间窗口
                for j in range(task.start_time - self.horizon_start, min(self.d_grids, task.end_time - task.duration + 1 - self.horizon_start)):
                    if self.state[action - 1][i][j] == 0:
                        success = True
                        #   更新状态
                        self.state[action - 1][i][j:j + task.duration] = 1
                        self.rs_list[action - 1][i]['E_left'] -= task.energy

                        #  和上一次任务比较，是否需要切换task satellite
                        if self.rs_list[action - 1][i]['last_ts'] >= 0 and self.rs_list[action - 1][i]['last_ts'] != task.ts_id:
                            if_switch = True
                        self.rs_list[action - 1][i]['last_ts'] = task.ts_id
                        break

                # # 如果不是按照开始时间依次调度，需要设置滑动窗口，检查是否有连续的时间窗口
                # for j in range(task.start_time, task.end_time - task.duration + 1):
                #     window = self.state[action - 1][i][j: j + task.duration]
                #     if all(value == 0 for value in window):
                #         return j  # 返回第一个可执行时间点

        reward = -1
        if success:
            # 更新reward
            reward = task.priority + 1 - if_switch
        # 判断是否终止
        done = self.cur_task >= len(self.tasks)
        if done:
            return self.state, reward, done, False, {}

        # 检查是否需要滑动horizon
        if self.tasks[self.cur_task].start_time - self.horizon_start +  self.tasks[self.cur_task].duration >= self.d_grids:
            self.move_horizon(self.d_grids)
        if self.horizon_start >= self.stop_time:
            done = True
            return self.state, reward, done, True, {}

        # 根据下一个task计算下一个state
        next_state = self.get_next_state()
        return next_state, reward, done, False, {}

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.horizon_start = 0
        self.state = np.zeros((self.rs_num, self.beam_num, self.d_grids))
        self.cur_task = 0
        self.rs_list = self.create_resource_satellites()
        self.tasks = create_tasks(self.ts_num, self.config.MAX_PRIORITY, self.config.MAX_TASK_TIME,
                                  self.config.MAX_TASK_E, self.stop_time)
        next_state = self.get_next_state()
        return next_state, {}

    def create_resource_satellites(self):
        satellites = []
        for i in range(self.config.RS_NUM):
            satellite = []
            for i in range(self.config.BEAM_NUM):
                if self.config.ENERGY_POLICY == 'full':
                    satellite.append({'E_left': self.config.INIT_ENERGY,
                                      'E_min': self.config.E_MIN,
                                      'last_ts': -1})
                else:  # random
                    satellite.append(
                        {'E_left': random() % (self.config.INIT_ENERGY - self.config.E_MIN) + self.config.E_MIN,
                         'E_min': self.config.E_MIN,
                         'last_ts': -1})
            satellites.append(satellite)
        return satellites

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def move_horizon(self, len):
        self.state = np.zeros((self.rs_num, self.beam_num, self.d_grids))
        self.horizon_start += len

    def get_next_state(self):
        next_state = self.state.copy()
        task = self.tasks[self.cur_task]
        for i in range(self.rs_num):
            for j in range(self.beam_num):
                next_state[i][j][0:task.start_time] = 1
                next_state[i][j][task.end_time:] = 1
        return next_state
