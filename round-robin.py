from simulator import Task
from config.default import Config
import numpy as np
import random

class RoundRobin:
    """
    Description
    RoundRobin will dispatch tasks evenly among all the resource satellites
    """
    def __init__(self,
                 time_period=Config.TOTAL_TIME, rs_num=Config.RS_NUM,
                 beam_num=Config.BEAM_NUM, ts_num=Config.TS_NUM,
                 max_duration=Config.MAX_TASK_TIME, E_capacity=Config.INIT_ENERGY,
                 max_priority=Config.MAX_PRIORITY, max_task_E = Config.MAX_TASK_E,
                 engy_th = Config.E_MIN
                 ):
        # Scenarios settings
        self.time_period = time_period
        self.rs_num = rs_num
        self.beam_num = beam_num
        self.ts_num = ts_num
        self.max_duration = max_duration
        self.E_capacity = E_capacity
        self.max_priority = max_priority
        self.max_task_E = max_task_E
        self.engy_th = engy_th

        self.tasks = None   # The queue to store tasks
        self.com_state = None
        self.engy_state = None

        # statistics
        # record succrss ratio, sum of priority and switch time
        self.total_tasks = None
        self.success_cnt = 0
        self.total_priority = 0
        self.switches = 0
        # record info about the position of scheduled/failed tasks
        self.schedu_tasks = {}
        self.failed_tasks = {}
    
    def evaluate(self):
        # generate tasks
        # random.seed(2)
        self.tasks = Task.create_tasks(self.ts_num, self.max_priority,
                                  self.max_duration, self.max_task_E,
                                  self.time_period)
        self.total_tasks = len(self.tasks)
        
        # which rs the ts is connected to 
        ts_connect = [-1] * self.ts_num

        # initialize the resource state of each satellite
        self.com_state, self.engy_state= self._state_initialize()
        rs_cursor = 0
        for task in self.tasks:
            rs = rs_cursor % self.rs_num
            rs_state = self.com_state[rs]
            is_success, rs_state = self._allo_task(rs_state, task, self.beam_num,
                                                   rs, self.engy_state)
            if is_success:
                # update connect state
                if rs != ts_connect[task.ts_id]:
                    self.switches += 1
                ts_connect[task.ts_id] = rs
                # update performance indicators
                self.success_cnt += 1
                self.total_priority += task.priority

            #update sat_state
            self.com_state[rs] = rs_state
            # update cursor
            rs_cursor += 1
    
    def _state_initialize(self, E_capacity=None, rs_num=None, beam_num=None, time_period=None):
        E_capacity = self.E_capacity if E_capacity is None else E_capacity
        rs_num = self.rs_num if rs_num is None else rs_num
        beam_num = self.beam_num if beam_num is None else beam_num
        time_period = self.time_period if time_period is None else time_period

        com_state = [np.zeros((beam_num, time_period)) for _ in range(rs_num)]
        energy_state = [E_capacity] * rs_num
        return com_state, energy_state
    
    def _allo_task(self, rs_state, task, beam_num, rs, engy_state):
        is_success = False
        is_coninue = True
        if engy_state[rs] >= self.engy_th + task.energy:
            for b in range(beam_num):
                for t in range(task.start_time, task.end_time-task.duration+1):
                    if rs_state[b, t] == 1:
                        continue
                    is_success = True
                    is_coninue = False
                    rs_state[b, t: t+task.duration] = 1
                    engy_state[rs] -= task.energy   # update energy state 
                    self.schedu_tasks[task] = {
                        'start_time': t,
                        'end_time': t+task.duration-1,
                        'rs': rs,
                        'ts': task.ts_id,
                        'beam': b,
                        'priority': task.priority
                        }
                if not is_coninue:
                    break
        if not is_success:
            self.failed_tasks[task] = {
                'start_time': task.start_time,
                'end_time': task.end_time,
                'ts': task.ts_id,
                'priority': task.priority
            }
        return is_success, rs_state

if __name__ == '__main__':
    algTest = RoundRobin()
    algTest.evaluate()
    print(f'The total number of tasks is {algTest.total_tasks}')
    print('Performance is as follows:')
    print(f'Success rate: {algTest.success_cnt/algTest.total_tasks*100:.2f}%', end=' ') 
    print(f'total priority: {algTest.total_priority}, switches: {algTest.switches}')
