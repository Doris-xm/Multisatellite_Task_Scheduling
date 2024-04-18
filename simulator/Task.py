import random


class Task:
    def __init__(self, id, priority, start_time, end_time, time, E):
        # start_time, end_time 是可执行时间窗口
        self.id = id
        self.priority = priority
        self.start_time = start_time
        self.end_time = end_time
        self.time = time
        self.E = E


def create_tasks(num):
    tasks = []
    for i in range(num):
        task = Task(i, random.randint(1, 10), 0, 100, random.randint(1, 10), random.randint(1, 10))
        tasks.append(task)
    return tasks