import random


class Task:
    def __init__(self, ts_id, priority, start_time, end_time, time, E):
        self.ts_id = ts_id  # 生成它的task satellite的id
        self.priority = priority  # 优先级
        self.start_time = start_time  # 可执行的时间窗口
        self.end_time = end_time  # 可执行的时间窗口
        self.duration = time  # 执行时间
        self.energy = E  # 能耗


def create_tasks(ts_num, max_priority, max_time, max_E, stop_time):
    tasks = []
    for i in range(ts_num):
        # 每一个task satellite依次随机生成任务序列
        # 目前保证task的可执行时间窗口不重叠
        start = 0
        while start < stop_time:
            time_consume = random.randint(1, max_time)
            if start + time_consume > stop_time:
                break
            task = Task(ts_id=i,
                        priority=random.randint(1, max_priority),
                        start_time=start,
                        end_time=min(start + time_consume + random.randint(1, 20), stop_time),
                        time=time_consume,
                        E=random.randint(1, max_E))
            tasks.append(task)
            start = task.end_time
        # time_consume = random.randint(1, max_time)
        # start_time = random.randint(0, stop_time - time_consume)
        # task = Task(ts_id=random.randint(1, ts_num),
        #             priority=random.randint(1, max_priority),
        #             start_time=start_time,
        #             end_time=random.randint(start_time + time_consume, stop_time),
        #             time=time_consume,
        #             E=random.randint(1, max_E))
        # tasks.append(task)

    # 按照任务开始时间排序
    tasks.sort(key=lambda x: x.start_time)
    return tasks
