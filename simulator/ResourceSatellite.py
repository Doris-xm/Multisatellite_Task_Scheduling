class ResourceSatellite:
    def __init__(self, num, beam=4, time=0, E=100, E_min=40, resource_satellites=None):
        if resource_satellites is None:
            self.resource_satellites = self.create_resource_satellites()
        else:
            self.resource_satellites = resource_satellites
        self.num = num
        self.beam = beam
        self.time = time
        self.E = E
        self.E_min = E_min

    def create_resource_satellites(self):
        satellites = []
        for i in range(self.num):
            satellite = []
            for i in range(self.beam):
                satellite.append({'time': 0, 'E': 100})
            satellites.append(satellite)
        return satellites

    def schedule_task(self, task, index):
        beam = -1
        for i in range(self.beam):
            if self.resource_satellites[index][i]['time'] + task.time > task.end_time:
                continue
            if self.resource_satellites[index][i]['E'] - task.E < self.E_min:
                continue
            if self.resource_satellites[index][i]['time'] < task.start_time or self.resource_satellites[index][i]['time'] > task.end_time:
                continue
            beam = i
            break

        # 更新状态
        self.resource_satellites[index][beam]['time'] += task.time
        self.resource_satellites[index][beam]['E'] -= task.E
        return beam
