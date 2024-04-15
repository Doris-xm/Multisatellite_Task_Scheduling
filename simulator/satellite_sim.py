class SatelliteSim:
    def __init__(self):
        self.time = 0
        self.last_scheduling_time = 0

    def get_current_time(self):
        return self.time

    def get_last_scheduling_time(self):
        return self.last_scheduling_time

    def perform_scheduling(self):
        current_time = self.get_current_time()
        self.last_scheduling_time = current_time

        return {}