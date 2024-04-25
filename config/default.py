
def Config():
    def __init__(self):
        # Resource Satellites
        self.RS_NUM = 40
        self.BEAM_NUM = 4
        self.E_MIN = 40   # minimum energy
        self.ENERGY_POLICY = 'full'  # how to initialize energy   ['random', 'full']
        self.INIT_ENERGY = 200

        # Task Satellites
        self.TS_NUM = 10

        # Tasks
        self.MAX_PRIORITY = 10
        self.MAX_TASK_TIME = 50
        self.MAX_TASK_E = 30


        #  Time Window
        self.D_GRIDS = 100  # how many grids in a horizon
        self.TOTAL_TIME = 1000  # total time of simulation
