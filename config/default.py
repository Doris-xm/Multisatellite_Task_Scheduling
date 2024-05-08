class Config:
    # Resource Satellites
    RS_NUM = 6
    BEAM_NUM = 4
    E_MIN = 40   # minimum energy
    ENERGY_POLICY = 'full'  # how to initialize energy   ['random', 'full']
    INIT_ENERGY = 200

    # Task Satellites
    TS_NUM = 20

    # Tasks
    MAX_PRIORITY = 10
    MAX_TASK_TIME = 50
    MAX_TASK_E = 30

    # Time Window
    D_GRIDS = 100  # how many grids in a horizon
    TOTAL_TIME = 1000  # total time of simulation