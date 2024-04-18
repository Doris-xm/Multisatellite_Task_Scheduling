from model.ddqn_agent import DDQN_agent
from simulator.Env import SatelliteTaskSchedulingEnv
from simulator.ResourceSatellite import ResourceSatellite
from simulator.Task import create_tasks

resources = ResourceSatellite(num=20)
tasks = create_tasks(10)
env = SatelliteTaskSchedulingEnv(resource_satellites=resources, tasks=tasks)

model = DDQN_agent(env)
model.train(1000)
