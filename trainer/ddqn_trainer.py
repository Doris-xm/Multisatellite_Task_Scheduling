from model.ddqn_agent import DDQN_agent
from simulator.Env import SatelliteSim


class DDQNTrainer:
    def __init__(self, agent: DDQN_agent, total_iterations: int):
        self.agent = agent
        self.total_iterations = total_iterations
        self.window_size = 10
        self.sim = SatelliteSim()

    def check_optim(self):

        return self.sim.get_current_time() - self.sim.get_last_scheduling_time() >= self.window_size

    def receding_horizon_optimization(self):
        # Update the current scheduling horizon
        # Perform Algorithm 1 for the K + 1-th scheduling horizon
        self.agent.train()

    def event_triggered_optimization(self):
        # Execute the event-triggered optimization strategy
        # Perform Algorithm 1 for the K + 1-th scheduling
        self.agent.train()

    def urgent_task_appears(self):
        # Check if an urgent task appears with the priority Pmax as in (23)
        return 1

    def train(self):
        for iteration in range(1, self.total_iterations + 1):
            # Perform one iteration of training
            self.agent.train(iteration)

            # Print progress
            print(f"Iteration {iteration} completed.")

            # receding horizon optimization(22)
            if self.check_optim():
                # Execute the receding horizon optimization
                self.receding_horizon_optimization()

            # Check if an urgent task appears with the priority Pmax as in (23)
            elif self.urgent_task_appears():
                # Execute the event-triggered optimization strategy
                self.event_triggered_optimization()

