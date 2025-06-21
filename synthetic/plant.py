# Libraries
import numpy as np

# Synthetic Markov chain to emulate vision-based autonomous system
class SyntheticSystem:

    # Define state, noise, and controller parameters
    def __init__(self, args):
        self.bias = np.array(args["bias"])
        self.cov_lb = np.array(args["cov_lb"])
        self.cov_ub = np.array(args["cov_ub"])
        self.init_state = np.array(args["init_state"])
        self.goal_state = np.array(args["goal_state"])
        self.gain = args["gain"]
        self.max_step = args["max_step"]
        self.success_dist = args["success_dist"]
        self.time_limit = args["time_limit"]
        self.barricades = args.get("barricades", [])
        self.state = self.init_state.copy()
        self.init_dist = np.linalg.norm(self.init_state - self.goal_state)
        self.failed = False
        self.terminal = False
        self.time = 0

    # Sample waypoint estimates at any state
    def sample(self, state):
        dist_to_goal = np.linalg.norm(state - self.goal_state)
        cov = ((dist_to_goal - self.success_dist) / (self.init_dist - self.success_dist) *
              (self.cov_ub - self.cov_lb)) + self.cov_lb
        waypoint_est = np.random.multivariate_normal(self.goal_state + self.bias, cov, size=1)[0]
        return waypoint_est
    
    # Compute state estimate and control action; execute dynamics
    def step(self):

        # Terminate and fail if time has expired
        if self.time >= self.time_limit:
            self.failed = True
            self.terminal = True
            print("Time expired!")

        # Draw waypoint estimate from gaussian (params are a function of distance to goal)
        dist_to_goal = np.linalg.norm(self.state - self.goal_state)
        cov = ((dist_to_goal - self.success_dist) / (self.init_dist - self.success_dist) *
              (self.cov_ub - self.cov_lb)) + self.cov_lb
        waypoint_est = np.random.multivariate_normal(self.goal_state + self.bias, cov, size=1)[0]

        # Compute action and normalize if it exceeds max_step
        action = self.gain * (waypoint_est - self.state)
        action_norm = np.linalg.norm(action)
        if action_norm > self.max_step:
            action = action * (self.max_step / action_norm)

        # Propagate
        self.time += 1
        self.state += action
        
        # Terminate if goal is reached
        if np.linalg.norm(self.state - self.goal_state) <= self.success_dist:
            self.terminal = True
        
        # Terminate and fail if barricade is hit
        if self.check_barricades(self.state):
            self.failed = True
            self.terminal = True
            print("Collision!")
        
        return waypoint_est, action, self.state, self.terminal, self.failed
    
    def pseudostep(self, state, waypoint):
        state = np.array(state)
        waypoint = np.array(waypoint)
        action = self.gain * (waypoint - state)
        action_norm = np.linalg.norm(action)
        if action_norm > self.max_step:
            action = action * (self.max_step / action_norm)
        return action
    
    def reset(self):
        self.terminal = False
        self.failed = False
        self.time = 0
        self.state = self.init_state.copy()

    def check_barricades(self, state):
        for b in self.barricades:
            if (state[0] >= b["x_min"] and state[0] <= b["x_max"] and
                state[1] >= b["y_min"] and state[1] <= b["y_max"]):
                return True
        return False

