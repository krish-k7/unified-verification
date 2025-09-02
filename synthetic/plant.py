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
            # print("Time expired!")

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
            # print("Collision!")
        
        return waypoint_est, action, self.state, self.terminal, self.failed
    
    
    def pseudostep(self, state, waypoint):
        state = np.array(state)
        waypoint = np.array(waypoint)
        action = self.gain * (waypoint - state)
        action_norm = np.linalg.norm(action)
        if action_norm > self.max_step:
            action = action * (self.max_step / action_norm)
        return action
    
    def pseudostep_1(self, state):
        """
        Deterministic pseudostep from an arbitrary state:
        - Computes action towards nominal waypoint
        - Returns the action (displacement vector for one step)
        """
        state = np.array(state)
        waypoint = self.goal_state + self.bias
        action = self.gain * (waypoint - state)
        action_norm = np.linalg.norm(action)
        if action_norm > self.max_step:
            action = action * (self.max_step / action_norm)
        return action
    
    def pseudostep_2(self, state):
        """
        Stochastic pseudostep from an arbitrary state:
        - If distance to goal > success_dist: sample fuzzy waypoint with distance-dependent covariance
        - If distance to goal <= success_dist: use nominal waypoint (goal_state + bias)
        - Computes the action toward that waypoint with gain and max_step clipping
        - Returns the action (displacement vector for one step)
        """
        state = np.array(state, dtype=float)
        dist_to_goal = np.linalg.norm(state - self.goal_state)

        # If within success distance, use nominal waypoint
        if dist_to_goal <= self.success_dist:
            waypoint = self.goal_state + self.bias
        else:
            # Compute distance ratio
            denom = (self.init_dist - self.success_dist)
            if denom == 0:
                t = 0.0
            else:
                t = (dist_to_goal - self.success_dist) / denom
            t = np.clip(t, 0.0, 1.0)

            # Distance-dependent covariance
            cov = self.cov_lb + t * (self.cov_ub - self.cov_lb)
            cov = 0.5 * (cov + cov.T)  # symmetrize
            cov += np.eye(cov.shape[0]) * 1e-12

            # Sample waypoint from Gaussian
            waypoint = np.random.multivariate_normal(self.goal_state + self.bias, cov, size=1)[0]

        # Proportional step toward chosen waypoint
        action = self.gain * (waypoint - state)
        action_norm = np.linalg.norm(action)
        if action_norm > self.max_step and action_norm > 0:
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

# Sample usage
if __name__ == "__main__":

    # Instantiate the synthetic system
    args = {
        "bias": [0.0, 0.0],
        "cov_lb": [[0.5, 0.0], [0.0, 0.5]],
        "cov_ub": [[2.0, 0.0], [0.0, 2.0]],
        "init_state": [0.0, 0.0],
        "goal_state": [10.0, 10.0],
        "gain": 0.5,
        "max_step": 0.7,
        "success_dist": 2.0,
        "time_limit": 25,
        "barricades": [
                {"x_min": 0.0, "x_max": 15.0, "y_min": 12.0, "y_max": 15.0},
                {"x_min": 12.0, "x_max": 15.0, "y_min": 0.0, "y_max": 15.0}
            ]
    }
    system = SyntheticSystem(args)
    for _ in range(100):
        est, action, state, terminal, failed = system.step()
        print(f"State: {np.round(state, 4).astype(float)}, Estimate: {np.round(est, 4).astype(float)}, Action: {np.round(action, 4).astype(float)}, Terminal: {terminal}, Failed: {failed}")
        if terminal or failed:
            break
