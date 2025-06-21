# Libraries
from plant import SyntheticSystem
import numpy as np
import itertools

# State machine
class PA:

    def __init__(self):

        # Initialize VBAS environment
        args = {
            "bias": [0.0, 0.0],
            "cov_lb": [[0.5, 0],[0, 0.5]],
            "cov_ub": [[2.0, 0],[0, 2.0]],
            "init_state": [0.0, 0.0],
            "goal_state": [10.0, 10.0],
            "gain": 0.5,
            "max_step": 0.7,
            "success_dist": 2.0,
            "time_limit": np.inf,
            "barricades": [
                # {"x_min": 0.0, "x_max": 15.0, "y_min": 12.0, "y_max": 15.0}
            ]}
        self.sys = SyntheticSystem(args)

        # Preallocate state machine data
        self.next_states = {}
        self.state_keys = []

        # State space abstraction parameters
        self.sx = 0.5
        self.x_space = [0.0, 12.0]
        self.sy = 0.5
        self.y_space = [0.0, 12.0]

        # Estimation space abstraction parameters
        self.sxhat = 1.0
        self.xhat_space = [8.0, 12.0]
        self.syhat = 1.0
        self.yhat_space = [8.0, 12.0]

        # Simulation and CI construction parameters
        self.lower_bounds = []
        self.upper_bounds = []

    # Abstract dynamics and state space into overconservative state machine
    def abstract(self):
        
        # Access abstraction parameters
        sx = self.sx
        x_space = self.x_space
        sy = self.sy
        y_space = self.x_space
        sxhat = self.sxhat
        xhat_space = self.xhat_space
        syhat = self.syhat
        yhat_space = self.yhat_space

        # Define discrete space
        xbar_space = np.arange(int(x_space[0]/sx), int(x_space[1]/sx))
        ybar_space = np.arange(int(y_space[0]/sy), int(y_space[1]/sy))
        xhatbar_space = np.arange(int(xhat_space[0]/sxhat), int(xhat_space[1]/sxhat))
        yhatbar_space = np.arange(int(yhat_space[0]/syhat), int(yhat_space[1]/syhat))

        # Simulate each state; overapproximiate transition relations
        total = len(xbar_space)*len(ybar_space)*len(xhatbar_space)*len(yhatbar_space)
        count = 0
        for xbar, ybar, xhatbar, yhatbar in itertools.product(xbar_space, ybar_space, xhatbar_space, yhatbar_space):
            
            if count % 10000 == 0:
                print(f"{count} of {total} state/action abstractions made ({np.round(count/total*100, 2)}%)")

            xlb, xub = (xbar * sx), (xbar * sx) + sx - 1e-5
            ylb, yub = (ybar * sy), (ybar * sy) + sy - 1e-5
            xhatlb, xhatub = (xhatbar * sxhat), (xhatbar * sxhat) + sxhat - 1e-5
            yhatlb, yhatub = (yhatbar * syhat), (yhatbar * syhat) + syhat - 1e-5

            deltas_x = []
            deltas_y = []
            for xb, yb, xhatb, yhatb in itertools.product([xlb, xub], [ylb, yub], [xhatlb, xhatub], [yhatlb, yhatub]):
                deltas = self.sys.pseudostep([xb, yb], [xhatb, yhatb])
                deltas_x.append(deltas[0])
                deltas_y.append(deltas[1])

            dxmin, dxmax = min(deltas_x), max(deltas_x)
            dymin, dymax = min(deltas_y), max(deltas_y)

            x_next = [xlb + dxmin, xub + dxmax]
            y_next = [ylb + dymin, yub + dymax]

            xbar_next = list(np.arange(int(np.floor(x_next[0] / sx)), int(np.floor(x_next[1] / sx))))
            ybar_next = list(np.arange(int(np.floor(y_next[0] / sy)), int(np.floor(y_next[1] / sy))))

            current_state = (xbar, ybar, xhatbar, yhatbar)
            possible_states = []
            for xn, yn in itertools.product(xbar_next, ybar_next):
                possible_states.append((xn, yn))

            # Clamp states between respective upper/lower bounds
            possible_states_clamped = []
            lb_states = (min(xbar_space), min(ybar_space), min(xhatbar_space), min(yhatbar_space))
            ub_states = (max(xbar_space), max(ybar_space), max(xhatbar_space), max(yhatbar_space))
            for possible_state in possible_states:
                possible_state_clamped = [int(max(lb, min(s, ub))) for s, lb, ub in zip(possible_state, lb_states, ub_states)]
                possible_state_clamped = tuple(possible_state_clamped)
                if possible_state_clamped not in possible_states_clamped:
                    possible_states_clamped.append(possible_state_clamped)

            # Cull self loops
            possible_states = []
            for possible_state in possible_states_clamped:
                if possible_state == current_state[:2]:
                    # print(f"Self loop prevented: {current_state[:2]} -> {possible_state}")
                    continue
                possible_states.append(possible_state)
        
            self.state_keys.append(current_state)
            self.next_states[current_state] = possible_states

            count += 1
        