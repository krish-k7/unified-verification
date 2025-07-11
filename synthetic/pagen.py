# Libraries
from plant import SyntheticSystem
import numpy as np
import itertools

# State machine
class PA:

    def __init__(self, args):

        # Unpack arguments
        self.system_args = args['system_args']
        self.sx = args['sx']
        self.x_space = args['x_space']
        self.sy = args['sy']
        self.y_space = args['y_space']
        self.sxhat = args['sxhat']
        self.xhat_space = args['xhat_space']
        self.syhat = args['syhat']
        self.yhat_space = args['yhat_space']

        # Initialize synthetic environment
        self.sys = SyntheticSystem(self.system_args)

        # Preallocate state machine data
        self.next_states = {}
        self.state_keys = []

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
        for xbar, ybar, xhatbar, yhatbar in itertools.product(xbar_space, ybar_space, xhatbar_space, yhatbar_space):

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

# Sample usage
if __name__ == "__main__":

    # Additional libraries
    import random

    # Build the automaton
    args = {
        'system_args': {
            "bias": [0.0, 0.0],
            "cov_lb": [[0.5, 0],[0, 0.5]],
            "cov_ub": [[2.0, 0],[0, 2.0]],
            "init_state": [0.0, 0.0],
            "goal_state": [10.0, 10.0],
            "gain": 0.5,
            "max_step": 0.7,
            "success_dist": 2.0,
            "time_limit": np.inf,
            "barricades": []
        },
        'sx': 0.5,
        'x_space': [0.0, 12.0],
        'sy': 0.5,
        'y_space': [0.0, 12.0],
        'sxhat': 1.0,
        'xhat_space': [8.0, 12.0],
        'syhat': 1.0,
        'yhat_space': [8.0, 12.0],
    }
    pa = PA(args)
    pa.abstract()
    automaton_mappings = pa.next_states # dictionary of state/estimate -> state mappings
    
    # Simulate the automaton in the abstract state space
    state = (0, 0) # Example initial state in the abstract state space
    for i in range(30):
        est = (10, 10) # Example estimate in the abstract state space
        state_key = state + est # tuple concatenation
        next_states = automaton_mappings[state_key]
        state = random.choice(next_states) # non-determinism resolved randomly
        print(f"Step {i}: State = {state}")