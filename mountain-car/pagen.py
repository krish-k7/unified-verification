# Libraries
from plant import clsysPseudo
import numpy as np
import itertools

# State machine
class PA:

    def __init__(self, args):

        # Unpack arguments
        self.sx = args['sx']  # state space resolution
        self.x_space = args['x_space']  # state space range
        self.sv = args['sv']  # velocity space resolution
        self.v_space = args['v_space']  # velocity space range
        self.serr = args['serr']  # error space resolution
        self.err_space = args['err_space']  # error space range
        self.method = args.get('method', "conserve")

        # Initialize MountainCar
        seed = 23
        self.clsys = clsysPseudo(seed)

        # Preallocate state machine data
        self.next_states = {}
        self.state_keys = []

    # Abstract dynamics and state space into overconservative state machine
    def abstract(self):
        
        # Access abstraction parameters
        sx = self.sx
        x_space = self.x_space
        sv = self.sv
        v_space = self.v_space
        serr = self.serr
        err_space = self.err_space

        # Define discrete space
        xbar_space = np.arange(int(x_space[0]/sx), int(x_space[1]/sx)+1)
        vbar_space = np.arange(int(v_space[0]/sv), int(v_space[1]/sv)+1)
        errbar_space = np.arange(int(err_space[0]/serr), int(err_space[1]/serr))

        # Simulate each state; overapproximiate transition relations
        total = len(xbar_space)*len(vbar_space)*len(errbar_space)#*len(vhatbar_space)
        count = 0
        for xbar, vbar, errbar in itertools.product(xbar_space, vbar_space, errbar_space):
            
            if count % 10000 == 0:
                print(f"{count} of {total} state/action abstractions made ({np.round(count/total*100, 2)}%)")

            xlb, xub = (xbar * sx), (xbar * sx) + sx - 1e-10
            vlb, vub = (vbar * sv), (vbar * sv) + sv - 1e-10
            errlb, errub = (errbar * serr), (errbar * serr) + serr - 1e-10
            
            if self.method == "approx":
                xb, vb, errb = (xlb + xub)/2, (vlb + vub)/2, (errlb + errub)/2
                next_state, _, _ = self.clsys.pseudostep(np.array([xb, vb], dtype=np.float32), errb)
                x_next = np.floor(next_state[0] / sx)
                v_next = np.floor(next_state[1] / sv)
                current_state = (int(xbar), int(vbar), int(errbar))
                xlb, xub = min(xbar_space), max(xbar_space)
                vlb, vub = min(vbar_space), max(vbar_space)
                next_state = (int(max(xlb, min(int(x_next), xub))), int(max(vlb, min(int(v_next), vub))))
                self.state_keys.append(current_state)
                self.next_states[current_state] = next_state

            elif self.method == "conserve":
                deltas_x = []
                deltas_v = []
                for xb, vb, errb in itertools.product([xlb, xub], [vlb, vub], [errlb, errub]):
                    next_state, _, done = self.clsys.pseudostep(np.array([xb, vb], dtype=np.float32), errb)
                    deltas = next_state - np.array([xb, vb])
                    deltas_x.append(deltas[0])
                    deltas_v.append(deltas[1])

                dxmin, dxmax = min(deltas_x), max(deltas_x)
                dvmin, dvmax = min(deltas_v), max(deltas_v)

                x_next = [xlb + dxmin, xub + dxmax]
                v_next = [vlb + dvmin, vub + dvmax]

                xbar_next = list(np.arange(int(np.floor(x_next[0] / sx)), int(np.floor(x_next[1] / sx)) + 1))
                vbar_next = list(np.arange(int(np.floor(v_next[0] / sv)), int(np.floor(v_next[1] / sv)) + 1))

                current_state = (xbar, vbar, errbar)
                current_state = tuple(int(x) for x in current_state)
                possible_states = []
                for xn, yn in itertools.product(xbar_next, vbar_next):
                    possible_state = (int(xn), int(yn))
                    possible_states.append(possible_state)

                # Clamp states between respecive upper/lower bounds
                possible_states_clamped = []
                lb_states = (min(xbar_space), min(vbar_space))
                ub_states = (max(xbar_space), max(vbar_space))
                for possible_state in possible_states:
                    possible_state_clamped = [int(max(lb, min(s, ub))) for s, lb, ub in zip(possible_state, lb_states, ub_states)]
                    possible_state_clamped = tuple(possible_state_clamped)
                    if possible_state_clamped not in possible_states_clamped:
                        possible_states_clamped.append(possible_state_clamped)
            
                self.state_keys.append(current_state)
                self.next_states[current_state] = possible_states_clamped

            count += 1
        self.clsys.close()
