# Libraries
from pagen import PA
from fractions import Fraction
import numpy as np
import pickle as pkl

# Prism model
class pm:

    def __init__(self, args):

        # Unpack some arguments
        self.modelpath = args['modelpath']
        self.intpath = args['intpath']
        self.method = args.get('method', "conserve")

        # Preallocate command lines
        self.init_command_lines = [] # initial distribution if desired
        self.est_command_lines = [] # main estimation prism commands
        self.act_command_lines = [] # main actuation prism commands
        self.term_command_line = "" # the conditions for which "term" becomes true
        self.all_lines = [] # all prism code lines

        # Instantiate probabilistic automaton model
        self.model = PA(args) # Probabilistic automaton
        self.minxbar, self.maxxbar = int(self.model.x_space[0]/self.model.sx), int(self.model.x_space[1]/self.model.sx)
        self.minvbar, self.maxvbar = int(self.model.v_space[0]/self.model.sv), int(self.model.v_space[1]/self.model.sv)
        self.minerrbar, self.maxerrbar = int(self.model.err_space[0]/self.model.serr), int(self.model.err_space[1]/self.model.serr)
        self.goal_set = []
        self.model.abstract() # Abstract dynamics into probabilistic automaton 

        # Model details
        num_transitions = [len(state_list) for state_list in self.model.next_states.values()]
        print(f"    > Deadlocks = {any([num==0 for num in num_transitions])}")
        print(f"    > Transition stats: avg={round(np.mean(num_transitions), 1)}, min={np.min(num_transitions)}, max={np.max(num_transitions)}")
        
        # Get confidence intervals from the intervals file
        with open(self.intpath, "rb") as f:
            DATA = pkl.load(f)
        self.lower_bounds = DATA["lower_bounds"]
        self.upper_bounds = DATA["upper_bounds"]

    # Generate an initial distribution in PRISM syntax
    # Re-running results in erasure of existing lines
    def gen_init_commands(self, x_inits, v_inits):
        
        self.init_command_lines = []
        n = len(x_inits) * len(v_inits)
        prob = Fraction(1, n)
        for x in x_inits:
            for v in v_inits:
                # Build one line with probability prob and the updates
                line = f"{prob}:(x'={x})&(v'={v})&(setup'=false)"
                self.init_command_lines.append(line)

    # Estimation commands (imprecisely probabilistic; builds in Wilson confidence intervals)
    # Estimation stage in PRISM code (from state -> state & estimate probabilisitically)
    # Re-running results in erasure of existing liness
    def gen_est_commands(self):

        self.est_command_lines = []

        # # Access confidence interval upper/lower bounds from PA
        lower_bounds = self.lower_bounds
        upper_bounds = self.upper_bounds

        # Access abstraction parameters
        sx = self.model.sx
        x_space = self.model.x_space
        serr = self.model.serr
        err_space = self.model.err_space

        # Define discrete space
        xbar_space = np.arange(int(x_space[0]/sx), int(x_space[1]/sx)+1)
        errbar_space = np.arange(int(err_space[0]/serr), int(err_space[1]/serr))

        # Loop through discrete space and construct PRISM commands
        for i in range(len(xbar_space)):
            xbar = xbar_space[i]
            command_str = f"[](est=true)&(term=false)&(setup=false)&(x={xbar})->"
            lowers = lower_bounds[i] # lower bounds of CI indexed at this state
            uppers = upper_bounds[i] # upper bounds of CI indexed at this state

            # Loop through estimation space; write probabilistic mapping command with CI
            for k in range(len(errbar_space)):
                err_bar = errbar_space[k]
                lower = lowers[k]
                upper = uppers[k]
                command_str += f"[{lower}, {upper}]:(err'={err_bar})&(est'=false)+"
                
            # Add final touches and append to external list
            command_str = command_str[:-1] + ";\n" # replaces last + sign with ;
            self.est_command_lines.append(command_str)

    # Actuation commands (non-deterministic uncertainty from state/estimate abstraction)
    # Dynamics stage in PRISM code (from state & estimate -> state non-deterministically)
    def gen_act_commands(self):
        
        self.command_lines = []
        state_keys = self.model.state_keys
        next_states = self.model.next_states

        # Format states state_keys and next_states set as PRISM command lines
        for curr_state in state_keys:

            # Unpack and get the list of next possible states for this current state
            xbar, vbar, errbar = curr_state
            possible_states = next_states[curr_state]
            
            # Build a guard string for the current state
            guard_str = f"(est=false)&(term=false)&(setup=false)&(x={xbar})&(v={vbar})&(err={errbar})"
            
            if self.method == "approx":
                xn, vn = possible_states
                
                if xn in self.goal_set:
                    update_str = f"(x'={xn})&(v'={vn})&(term'=true)" # Build in terminal flag using goal set &(step'=min(100, step+1))
                else:
                    update_str = f"(x'={xn})&(v'={vn})&(est'=true)"
                
                # Combine into PRISM command syntax
                cmd_str = f"[] {guard_str} -> {update_str};\n"
                self.act_command_lines.append(cmd_str)

            elif self.method == "conserve":
                # Build a command line for each possible next state
                for possible_state in possible_states:
                    xn, vn = possible_state

                    # Clamp the states to maintain PRISM state space range
                    # xn = max(self.minxbar, min(xn, self.maxxbar))
                    # vn = max(self.minvbar, min(vn, self.maxvbar))
                    
                    if xn in self.goal_set:
                        update_str = f"(x'={xn})&(v'={vn})&(term'=true)" # Build in terminal flag using goal set &(step'=min(100, step+1))
                    else:
                        update_str = f"(x'={xn})&(v'={vn})&(est'=true)"
                    
                    # Combine into PRISM command syntax
                    cmd_str = f"[] {guard_str} -> {update_str};\n"
                    self.act_command_lines.append(cmd_str)

    def gen_goal_set(self):

        # Access abstraction parameters
        sx = self.model.sx
        x_space = self.model.x_space

        # Define discrete space
        xbar_space = np.arange(int(x_space[0]/sx), int(x_space[1]/sx)+1)

        # Generate goal set
        gx = 0.5
        self.goal_set = []
        for xb in xbar_space:
            corners = [xb*sx, (xb*sx)+sx]
            if all(x >= gx for x in corners):
                self.goal_set.append(int(xb))

    # Create new .pm file
    # Re-running results in erasure of existing lines
    def make(self):
        
        # Append model type and variables
        self.all_lines = []
        self.all_lines.append("mdp\n")
        self.all_lines.append("module dynamics\n")
        self.all_lines.append(f"x : [{self.minxbar}..{self.maxxbar}] init {0};\n")
        self.all_lines.append(f"v : [{self.minvbar}..{self.maxvbar}] init {0};\n")
        self.all_lines.append(f"err : [{self.minerrbar}..{self.maxerrbar}] init {self.minerrbar};\n")
        self.all_lines.append(f"term : bool init false;\n")
        self.all_lines.append(f"setup : bool init true;\n")
        self.all_lines.append(f"est : bool init true;\n")
        
        # Create and append initial distribution commands
        body = "+\n".join(self.init_command_lines) + ";"
        init_command = f"[](setup=true)->\n{body}"
        self.all_lines.append(init_command)
        self.all_lines.append("\n")

        # Append main commands
        for cmd in self.est_command_lines:
            self.all_lines.append(cmd)
        for cmd in self.act_command_lines:
            self.all_lines.append(cmd)
        self.all_lines.append("endmodule")

        # Write/rewrite .pm model file
        with open(self.modelpath, 'w') as file:
            file.writelines(self.all_lines)


def generate_prism_model(args):

    # Generate PRISM code for MountainCar
    init_state = args['init_state']
    prism_model = pm(args)
    prism_model.gen_goal_set()
    prism_model.gen_init_commands([init_state[0]], [init_state[1]]) # adjust initial state set as desired
    prism_model.gen_est_commands()
    prism_model.gen_act_commands()
    prism_model.make()

# Sample usage
if __name__ == "__main__":

    # Find paths with sample data and intervals
    import sys
    import os
    PATH = os.path.dirname(os.path.abspath(sys.argv[0]))

    # Make the custom PRISM model
    args = {
        'modelpath': f'{PATH}/prism-models/sample-model.pm',
        'intpath': f'{PATH}/data/sample-intervals.pkl',
        'method': 'conserve',
        'x_space': [-1.2, 0.6],
        'v_space': [-0.07, 0.07],
        'serr': 0.1,
        'err_space': [-0.2, 0.4],
        'sx': 0.05,
        'sv': 0.01,
        'init_state': [6, 7],
    }
    print("Generating PRISM model...")
    generate_prism_model(args)
    print("PRISM model generated at:", args['modelpath'])
    