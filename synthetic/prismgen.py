# Libraries
from pagen import PA
from fractions import Fraction
import numpy as np
import pickle as pkl
import os
import sys

# Prism model
class pm:

    def __init__(self, modelpath):
        PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # file folder path
        self.modelpath = modelpath
        self.init_command_lines = [] # initial distribution if desired
        self.est_command_lines = [] # main estimation prism commands
        self.act_command_lines = [] # main actuation prism commands
        self.term_command_line = "" # the conditions foer which "term" becomes true
        self.all_lines = [] # all prism code lines
        self.model = PA() # Probabilistic automaton
        self.minxbar, self.maxxbar = int(self.model.x_space[0]/self.model.sx), int(self.model.x_space[1]/self.model.sx)
        self.minybar, self.maxybar = int(self.model.y_space[0]/self.model.sy), int(self.model.y_space[1]/self.model.sy)
        self.minxhatbar, self.maxxhatbar = int(self.model.xhat_space[0]/self.model.sxhat), int(self.model.xhat_space[1]/self.model.sxhat)
        self.minyhatbar, self.maxyhatbar = int(self.model.yhat_space[0]/self.model.syhat), int(self.model.yhat_space[1]/self.model.syhat)
        self.goal_set = []
        self.model.abstract() # Abstract dynamics into probabilistic automaton 
        num_transitions = [len(state_list) for state_list in self.model.next_states.values()]
        print(f"deadlocks = {any([num==0 for num in num_transitions])}")
        print(f"transition stats: avg={round(np.mean(num_transitions), 1)}, min={np.min(num_transitions)}, max={np.max(num_transitions)}")
        with open(f"{PATH}/data/sample-intervals.pkl", "rb") as f:
            DATA = pkl.load(f)
        self.lower_bounds = DATA["lower_bounds"]
        self.upper_bounds = DATA["upper_bounds"]

    # Generate an initial distribution in PRISM syntax
    # Re-running results in erasure of existing lines
    def gen_init_commands(self, x_inits, y_inits):
        
        self.init_command_lines = []
        n = len(x_inits) * len(y_inits)
        prob = Fraction(1, n)
        for x in x_inits:
            for y in y_inits:
                # Build one line with probability prob and the updates
                line = f"{prob}:(x'={x})&(y'={y})&(setup'=false)"
                self.init_command_lines.append(line)

    # Estimation commands (imprecisely probabilistic; builds in Wilson confidence intervals)
    # Estimation stage in PRISM code (from state -> state & estimate probabilisitically)
    # Re-running results in erasure of existing liness
    def gen_est_commands(self):

        self.est_command_lines = []

        # Access confidence interval upper/lower bounds from PA
        lower_bounds = self.lower_bounds
        upper_bounds = self.upper_bounds

        # Access abstraction parameters
        sx = self.model.sx
        x_space = self.model.x_space
        sy = self.model.sy
        y_space = self.model.y_space
        sxhat = self.model.sxhat
        xhat_space = self.model.xhat_space
        syhat = self.model.syhat
        yhat_space = self.model.yhat_space

        # Define discrete space
        xbar_space = np.arange(int(x_space[0]/sx), int(x_space[1]/sx))
        ybar_space = np.arange(int(y_space[0]/sy), int(y_space[1]/sy))
        xhatbar_space = np.arange(int(xhat_space[0]/sxhat), int(xhat_space[1]/sxhat))
        yhatbar_space = np.arange(int(yhat_space[0]/syhat), int(yhat_space[1]/syhat))

        # Loop through discrete space and construct PRISM commands
        for i in range(len(xbar_space)):
            xbar = xbar_space[i]
            for j in range(len(ybar_space)):
                ybar = ybar_space[j]
                command_str = f"[](est=true)&(term=false)&(setup=false)&(x={xbar})&(y={ybar})->"
                lowers = lower_bounds[i, j] # lower bounds of CI indexed at this state
                uppers = upper_bounds[i, j] # upper bounds of CI indexed at this state

                # Loop through estimation space; write probabilistic mapping command with CI
                for k in range(len(xhatbar_space)):
                    xhat_bar = xhatbar_space[k]
                    for p in range(len(yhatbar_space)):
                        yhat_bar = yhatbar_space[p]
                        lower = lowers[k][p]
                        upper = uppers[k][p]
                        command_str += f"[{lower}, {upper}]:(xhat'={xhat_bar})&(yhat'={yhat_bar})&(est'=false)+"
                
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
            xbar, ybar, xhatbar, yhatbar = curr_state
            possible_states = next_states[curr_state]
            
            # Build a guard string for the current state
            guard_str = f"(est=false)&(term=false)&(setup=false)&(x={xbar})&(y={ybar})&(xhat={xhatbar})&(yhat={yhatbar})"
            
            # Build a command line for each possible next state
            for possible_state in possible_states:
                xn, yn = possible_state

                # Clamp the states to maintain PRISM state space range
                xn = max(self.minxbar, min(xn, self.maxxbar))
                yn = max(self.minybar, min(yn, self.maxybar))
                
                if possible_state in self.goal_set:
                    update_str = f"(x'={xn})&(y'={yn})&(term'=true)" # Build in terminal flag using goal set &(step'=min(100, step+1))
                else:
                    update_str = f"(x'={xn})&(y'={yn})&(est'=true)"
                
                # Combine into PRISM command syntax
                cmd_str = f"[] {guard_str} -> {update_str};\n"
                self.act_command_lines.append(cmd_str)

    def gen_goal_set(self):

        # Access abstraction parameters
        sx = self.model.sx
        sy = self.model.sy
        x_space = self.model.x_space
        y_space = self.model.y_space

        # Define discrete space
        xbar_space = list(range(int(x_space[0]/sx), int(x_space[1]/sx)))
        ybar_space = list(range(int(y_space[0]/sy), int(y_space[1]/sy)))
        x_min_index, x_max_index = min(xbar_space), max(xbar_space)
        y_min_index, y_max_index = min(ybar_space), max(ybar_space)

        # Generate goal set
        gx, gy = self.model.sys.goal_state
        dist_threshold = self.model.sys.success_dist
        self.goal_set = []
        for i in range(x_min_index, x_max_index + 1):
            for j in range(y_min_index, y_max_index + 1):
                # Define the corners of the bin (assuming bins are [i*dx, (i+1)*dx) and similarly for y).
                corners = [
                    (i * sx, j * sy),
                    ((i + 1) * sx, j * sy),
                    (i * sx, (j + 1) * sy),
                    ((i + 1) * sx, (j + 1) * sy)]
                if all(np.sqrt((x - gx)**2 + (y - gy)**2) <= dist_threshold for x, y in corners):
                    self.goal_set.append((i, j))
        print(self.goal_set)

    # Create new .pm file
    # Re-running results in erasure of existing lines
    def make(self):
        
        # Append model type and variables
        self.all_lines = []
        self.all_lines.append("mdp\n")
        self.all_lines.append("module dynamics\n")
        self.all_lines.append(f"x : [{self.minxbar}..{self.maxxbar}] init {0};\n")
        self.all_lines.append(f"y : [{self.minybar}..{self.maxybar}] init {0};\n")
        self.all_lines.append(f"xhat : [{self.minxhatbar}..{self.maxxhatbar}] init {self.minxhatbar};\n")
        self.all_lines.append(f"yhat : [{self.minyhatbar}..{self.maxyhatbar}] init {self.minyhatbar};\n")
        # self.all_lines.append(f"step : [{0}..{100}] init {0};\n")
        self.all_lines.append(f"term : bool init false;\n")
        self.all_lines.append(f"setup : bool init true;\n")
        self.all_lines.append(f"est : bool init true;\n")

        # Append terminal command
        # self.all_lines.append(self.term_command_line)
        
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

PATH = os.path.dirname(os.path.abspath(sys.argv[0])) # file folder path
prism_model = pm(f"{PATH}/prism-models/sample-model.pm")
prism_model.gen_goal_set()
prism_model.gen_init_commands([0], [0]) # specify any initial set over valid state space
prism_model.gen_est_commands()
prism_model.gen_act_commands()
prism_model.make()
