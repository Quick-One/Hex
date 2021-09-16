from Hex_class import HexState
from config import RAVE_constants
from math import sqrt, log
from random import choice
from copy import deepcopy

"""
NODE:
    VALUE STORAGE:
        1) Number of times visited (N)
        2) Total reward obtained (Q)
        3) Parent (default = None)
        4) children - empty list
        5) N rave
        6) Q rave
        7) move required to move parent to current node
    FUNCTIONS:
        1) Value of node using formula
        2) add_children
"""

"""
MCTS Agent:
    VALUE STORAGE:
        1) Root node
        2) root_state
        3) run time limit
    FUNCTIONS:
        1) search
        2) select_node
        3) rollout
        4) backpropogate
        5) expansion
"""

class Node:
    def __init__(self, move: tuple, parent = None):
        self.parent = parent
        self.move = move
        #move : Node()
        self.children = {}

        self.N = 0
        self.Q = 0

        self.N_rave = 0
        self.Q_rave = 0

    @property
    def value(self, rave_cons, explore):
        rave_weight = max(0, 1 - (self.N/RAVE_constants.rave_const))
        UCT_value = self.Q/self.N + explore*sqrt(2 * log(self.parent.N/self.N))
        rave_value = self.Q_rave/self.N_rave if self.N_rave != 0 else 0
        
        value = (1 - rave_weight)*UCT_value + rave_weight*rave_value
        return value

    def add_children(self, new_children: dict):
        self.children.update(new_children)

"""
Each iteration:
1) Selection
2) Expansion
3) Simulation(Rollout)
4) Backpropogation
"""

class MCTSAgent:
    def __init__(self, root_state) -> None:
        root_node = Node()

    def simulate(self, state: HexState):
        curr_state = deepcopy(state)
        possible_moves = curr_state.possible_actions()
        
        while curr_state.winner == None:
            curr_action = choice(possible_moves)
            curr_state.step(curr_action)
            possible_moves.remove(curr_action)

    
        