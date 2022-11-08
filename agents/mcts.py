import logging
from copy import deepcopy
from math import log, sqrt
from random import choice
from time import perf_counter

from hex.agent import HexAgent
from hex.board import P1, P2, Hex, HexBase, possible_moves

EXPLORE = 0.5
RAVE_CONST = 300
ROLLOUT = 5000
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
        4) backpropagate
        5) expand
"""


class Node:
    def __init__(self, move: tuple = None, parent=None):
        self.parent = parent
        self.move = move
        self.children = []

        self.N = 0
        self.Q = 0
        self.N_rave = 0
        self.Q_rave = 0

    def get_value(self):
        return self.N

    @property
    def value(self, explore=EXPLORE):
        if self.N == 0:
            return 0 if explore == 0 else float('inf')
        else:
            rave_weight = max(0, 1 - (self.N_rave/RAVE_CONST))
            UCT_value = self.Q/self.N + explore * \
                sqrt(2 * log(self.parent.N/self.N))
            rave_value = self.Q_rave/self.N_rave if self.N_rave != 0 else 0

            value = (1 - rave_weight)*UCT_value + rave_weight*rave_value
            return value

    def add_children(self, new_children: dict):
        self.children.extend(new_children)

    @property
    def isleaf(self):
        return True if len(self.children) == 0 else False


"""
Each iteration:
1) Selection
2) Expansion
3) Simulation(Rollout)
4) Backpropogation
"""


class MCTSAgent:
    def __init__(self, root_state: HexBase) -> None:
        self.root_node = Node()
        self.root_state = deepcopy(root_state)
        self.num_rollouts = 0

    def simulate(self, state: HexBase):
        curr_state = deepcopy(state)
        legal_moves = possible_moves(curr_state.size, curr_state.board)

        while curr_state.winner is None:
            curr_action = choice(legal_moves)
            curr_state.step(curr_action)
            legal_moves.remove(curr_action)

        black_rave_pts = []
        white_rave_pts = []

        for x in range(state.size):
            for y in range(state.size):
                if state.board[(x, y)] == P1:
                    black_rave_pts.append((x, y))
                elif state.board[(x, y)] == P2:
                    white_rave_pts.append((x, y))

        return curr_state.winner, black_rave_pts, white_rave_pts

    def expand(self, node: Node, state):
        if state.winner is not None:
            return False

        node.add_children([Node(move, node)
                          for move in possible_moves(state.size, state.board)])
        return True

    def select_node(self):
        node = self.root_node
        state = deepcopy(self.root_state)

        while not node.isleaf:
            benchmark = float('-inf')
            max_children = []
            for child_node in node.children:
                curr_value = child_node.value
                if curr_value > benchmark:
                    benchmark = curr_value
                    max_children = []
                    max_children.append(child_node)
                elif curr_value == benchmark:
                    max_children.append(child_node)

            node = choice(max_children)
            state.step(node.move)

            if node.N == 0:
                return node, state

        if self.expand(node, state):
            node = choice(node.children)
            state.step(node.move)
        return node, state

    def backpropagate(self, node, outcome, turn, black_rave_pts, white_rave_pts):
        reward = -1 if outcome == turn else 1

        while node is not None:
            for i, child_node in enumerate(node.children):
                if turn == P2:
                    if child_node.move in white_rave_pts:
                        node.children[i].Q_rave += -reward
                        node.children[i].N_rave += 1
                else:
                    if child_node.move in black_rave_pts:
                        node.children[i].Q_rave += -reward
                        node.children[i].N_rave += 1

            node.N += 1
            node.Q += reward

            turn = -turn
            reward = -reward

            node = node.parent

    def search(self, num_rollout=ROLLOUT):
        rollout_limit = num_rollout
        num_rollouts = 0
        while num_rollouts < rollout_limit:
            node, state = self.select_node()
            turn = state.turn
            outcome, black, white = self.simulate(state)
            self.backpropagate(node, outcome, turn, black, white)
            num_rollouts += 1
        self.num_rollouts = num_rollouts

    def best_move(self) -> tuple:
        benchmark = float('-inf')
        max_children = []
        for child_node in self.root_node.children:
            curr_value = child_node.get_value()
            if curr_value > benchmark:
                benchmark = curr_value
                max_children = []
                max_children.append(child_node)
            elif curr_value == benchmark:
                max_children.append(child_node)

        return choice(max_children).move


def best_move(board: Hex):
    board = board.get_base()
    agent = MCTSAgent(board)
    start = perf_counter()
    agent.search()
    best_move = agent.best_move()
    logging.info(f'Completed {agent.num_rollouts} rollouts in {(perf_counter()-start):.3f}s.')
    return best_move


HexAgent('MCTS', best_move)
