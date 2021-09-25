from settings import game_settings
from Numba_hex_class import _simulate, create_empty_board
from random import randint
from math import sqrt, log
from numba import njit
from numba import int64, deferred_type, optional, float64
from numba.experimental import jitclass
import numpy as np

# NODE CLASS
Node_type = deferred_type()

spec = (

    ('parent', optional(Node_type)),
    ('move', int64),
    ('children', int64[:]),

    ('N', int64),
    ('Q', int64),
    ('N_rave', int64),
    ('Q_rave', int64),

)


@jitclass(spec)
class Node:
    def __init__(self, parent, move):
        self.parent = parent
        self.move = move
        self.children = np.zeros(0, dtype=np.int64)

        self.N = 0
        self.Q = 0
        self.N_rave = 0
        self.Q_rave = 0

    def value(self):
        if self.N == 0:
            return float64(100000)

        rave_weight = max(0, 1 - (self.N/300))
        UCT_value = (self.Q/self.N) + 0.5 * sqrt(2 * log(self.parent.N/self.N))
        if self.N_rave != 0:
            rave_value = self.Q_rave/self.N_rave
        else:
            rave_value = 0
        value = (1 - rave_weight)*UCT_value + rave_weight*rave_value

        return float64(value)

    def set_children(self, new_children):
        self.children = np.append(self.children, new_children)

    def get_stats(self):
        print(
            f'N:{self.N}, Q:{self.Q}, Q_r:{self.Q_rave}, N_r:{self.N_rave}, M:{self.move}')
        print(self.value())

    def QbyN(self):
        return self.Q/self.N

    
Node_type.define(Node.class_type.instance_type)

# EXPAND


@njit
def expand(parent: Node, state, mem, mem_address):

    if state.winner() != 0:
        return (0, mem_address)

    possible_moves = state.possible_moves()
    children = np.zeros_like(possible_moves)

    for index, move in enumerate(possible_moves):
        child_node = Node(parent, move)
        mem_address += 1
        mem[mem_address] = child_node
        children[index] = mem_address

    parent.set_children(children)
    return (1, mem_address)

# lead node


@njit
def leaf_node(root_node: Node, root_state, mem, mem_addrs):

    node = root_node
    state = root_state

    while node.children.size != 0:

        array = np.zeros(0, np.int64)
        benchmark = float64(-1000000)

        for child_mem_addrs in node.children:
            child = mem[child_mem_addrs]
            value_of_child = child.value()

            if value_of_child > benchmark:
                benchmark = value_of_child
                array = np.zeros(0, np.int64)
                array = np.append(array, child_mem_addrs)

            elif value_of_child == benchmark:
                array = np.append(array, child_mem_addrs)

        selected_index_from_array = randint(0, array.size-1)
        node = mem[array[selected_index_from_array]]
        move = node.move

        state.step(move)

        if node.N == 0:
            return (node, state, mem_addrs)

    bool_expand, mem_addrs = expand(node, state, mem, mem_addrs)
    if bool_expand:
        children_addrs = node.children
        child_index = randint(0, children_addrs.size-1)
        child_addrs = children_addrs[child_index]
        node = mem[child_addrs]
        state.step(node.move)

    return (node, state, mem_addrs)

# rollout


@njit
def rollout(state):

    moves = state.possible_moves()

    while state.winner() == 0:
        move_index = randint(0, moves.size-1)
        move = moves[move_index]
        state.step(move)
        moves = np.delete(moves, move_index)

    blk_rave_pieces = np.zeros(0, dtype=np.int64)
    wht_rave_pieces = np.zeros(0, dtype=np.int64)

    board = state.get_board()
    blk_rave_pieces = np.append(blk_rave_pieces, np.where(board == 1)[0])
    wht_rave_pieces = np.append(wht_rave_pieces, np.where(board == -1)[0])

    return (state.winner(), blk_rave_pieces, wht_rave_pieces)

# BACKUP


@njit
def backup(outcome, turn, node, blk_rave_pieces, wht_rave_pieces, mem):

    if outcome == turn:
        reward = -1
    else:
        reward = 1

    while node is not None:
        if turn == -1:
            for child_addrs in node.children:
                child = mem[child_addrs]

                if child.move in wht_rave_pieces:
                    child.N_rave += 1
                    child.Q_rave += -reward
        else:
            for child_addrs in node.children:
                child = mem[child_addrs]

                if child.move in blk_rave_pieces:
                    child.N_rave += 1
                    child.Q_rave += -reward

        node.N += 1
        node.Q += reward

        turn = -turn
        reward = -reward
        node = node.parent

# fetch_best_move


@njit
def fetch_best_move(state, limit):

    memory = {}
    memory_address = 0

    root_state = state.copy()
    root_node = Node(None, 20)
    memory[memory_address] = root_node

    num_simulation = 0
    blk = 0
    wht = 0
    while num_simulation < limit:

        state_copy = root_state.copy()

        node, new_state, memory_address = leaf_node(
            root_node, state_copy, memory, memory_address)
        turn = new_state.to_play
        winner, blk_rave_pieces, wht_rave_pieces = rollout(new_state)

        if new_state.winner() == 1:
            blk += 1
        else:
            wht += 1

        backup(winner, turn, node, blk_rave_pieces, wht_rave_pieces, memory)

        num_simulation += 1
    print(blk, wht)
    
    array = np.zeros(0, dtype=np.int64)
    benchmark = float64(-1000000)
    
    for child_mem_addrs in root_node.children:
        child = memory[child_mem_addrs]
        value_of_child = child.QbyN()

        if value_of_child > benchmark:
            benchmark = value_of_child
            array = np.zeros(0, np.int64)
            array = np.append(array, child_mem_addrs)

        elif value_of_child == benchmark:
            array = np.append(array, child_mem_addrs)

    selected_index_from_array = randint(0, array.size-1)
    node = memory[array[selected_index_from_array]]
    move = node.move
    return move
    


# Compiling JIT classes and functions
_simulate(100)
print('compiled board class')
board = create_empty_board(game_settings.board_size)
fetch_best_move(board, 1000)
print('compiled rave functions')