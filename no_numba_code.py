from random import randint

import numpy as np
from Hex_utils import intmove_to_tupl

from settings import game_settings
from math import sqrt, log

size = game_settings.board_size

TL = np.array([size, +1], np.int64)
TC = np.array([-1, +1, size-1, size], np.int64)
TR = np.array([-1, size-1, size], np.int64)
ML = np.array([-size, -size+1, +1, size], np.int64)
MC = np.array([-size, -size+1, -1, +1, size, size-1], np.int64)
MR = np.array([-size, -1, size-1, size], np.int64)
BL = np.array([-size, -size+1, +1], np.int64)
BC = np.array([-size, -size+1,  -1, +1], np.int64)
BR = np.array([-size, -1], np.int64)



def wht_nfind(x, wht_parent, wht_rank, wht_groups):
    if x not in wht_parent:
        wht_parent[x] = x
        wht_rank[x] = 0
        wht_groups[x] = np.array([x], dtype=np.int64)

    # If root node then return node itself.
    parent_x = wht_parent[x]
    if x == parent_x:
        return x

    parent_parent_x = wht_parent[parent_x]
    if parent_parent_x == parent_x:
        return parent_x

    # Compress treee by bringing passed node above by making it's parent, it's parent's parent
    wht_parent[x] = parent_parent_x

    return wht_nfind(x, wht_parent, wht_rank, wht_groups)


def blk_nfind(x, blk_parent, blk_rank, blk_groups):
    if x not in blk_parent:
        blk_parent[x] = x
        blk_rank[x] = 0
        blk_groups[x] = np.array([x], dtype=np.int64)

    # If root node then return node itself.
    parent_x = blk_parent[x]
    if x == parent_x:
        return x

    parent_parent_x = blk_parent[parent_x]
    if parent_parent_x == parent_x:
        return parent_x

    # Compress treee by bringing passed node above by making it's parent, it's parent's parent
    blk_parent[x] = parent_parent_x

    return blk_nfind(x, blk_parent, blk_rank, blk_groups)



def fetch_neighbours(cell, size):
    x = cell//size
    y = cell % size

    if x == 0:
        if y == 0:
            return cell + TL
        elif y == size-1:
            return cell + TR
        else:
            return cell + TC
    elif x == size-1:
        if y == 0:
            return cell + BL
        elif y == size-1:
            return cell + BR
        else:
            return cell + BC
    else:
        if y == 0:
            return cell + ML
        elif y == size-1:
            return cell + MR
        else:
            return cell + MC



class HexState:

    def __init__(self,
                 size, 
                 board,
                 to_play,
                 EDGE_START,
                 EDGE_FINISH,
                 blk_parent,
                 blk_rank,
                 blk_groups,
                 wht_parent,
                 wht_rank,
                 wht_groups,
                 ):
        self.size = size
        self.board = board
        self.to_play = to_play
        self.EDGE_START = EDGE_START
        self.EDGE_FINISH = EDGE_FINISH

        self.blk_parent = blk_parent
        self.blk_rank = blk_rank
        self.blk_groups = blk_groups

        self.wht_parent = wht_parent
        self.wht_rank = wht_rank
        self.wht_groups = wht_groups

    def copy(self):
        copy_blk_groups = {}
        for key, value in self.blk_groups.items():
            copy_blk_groups[key] = value.copy()
            
        copy_wht_groups = {}
        for key, value in self.wht_groups.items():
            copy_wht_groups[key] = value.copy()
            
        
        
        return HexState(self.size, 
                        self.board.copy(), 
                        self.to_play, 
                        self.EDGE_START, 
                        self.EDGE_FINISH, 
                        self.blk_parent.copy(), 
                        self.blk_rank.copy(), 
                        copy_blk_groups, 
                        self.wht_parent.copy(), 
                        self.wht_rank.copy(), 
                        copy_wht_groups)
    
    def show_board(self):
        self.wht_groups[1] = np.zeros(0, dtype=np.int64)
        self.wht_groups[1] = np.append(self.wht_groups[1], 2)
        return self.wht_groups

    def possible_moves(self):
        return np.where(self.board == 0)[0]

    def blk_find(self, x):
        return blk_nfind(x, self.blk_parent, self.blk_rank, self.blk_groups)

    def wht_find(self, x):
        return wht_nfind(x, self.wht_parent, self.wht_rank, self.wht_groups)

    def blk_join(self, x, y):
        # Find root/parent node of x and y
        root_x = self.blk_find(x)
        root_y = self.blk_find(y)

        # Return false if x and y are in the same group
        if root_x == root_y:
            return False

        # Add the tree with the smaller rank to the one with the higher rank and delete the smaller tree.
        if self.blk_rank[root_x] < self.blk_rank[root_y]:
            self.blk_parent[root_x] = root_y

            self.blk_groups[root_y] = np.append(
                self.blk_groups[root_y], self.blk_groups[root_x])
            del self.blk_groups[root_x]

        elif self.blk_rank[root_x] > self.blk_rank[root_y]:
            self.blk_parent[root_y] = root_x

            self.blk_groups[root_x] = np.append(
                self.blk_groups[root_x], self.blk_groups[root_y])
            del self.blk_groups[root_y]

        # If the trees have same rank, add one to the other and increase it's rank.
        else:
            self.blk_parent[root_x] = root_y
            self.blk_rank[root_y] += 1

            self.blk_groups[root_y] = np.append(
                self.blk_groups[root_y], self.blk_groups[root_x])
            del self.blk_groups[root_x]

        return True

    def wht_join(self, x, y):
        # Find root/parent node of x and y
        root_x = self.wht_find(x)
        root_y = self.wht_find(y)

        # Return false if x and y are in the same group
        if root_x == root_y:
            return False

        # Add the tree with the smaller rank to the one with the higher rank and delete the smaller tree.
        if self.wht_rank[root_x] < self.wht_rank[root_y]:
            self.wht_parent[root_x] = root_y

            self.wht_groups[root_y] = np.append(
                self.wht_groups[root_y], self.wht_groups[root_x])
            del self.wht_groups[root_x]

        elif self.wht_rank[root_x] > self.wht_rank[root_y]:
            self.wht_parent[root_y] = root_x

            self.wht_groups[root_x] = np.append(
                self.wht_groups[root_x], self.wht_groups[root_y])
            del self.wht_groups[root_y]

        # If the trees have same rank, add one to the other and increase it's rank.
        else:
            self.wht_parent[root_x] = root_y
            self.wht_rank[root_y] += 1

            self.wht_groups[root_y] = np.append(
                self.wht_groups[root_y], self.wht_groups[root_x])
            del self.wht_groups[root_x]

        return True

    def place_stone(self, cell, player):
        if self.board[cell] == 0:
            self.board[cell] = player

        if player == 1:
            if cell//self.size == 0:
                self.blk_join(self.EDGE_START, cell)

            elif cell//self.size == self.size - 1:
                self.blk_join(self.EDGE_FINISH, cell)

            for neighbour in fetch_neighbours(cell, self.size):
                if self.board[neighbour] == 1:
                    self.blk_join(neighbour, cell)

        elif player == -1:
            if cell % self.size == 0:
                self.wht_join(self.EDGE_START, cell)

            elif cell % self.size == self.size - 1:
                self.wht_join(self.EDGE_FINISH, cell)

            for neighbour in fetch_neighbours(cell, self.size):
                if self.board[neighbour] == -1:
                    self.wht_join(neighbour, cell)

    def step(self, cell):

        if self.to_play == 1:
            self.place_stone(cell, 1)
            self.to_play = -1
        elif self.to_play == -1:
            self.place_stone(cell, -1)
            self.to_play = 1

    def winner(self):
        """
        Return a number corresponding to the winning player,
        or none if the game is not over.
        """
        if self.wht_find(self.EDGE_START) == self.wht_find(self.EDGE_FINISH):
            return -1
        elif self.blk_find(self.EDGE_START) == self.blk_find(self.EDGE_FINISH):
            return 1
        else:
            return 0

    def get_board(self):
        return self.board

    def get_to_play(self):
        return self.to_play

    # def printer(self):
    #     print(self.blk_parent)
    #     print(self.blk_rank)
    #     print(self.blk_groups)

    # def dummy_operation(self):
    #     for key in self.blk_groups:
    #         val = self.blk_groups[key] + 100
    #         self.blk_groups[key] = val
    #     print(self.blk_groups)
            
def create_empty_board(size):


    brd = np.zeros(size**2, dtype=np.int64)
    to_play = 1
    EDGE_START = 1000
    EDGE_FINISH = -1000
    blk_parent = {}
    blk_rank = {}
    blk_groups = {}
    wht_parent = {}
    wht_rank = {}
    wht_groups = {}

    board = HexState(size, brd, to_play, EDGE_START, EDGE_FINISH, blk_parent, blk_rank, blk_groups, wht_parent, wht_rank, wht_groups)
    return board


class Node:
    def __init__(self, parent, move):
        self.parent = parent
        self.move = move
        self.children = np.zeros(0, dtype=np.int64)

        self.N = 0
        self.Q = 0
        self.N_rave = 0
        self.Q_rave = 0

    def value(self, turn):
        if self.N == 0:
            return 1000

        Q = turn*self.Q
        UCT_value = (Q/self.N) + 0.5 * sqrt(2 * log(self.parent.N/self.N))
        return UCT_value

    def set_children(self, new_children):
        self.children = np.append(self.children, new_children)

    def get_stats(self):
        print(
            f'N:{self.N}, Q:{self.Q}, Q_r:{self.Q_rave}, N_r:{self.N_rave}, M:{self.move}')
        print(self.value())

    def QbyN(self):
        return self.Q/self.N

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

def leaf_node(root_node: Node, root_state, mem, mem_addrs):

    node = root_node
    state = root_state

    while node.children.size != 0:

        array = np.zeros(0, np.int64)
        benchmark = -1000000

        for child_mem_addrs in node.children:
            child = mem[child_mem_addrs]
            value_of_child = child.value(state.to_play)

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

def rollout(state):

    moves = state.possible_moves()

    while state.winner() == 0:
        move_index = randint(0, moves.size-1)
        move = moves[move_index]
        state.step(move)
        moves = np.delete(moves, move_index)

    return state.winner()

def backup(outcome, node):

    while node is not None:
        node.N += 1
        node.Q += outcome
        node = node.parent

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
        winner = rollout(new_state)

        if new_state.winner() == 1:
            blk += 1
        else:
            wht += 1

        backup(winner,node)

        num_simulation += 1
    print(len(memory))
    # array = np.zeros(0, dtype=np.int64)
    # benchmark = float('-inf')
    
    # for child_mem_addrs in root_node.children:
    #     child = memory[child_mem_addrs]
    #     value_of_child = child.QbyN()


    #     if value_of_child > benchmark:
    #         benchmark = value_of_child
    #         array = np.zeros(0, np.int64)
    #         array = np.append(array, child_mem_addrs)

    #     elif value_of_child == benchmark:
    #         array = np.append(array, child_mem_addrs)

    # selected_index_from_array = randint(0, array.size-1)
    # node = memory[array[selected_index_from_array]]
    # move = node.move
    # return move
    for child_mem_addrs in root_node.children:
        child = memory[child_mem_addrs]
        print(child.N, child.value(root_state.to_play),intmove_to_tupl(child.move, root_state.size))

board = create_empty_board(game_settings.board_size)

from time import perf_counter
start = perf_counter()
fetch_best_move(board, 10000)
print(perf_counter()-start)