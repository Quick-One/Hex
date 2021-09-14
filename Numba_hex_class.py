from random import choice

import numpy as np
from numba import int64, njit, typed, types
from numba.experimental import jitclass

from settings import game_settings

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


parent_type = (types.int64, types.int64)
rank_type = (types.int64, types.int64)
groups_type = (types.int64, types.int64[:])

spec = [
    ('size', int64),
    ('board', int64[:]),
    ('to_play', int64),
    ('EDGE_START', int64),
    ('EDGE_FINISH', int64),

    ('blk_parent', types.DictType(*parent_type)),
    ('blk_rank', types.DictType(*rank_type)),
    ('blk_groups', types.DictType(*groups_type)),

    ('wht_parent', types.DictType(*parent_type)),
    ('wht_rank', types.DictType(*rank_type)),
    ('wht_groups', types.DictType(*groups_type)),

]


@njit(int64(int64, types.DictType(*parent_type), types.DictType(*rank_type), types.DictType(*groups_type)))
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


@njit(int64(int64, types.DictType(*parent_type), types.DictType(*rank_type), types.DictType(*groups_type)))
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


@njit(int64[:](int64, int64))
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


@jitclass(spec)
class HexState:

    def __init__(self, size):
        self.size = size
        self.board = np.zeros(size**2, dtype=np.int64)
        self.to_play = 1
        self.EDGE_START = 1000
        self.EDGE_FINISH = -1000

        self.blk_parent = typed.Dict.empty(*parent_type)
        self.blk_rank = typed.Dict.empty(*rank_type)
        self.blk_groups = typed.Dict.empty(*groups_type)

        self.wht_parent = typed.Dict.empty(*parent_type)
        self.wht_rank = typed.Dict.empty(*rank_type)
        self.wht_groups = typed.Dict.empty(*groups_type)

    def show_board(self):
        self.wht_groups[1] = np.zeros(0, dtype=np.int64)
        self.wht_groups[1] = np.append(self.wht_groups[1], 2)
        return self.wht_groups

    def possible_moves(self):
        return np.where(self.board == 0)[0]

    def blk_find(self, x):
        return blk_nfind(x, self.blk_parent, self.blk_rank, self.blk_groups)

    def wht_find(self, x) -> int:
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
            return None

    def get_board(self):
        return self.board


def _simulate(n):
    for _ in range(n):
        board = HexState(6)
        while board.winner() == None:
            action = choice(board.possible_moves())
            board.step(action)

        # from Hex_utils import visualize_board
        # print(board.winner())
        # d = board.get_board()
        # visualize_board(d.reshape(6,6))


def _main():
    from time import perf_counter
    a = perf_counter()
    _simulate(10)
    print(perf_counter()-a)

    for i in range(10):
        a = perf_counter()
        _simulate(10000)
        print(i, perf_counter()-a)


if __name__ == '__main__':
    _main()
    # _simulate(10)
    pass
