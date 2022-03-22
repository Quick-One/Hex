from collections import defaultdict
from queue import Queue
from random import choice
from string import ascii_letters
from typing import NamedTuple

import numpy as np

__all__ = ['Hex']

P1 = 1
EMPTY = 0
P2 = -1

EDGE_START = 's'
EDGE_FINISH = 'f'

NEIGHBOUR_PATTERNS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))

RULES_OF_HEX = (f'{"_"*80}',
                'Rules of Hex: ',
                '1. Players choose a color and take turns. Black moves first.',
                "2. On each turn one piece of player's color is placed in an empty hexagonal cell.",
                '3. The first player to form a connected path of their pieces linking the opposing sides of the board marked by their colour wins.',
                f'{"‾"*80}')


class player(NamedTuple):
    name: str
    is_AI: bool


def shortest_connection(board, size, player):
    """
    If board is terminated returns the winning connection.
    Uses BFS to find the shortest path.

    Args:
        board (np.ndarray): A numpy array representing the board in NxN shape.
        size (int): board size (N)
        player (int): player number

    Returns:
        List containg tuples representing cells in the shortest path.
    """
    EDGE_START = 'START'
    EDGE_FINISH = 'END'

    # Generating graph upon which BFS is applied
    graph_dict = defaultdict(list)
    for i, j in np.ndindex(size, size):
        if board[i, j] != player:
            continue

        for neighbour in neighbours((i, j), size):
            if board[neighbour] == player:
                graph_dict[(i, j)].append(neighbour)

        if player == P1:
            if i == 0:
                graph_dict[EDGE_START].append((i, j))
                graph_dict[(i, j)].append(EDGE_START)
            elif i == (size - 1):
                graph_dict[EDGE_FINISH].append((i, j))
                graph_dict[(i, j)].append(EDGE_FINISH)

        if player == P2:
            if j == 0:
                graph_dict[EDGE_START].append((i, j))
                graph_dict[(i, j)].append(EDGE_START)
            elif j == (size - 1):
                graph_dict[EDGE_FINISH].append((i, j))
                graph_dict[(i, j)].append(EDGE_FINISH)

    # BFS algorithm to find the cost of each node.
    cost_dict = {}
    cost_dict[EDGE_START] = 0
    queue = Queue()
    queue.put(EDGE_START)
    while not(queue.empty()):
        node = queue.get()
        node_cost = cost_dict[node]
        for neighbour in graph_dict[node]:
            if cost_dict.get(neighbour, None) == None:
                cost_dict[neighbour] = node_cost + 1
                if neighbour == EDGE_FINISH:
                    break
                queue.put(neighbour)

    # Iterating backword to get the shortest path.
    shortest_path = []
    node = EDGE_FINISH
    while node != EDGE_START:
        next_node = None
        next_node_value = float('inf')
        for neighbour in graph_dict[node]:
            neighbour_value = cost_dict.get(neighbour, float('inf'))
            if neighbour_value < next_node_value:
                next_node_value = neighbour_value
                next_node = neighbour
        node = next_node
        shortest_path.append(node)
    shortest_path.remove(EDGE_START)

    return shortest_path


def move_to_string(move):
    """
    Returns the coordinate in '$#' form where 
    $ = letter and # = digit.
    """
    i, j = move
    return f"{ascii_letters[j]}{i+1}"


def opponent(player):
    ''' Returns opponent of the given player '''
    if player == P1:
        return P2
    return P1


def neighbours(cell, size):
    """
    Returns a list of neighbours of the cell.
    Args:
        cell(tuple): Tuple of x, y coordinates of the cell.
        size(int): Size of the board.
    Returns:
        list: List of neighbours of the cell.
    """
    x, y = cell
    return [(dx+x, dy+y) for dx, dy in NEIGHBOUR_PATTERNS
            if (0 <= dx+x < size and 0 <= dy+y < size)]


def possible_moves(size, board):
    """
    Return a list of all moves possible in the current board state.
    Args:
        size(int): Size of the board.
        board(numpy.ndarray): Current board state.
    returns:
        list: List of all possible moves.
    """
    moves = [(x, y) for x in range(size)
             for y in range(size) if board[x][y] == EMPTY]
    return moves


class HexException(Exception):
    pass


class UnionFind:
    """
    Unionfind data structure specialized for finding hex connections.

    Attributes:
        parent (dict): Each group parent
        rank (dict): Each group rank
        groups (dict): Stores the groups and chain of cells
    """

    def __init__(self) -> None:
        self.parent = {}
        self.rank = {}
        self.groups = {}

    def join(self, x, y) -> bool:
        """
        Merge the groups of x and y if they were not already,
        return False if they were already merged, true otherwise.

        Args:
            x (tuple): game board cell
            y (tuple): game board cell

        """
        # Find root/parent node of x and y
        root_x = self.find(x)
        root_y = self.find(y)

        # Return false if x and y are in the same group
        if root_x == root_y:
            return False

        # Add the tree with the smaller rank to the one with the higher rank and delete the smaller tree.
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.groups[root_y].extend(self.groups[root_x])
            del self.groups[root_x]

        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.groups[root_x].extend(self.groups[root_y])
            del self.groups[root_y]

        # If the trees have same rank, add one to the other and increase it's rank.
        else:
            self.parent[root_x] = root_y
            self.rank[root_y] += 1
            self.groups[root_y].extend(self.groups[root_x])
            del self.groups[root_x]

        return True

    def find(self, x):
        """
        Get the root element of the group in which element x resides. 
        Uses grandparent compression to compress the tree on each find 
        operation so that future find operations are faster.
        Args:
            x (tuple): game board cell
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.groups[x] = [x]

        # If root node then return node itself.
        parent_x = self.parent[x]
        if x == parent_x:
            return x

        parent_parent_x = self.parent[parent_x]
        if parent_parent_x == parent_x:
            return parent_x

        # Compress treee by bringing passed node above by making it's parent, it's parent's parent
        self.parent[x] = parent_parent_x

        return self.find(parent_parent_x)

    def connected(self, x, y) -> bool:
        """
        Check if two elements are conneced.

        Args:
            x (tuple): game board cell
            y (tuple): game board cell
        """
        return self.find(x) == self.find(y)


class HexBase:

    def __init__(self, size):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.turn = P1
        self.groups = {P1: UnionFind(), P2: UnionFind()}

    @property
    def winner(self):
        """
        Return a number corresponding to the winning player,
        or none if the game is not over.
        """
        if self.groups[P2].connected(EDGE_START, EDGE_FINISH):
            return P2
        elif self.groups[P1].connected(EDGE_START, EDGE_FINISH):
            return P1
        else:
            return None

    def step(self, cell):
        """
        Place a stone on the board.
        And changes the underlying board state and unionfind data structures.

        Args:
            cell (tuple): row and column of the cell
            player (int): player by which stone has been placed
            coord_index: 0 for x, 1 for y to check for end condition row wise or column wise respectively
        """
        player = self.turn
        if self.board[cell] == EMPTY:
            self.board[cell] = player
        else:
            raise HexException(f"Cell {cell} already occupied.")

        coord_index = 0 if player == P1 else 1
        if cell[coord_index] == 0:
            self.groups[player].join(EDGE_START, cell)
        elif cell[coord_index] == self.size - 1:
            self.groups[player].join(EDGE_FINISH, cell)

        for neighbour in neighbours(cell, self.size):
            if self.board[neighbour] == player:
                self.groups[player].join(neighbour, cell)

        self.turn = opponent(player)


class Hex(HexBase):
    legend = {
        P1: "Player_1",
        P2: "Player_2",
    }

    def __init__(self, size,
                 player_1=player('Undefined', False),
                 player_2=player('Undefined', False)):
        super().__init__(size)
        self.move_history = []
        self.players = {P1: player_1, P2: player_2}

    def step(self, cell):
        """
        Modified step function to include adding moves to move_history list.
        """
        self.move_history.append(cell)
        super().step(cell)

    @property
    def current_player(self):
        return self.players[self.turn].name

    def __repr__(self):
        string = ""
        sep = ' - '
        for i in range(self.size-1, -self.size, -1):
            l = []
            for element in self.board.diagonal(i):
                if element == P1:
                    l.append('B')
                elif element == P2:
                    l.append('W')
                else:
                    l.append('*')
            row = sep.join(l).center(40)
            string += row + '\n'
        return string

    def get_base(self):
        base = HexBase(self.size)
        base.board = self.board
        base.groups = self.groups
        base.turn = self.turn
        return base
