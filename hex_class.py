from typing import List
import numpy as np
from string import ascii_letters
from collections import defaultdict
from queue import Queue

import random

BOARD_SIZE = 6

from unionfind import UnionFind

EDGE1 = 1
EDGE2 = 2

NEIGHBOUR_PATTERNS = ((-1, 0), (0, -1), (-1, 1), (0, 1), (1, 0), (1, -1))

# White = 1; Black = -1

class HexState:
    def __init__(self, size=BOARD_SIZE) -> None:
        self.size = size
        self.board = np.zeros((size, size))

        self.to_play = -1

        self.n_moves = {-1: 0, 1: 0}

        self.white_groups = UnionFind()
        self.black_groups = UnionFind()
        self.white_groups.set_ignored_elements([EDGE1, EDGE2])
        self.black_groups.set_ignored_elements([EDGE1, EDGE2])

    def neighbours(self, cell: tuple) -> list:
        """
        Return list of neighbors of the passed cell.

        Args:
            cell tuple):
        """
        x = cell[0]
        y = cell[1]
        return [(n[0] + x, n[1] + y) for n in NEIGHBOUR_PATTERNS
                if (0 <= n[0] + x < self.size and 0 <= n[1] + y < self.size)]

    def possible_actions(self) -> list:
        """
        Get a list of all moves possible in the current board state.
        """
        moves = [(x, y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == 0]
        return moves

    @staticmethod
    def move_to_string(move: tuple) -> str:
        i, j = move
        return f"{ascii_letters[j]}{i+1}"

    @property
    def winner(self) -> int:
        """
        Return a number corresponding to the winning player,
        or none if the game is not over.
        """
        if self.white_groups.connected(EDGE1, EDGE2):
            return 1
        elif self.black_groups.connected(EDGE1, EDGE2):
            return -1
        else:
            return None

    def set_turn(self, player: int) -> None:
        """
        Set the player to take the next move.
        Raises:
            ValueError if player is not -1 or 1
        """
        if (player == 1 or player == -1) and player != 0:
            self.to_play = player
        else:
            raise ValueError(f'Invalid player {player}')

    def turn(self) -> int:
        return self.to_play
    
    def step(self, cell: tuple) -> None:
        if self.to_play == 1:
            self.place_white_stone(cell)
            self.to_play = -1
        elif self.to_play == -1:
            self.place_black_stone(cell)
            self.to_play = 1

    def place_white_stone(self, cell):
        """
        Place a white stone.

        Args:
            cell (tuple): row and column of the cell
        """
        if self.board[cell] == 0:
            self.board[cell] = 1
            self.n_moves[1] += 1
        else:
            raise ValueError(f"Cell {cell} already occupied.")

        # If the placed cell touches any edge cell, connect it
        if cell[0] == 0:
            self.white_groups.join(EDGE1, cell)
        elif cell[0] == self.size - 1:
            self.white_groups.join(EDGE2, cell)

        # Join any groups connected by the new stone
        for neighbor in self.neighbours(cell):
            if self.board[neighbor] == 1:
                self.white_groups.join(neighbor, cell)

    def place_black_stone(self, cell):
        """
        Place a black stone.

        Args:
            cell (tuple): row and column of the cell
        """
        if self.board[cell] == 0:
            self.board[cell] = -1
            self.n_moves[-1] += 1
        else:
            raise ValueError(f"Cell {cell} already occupied.")

        # If the placed cell touches any edge cell, connect it
        if cell[1] == 0:
            self.black_groups.join(EDGE1, cell)
        elif cell[1] == self.size - 1:
            self.black_groups.join(EDGE2, cell)

        # Join any groups connected by the new stone
        for neighbor in self.neighbours(cell):
            if self.board[neighbor] == -1:
                self.black_groups.join(neighbor, cell)

    def shortest_connection(self, board: np.ndarray, size: int, color: int) -> list:
        """
        If board is terminated returns the winning connection.
        """
        START_NODE = 'START'
        END_NODE = 'END'

        graph_dict = defaultdict(list)
        for i, j in ((x, y) for x in range(size) for y in range(size)):
            if board[i,j] != color:
                continue
            
            for neighbour in self.neighbours((i, j), color, size, board):
                graph_dict[(i, j)].append(neighbour)

            if color == 1:
                if i == 0:
                    graph_dict[START_NODE].append((i, j))
                    graph_dict[(i, j)].append(START_NODE)
                elif i == (size - 1):
                    graph_dict[END_NODE].append((i, j))
                    graph_dict[(i, j)].append(END_NODE)

            if color == -1:
                if j == 0:
                    graph_dict[START_NODE].append((i, j))
                    graph_dict[(i, j)].append(START_NODE)
                elif j == (size - 1):
                    graph_dict[END_NODE].append((i, j))
                    graph_dict[(i, j)].append(END_NODE)

        cost_dict = {}
        cost_dict[START_NODE] = 0
        queue = Queue()
        queue.put(START_NODE)
        while not(queue.empty()):
            node = queue.get()
            node_cost = cost_dict[node]
            for neighbour in graph_dict[node]:
                if cost_dict.get(neighbour, None) == None:
                    cost_dict[neighbour] = node_cost + 1
                    if neighbour == END_NODE:
                        break
                    queue.put(neighbour)

        shortest_path = []
        node = END_NODE
        while node != START_NODE:
            next_node = None
            next_node_value = float('inf')

            for neighbour in graph_dict[node]:
                neighbour_value = cost_dict.get(neighbour, float('inf'))
                if neighbour_value < next_node_value:
                    next_node_value = neighbour_value
                    next_node = neighbour
            node = next_node
            shortest_path.append(node)
        shortest_path.remove(START_NODE)

        return shortest_path

        
def test_agent():
    board = HexState()
    while board.winner == None:
        action = random.choice(board.possible_actions())
        board.step(action)

    print(board.winner)

# test_agent()