from collections import defaultdict
from queue import Queue
from random import choice
from string import ascii_letters

import numpy as np

from Hex_utils import visualize_board
from unionfind import UnionFind

# STATIC VARIABLES
EDGE_START = 'start'
EDGE_FINISH = 'finish'

NEIGHBOUR_PATTERNS = ((-1, 0), (0, -1), (0, 1), (1, 0), (1, -1), (-1, 1))

BLACK = 1
WHITE = -1


class HexState:
    def __init__(self, size) -> None:
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.to_play = BLACK
        self.groups = {BLACK: UnionFind(), WHITE: UnionFind()}

    @staticmethod
    def neighbours(cell: tuple, size: int) -> list:
        """
        Returns a list of the neighbours of the passed cell.

        Args:
            size (int): size of the board
            cell (tuple):
        """
        x, y = cell
        return [(n[0] + x, n[1] + y) for n in NEIGHBOUR_PATTERNS
                if (0 <= n[0] + x < size and 0 <= n[1] + y < size)]

    def possible_actions(self) -> list:
        """
        Get a list of all moves possible in the current board state.
        """
        moves = [(x, y) for x in range(self.size)
                 for y in range(self.size) if self.board[x][y] == 0]
        return moves

    @property
    def winner(self) -> int:
        """
        Return a number corresponding to the winning player,
        or none if the game is not over.
        """
        if self.groups[WHITE].connected(EDGE_START, EDGE_FINISH):
            return WHITE
        elif self.groups[BLACK].connected(EDGE_START, EDGE_FINISH):
            return BLACK
        else:
            return None

    def turn(self) -> int:
        """
        Returns the player who's turn it is.
        """
        return self.to_play

    def step(self, cell: tuple) -> None:
        """
        Makes the move passed in, according to who's turn it is.
        """
        if self.to_play == BLACK:
            self.place_stone(cell, BLACK, 0)
            self.to_play = WHITE
        elif self.to_play == WHITE:
            self.place_stone(cell, WHITE, 1)
            self.to_play = BLACK

    def place_stone(self, cell, player, coord_index) -> None:
        """
        Places a stone.
        coord_index = 0; corresponds to row. Assign this for checking connection from top to bottom.
        coord_index = 1; corresponds to column. Assign this for checking connection from left to right.

        Args:
            cell (tuple): row and column of the cell
            player (int): player by which stone has been placed
            coord_index: 0 for x, 1 for y to check for end condition row wise or column wise respectively
        """

        if self.board[cell] == 0:
            self.board[cell] = player
        else:
            raise ValueError(f"Cell {cell} already occupied.")

        if cell[coord_index] == 0:
            self.groups[player].join(EDGE_START, cell)
        elif cell[coord_index] == self.size - 1:
            self.groups[player].join(EDGE_FINISH, cell)

        for neighbour in HexState.neighbours(cell, self.size):
            if self.board[neighbour] == player:
                self.groups[player].join(neighbour, cell)


class GuiHexState(HexState):
    color_legend = {1: 'Black', -1: 'White'}

    def __init__(self, size) -> None:
        super().__init__(size)
        self.move_history = []

    def step(self, cell: tuple) -> None:
        """
        Modified step function to include adding moves to move_history list.
        """
        self.move_history.append(cell)

        if self.to_play == BLACK:
            self.place_stone(cell, BLACK, 0)
            self.to_play = WHITE
        elif self.to_play == WHITE:
            self.place_stone(cell, WHITE, 1)
            self.to_play = BLACK

    def get_move_history(self) -> list:
        """
        Returns the move_history of the game.
        Note: Used in GUI to display move numbers.
        """
        return self.move_history

    @staticmethod
    def move_to_string(move: tuple) -> str:
        """
        Returns the coordinate in '$#' form where $ - letter and # - digit.
        """
        i, j = move
        return f"{ascii_letters[j]}{i+1}"

    @staticmethod
    def shortest_connection(board: np.ndarray, size: int, player: int) -> list:
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
        START_NODE = 'START'
        END_NODE = 'END'

        graph_dict = defaultdict(list)
        for i, j in ((x, y) for x in range(size) for y in range(size)):
            if board[i, j] != player:
                continue

            for neighbour in GuiHexState.neighbours((i, j), size):
                if board[neighbour] == player:
                    graph_dict[(i, j)].append(neighbour)

            if player == BLACK:
                if i == 0:
                    graph_dict[START_NODE].append((i, j))
                    graph_dict[(i, j)].append(START_NODE)
                elif i == (size - 1):
                    graph_dict[END_NODE].append((i, j))
                    graph_dict[(i, j)].append(END_NODE)

            if player == WHITE:
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


def _main(visualise=True) -> None:
    '''
    Plays a random game of hex.
    if visualise is True then board is drawn via GUI.
    '''
    board = GuiHexState(6)
    while board.winner == None:
        action = choice(board.possible_actions())
        board.step(action)
    if board.winner == BLACK:
        print("BLACK WON!")
    else:
        print("WHITE WON!")

    print(board.board)
    if visualise:
        visualize_board(board.board, board.get_move_history())


if __name__ == '__main__':
    _main()
    pass
    # Code to generate call graph
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput

    # with PyCallGraph(output=GraphvizOutput()):
    #     _main(visualise=False)
