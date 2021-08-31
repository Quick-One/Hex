from Hex_utils import visualize_board
import numpy as np
delta_neighbouring_cell = [(-1, 0), (-1, +1), (0, -1), (0, 1), (1, -1), (1, 0)]


class Stack:
    def __init__(self):
        self.memory = []

    def add(self, element):
        self.memory.append(element)

    def pop(self):
        return self.memory.pop()

    def isNotEmpty(self):
        if len(self.memory) != 0:
            return True
        return False

    def extend(self, arg):
        self.memory.extend(arg)


def fetch_neighbours(coords: tuple, player: int, board: np.ndarray):
    size_row, size_column = board.shape
    x, y = coords
    neighbours = []
    for delta_x, delta_y in delta_neighbouring_cell:
        x_new = x + delta_x
        y_new = y + delta_y
        if (0 <= x_new <= size_row-1) and (0 <= y_new <= size_column-1) and board[x_new, y_new] == player:
            neighbours.append((x_new, y_new))
    return neighbours


def Check_end(board: np.ndarray, nodes_to_be_connected: list):

    visited = []
    dfs_stack = Stack()

    start_node = nodes_to_be_connected[0]
    dfs_stack.add(start_node)
    cluster_connected_bool = {node: False for node in nodes_to_be_connected}

    while dfs_stack.isNotEmpty():
        current_node = dfs_stack.pop()
        visited.append(current_node)

        # Check the end condition
        if current_node in cluster_connected_bool:
            cluster_connected_bool[current_node] = True
            if all(cluster_connected_bool.values()):
                return (True, 1)

        neighbours = fetch_neighbours(current_node, 1, board)
        for neighbour in neighbours:
            # adding only those neighbours who have not been visited
            if neighbour not in visited:
                dfs_stack.add(neighbour)

    # finding if there is any possible moves
    if np.count_nonzero(board == 0) == 0:
        return (True, 0)
    return (False, None)
