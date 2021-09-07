import numpy as np
# Working code
# Convert c_isterminal.py logic to c
# Changes have been made to env.py so as to accomodate the usage of this file
# Refer to line 37 and line 52 of env.py
# Note the 'import c_isterminal'

# CAUTION while copying code for both players
# Although the while statement looks almost alike, the end condition for both the players is different

# Saving delta list since will be used many times
# Excludes the top left and bottom right element
delta_neighbouring_cell = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]


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


def fetch_neighbours(coords: tuple, player: int, size: int, board: np.ndarray):
    x, y = coords
    neighbours = []
    for delta_x, delta_y in delta_neighbouring_cell:
        x_new = x + delta_x
        y_new = y + delta_y
        if (0 <= x_new <= size-1) and (0 <= y_new <= size-1) and board[x_new, y_new] == player:
            neighbours.append((x_new, y_new))
    return neighbours


def hex_IsTerminal(board: np.ndarray, size: int, player: int):

    visited = []

    dfs_stack = Stack()

    if player == 1:
        # Basic test for the terminal state.
        # Each row must have atleast a single piece for state to be terminal
        for row in board:
            if player not in row:
                return (False, None)

        # Iterate through the first row of board to find the coordinate of player's piece in the first row
        top = np.where(board[0, :] == player)[0]
        top_list = []
        for column_coord in top:
            top_list.append((0, column_coord))

        # Add the initial elements to stack
        for element in top_list:
            dfs_stack.add(element)

        while dfs_stack.isNotEmpty():
            current_node = dfs_stack.pop()
            x, y = current_node
            visited.append(current_node)

            # Check the end condition
            if x == size - 1:
                return (True, player)

            neighbours = fetch_neighbours(current_node, player, size, board)
            for neighbour in neighbours:
                # adding only those neighbours who have not been visited
                if neighbour not in visited:
                    dfs_stack.add(neighbour)

        # Since the stack is empty and we couldnt find anything hence
        return (False, None)

    elif player == -1:
        # Basic test for the terminal state.
        # Each column must have atleast a single piece for state to be terminal
        for index in range(size):
            coulumn = board[:, index]
            if player not in coulumn:
                return (False, None)

        # Iterate through the first column of board to find the coordinate of player's piece in the first column
        left = np.where(board[:, 0] == player)[0]
        left_list = []
        for row_coord in left:
            left_list.append((row_coord, 0))

        # Add the initial elements to stack
        for element in left_list:
            dfs_stack.add(element)

        while dfs_stack.isNotEmpty():
            current_node = dfs_stack.pop()
            x, y = current_node
            visited.append(current_node)

            # Check the end condition
            if y == size - 1:
                return (True, player)

            neighbours = fetch_neighbours(current_node, player, size, board)
            for neighbour in neighbours:
                # adding only those neighbours who have not been visited
                if neighbour not in visited:
                    dfs_stack.add(neighbour)

        # Since the stack is empty and we couldnt find anything hence
        return (False, None)
