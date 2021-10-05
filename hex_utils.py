import warnings
from string import ascii_letters

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, RegularPolygon

warnings.filterwarnings("ignore")

NEIGHBOUR_PATTERNS = ((-1, 0), (0, -1), (0, 1), (1, 0), (1, -1), (-1, 1))
circumradius = 1
iplusj_multiplier = circumradius * 1.5
jminusi_multiplier = circumradius * np.sqrt(3)/2
piby6 = np.pi/6


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


def get_coords(index: tuple) -> tuple:
    i, j = index
    return ((i+j)*iplusj_multiplier, (j-i)*jminusi_multiplier)


def visualize_board(board: np.ndarray, moves_order: list = None, filename=None, filter: np.ndarray = None):

    # For 1D board format
    if len(board.shape) == 1:
        size = int((board.size)**(0.5))
        board = board.reshape(size, size)

    size_row, size_column = board.shape
    coords = np.zeros((size_row, size_column), dtype=object)

    # Generate the coordinates of centres of the isometric grid
    for i, j in ((x, y) for x in range(size_row) for y in range(size_column)):
        coords[i, j] = get_coords((i, j))

    # Creating a 2D array containing move order
    if moves_order != None:
        move_matrix = np.zeros((size_row, size_column), int)
        for move_number, move_coordinate in enumerate(moves_order, 1):
            move_matrix[move_coordinate] = move_number

    ax = plt.axes()
    ax.set_aspect('equal')

    L = coords[0, 0]
    R = coords[size_row-1, size_column-1]
    U = coords[0, size_column-1]
    D = coords[size_row-1, 0]
    MID = ((L[0]+R[0])/2, (L[1]+R[1])/2)
    M1 = (coords[0, 1][1]-coords[0, 0])[1]/coords[0, 1][0]-coords[0, 0][0]
    M2 = (coords[1, 0][1]-coords[0, 0])[1]/coords[1, 0][0]-coords[0, 0][0]
    OFFSET = 3
    L_offset = (L[0] - OFFSET, L[1])
    R_offset = (R[0] + OFFSET, R[1])
    U_offset = (U[0], L_offset[1] + M1*(U[0] - L_offset[0]))
    D_offset = (D[0], L_offset[1] + M2*(D[0] - L_offset[0]))

    triangleNW = Polygon([L_offset, U_offset, MID], closed=True, color='k')
    triangleNE = Polygon([R_offset, U_offset, MID], closed=True, color='0.90')
    triangleSW = Polygon([L_offset, D_offset, MID], closed=True, color='0.90')
    triangleSE = Polygon([R_offset, D_offset, MID], closed=True, color='k')
    ax.add_patch(triangleNW)
    ax.add_patch(triangleNE)
    ax.add_patch(triangleSW)
    ax.add_patch(triangleSE)

    LABEL_OFFSET = 0.7
    label_alpha = [get_coords((-1, j)) for j in range(size_column)]
    label_numeric = [get_coords((i, -1)) for i in range(size_row)]
    for num, coord in enumerate(label_alpha, 1):

        x_coord, y_coord = coord
        ax.text(x_coord - LABEL_OFFSET, y_coord - LABEL_OFFSET * M2, ascii_letters[num-1], ha='center',
                va='center', size=10, color='k', family='sans-serif', weight='light')
    for num, coord in enumerate(label_numeric, 1):

        x_coord, y_coord = coord
        ax.text(x_coord - LABEL_OFFSET, y_coord - LABEL_OFFSET * M1, str(num), ha='center',
                va='center', size=10, color='k', family='sans-serif', weight='light')

    for i, j in ((x, y) for x in range(size_row) for y in range(size_column)):
        coord_x, coord_y = coords[i, j]
        piece = board[i, j]

        # ADD OUTLINE
        if filter is None:
            hexagon = RegularPolygon((coord_x, coord_y), numVertices=6, radius=circumradius,
                                     facecolor='1', edgecolor='darkgrey', linewidth='1.5', orientation=piby6)
        else:
            if filter[i, j]:
                hexagon = RegularPolygon((coord_x, coord_y), numVertices=6, radius=circumradius,
                                         facecolor='0.85', edgecolor='darkgrey', linewidth='1.5', orientation=piby6)
            else:
                hexagon = RegularPolygon((coord_x, coord_y), numVertices=6, radius=circumradius,
                                         facecolor='1', edgecolor='darkgrey', linewidth='1.5', orientation=piby6)

        ax.add_patch(hexagon)

        # ADD PIECE
        if piece == 0:
            continue
        elif piece == 1:
            # Black Piece
            color = '0.2'
        elif piece == -1:
            # White Piece
            color = 'w'
        hexagon = Circle((coord_x, coord_y), radius=0.6,
                         facecolor=color, edgecolor='k', linewidth='2')
        ax.add_patch(hexagon)

        if moves_order is not None:
            move_number = move_matrix[i, j]
            if piece == 1:
                color = 'w'
            elif piece == -1:
                color = 'k'
            plt.text(coord_x, coord_y, str(move_number), ha='center',
                     va='center', size=12, fontfamily='Comic Sans MS', color=color)

    plt.axis('off')
    plt.autoscale(enable=True)

    if filename != None:
        plt.savefig(f'game_{str(filename)}.png', bbox_inches='tight')
    else:
        plt.show()


def fetch_neighbours(coords: tuple, player: int, size: int, board: np.ndarray):
    '''
    Gets those neighbours of a cell whose adjacent cell is also occupied by the same color.
    '''
    x, y = coords
    neighbours = []
    for delta_x, delta_y in NEIGHBOUR_PATTERNS:
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


def _main():
    board = np.zeros((6, 6), dtype=int)
    board[0][0] = 1
    board[0][1] = -1
    board[1][2] = 1
    order = [(0, 0), (0, 1), (1, 2)]
    visualize_board(board, order)

    print(hex_IsTerminal(board, 6, 1))


def intmove_to_tupl(move: int, size: int) -> tuple:
    return (move//size, move % size)


def tuplemove_to_int(move: tuple, size: int) -> int:
    x, y = move
    return x*size + y


def move_to_string(move: tuple) -> str:
    """
    Returns the coordinate in '$#' form where $ - letter and # - digit.
    """
    i, j = move
    return f"{ascii_letters[j]}{i+1}"


if __name__ == '__main__':
    _main()
