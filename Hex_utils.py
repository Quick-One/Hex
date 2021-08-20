import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sin30 = 0.5
cos30 = np.sqrt(3)/2

def get_coords(index : tuple) -> tuple:
    row,column = index
    return (-sin30*row + column, -cos30*row)


def visualize_board(board: np.ndarray, moves_order: list = None, filename=None, filter: np.ndarray = None):
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

    for i, j in ((x, y) for x in range(size_row) for y in range(size_column)):
        coord_x, coord_y = coords[i, j]
        piece = board[i, j]

        # ADD OUTLINE
        if filter is None:
            hexagon = RegularPolygon((coord_x, coord_y), numVertices=6, radius=np.sqrt(
                1/3), facecolor='0.95', edgecolor='darkgrey', linewidth='1.5')
        else:
            if filter[i, j]:
                hexagon = RegularPolygon((coord_x, coord_y), numVertices=6, radius=np.sqrt(
                    1/3), facecolor='0.85', edgecolor='darkgrey', linewidth='1.5')
            else:
                hexagon = RegularPolygon((coord_x, coord_y), numVertices=6, radius=np.sqrt(
                    1/3), facecolor='0.95', edgecolor='darkgrey', linewidth='1.5')

        ax.add_patch(hexagon)
        
        #ADD PIECE
        if piece == 0:
            continue
        elif piece == 1:
            # Black Piece
            color = '0.2'
        elif piece == -1:
            #White Piece
            color  = 'w'
        hexagon = Circle((coord_x, coord_y), radius=0.42,facecolor = color, edgecolor='k', linewidth = '2')
        ax.add_patch(hexagon)

        if moves_order is not None:
            move_number = move_matrix[i, j]
            if piece == 1:
                color = 'w'
            elif piece == -1:
                color = 'k'
            plt.text(coord_x, coord_y, str(move_number), ha='center',
                     va='center', size=17, fontfamily='Comic Sans MS', color=color)

    plt.axis('off')
    plt.autoscale(enable=True)

    if filename != None:
        plt.savefig(f'game_{str(filename)}.png', bbox_inches='tight')
    else:
        plt.show()
    plt.cla()


if __name__ == '__main__':
    board = np.zeros((6, 6), dtype=int)
    board[0][0] = 1
    board[0][1] = -1
    board[1][2] = 1
    order = [(0, 0), (0, 1), (1, 2)]
    visualize_board(board, order, filename=1)
