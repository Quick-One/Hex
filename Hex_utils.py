import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
import numpy as np

sin30 = 0.5
cos30 = np.sqrt(3)/2


def get_coords(index: tuple) -> tuple:
    row, column = index
    return (-sin30*row + column, -cos30*row)


def visualize_board(board: np.ndarray):
    size = board.shape[0]
    coords = np.zeros((size, size), dtype=object)
    for i, j in ((x, y) for x in range(size) for y in range(size)):
        coords[i, j] = get_coords((i, j))

    ax = plt.axes()

    ax.set_aspect('equal')
    for i, j in ((x, y) for x in range(size) for y in range(size)):
        coord_x, coord_y = coords[i, j]
        piece = board[i, j]

        # ADD OUTLINE
        hexagon = RegularPolygon((coord_x, coord_y), numVertices=6, radius=np.sqrt(
            1/3), facecolor='whitesmoke', edgecolor='darkgrey', linewidth='1.5')
        ax.add_patch(hexagon)

        # ADD PIECE
        if piece == 0:
            continue
        elif piece == 1:
            # Black Piece
            color = 'k'
        elif piece == -1:
            # White Piece
            color = 'w'
        hexagon = Circle((coord_x, coord_y), radius=0.42,
                         facecolor=color, edgecolor='k', linewidth='2')
        ax.add_patch(hexagon)

    plt.axis('off')
    plt.autoscale(enable=True)
    plt.show()


if __name__ == '__main__':
    board = np.zeros((6, 6), dtype=int)
    board[0][0] = 1
    board[0][1] = -1
    visualize_board(board)
