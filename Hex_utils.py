import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle, Polygon
import numpy as np
import warnings
warnings.filterwarnings("ignore")

radius = 1
iplusj_multiplier = radius * 1.5
jminusi_multiplier = radius * np.sqrt(3)/2
piby6 = np.pi/6


def get_coords(index: tuple) -> tuple:
    i, j = index
    return ((i+j)*iplusj_multiplier, (j-i)*jminusi_multiplier)


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
   
    triangleNW = Polygon([L_offset, U_offset, MID], closed = True, color = 'k')
    triangleNE = Polygon([R_offset, U_offset, MID], closed = True, color = '0.90')
    triangleSW = Polygon([L_offset, D_offset, MID], closed = True, color = '0.90')
    triangleSE = Polygon([R_offset, D_offset, MID], closed = True, color = 'k')
    ax.add_patch(triangleNW)
    ax.add_patch(triangleNE)
    ax.add_patch(triangleSW)
    ax.add_patch(triangleSE)
    

    for i, j in ((x, y) for x in range(size_row) for y in range(size_column)):
        coord_x, coord_y = coords[i, j]
        piece = board[i, j]

        # ADD OUTLINE
        if filter is None:
            hexagon = RegularPolygon((coord_x, coord_y), numVertices=6, radius=radius,
                                     facecolor='1', edgecolor='darkgrey', linewidth='1.5', orientation=piby6)
        else:
            if filter[i, j]:
                hexagon = RegularPolygon((coord_x, coord_y), numVertices=6, radius=radius,
                                         facecolor='0.85', edgecolor='darkgrey', linewidth='1.5', orientation=piby6)
            else:
                hexagon = RegularPolygon((coord_x, coord_y), numVertices=6, radius=radius,
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
    visualize_board(board, order)
