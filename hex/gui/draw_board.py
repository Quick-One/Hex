from functools import lru_cache
from pathlib import Path
from string import ascii_letters

import matplotlib.pyplot as plt
import numpy as np
from hex.board import EMPTY, P1, P2
from hex.gui.theme import GUI_PARAMS
from matplotlib.patches import Circle, Polygon, RegularPolygon


def get_coords(cell):
    i, j = cell
    iplusj_multiplier = 1.5
    jminusi_multiplier = np.sqrt(3)/2
    return (iplusj_multiplier*(i+j), jminusi_multiplier*(j-i))


@lru_cache(maxsize=5)
def coord_matrix(size):
    coords = np.zeros((size, size), dtype=object)
    for i, j in np.ndindex(size, size):
        coords[i, j] = get_coords((i, j))
    return coords


def HexPlot(size):
    with plt.rc_context(GUI_PARAMS['rcParams']):
        fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()

    try:
        manager = fig.canvas.manager
        manager.set_window_title('HEX')
        manager.window.wm_iconbitmap(Path(__file__).parent / 'hex.ico')
    except:
        pass

    coords = coord_matrix(size)

    # Adding background triangles
    L = coords[0, 0]
    R = coords[size-1, size-1]
    U = coords[0, size-1]
    D = coords[size-1, 0]
    MID = ((L[0]+R[0])/2, (L[1]+R[1])/2)

    M1 = (coords[0, 1][1]-coords[0, 0])[1]/coords[0, 1][0]-coords[0, 0][0]
    M2 = (coords[1, 0][1]-coords[0, 0])[1]/coords[1, 0][0]-coords[0, 0][0]

    OFFSET = GUI_PARAMS['BOARD_OFFSET']
    L_offset = (L[0] - OFFSET, L[1])
    R_offset = (R[0] + OFFSET, R[1])
    U_offset = (U[0], L_offset[1] + M1*(U[0] - L_offset[0]))
    D_offset = (D[0], L_offset[1] + M2*(D[0] - L_offset[0]))

    triangleNW = Polygon([L_offset, U_offset, MID], **GUI_PARAMS['P1_triangle'])
    triangleSE = Polygon([R_offset, D_offset, MID], **GUI_PARAMS['P1_triangle'])
    triangleNE = Polygon([R_offset, U_offset, MID], **GUI_PARAMS['P2_triangle'])
    triangleSW = Polygon([L_offset, D_offset, MID], **GUI_PARAMS['P2_triangle'])
    for patch in (triangleNW, triangleNE, triangleSW, triangleSE):
        ax.add_patch(patch)

    # Adding board borders
    border = Polygon([L_offset, U_offset, R_offset, D_offset], **GUI_PARAMS['boardBorder'])
    ax.add_patch(border)

    # Adding Labels to the board
    LABEL_OFFSET = GUI_PARAMS['LABEL_OFFSET']
    label_alpha = [get_coords((-1, j)) for j in range(size)]
    label_numeric = [get_coords((i, -1)) for i in range(size)]

    # Making column labels
    for num, coord in enumerate(label_alpha):
        x_coord, y_coord = coord
        lx, ly = x_coord - LABEL_OFFSET, y_coord - LABEL_OFFSET * M2
        ax.text(lx, ly, ascii_letters[num], **GUI_PARAMS['axLabel'])

    # Making row labels
    for num, coord in enumerate(label_numeric, 1):
        x_coord, y_coord = coord
        lx, ly = x_coord - LABEL_OFFSET, y_coord - LABEL_OFFSET * M1
        ax.text(lx, ly, str(num), **GUI_PARAMS['axLabel'])

    # Adding hexagonal tiles
    tiles = np.empty((size, size), dtype=object)
    for i, j in np.ndindex(size, size):
        hexagon = RegularPolygon(coords[i, j], **GUI_PARAMS['hexTile'])
        tiles[i, j] = hexagon
        ax.add_patch(hexagon)

    ax.autoscale(enable=True)
    return fig, ax, tiles


def add_pieces(ax, board):
    coords = coord_matrix(board.shape[0])
    pieces = np.empty(board.shape, dtype=object)
    for i, j in np.ndindex(board.shape):
        if board[i, j] == EMPTY:
            continue
        elif board[i, j] == P1:
            piece = Circle(coords[i, j], **GUI_PARAMS['P1_piece'])
        elif board[i, j] == P2:
            piece = Circle(coords[i, j], **GUI_PARAMS['P2_piece'])
        pieces[i, j] = piece
        ax.add_patch(piece)
    return pieces


def highlight_tiles(ax, locs: list):
    high_tiles = np.empty(len(locs), dtype=object)
    for i, loc in enumerate(locs):
        coord = get_coords(loc)
        highlighted_tile = RegularPolygon(coord, **GUI_PARAMS['highlightedTile'])
        high_tiles[i] = highlighted_tile
        ax.add_patch(highlighted_tile)
    return high_tiles


def add_piece(ax, loc, player, ghost = False):
    coord = get_coords(loc)
    if player == P1:
        piece = Circle(coord, **GUI_PARAMS['P1_piece'])
    else:
        piece = Circle(coord, **GUI_PARAMS['P2_piece'])
    if ghost:
        piece.set_alpha(0.4)
        piece.set_linestyle('--')
    ax.add_patch(piece)
    return piece


def add_move_order(ax, board, move_order):
    coords = coord_matrix(board.shape[0])
    for move_num, move in enumerate(move_order, 1):
        coord = coords[move]
        if board[move] == P1:
            ax.text(*coord, str(move_num), **GUI_PARAMS['P1MoveNo'])
        else:
            ax.text(*coord, str(move_num), **GUI_PARAMS['P2MoveNo'])


def visualize_board(board, move_order=None, plot=False):
    fig, ax, _ = HexPlot(board.shape[0])
    add_pieces(ax, board)
    if move_order is not None:
        add_move_order(ax, board, move_order)
    if plot:
        plt.show()
    else:
        return fig, ax
