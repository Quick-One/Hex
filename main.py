import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle, Polygon
import numpy as np
from env import Hex
import warnings
import agent
import time
warnings.filterwarnings("ignore")

class CustomException(Exception):
    pass

CIRCUMRADIUS = 1
INRADIUS = 0.6
SIZE = 6
iplusj_multiplier = CIRCUMRADIUS * 1.5
jminusi_multiplier = CIRCUMRADIUS * np.sqrt(3)/2
piby6 = np.pi/6

fig, ax = plt.subplots()
ax.axis('off')
ax.autoscale(enable=True)
ax.set_aspect('equal')
fig.suptitle('Different types of oscillations', fontsize=16)
ax.set_title('Test')

board = np.zeros((SIZE, SIZE), int)
size_row, size_column = board.shape
coords = np.zeros((size_row, size_column), dtype=object)


def get_coords(index: tuple) -> tuple:
    '''
    Return the (x,y) coordinates of the centre of hexagon for an index of 2D array
    '''
    i, j = index
    return ((i+j)*iplusj_multiplier, (j-i)*jminusi_multiplier)


# Generate the coordinates of centres of the isometric grid
for i, j in ((x, y) for x in range(size_row) for y in range(size_column)):
    coords[i, j] = get_coords((i, j))


# Adding the four triangles
OFFSET = 3
L = coords[0, 0]
R = coords[size_row-1, size_column-1]
U = coords[0, size_column-1]
D = coords[size_row-1, 0]
MID = ((L[0]+R[0])/2, (L[1]+R[1])/2)
M1 = (coords[0, 1][1]-coords[0, 0])[1]/coords[0, 1][0]-coords[0, 0][0]
M2 = (coords[1, 0][1]-coords[0, 0])[1]/coords[1, 0][0]-coords[0, 0][0]

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
fig.show()



for i, j in ((x, y) for x in range(size_row) for y in range(size_column)):
    coord_x, coord_y = coords[i, j]
    piece = board[i, j]

    hexagon = RegularPolygon((coord_x, coord_y), numVertices=6, radius=CIRCUMRADIUS,
                             facecolor='1', edgecolor='darkgrey', linewidth='1.5', orientation=piby6)

    ax.add_patch(hexagon)
    fig.canvas.draw()
    fig.canvas.flush_events()


def pt_on_board(x, y, board_coord=coords, inrad=INRADIUS):
    for i, j in ((n, m) for n in range(SIZE) for m in range(SIZE)):
        x_coord, y_coord = board_coord[i, j]
        if ((x-x_coord)**2 + (y-y_coord)**2 - inrad**2) < 0:
            return (True, (i, j))
    return (False, None)


players_dict = {1: 'Human', -1:'Agent' }
Game = Hex()
while Game.terminated == False:
    turn  = Game.fetch_turn()

    if players_dict[turn] == 'Human':
        print('Human', turn)
        
        def onclick(event):
            is_find, index = pt_on_board(event.xdata, event.ydata)
            if is_find :
                orignal_piece_on_board = Game.board[index]
                if orignal_piece_on_board != 0:
                    print(f"Index {index}, is already occupied by {orignal_piece_on_board}")
                else:
                    #VALID MOVE DONE
                    fig.canvas.mpl_disconnect(cid)
                    
                    Game.step(index)
                    
                    if turn == 1:
                        piece = Circle(coords[index], radius=INRADIUS,
                                facecolor='0.2', edgecolor='k', linewidth='2')
                    elif turn == -1:
                        piece = Circle(coords[index], radius=INRADIUS,
                                facecolor='w', edgecolor='k', linewidth='2')
                    ax.add_patch(piece)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        
        while Game.fetch_turn() == turn:
            plt.pause(0.1) 

    if players_dict[turn] == 'Agent':
        move  = agent.func(Game.board)
        Game.step(move)  
        if turn == 1:
            piece = Circle(coords[move], radius=INRADIUS,
                    facecolor='0.2', edgecolor='k', linewidth='2')
        elif turn == -1:
            piece = Circle(coords[move], radius=INRADIUS,
                    facecolor='w', edgecolor='k', linewidth='2')
        ax.add_patch(piece)
        fig.canvas.draw()
        fig.canvas.flush_events()

for move_number,move in enumerate(Game.move_history,1):
    piece_player = Game.board[move]
    coord_x,coord_y = coords[move]
    if piece_player == 1:
        ax.text(coord_x,coord_y, str(move_number), ha='center',
                     va='center', size=12, fontfamily='Comic Sans MS', color='w')
    else:
        ax.text(coord_x,coord_y, str(move_number), ha='center',
                     va='center', size=12, fontfamily='Comic Sans MS', color='k')
    fig.canvas.draw()
    fig.canvas.flush_events()
        
    
        
         



input('Press ENTER to exit')