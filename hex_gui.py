from platform import system
from string import ascii_letters

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, RegularPolygon

from hex_agents import HexAgents
from hex_class import GuiHexState, HexState
from hex_utils import intmove_to_tupl, tuplemove_to_int
from settings import (Numba_agent_settings, board_settings, game_settings,
                      py_agent_settings)

# Removes icons from mpl window
mpl.rcParams['toolbar'] = 'None'


class GUI_element:
    '''
    Class to customise GUI elements like board and pieces.
    '''

    @staticmethod
    def Black_board_piece(xy: list):
        kwargs = {
            'closed': True,
            'color': 'k'
        }
        return Polygon(xy, **kwargs)

    @staticmethod
    def White_board_piece(xy: list):
        kwargs = {
            'closed': True,
            'color': '0.90'
        }
        return Polygon(xy, **kwargs)

    @staticmethod
    def Hex_tile(coords: tuple, radius: float):
        kwargs = {
            'numVertices': 6,
            'facecolor': '1',
            'edgecolor': 'darkgrey',
            'linewidth': '1.5',
            'orientation': np.pi/6,
        }

        return RegularPolygon(coords, radius=radius, **kwargs)

    @staticmethod
    def White_piece(x: float, y: float, radius: float):
        kwargs = {
            'facecolor': 'w',
            'edgecolor': 'k',
            'linewidth': '2'
        }

        return Circle((x, y), radius=radius, **kwargs)

    @staticmethod
    def Black_piece(x: float, y: float, radius: float):
        kwargs = {
            'facecolor': '0.2',
            'edgecolor': 'k',
            'linewidth': '2'
        }

        return Circle((x, y), radius=radius, **kwargs)

    @staticmethod
    def Highlighted_tile(coords: tuple, radius: int):
        kwargs = {
            'numVertices': 6,
            'facecolor': 'none',
            'edgecolor': 'lime',
            'linewidth': '4',
            'orientation': np.pi/6
        }

        return RegularPolygon(coords, radius=radius, **kwargs)


class GUI():

    def __init__(self, game: GuiHexState,
                 agent_game: HexState,
                 player1: str, player2: str,
                 numba_agent_game=None,
                 heatmap=False,
                 CIRCUMRADIUS: int = board_settings.CIRCUMRADIUS,
                 INRADIUS: int = board_settings.INRADIUS,
                 BOARD_OFFSET: int = board_settings.BOARD_OFFSET,
                 LABEL_OFFSET: int = board_settings.LABEL_OFFSET):
        self.game = game
        self.agent_game = agent_game
        self.numba_agent_game = numba_agent_game
        self.board_size = game.size
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        self.ax.autoscale(enable=True)
        self.ax.set_aspect('equal')
        self.player1 = player1
        self.player2 = player2
        self.coords = np.zeros(
            (self.board_size, self.board_size), dtype=object)
        self.CIRCUMRADIUS = CIRCUMRADIUS
        self.hexagonal_tiles = np.zeros(
            (self.board_size, self.board_size), dtype=object)
        self.pieces = np.zeros(
            (self.board_size, self.board_size), dtype=object)
        self.INRADIUS = INRADIUS
        self.BOARD_OFFSET = BOARD_OFFSET
        self.LABEL_OFFSET = LABEL_OFFSET
        self.heatmap = heatmap

    def configure_mpl_window(self) -> None:
        '''
        1. Sets matplotlib window label. 
        2. Sets icon to hex.ico 
        '''
        try:
            self.fig.canvas.set_window_title('HEX')
        except Exception as e:
            pass

        try:
            thismanager = plt.get_current_fig_manager()
            thismanager.window.wm_iconbitmap("hex.ico")
        except Exception as e:
            pass

    def show(self, fullscreen=True) -> None:
        '''
        Shows the main GUI window.
        '''
        if system() == 'Windows':
            if fullscreen:
                plt.get_current_fig_manager().window.state('zoomed')
            else:
                self.fig.show()
        else:
            self.fig.show()

    def refresh(self) -> None:
        '''
        Refreshes the GUI window.
        '''
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if system() == 'Windows':
            pass
        else:
            plt.pause(0.01)

    def get_coords(self, index: tuple) -> tuple:
        '''
        Return coordinates of the centre of hexagonal tile on the XY plane for an index on board.
        '''
        i, j = index
        iplusj_multiplier = self.CIRCUMRADIUS * 1.5
        jminusi_multiplier = self.CIRCUMRADIUS * np.sqrt(3)/2
        return ((i+j)*iplusj_multiplier, (j-i)*jminusi_multiplier)

    def render_coords(self) -> None:
        '''
        Generate the coordinates for the centres of the each tile on the board.
        '''
        for i, j in ((x, y) for x in range(self.board_size) for y in range(self.board_size)):
            self.coords[i, j] = self.get_coords((i, j))

    def render_board(self) -> None:
        '''
        Generates the main board layout by forming the 4 main triangles to demarcate the colors.
        '''
        L = self.coords[0, 0]
        R = self.coords[self.board_size-1, self.board_size-1]
        U = self.coords[0, self.board_size-1]
        D = self.coords[self.board_size-1, 0]

        M1 = (self.coords[0, 1][1]-self.coords[0, 0])[1] / \
            self.coords[0, 1][0]-self.coords[0, 0][0]
        M2 = (self.coords[1, 0][1]-self.coords[0, 0])[1] / \
            self.coords[1, 0][0]-self.coords[0, 0][0]

        L_offset = (L[0] - self.BOARD_OFFSET, L[1])
        R_offset = (R[0] + self.BOARD_OFFSET, R[1])
        U_offset = (U[0], L_offset[1] + M1*(U[0] - L_offset[0]))
        D_offset = (D[0], L_offset[1] + M2*(D[0] - L_offset[0]))
        MID = ((L[0]+R[0])/2, (L[1]+R[1])/2)

        triangleNW = GUI_element.Black_board_piece([L_offset, U_offset, MID])
        triangleNE = GUI_element.White_board_piece([R_offset, U_offset, MID])
        triangleSW = GUI_element.White_board_piece([L_offset, D_offset, MID])
        triangleSE = GUI_element.Black_board_piece([R_offset, D_offset, MID])

        self.ax.add_patch(triangleNW)
        self.ax.add_patch(triangleNE)
        self.ax.add_patch(triangleSW)
        self.ax.add_patch(triangleSE)

    def render_labels(self) -> None:
        '''
        Renders labels for the board.
        '''
        label_alpha = [self.get_coords((-1, j))
                       for j in range(self.board_size)]
        label_numeric = [self.get_coords((i, -1))
                         for i in range(self.board_size)]

        M1 = (self.coords[0, 1][1]-self.coords[0, 0])[1] / \
            self.coords[0, 1][0]-self.coords[0, 0][0]
        M2 = (self.coords[1, 0][1]-self.coords[0, 0])[1] / \
            self.coords[1, 0][0]-self.coords[0, 0][0]

        # Rendering top labels
        for num, coord in enumerate(label_alpha, 1):
            x_coord, y_coord = coord
            self.ax.text(x_coord - self.LABEL_OFFSET, y_coord - self.LABEL_OFFSET * M2, ascii_letters[num-1], ha='center',
                         va='center', size=10, color='k', family='sans-serif', weight='light')

        # Rendering left labels
        for num, coord in enumerate(label_numeric, 1):
            x_coord, y_coord = coord
            self.ax.text(x_coord - self.LABEL_OFFSET, y_coord - self.LABEL_OFFSET * M1, str(num), ha='center',
                         va='center', size=10, color='k', family='sans-serif', weight='light')

    def render_tiles(self) -> None:
        '''
        Generates all the tiles for the board
        '''
        for i, j in ((x, y) for x in range(self.board_size) for y in range(self.board_size)):
            coord_x, coord_y = self.coords[i, j]

            tile = GUI_element.Hex_tile((coord_x, coord_y), self.CIRCUMRADIUS)

            self.hexagonal_tiles[(i, j)] = self.ax.add_patch(tile)

    def pt_on_board(self, x: int, y: int) -> tuple:
        '''
        Determines the tile closest to the user's mouse click. 
        '''
        if (x == None) or (y == None):
            return (False, None)

        for i, j in ((n, m) for n in range(self.board_size) for m in range(self.board_size)):
            x_coord, y_coord = self.coords[i, j]
            if ((x-x_coord)**2 + (y-y_coord)**2 - self.INRADIUS**2) < 0:
                return (True, (i, j))
        return (False, None)

    def generate_heatmap(self, heatmap, filename):
        total_n = sum(heatmap.values())
        initial_color = self.hexagonal_tiles[0, 0].get_facecolor()
        tiles_changed = []

        for key, value in heatmap.items():
            if isinstance(key, int):
                move = intmove_to_tupl(key, self.board_size)
            tiles_changed.append(move)

            # color_heat = 1 - (value/total_n)
            color_heat = (1/(1 + np.exp(-(1-(value/total_n)))))
            print(move, color_heat)

            if self.game.to_play == -1:
                col = (color_heat, color_heat, 1)
            else:
                col = (1, color_heat, color_heat)
            self.hexagonal_tiles[move].set_facecolor(col)
        plt.savefig(f'game_{str(filename)}.png', bbox_inches='tight')

        for move in tiles_changed:
            self.hexagonal_tiles[move].set_facecolor(initial_color)

    def get_human_move(self, turn: int) -> tuple:
        '''
        Gets human move from the GUI.
        '''
        human_move = {'Found_valid_move': False, 'move': None}

        def getmove_onclick(event):
            is_find, move = self.pt_on_board(event.xdata, event.ydata)
            if is_find:
                orignal_piece_on_board = self.game.board[move]
                if orignal_piece_on_board != 0:
                    print(
                        f"WARNING: {GuiHexState.move_to_string(move)} is already occupied by {GuiHexState.color_legend[orignal_piece_on_board]} piece.")
                else:
                    self.fig.canvas.mpl_disconnect(event_handler_id)
                    human_move['Found_valid_move'] = True
                    human_move['move'] = move
                    print(
                        f'INFO: Move on {GuiHexState.move_to_string(move)} registered for {GuiHexState.color_legend[turn]}.')

        event_handler_id = self.fig.canvas.mpl_connect(
            'button_press_event', getmove_onclick)

        while human_move['Found_valid_move'] == False:
            plt.pause(0.1)
        return human_move['move']

    def get_agent_move(self) -> tuple:
        '''
        Gets agent's move. 
        '''
        if self.numba_agent_game is not None:
            move, heatmap = HexAgents.numba_MCTS_RAVE(
                self.numba_agent_game, Numba_agent_settings.num_rollouts)
        else:
            return HexAgents.MCTS_RAVE(self.agent_game, py_agent_settings.num_rollout, py_agent_settings.time_control)
        if self.heatmap:
            self.generate_heatmap(heatmap, 1)
        return move

    def simulate_game(self) -> None:
        '''
        Simulates Hex game.
        '''
        while self.game.winner == None:
            turn = self.game.turn()
            self.fig.suptitle(
                f'{GuiHexState.color_legend[turn]} to move', fontsize=16)

            if turn == 1:
                player = self.player1
            elif turn == -1:
                player = self.player2

            if player == 'Human':
                self.ax.set_title('Click on empty square to register a move')
                self.refresh()
                move = self.get_human_move(turn)

            elif player == 'Agent':
                self.ax.set_title('Computer is thinking ...')
                self.refresh()
                move = self.get_agent_move()

            self.game.step(move)
            self.agent_game.step(move)
            if self.numba_agent_game is not None:
                self.numba_agent_game.step(
                    tuplemove_to_int(move, self.board_size))
            self.place_piece(move, turn)
            self.refresh()

    def place_piece(self, move: tuple, turn: int) -> None:
        '''
        Generates piece on the GUI window.
        '''
        if turn == 1:
            piece = GUI_element.Black_piece(*self.coords[move], self.INRADIUS)
        elif turn == -1:
            piece = GUI_element.White_piece(*self.coords[move], self.INRADIUS)

        self.pieces[move] = self.ax.add_patch(piece)

    def render_game_end(self, label_moves=True, connection=True) -> None:
        '''
        Renders end screen for the game.
        '''
        if self.game.winner == None:
            return

        self.fig.suptitle(
            f'{GuiHexState.color_legend[self.game.winner]} won the game')
        self.ax.set_title('')

        if label_moves == True:
            self.render_move_order()

        self.refresh()

        if connection == True:
            self.render_connection()

        self.refresh()

    def render_move_order(self) -> None:
        '''
        Renders the move order on the game played.
        '''
        if self.game.winner == None:
            return

        for move_number, move in enumerate(self.game.get_move_history(), 1):
            piece_player = self.game.board[move]
            coord_x, coord_y = self.coords[move]
            if piece_player == 1:
                self.ax.text(coord_x, coord_y, str(move_number), ha='center',
                             va='center', size=12, fontfamily='Comic Sans MS', color='w')
            else:
                self.ax.text(coord_x, coord_y, str(move_number), ha='center',
                             va='center', size=12, fontfamily='Comic Sans MS', color='k')

    def render_connection(self) -> None:
        '''
        Renders the winning connection.
        '''
        connection = GuiHexState.shortest_connection(
            self.game.board, self.board_size, self.game.winner)
        for i, j in connection:
            coord_x, coord_y = self.coords[(i, j)]
            Higlighted_tile = GUI_element.Highlighted_tile(
                (coord_x, coord_y), self.CIRCUMRADIUS)
            self.ax.add_patch(Higlighted_tile)


def _main():
    game = GuiHexState(size=game_settings.board_size)
    agent_game = HexState(size=game_settings.board_size)
    # numba_agent_game = Numba_hex_class.create_empty_board(
    #     game_settings.board_size)
    gui = GUI(game, agent_game, game_settings.player_1, game_settings.player_2)
    # gui = GUI(game, agent_game, game_settings.player_1,
    #           game_settings.player_2, numba_agent_game=numba_agent_game)
    gui.render_coords()
    gui.render_board()
    gui.render_labels()
    gui.render_tiles()
    gui.show(fullscreen=False)
    gui.simulate_game()
    gui.render_game_end()
    input('Press ENTER to QUIT')


if __name__ == '__main__':
    _main()

    # Code to generate call graph
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput

    # with PyCallGraph(output=GraphvizOutput()):
    #     main()
