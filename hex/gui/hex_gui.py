import logging as log

import matplotlib.pyplot as plt
import numpy as np
from hex.agent import HexAgent
from hex.board import EMPTY, Hex, move_to_string, shortest_connection, P1
from hex.gui.draw_board import (HexPlot, add_move_order, add_piece, add_pieces,
                                coord_matrix, highlight_tiles)
from hex.gui.theme import GUI_PARAMS


def player_color(p_no):
    if p_no == P1:
        return GUI_PARAMS['P1_color']
    return GUI_PARAMS['P2_color']


class HexGUI():

    def __init__(self, game: Hex):
        self.game = game
        self.fig, self.ax, _ = HexPlot(game.size)
        add_pieces(self.ax, game.board)

    def show(self):
        '''
        Shows the main GUI window.
        '''
        self.fig.show()

    def refresh(self):
        '''
        Refreshes the GUI window.
        '''
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def pt_on_board(self, x: int, y: int) -> tuple:
        '''
        Determines the tile closest to the user's mouse click. 
        '''
        if (x == None) or (y == None):
            return (False, None)

        coords = coord_matrix(self.game.size)
        for i, j in np.ndindex(coords.shape):
            x_coord, y_coord = coords[i, j]
            inradius = GUI_PARAMS['P1_piece']['radius']
            if ((x-x_coord)**2 + (y-y_coord)**2 - inradius**2) < 0:
                return (True, (i, j))
        return (False, None)

    def get_human_move(self) -> tuple:
        '''
        Gets human move from the GUI.
        '''
        human_move = {'Found_valid_move': False, 'move': None}

        def getmove_onclick(event):
            is_find, move = self.pt_on_board(event.xdata, event.ydata)
            if is_find:
                orignal_piece_on_board = self.game.board[move]
                if orignal_piece_on_board != EMPTY:
                    occ_piece_owner = self.game.players[orignal_piece_on_board].name
                    log.warning(f"{move_to_string(move)} is already occupied by {occ_piece_owner}'s piece.")
                else:
                    self.fig.canvas.mpl_disconnect(event_handler_id)
                    human_move['Found_valid_move'] = True
                    human_move['move'] = move
                    curr_player = self.game.current_player
                    log.info(f'Move on {move_to_string(move)} registered for {curr_player}.')

        event_handler_id = self.fig.canvas.mpl_connect('button_press_event', getmove_onclick)
        while human_move['Found_valid_move'] == False:
            plt.pause(0.1)
        return human_move['move']

    def get_agent_move(self) -> tuple:
        '''
        Gets agent's move. 
        '''
        agent = HexAgent.get_agent(self.game.current_player)
        return agent.best_move(self.game)

    def simulate_game(self) -> None:
        '''
        Simulates Hex game.
        '''
        while self.game.winner is None:
            turn = self.game.turn
            player = self.game.players[turn]
            self.fig.suptitle(f'{player.name} ({player_color(turn)}) to move.')

            if player.is_AI:
                self.ax.set_title('Computer is thinking ...')
                self.refresh()
                move = self.get_agent_move()

            else:
                self.ax.set_title('Click on empty square to register a move')
                self.refresh()
                move = self.get_human_move()

            self.game.step(move)
            add_piece(self.ax, move, turn)
            self.refresh()

    def render_game_end(self, label_moves=True, connection=True) -> None:
        '''
        Renders end screen for the game.
        '''
        winner = self.game.winner
        winner_name = self.game.players[winner].name

        self.fig.suptitle(f'{winner_name}({player_color(winner)}) won the game')
        self.ax.set_title('')
        self.refresh()

        if label_moves == True:
            add_move_order(self.ax, self.game.board, self.game.move_history)
        self.refresh()

        if connection == True:
            highlight_tiles(self.ax, shortest_connection(self.game.board, self.game.size, self.game.winner))
        self.refresh()

    @classmethod
    def start_game(cls, game: Hex):
        '''
        Starts the game.
        '''
        gui = cls(game)
        gui.show()
        gui.simulate_game()
        gui.render_game_end()
        input('Press ENTER to QUIT')
        plt.close()
