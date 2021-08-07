
# PLEASE note that Hex.isterminal returns a bool,winner tuple, this is to ensure flexibility, More details in its docstring

import numpy as np


BOARD_SIZE = 6

# Saving delta list since will be used many times
# Excludes the top left and bottom right element
delta_neighbouring_cell = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]


class Hex:
    """"
    Hex class contains all the information regarding the current state of board.
    Rules:
    BLACK MOVES FIRST, whites moves second
    Denote Black in board as '1'
    White as '-1', empty cells as '0'


    """

    def __init__(self, size=BOARD_SIZE, multi_agent=False):
        self.board = np.zeros(size, size)
        self.history = []
        self.terminated = False
        self.multi_agent = multi_agent
        self.winner = None

    def IsTerminal(self):
        """
        Returns if board state is terminal
        Possibly implement a DFS here
        Start from one edge (edge of the player who played last move) of the board
        Return a tuple ==> (boolean ( if game has ended ), winner)
        """
        pass

    def fetch_turn(self):
        """" 
        Fetches the side of the player who should move in the current state
        Helper function for Isterminal and Step
        By determining the the player who played the last turn 
        we can run the DFS from only one side instad of 2
        Can be solved by determing from the number of black and white pieces on board. 
        if black > white; then white's turn, else black's
        """
        pass

    def step(self, nn):
        '''
        Emulates a move on the board.
        In case of multi_agent add another argument for Neural network to determine the next step
        For random case pick a random action from possible actions
        pick a random action from possible_action

        Step appends to self.history
        bool_game_end,winner = self.interminal() 
        if bool_game_end == True:
            self.terminated = True
            self.winner = winner
        '''
        pass

    def possible_actions(self):
        '''
        Helper function for step
        Return all possible action available for the player to pick upon
        '''
        pass


def generate_games(self, batchsize=100):
    '''
    Generates games for the NN to train upon

    '''
    # Pseudocode for this function
    # for n in range(batchsize):
    #   game = Hex():
    #   while game.terminated == False:
    #       game.step()
    # Access the game.history and game.winner to generate final dataset

    pass
