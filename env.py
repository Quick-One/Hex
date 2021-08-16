import random

import numpy as np

from c_isterminal import c_IsTerminal

import cProfile
BOARD_SIZE = 6

# Saving delta list since will be used many times
# Excludes the top left and bottom right element
delta_neighbouring_cell = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]


class DFScomplete(Exception):
    pass


class Hex:
    """"
    Hex class contains all the information regarding the current state of board.
    Rules:
    BLACK MOVES FIRST, whites moves second
    Denote Black in board as '1'
    White as '-1', empty cells as '0'
    """

    def __init__(self, size=BOARD_SIZE, multi_agent=False):
        self.board = np.zeros((size, size), np.int8)
        self.history = []
        self.terminated = False
        self.multi_agent = multi_agent
        self.winner = None
        self.size = size
        self.total_moves = 0

    def IsTerminal(self, p=None) -> tuple:
        """
        Returns if board state is terminal
        Possibly implement a DFS here
        Start from one edge (edge of the player who played last move) of the board
        Return a tuple ==> (boolean ( if game has ended ), winner)
        param p: Checks for a specific player
        """

        if p == None:
            # The player whose win is to be checked
            player = self.fetch_turn(inverse=True)
        else:
            player = p

        return c_IsTerminal(self.board, self.size, player)

    def fetch_turn(self, inverse=False) -> int:
        """" 
        Fetches the side of the player who should move in the current state
        Helper function for Isterminal and Step
        By determining the the player who played the last turn 
        we can run the DFS from only one side instad of 2
        Can be solved by determing from the number of black and white pieces on board. 
        if black > white; then white's turn, else black's
        """
        if np.sum(self.board) == 0:
            turn = 1
        else:
            turn = -1
        if inverse == True:
            return (turn * -1)
        return turn

    def step(self, agent_move=None):
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

        if self.terminated == True:
            print('Shouldnt come here')

        else:
            board_state_copy = self.board.copy()
            self.history.append(board_state_copy)
            self.total_moves += 1

            if agent_move == None:
                player = self.fetch_turn()
                action = random.choice(self.possible_actions())
                self.board[action] = player

                boolean, winner = self.IsTerminal()

                if boolean:
                    self.terminated = True
                    self.winner = winner
                    self.history.append(self.board)

    def possible_actions(self) -> list:
        '''
        Helper function for step
        Return all possible action available for the player to pick upon
        '''
        # finding where element is 0
        row_coords, column_coords = np.where(self.board == 0)
        possible_actions = []
        for n in range(len(row_coords)):
            possible_actions.append((row_coords[n], column_coords[n]))
        return possible_actions

    def data_for_nn(self):
        '''
        Generates data for nn in the correct format
        '''
        return (self.history, self.winner)


def generate_games(batchsize=100):
    '''
    Generates games for the NN to train upon

    '''

    data = []
    for n in range(batchsize):
        game = Hex()
        while game.terminated == False:
            game.step()
        data.append(game.data_for_nn())
    return data


if __name__ == "__main__":
    pass

    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput

    # with PyCallGraph(output=GraphvizOutput()):
    cProfile.run('c = generate_games(batchsize=100)')
