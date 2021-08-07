import random
from collections import defaultdict

import numpy as np

BOARD_SIZE = 6


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
        self.size == size

    def IsTerminal(self) -> tuple:
        """
        Returns if board state is terminal
        Possibly implement a DFS here
        Start from one edge (edge of the player who played last move) of the board
        Return a tuple ==> (boolean ( if game has ended ), winner)
        """
        if self.size % 2 == 1:
            raise Exception('IsTerminal cannot handle odd board sizes yet')
        # The player whose win is to be checked
        player = self.fetch_turn(inverse=True)

        def righttree_neighbour(index):
            row, column = index
            return [(row+1, column-1), (row, column-1)]

        def lefttree_neighbour(index):
            row, column = index
            return [(row, column+1), (row-1, column+1)]

        def uppertree_neighbour(index):
            row, column = index
            return [(row+1, column-1), (row+1, column)]

        def lowertree_neighbour(index):
            row, column = index
            return [(row-1, column+1), (row-1, column)]

        connected = defaultdict(set)

        if player == 1:
            top = np.where(self.board[0, :] == player)[0]
            bottom = np.where(self.board[(self.size-1), :] == player)[0]

            if top.size == 0:
                return (False, None)
            else:
                for column_coord in top:
                    connected[0].add((0, column_coord))

            if bottom.size == 0:
                return (False, None)
            else:
                for column_coord in bottom:
                    connected[self.size-1].add((self.size-1, column_coord))

            # Top to down search
            for row in range(self.size//2-1):  # Range(0, 2)
                dummy_append_list = []
                for coord in connected[row]:
                    for down_neighbour in uppertree_neighbour(coord):
                        if down_neighbour == player:
                            dummy_append_list.append(down_neighbour)
                if len(dummy_append_list) == 0:
                    return (False, None)
                else:
                    connected[row+1].update(dummy_append_list)

            # Bottom to up search
            for row in range(self.size-1, self.size//2, -1):  # Range(5, 3,-1)
                dummy_append_list = []
                for coord in connected[row]:
                    for up_neighbour in lowertree_neighbour(coord):
                        if up_neighbour == player:
                            dummy_append_list.append(up_neighbour)
                if len(dummy_append_list) == 0:
                    return (False, None)
                else:
                    connected[row-1].update(dummy_append_list)

            # Checking connectivity of the top and bottom trees
            for upper_coord in connected[self.size//2-1]:
                for down_neighbour in uppertree_neighbour(upper_coord):
                    if down_neighbour in connected[self.size//2]:
                        return (True, player)
            return (False, None)

        elif player == -1:
            left = np.where(self.board[:, 0] == player)[0]
            right = np.where(self.board[:, self.size-1] == player)[0]

            if left.size == 0:
                return (False, None)
            else:
                for row_coord in left:
                    connected[0].add((row_coord, 0))

            if right.size == 0:
                return (False, None)
            else:
                for row_coord in right:
                    connected[self.size-1].add((row_coord, self.size-1))

            # Left to right
            for column in range(self.size//2-1):  # Range(0, 2)
                dummy_append_list = []
                for coord in connected[column]:
                    for right_neighbour in lefttree_neighbour(coord):
                        if right_neighbour == player:
                            dummy_append_list.append(right_neighbour)
                if len(dummy_append_list) == 0:
                    return (False, None)
                else:
                    connected[column+1].update(dummy_append_list)

            # Right to left
            for column in range(self.size-1, self.size//2, -1):  # Range(5, 3,-1)
                dummy_append_list = []
                for coord in connected[column]:
                    for left_neighbour in righttree_neighbour(coord):
                        if left_neighbour == player:
                            dummy_append_list.append(left_neighbour)
                if len(dummy_append_list) == 0:
                    return (False, None)
                else:
                    connected[column-1].update(dummy_append_list)

            # Checking connectivity of the left and right trees
            for left_coord in connected[self.size//2-1]:
                for right_neighbour in uppertree_neighbour(left_coord):
                    if right_neighbour in connected[self.size//2]:
                        return (True, player)
            return (False, None)

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

    def step(self):
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
        board_state_copy = self.board.copy()
        self.history.append(board_state_copy)

        boolean, winner = self.IsTerminal()
        if boolean:
            self.terminated = True
            self.winner = winner
        else:
            player = self.fetch_turn()
            action = random.choice(self.possible_actions())
            self.board[action] = player

    def possible_actions(self) -> list:
        '''
        Helper function for step
        Return all possible action available for the player to pick upon
        '''
        # finding where element is 0
        row_coords, column_coords = np.where(self.board == 0)
        print(row_coords, column_coords)
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


if __name__ == "__main__":
    pass
    # game = Hex()
    # print(game.board)
    # game.board[1, 1] = 1
    # print(game.possible_actions())
