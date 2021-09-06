import random
from collections import defaultdict
from queue import Queue
from string import ascii_letters

import numpy as np

from hex_isterminal import fetch_neighbours, hex_IsTerminal


class Hex:
    """"
    Hex class contains all the information regarding the current state of board.
    Rules:
    BLACK MOVES FIRST, whites moves second
    Denote Black in board as '1'
    White as '-1', empty cells as '0'
    """
    color_dict = {1: 'Black', -1: 'White'}

    def __init__(self, size, multi_agent=False):
        self.board = np.zeros((size, size), np.int8)
        self.history = []
        self.terminated = False
        self.multi_agent = multi_agent
        self.winner = None
        self.size = size
        self.total_moves = 0
        self.move_history = []

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

        return hex_IsTerminal(self.board, self.size, player)

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

    def step(self, agent_move: tuple = None):
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
            raise Exception('Shouldnt come to step when game is terminated')

        else:
            board_state_copy = self.board.copy()
            self.history.append(board_state_copy)
            self.total_moves += 1

            if agent_move == None:
                action = random.choice(self.possible_actions())
            else:
                action = agent_move
            player = self.fetch_turn()
            self.board[action] = player
            self.move_history.append(action)

            iswin, winner = self.IsTerminal()

            if iswin:
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

    @staticmethod
    def move_to_string(move: tuple) -> str:
        i, j = move
        return f"{ascii_letters[j]}{i+1}"

    def data_for_nn(self):
        '''
        Generates data for nn in the correct format
        '''
        return (self.history, self.winner)

    @staticmethod
    def shortest_connection(board: np.ndarray, size: int, color: int) -> list:
        '''
        If board is terminated returns the winning connection.
        '''
        START_NODE = 'START'
        END_NODE = 'END'

        graph_dict = defaultdict(list)
        for i, j in ((x, y) for x in range(size) for y in range(size)):
            for neighbour in fetch_neighbours((i, j), color, size, board):
                graph_dict[(i, j)].append(neighbour)

            if color == 1:
                if i == 0:
                    graph_dict[START_NODE].append((i, j))
                    graph_dict[(i, j)].append(START_NODE)
                elif i == (size - 1):
                    graph_dict[END_NODE].append((i, j))
                    graph_dict[(i, j)].append(END_NODE)

            if color == -1:
                if j == 0:
                    graph_dict[START_NODE].append((i, j))
                    graph_dict[(i, j)].append(START_NODE)
                elif j == (size - 1):
                    graph_dict[END_NODE].append((i, j))
                    graph_dict[(i, j)].append(END_NODE)

        cost_dict = {}
        cost_dict[START_NODE] = 0
        queue = Queue()
        queue.put(START_NODE)
        while not(queue.empty()):
            node = queue.get()
            node_cost = cost_dict[node]
            for neighbour in graph_dict[node]:
                if cost_dict.get(neighbour, None) == None:
                    cost_dict[neighbour] = node_cost + 1
                    if neighbour == END_NODE:
                        break
                    queue.put(neighbour)

        shortest_path = []
        node = END_NODE
        while node != START_NODE:
            next_node = None
            next_node_value = float('inf')

            for neighbour in graph_dict[node]:
                neighbour_value = cost_dict.get(neighbour, float('inf'))
                if neighbour_value < next_node_value:
                    next_node_value = neighbour_value
                    next_node = neighbour
            node = next_node
            shortest_path.append(node)
        shortest_path.remove(START_NODE)

        return shortest_path


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
        # print(game.winner)
        # visualize_board(game.board, moves_order=game.move_history)
    return data


if __name__ == "__main__":
    c = generate_games(batchsize=10)
