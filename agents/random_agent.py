from random import choice

from hex.agent import HexAgent
from hex.board import Hex, possible_moves


def best_move(board: Hex):
    return choice(possible_moves(board.size, board.board))


HexAgent('random', best_move, description='plays random moves')
