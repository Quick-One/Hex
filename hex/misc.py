from random import choice

from hex.board import Hex, possible_moves


def random_game():
    board = Hex(6)
    while board.winner is None:
        move = choice(possible_moves(board.size, board.board))
        board.step(move)
    return board
