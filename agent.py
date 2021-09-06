from time import sleep
import numpy as np
from random import choice


def best_move(board) -> tuple:
    row_coords, column_coords = np.where(board == 0)
    possible_actions = []
    for n in range(len(row_coords)):
        possible_actions.append((row_coords[n], column_coords[n]))
    sleep(2)
    return choice(possible_actions)
