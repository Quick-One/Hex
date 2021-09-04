from time import sleep, time
from env import Hex
import numpy as np
from random import choice

def func(board):
    row_coords, column_coords = np.where(board == 0)
    possible_actions = []
    for n in range(len(row_coords)):
            possible_actions.append((row_coords[n], column_coords[n]))
    sleep(3)    
    return choice(possible_actions)

