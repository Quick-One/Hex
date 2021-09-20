import numpy as np
from random import choice

from rave_class import MCTSAgent

from time import perf_counter
# def best_move(board) -> tuple:
#     row_coords, column_coords = np.where(board == 0)
#     possible_actions = []
#     for n in range(len(row_coords)):
#         possible_actions.append((row_coords[n], column_coords[n]))
#     return choice(possible_actions)

def best_move(board):
    agent = MCTSAgent(board)
    t1_start = perf_counter()
    agent.search()
    t1_stop = perf_counter()
    print("Elapsed time during the whole program in seconds:",
                                        t1_stop-t1_start)
    return agent.best_move()