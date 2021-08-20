import numpy as np
from seize_isTerminal import Check_end
from Hex_utils import visualize_board
import cProfile
from itertools import product, combinations
from tqdm import tqdm
import pickle


def state_value(state: np.ndarray, player: int, memory: dict, cluster: list):

    value = memory.get((state.tobytes(), player), None)
    if value == None:
        is_end, end_value = Check_end(state, cluster)

        # If end condition. store it in memory
        if is_end:
            memory[(state.tobytes(), player)] = end_value
            return end_value

        else:
            next_states = []
            for move in possible_moves(state):
                state_copy = state.copy()
                state_copy[move] = player
                next_states.append(state_copy)

            if player == -1:
                value = min([state_value(next_state, player*-1, memory, cluster)
                            for next_state in next_states])
            elif player == 1:
                value = max([state_value(next_state, player*-1, memory, cluster)
                            for next_state in next_states])

            memory[(state.tobytes(), player)] = value
            return value

    return value


def possible_moves(state):
    move_x, move_y = np.where(state == 0)
    return zip(move_x, move_y)


def filter_check(obs_pattern: np.ndarray, intrusion_zone_pattern: np.ndarray,  filter_list: list):

    # size of intrusion zone
    size_intru_zone_pattern = np.count_nonzero(intrusion_zone_pattern == 1)

    for filter in filter_list:
        # size of intrusion_zone of filter
        intrusion_zone_filter = filter[1]
        size_intru_zone_filter = np.count_nonzero(intrusion_zone_filter == 1)

        # if size of intrusion zone is same:
        # check by overlaying
        if size_intru_zone_pattern == size_intru_zone_filter:
            pattern_x, pattern_y = intrusion_zone_pattern.shape
            filter_x, filter_y = intrusion_zone_filter.shape

            # size of filter should be smaller than pattern
            if (filter_x <= pattern_x) and (filter_y <= pattern_y):

                for x, y in ((a, b) for a in range(pattern_x-filter_x+1) for b in range(pattern_y-filter_y+1)):
                    if np.array_equal(intrusion_zone_pattern[x:x+filter_x, y:y+filter_y], intrusion_zone_filter):
                        return False

    return True


if __name__ == '__main__':

    patterns_search_list = [(2, 2, 2),
                            (2, 3, 2), (3, 2, 2),
                            (2, 4, 2), (4, 2, 2), (2, 4, 3), (4, 2, 3),
                            (3, 3, 3), (3, 3, 2),
                            (4, 3, 3), (3, 4, 3),
                            (4, 4, 2), (4, 4, 3), (4, 4, 4),
                            (5, 5, 2)]

    filters_found = []

    for query in patterns_search_list:

        size_x, size_y, n_pieces = query
        empty_board = np.zeros((size_x, size_y), int)
        cnt_pattern = 0

        l = list(combinations(product(range(size_x), range(size_y)), n_pieces))

        for i in tqdm(range(len(l))):
            connect_me = l[i]

            # Create board with random pieces at begin
            board = empty_board.copy()
            for coord in connect_me:
                board[coord] = 1

            # If it is allready connected no pattern to recognize here
            if Check_end(board, connect_me)[0]:
                continue

            state_value_memory = {}
            initial_state_value = state_value(
                board, -1, state_value_memory, connect_me)

            # If it cant be connected then nothing interesting here
            if initial_state_value == 0:
                continue

            intrusion_zone = np.zeros((size_x, size_y), int)
            for move in possible_moves(board):

                state_copy = board.copy()
                state_copy[move] = -1

                # If move_coord is occupied and the state is no longer connectable hence it is in intrusion zone
                if state_value(state_copy, -1, state_value_memory, connect_me) == 0:
                    intrusion_zone[move] = 1

            # if filter is not present in the filter_found
            if filter_check(board, intrusion_zone, filters_found):
                filters_found.append(
                    (board, intrusion_zone, state_value_memory))

    with open('filters.pickle', 'wb') as handle:
        pickle.dump(filters_found, handle)
