from time import perf_counter

import numba_rave
from hex_utils import intmove_to_tupl, move_to_string
from rave import MCTSAgent


class HexAgents:

    @staticmethod
    def MCTS_RAVE(board, num_rollout, time_limit):
        agent = MCTSAgent(board)
        start = perf_counter()
        agent.search(num_rollout, time_limit)
        best_move = agent.best_move()
        print(
            f'Completed {agent.num_rollouts} rollouts in {(perf_counter()-start):.3f}s.')
        print(f'INFO: MCTS_RAVE plays {move_to_string(best_move)}.')
        return best_move

    @staticmethod
    def numba_MCTS_RAVE(board, n_rollout: int = 100_000):
        size = board.size
        start = perf_counter()
        best_move, heatmap = numba_rave.fetch_best_move(board, n_rollout)
        best_move = intmove_to_tupl(best_move, size)
        print(
            f'Completed {n_rollout} rollouts in {(perf_counter()-start):.3f}s.')
        print(f'INFO: MCTS_RAVE plays {move_to_string(best_move)}.')
        return best_move, heatmap
