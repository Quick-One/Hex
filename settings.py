
class game_settings:
    player_1 = 'Agent'
    player_2 = 'Human'
    board_size = 6


class board_settings:
    CIRCUMRADIUS = 1
    INRADIUS = 0.6
    BOARD_OFFSET = 3
    LABEL_OFFSET = 0.7


class py_agent_settings:
    time_control = None
    num_rollout = None


class Numba_agent_settings:
    num_rollouts = 100_000


class RAVE_constants:
    rave_const = 300
    explore = 0.5
