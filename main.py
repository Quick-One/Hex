from argparse import ArgumentParser

import numba_hex_class
import numba_rave
from hex_class import GuiHexState, HexState
from hex_gui import GUI
from settings import game_settings


def parse_flags():
    parser = ArgumentParser()
    parser.add_argument('-n', '-numba', action='store_true')
    args = parser.parse_args()
    return args.n


def print_rules():
    rules = (f'{"_"*80}',
             'Rules of Hex: ',
             '1. Players choose a color and take turns. Black moves first.',
             "2. On each turn one piece of player's color is placed in an empty hexagonal cell.",
             '3. The first player to form a connected path of their pieces linking the opposing sides of the board marked by their colour wins.',
             f'{"‾"*80}',
             )
    print('\n'.join(rules))


def main():
    print_rules()
    game = GuiHexState(size=game_settings.board_size)
    agent_game = HexState(size=game_settings.board_size)
    if parse_flags():
        # Using numba optimised rave
        print(
            f'Running Numba Optimised RAVE. \nThis may take upto 30 second to compile. \n{"_"*80}')
        numba_hex_class.compile_board()
        numba_rave.compile_RAVE()
        print(f'{"‾"*80}')

        numba_agent_game = numba_hex_class.create_empty_board(
            game_settings.board_size)
        gui = GUI(game, agent_game, game_settings.player_1,
                  game_settings.player_2, numba_agent_game=numba_agent_game, heatmap=False)
    else:
        print(f'{"_"*80}')
        print('Running non optimised MCTS_RAVE. To run optimised MCTS_RAVE run: ')
        print('"python main.py -n"'.center(80))
        print(f'{"‾"*80}')
        gui = GUI(game, agent_game, game_settings.player_1,
                  game_settings.player_2)
    gui.render_coords()
    gui.render_board()
    gui.render_labels()
    gui.render_tiles()
    gui.configure_mpl_window()
    gui.show(fullscreen=False)
    gui.simulate_game()
    gui.render_game_end()
    input('Press ENTER to QUIT')


if __name__ == '__main__':
    main()
