from argparse import ArgumentParser

from Hex_class import GuiHexState, HexState
from Hex_GUI import GUI
import Numba_hex_class
import Numba_rave
from settings import game_settings


def parse_flags():
    parser = ArgumentParser()
    parser.add_argument('-n', '-numba', action='store_true')
    args = parser.parse_args()
    return args.n


def main():
    game = GuiHexState(size=game_settings.board_size)
    agent_game = HexState(size=game_settings.board_size)
    if parse_flags():
        # Using numba optimised rave
        print(f'Running Numba Optimised RAVE. \nThis may take upto 30 second to compile. \n{"__"*15}')
        Numba_hex_class.compile_board(100, 6)
        Numba_rave.compile_RAVE(1000)
        
        numba_agent_game = Numba_hex_class.create_empty_board(
            game_settings.board_size)
        gui = GUI(game, agent_game, game_settings.player_1,
                  game_settings.player_2, numba_agent_game=numba_agent_game)
    else:
        gui = GUI(game, agent_game, game_settings.player_1, game_settings.player_2 )
    gui.render_coords()
    gui.render_board()
    gui.render_labels()
    gui.render_tiles()
    gui.set_window_label_icon()
    gui.show(fullscreen=False)
    gui.simulate_game()
    gui.render_game_end()
    input('Press ENTER to QUIT')


if __name__ == '__main__':
    main()
