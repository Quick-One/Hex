
from hex_class import GuiHexState, HexState
from hex_gui import GUI
from settings import game_settings

def prompt():
    
    pass

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
    gui = GUI(game, agent_game, game_settings.player_1, game_settings.player_2)
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
