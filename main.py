from Hex_class import GuiHexState
from Hex_GUI import GUI
from settings import game_settings


def main():
    game = GuiHexState(size=game_settings.board_size)
    gui = GUI(game, game_settings.player_1, game_settings.player_2)
    gui.render_coords()
    gui.render_board()
    gui.render_labels()
    gui.render_tiles()
    gui.show(fullscreen=False)
    gui.simulate_game()
    gui.render_game_end()
    input('Press ENTER to QUIT')


if __name__ == '__main__':
    main()
