from game_storage_api import DATAFILE_PATH, DB_PATH, TABLE_NAME, GameSaver, GameStorageAPI
from hex_class import GuiHexState, HexState
from hex_gui import GUI
from settings import game_settings
from random import randint


def print_rules():
    rules = (f'{"_"*80}',
             'Rules of Hex: ',
             '1. Players choose a color and take turns. Black moves first.',
             "2. On each turn one piece of player's color is placed in an empty hexagonal cell.",
             '3. The first player to form a connected path of their pieces linking the opposing sides of the board marked by their colour wins.',
             f'{"‾"*80}',
             )
    print('\n'.join(rules))

def menu():
    print("""
        1. View stored games
        2. Play game
    """)
    while True:
        option = input('Choose option(Enter 1 or 2): ')

        if option == '1':
            api = GameStorageAPI(DB_PATH, TABLE_NAME, DATAFILE_PATH)
            api.menu()
            break
        elif option == '2':
            print_rules()
            game_save_api = GameSaver(DB_PATH, TABLE_NAME, DATAFILE_PATH)
            first_turn = randint(1, 2)

            username1 = input('Enter username of player: ')
            agent_input = input('Do you want to play with computer (Y/N)? ')

            if agent_input == 'N':
                username2 = input('Enter username of second player: ')
                is_agent = False

                if first_turn == 1:
                    print(f'User {username1} plays as black and moves first.')
                    move_history, winner = play_game('Human', 'Human')
                    game_save_api.save(username2, username1, winner, is_agent, move_history, game_settings.board_size)
                else:
                    print(f'User {username2} plays as black and moves first.')
                    move_history, winner = play_game('Human', 'Human')
                    game_save_api.save(username1, username2, winner, is_agent, move_history, game_settings.board_size)

            else:
                is_agent = True

                if first_turn == 1:
                    print(f'User {username1} plays as black and moves first.')
                    move_history, winner = play_game('Human', 'Agent')
                    game_save_api.save(None, username1, winner, is_agent, move_history, game_settings.board_size)
                else:
                    print(f'User {username1} plays as white and moves second.')
                    move_history, winner = play_game('Agent', 'Human')
                    game_save_api.save(username1, None, winner, is_agent, move_history, game_settings.board_size)
            break
        else:
            print('Invalid option entered. Please try again')

def play_game(player_1, player_2):
    game = GuiHexState(size=game_settings.board_size)
    agent_game = HexState(size=game_settings.board_size)
    gui = GUI(game, agent_game, player_1, player_2)
    gui.render_coords()
    gui.render_board()
    gui.render_labels()
    gui.render_tiles()
    gui.configure_mpl_window()
    gui.show(fullscreen=False)
    gui.simulate_game()
    gui.render_game_end()
    input('Press ENTER to QUIT')
    return game.move_history, game.color_legend[game.winner]


if __name__ == '__main__':
    menu()
