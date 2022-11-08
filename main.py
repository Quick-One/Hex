import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
from json import load
from random import shuffle

import agents.mcts
from db.game_storage_api import (DATAFILE_PATH, DB_PATH, TABLE_NAME, GameSaver,
                                 GameStorageAPI)
from hex.board import RULES_OF_HEX, Hex, player
from hex.gui.hex_gui import HexGUI
from hex.gui.theme import GUI_PARAMS



theme = load(open('hex/themes/red_blue.json'))
GUI_PARAMS.override_theme(theme)


BOARD_SIZE = 6


def name_wrapper(person: player):
    if person.is_AI:
        return None
    return person.name


def print_rules():
    print('\n'.join(RULES_OF_HEX))


def menu():
    print("""
        1. View stored games
        2. Play game
    """)
    while True:
        option = input('Choose option (Enter 1 or 2): ')

        if option == '1':
            api = GameStorageAPI(DB_PATH, TABLE_NAME, DATAFILE_PATH)
            api.menu()
            break

        elif option == '2':
            print_rules()
            game_save_api = GameSaver(DB_PATH, TABLE_NAME, DATAFILE_PATH)

            p1 = player(input('Enter username of player: '), False)
            agent_input = input('Do you want to play with computer (Y/N)? ')
            if agent_input.upper() == 'N':
                is_agent = False
                p2 = player(input('Enter username of second player: '), False)
            else:
                is_agent = True
                p2 = player('MCTS', True)
            player_list = [p1, p2]
            shuffle(player_list)
            print(f'Player 1: {player_list[0].name} plays first')
            move_history, winner = play_game(*player_list)
            game_save_api.save(name_wrapper(player_list[0]), name_wrapper(player_list[1]),
                               winner, is_agent, move_history, BOARD_SIZE)
            break
        else:
            print('Invalid option entered. Please try again')


def play_game(player_1: player, player_2: player):
    game = Hex(BOARD_SIZE, player_1, player_2)
    HexGUI.start_game(game)
    return game.move_history, game.legend[game.winner]


if __name__ == '__main__':
    menu()
