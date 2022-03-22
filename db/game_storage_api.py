import pickle

import numpy as np
from hex.gui.draw_board import visualize_board
from tabulate import tabulate

from db.db_api import game_db_api

DB_PATH = 'db/game_storage_db.db'
TABLE_NAME = 'games'
DATAFILE_PATH = 'db/game_storage.dat'


class GameSaver:
    '''
    A class for saving game in the pickle file by indexing in the db
    '''

    def __init__(self, db_path, table_name, datafile_path):
        self.datafile_path = datafile_path
        self.db = game_db_api(db_path, table_name)

    def save_game_history(self, id, history, board_size):
        with open(self.datafile_path, 'ab') as dtfile:
            data = {'id': id, 'game': history, 'board_size': board_size}
            pickle.dump(data, dtfile)

    def save(self, player_white: str, player_black: str, winner: str, is_agent_play: bool, game_history: list, board_size: int):
        id = self.db.add_game(player_white, player_black, winner, is_agent_play)
        self.save_game_history(id, game_history, board_size)


class GameStorageAPI:
    '''
    User front front end for db.
    '''

    def __init__(self, db_path, table_name, datafile_path) -> None:
        self.datafile_path = datafile_path
        self.db = game_db_api(db_path, table_name)

    def search_datafile(self, id):
        with open(self.datafile_path, 'rb') as dtfile:
            while True:
                try:
                    curr_dict = pickle.load(dtfile)
                    if curr_dict['id'] == id:
                        return curr_dict['board_size'], curr_dict['game']
                except EOFError:
                    break

    @staticmethod
    def convert_to_board(move_history, size):
        board = np.zeros((size, size))
        curr = 1
        for x, y in move_history:
            board[x, y] = curr
            curr *= -1
        return board

    def menu(self):
        while True:
            print()
            username = input("Enter username to retrieve games: ")

            cursor = self.db.get_game_by_username(username)

            if len(list(self.db.get_game_by_username(username))) == 0:
                print("No game entries found with this username.\n")
            else:
                data = [[i + 1, *row] for i, row in enumerate(list(cursor))]

                for i in range(len(data)):
                    if bool(data[i][5]):
                        data[i][5] = 'Yes'
                    else:
                        data[i][5] = 'No'

                table = tabulate(
                    data,
                    headers=['S No.', 'DB_ID', 'player_black', 'player_white', 'winner', 'is_agent_play', 'time_added'],
                    tablefmt='orgtbl'
                )
                print(table)
                print()

                id = int(input("Select game by DB_ID: "))
                game_info = list(self.db.get_game_by_id(id))[0]
                board_size, game_history = self.search_datafile(id)

                visualize_board(GameStorageAPI.convert_to_board(game_history, board_size),
                                move_order=game_history, plot=True)
