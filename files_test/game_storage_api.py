from db_api import game_db_api
import pickle
from copy import deepcopy
from tabulate import tabulate
from hex_utils import visualize_board
import numpy as np

DB_PATH = 'test.db'
TABLE_NAME = 'games'
DATAFILE_PATH = 'game_storage_test.dat'

class GameSaver:
    def __init__(self, db_path, table_name, datafile_path):
        self.datafile_path = datafile_path
        self.db = game_db_api(db_path, table_name)

    def save_game_history(self, history, id):
        with open(self.datafile_path, 'ab') as dtfile:
            data = {'id': id, 'game': history}
            pickle.dump(data, dtfile)

    def save(self, username1: str, username2: str, winner: int, game_history: list):
        id = self.db.add_game(username1, username2, winner)
        self.save_game_history(game_history, id)

class GameStorageAPI:
    def __init__(self, db_path, table_name, datafile_path) -> None:
        self.datafile_path = datafile_path
        self.db = game_db_api(db_path, table_name)

    def search_datafile(self, id):
        with open(self.datafile_path, 'rb') as dtfile:
            while True:
                try:
                    curr_dict = pickle.load(dtfile)
                    print(curr_dict)
                    if curr_dict['id'] == id:
                        return curr_dict['game']
                except EOFError:
                    break
    
    @staticmethod
    def convert_to_board(move_history, size):
        board = np.zeros((size, size))
        curr = 1
        for x, y in move_history:
            board[x, y] = curr
            curr*=-1
        return board

    def menu(self):
        while True:
            username = input("Enter username to retrieve games: ")

            cursor = self.db.get_game_by_username(username)

            if len(list(self.db.get_game_by_username(username))) == 0:
                print("No game entries found with this username.\n")
            else:
                table = tabulate(
                    [[i + 1, *row] for i, row in enumerate(list(cursor))],
                    headers=['S No.', 'DB_ID', 'username1', 'username2', 'winner', 'time_added'],
                    tablefmt='orgtbl'
                )
                print(table)
                print()

                id = int(input("Select game by DB_ID: "))
                game_info = list(self.db.get_game_by_id(id))[0]
                game_history = self.search_datafile(id)
                print(game_history, game_info)

                visualize_board(GameStorageAPI.convert_to_board(game_history, 6), moves_order=game_history)


