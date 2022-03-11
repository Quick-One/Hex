from db_api import game_db_api
from game_storage_api import GameSaver, GameStorageAPI

DB_PATH = 'test.db'
TABLE_NAME = 'games'
DATAFILE_PATH = 'game_storage_test.dat'

# game_saver = GameSaver(DB_PATH, TABLE_NAME, DATAFILE_PATH)
# game_saver.save('avi', 'user2', 1, [(1, 1), (2, 2,), (1, 5)])

# db = game_db_api(DB_PATH, TABLE_NAME)
# cursor = db.get_game_by_username('avi')
# print(list(cursor))

api = GameStorageAPI(DB_PATH, TABLE_NAME, DATAFILE_PATH)
api.menu()