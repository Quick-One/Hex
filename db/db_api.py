import sqlite3
from datetime import datetime


class game_db_api:
    def __init__(self, db_path, table_name) -> None:
        self.db = sqlite3.connect(db_path)
        self.table_name = table_name

    def add_game(self, player_1: str, player_2: str, winner: str, is_agent_play: bool):
        cursor = self.db.execute(
            f'INSERT INTO {self.table_name} (player_black, player_white, winner, is_agent_play, curr_time) \
            VALUES (?, ?, ?, ?, ?);',
            (player_1, player_2, winner, int(is_agent_play), str(datetime.now()))
        )

        self.db.commit()
        return cursor.lastrowid

    def get_game_by_username(self, username):
        return self.db.execute(f'SELECT * FROM {self.table_name} \
            WHERE player_black="{username}" OR player_white="{username}";')

    def get_game_by_id(self, id):
        return self.db.execute(f"SELECT * FROM {self.table_name} \
            WHERE game_id={id}")

    def get_agent_games(self):
        return self.db.execute(f"SELCT * FROM {self.table_name} \
            WHERE is_agent_play=1")

    def get_games(self):
        return self.db.execute(f"SELECT * FROM {self.table_name}")
