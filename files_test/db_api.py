import sqlite3
from datetime import datetime

class game_db_api:
    def __init__(self, db_path, table_name) -> None:
        self.db = sqlite3.connect(db_path)
        self.table_name = table_name

    def add_game(self, username1: str, username2: str, winner: int):
        cursor = self.db.execute(
            f'INSERT INTO {self.table_name} (username1, username2, winner, curr_time) \
            VALUES (?, ?, ?, ?);',
            (username1, username2, winner, str(datetime.now()))
        )
        
        self.db.commit()
        return cursor.lastrowid

    def get_game_by_username(self, username):
        return self.db.execute(f'SELECT * FROM {self.table_name} \
            WHERE username1="{username}" OR username2="{username}";')

    def get_game_by_id(self, id):
        return self.db.execute(f"SELECT * FROM {self.table_name} \
            WHERE id={id}")

    def get_games(self):
        return self.db.execute(f"SELECT * FROM {self.table_name}")
    
