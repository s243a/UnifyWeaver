import sqlite3
import json
import struct
import sys

class PtImporter:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS objects (
                id TEXT PRIMARY KEY,
                type TEXT,
                data JSON
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                vector BLOB
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS links (
                source_id TEXT,
                target_id TEXT,
                PRIMARY KEY (source_id, target_id)
            )
        ''')
        self.conn.commit()

    def upsert_object(self, obj_id, obj_type, data):
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO objects (id, type, data)
                VALUES (?, ?, ?)
            ''', (obj_id, obj_type, json.dumps(data)))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error upserting object {obj_id}: {e}", file=sys.stderr)

    def upsert_embedding(self, obj_id, vector):
        # Pack floats into bytes (Little Endian float32)
        blob = struct.pack(f'<{len(vector)}f', *vector)
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO embeddings (id, vector)
                VALUES (?, ?)
            ''', (obj_id, blob))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error upserting embedding {obj_id}: {e}", file=sys.stderr)

    def upsert_link(self, source_id, target_id):
        try:
            self.cursor.execute('''
                INSERT OR IGNORE INTO links (source_id, target_id)
                VALUES (?, ?)
            ''', (source_id, target_id))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error upserting link {source_id}->{target_id}: {e}", file=sys.stderr)

    def close(self):
        self.conn.close()
