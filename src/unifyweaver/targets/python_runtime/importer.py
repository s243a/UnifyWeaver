"""
Multi-account Pearltrees importer with SQLite storage.

Stores trees, pearls, and embeddings with account awareness.
Extracts account from URI patterns like: https://www.pearltrees.com/s243a/...
"""

import sqlite3
import json
import struct
import sys
import re
from pathlib import Path
from typing import Optional, List, Dict, Any


def extract_account_from_uri(uri: str) -> str:
    """
    Extract account name from Pearltrees URI.
    
    Examples:
        https://www.pearltrees.com/s243a/... -> s243a
        https://www.pearltrees.com/s243a_groups/... -> s243a_groups
        http://www.pearltrees.com/s243a#sioc -> s243a
    """
    if not uri:
        return "unknown"
    match = re.search(r'pearltrees\.com/([^/#?]+)', uri)
    return match.group(1) if match else "unknown"


class PtMultiAccountImporter:
    """
    SQLite importer with multi-account support.
    
    Schema:
        objects: stores trees and pearls with account field
        embeddings: stores vectors for semantic search
        links: stores parent-child relationships
    """
    
    SCHEMA_VERSION = 2  # Incremented for multi-account support
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self.cursor = self.conn.cursor()
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema with account support."""
        
        # Check schema version
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        self.cursor.execute('SELECT value FROM schema_info WHERE key = ?', ('version',))
        row = self.cursor.fetchone()
        current_version = int(row['value']) if row else 0
        
        if current_version < self.SCHEMA_VERSION:
            self._migrate_schema(current_version)
        
        # Objects table with account
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS objects (
                id TEXT NOT NULL,
                account TEXT NOT NULL DEFAULT 'unknown',
                type TEXT,
                data JSON,
                PRIMARY KEY (id, account)
            )
        ''')
        
        # Index on account for fast filtering
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_objects_account ON objects(account)
        ''')
        
        # Index on type for filtering by trees vs pearls
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(type)
        ''')
        
        # Embeddings table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT NOT NULL,
                account TEXT NOT NULL DEFAULT 'unknown',
                vector BLOB,
                PRIMARY KEY (id, account)
            )
        ''')
        
        # Links table with account
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                account TEXT NOT NULL DEFAULT 'unknown',
                link_type TEXT DEFAULT 'contains',
                PRIMARY KEY (source_id, target_id, account)
            )
        ''')
        
        # Index for finding children of a parent
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_id, account)
        ''')
        
        # Update schema version
        self.cursor.execute('''
            INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)
        ''', ('version', str(self.SCHEMA_VERSION)))
        
        self.conn.commit()
    
    def _migrate_schema(self, from_version: int):
        """Migrate from older schema versions."""
        if from_version < 2:
            # Add account column if missing
            try:
                self.cursor.execute('ALTER TABLE objects ADD COLUMN account TEXT DEFAULT "unknown"')
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                self.cursor.execute('ALTER TABLE embeddings ADD COLUMN account TEXT DEFAULT "unknown"')
            except sqlite3.OperationalError:
                pass
            
            try:
                self.cursor.execute('ALTER TABLE links ADD COLUMN account TEXT DEFAULT "unknown"')
            except sqlite3.OperationalError:
                pass
            
            try:
                self.cursor.execute('ALTER TABLE links ADD COLUMN link_type TEXT DEFAULT "contains"')
            except sqlite3.OperationalError:
                pass
        
        self.conn.commit()
    
    def upsert_object(self, obj_id: str, obj_type: str, data: dict, account: Optional[str] = None):
        """
        Insert or update an object.
        
        Args:
            obj_id: Unique ID (can be URI or tree_id)
            obj_type: 'tree', 'pearl', 'page_pearl', etc.
            data: JSON-serializable data dict
            account: Account name (extracted from URI if not provided)
        """
        # Extract account from URI if not provided
        if account is None:
            # Try to get from data's uri or about field
            uri = data.get('uri') or data.get('about') or obj_id
            account = extract_account_from_uri(uri)
        
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO objects (id, account, type, data)
                VALUES (?, ?, ?, ?)
            ''', (obj_id, account, obj_type, json.dumps(data)))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error upserting object {obj_id}: {e}", file=sys.stderr)
    
    def upsert_embedding(self, obj_id: str, vector: List[float], account: Optional[str] = None):
        """Insert or update an embedding vector."""
        if account is None:
            account = extract_account_from_uri(obj_id)
        
        blob = struct.pack(f'<{len(vector)}f', *vector)
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO embeddings (id, account, vector)
                VALUES (?, ?, ?)
            ''', (obj_id, account, blob))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error upserting embedding {obj_id}: {e}", file=sys.stderr)
    
    def upsert_link(self, source_id: str, target_id: str, account: Optional[str] = None, 
                    link_type: str = "contains"):
        """Insert or update a link between objects."""
        if account is None:
            account = extract_account_from_uri(source_id)
        
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO links (source_id, target_id, account, link_type)
                VALUES (?, ?, ?, ?)
            ''', (source_id, target_id, account, link_type))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error upserting link {source_id}->{target_id}: {e}", file=sys.stderr)
    
    def get_object(self, obj_id: str, account: Optional[str] = None) -> Optional[dict]:
        """Get an object by ID, optionally filtered by account."""
        if account:
            self.cursor.execute(
                'SELECT * FROM objects WHERE id = ? AND account = ?',
                (obj_id, account)
            )
        else:
            self.cursor.execute('SELECT * FROM objects WHERE id = ?', (obj_id,))
        
        row = self.cursor.fetchone()
        if row:
            result = dict(row)
            result['data'] = json.loads(result['data']) if result['data'] else {}
            return result
        return None
    
    def get_children(self, parent_id: str, account: Optional[str] = None) -> List[dict]:
        """Get all children of a parent object."""
        if account:
            self.cursor.execute('''
                SELECT o.* FROM objects o
                JOIN links l ON o.id = l.target_id AND o.account = l.account
                WHERE l.source_id = ? AND l.account = ?
            ''', (parent_id, account))
        else:
            self.cursor.execute('''
                SELECT o.* FROM objects o
                JOIN links l ON o.id = l.target_id
                WHERE l.source_id = ?
            ''', (parent_id,))
        
        results = []
        for row in self.cursor.fetchall():
            result = dict(row)
            result['data'] = json.loads(result['data']) if result['data'] else {}
            results.append(result)
        return results
    
    def get_pearls_in_tree(self, tree_id: str, account: Optional[str] = None, 
                           limit: int = 20) -> List[dict]:
        """Get pearls (bookmarks) contained in a tree."""
        if account:
            self.cursor.execute('''
                SELECT o.* FROM objects o
                JOIN links l ON o.id = l.target_id AND o.account = l.account
                WHERE l.source_id = ? AND l.account = ? AND o.type LIKE '%pearl%'
                LIMIT ?
            ''', (tree_id, account, limit))
        else:
            self.cursor.execute('''
                SELECT o.* FROM objects o
                JOIN links l ON o.id = l.target_id
                WHERE l.source_id = ? AND o.type LIKE '%pearl%'
                LIMIT ?
            ''', (tree_id, limit))
        
        results = []
        for row in self.cursor.fetchall():
            result = dict(row)
            result['data'] = json.loads(result['data']) if result['data'] else {}
            results.append(result)
        return results
    
    def get_all_accounts(self) -> List[str]:
        """Get list of all accounts in the database."""
        self.cursor.execute('SELECT DISTINCT account FROM objects ORDER BY account')
        return [row['account'] for row in self.cursor.fetchall()]
    
    def get_trees_by_account(self, account: str, limit: int = 100) -> List[dict]:
        """Get all trees for an account."""
        self.cursor.execute('''
            SELECT * FROM objects 
            WHERE account = ? AND type = 'tree'
            LIMIT ?
        ''', (account, limit))
        
        results = []
        for row in self.cursor.fetchall():
            result = dict(row)
            result['data'] = json.loads(result['data']) if result['data'] else {}
            results.append(result)
        return results
    
    def search_by_title(self, query: str, account: Optional[str] = None, 
                        obj_type: Optional[str] = None, limit: int = 20) -> List[dict]:
        """Search objects by title (uses LIKE, case-insensitive)."""
        sql = "SELECT * FROM objects WHERE json_extract(data, '$.title') LIKE ?"
        params = [f'%{query}%']
        
        if account:
            sql += " AND account = ?"
            params.append(account)
        
        if obj_type:
            sql += " AND type = ?"
            params.append(obj_type)
        
        sql += " LIMIT ?"
        params.append(limit)
        
        self.cursor.execute(sql, params)
        
        results = []
        for row in self.cursor.fetchall():
            result = dict(row)
            result['data'] = json.loads(result['data']) if result['data'] else {}
            results.append(result)
        return results
    
    def stats(self) -> dict:
        """Get database statistics."""
        self.cursor.execute('SELECT COUNT(*) as count FROM objects')
        total_objects = self.cursor.fetchone()['count']
        
        self.cursor.execute('SELECT COUNT(*) as count FROM embeddings')
        total_embeddings = self.cursor.fetchone()['count']
        
        self.cursor.execute('SELECT COUNT(*) as count FROM links')
        total_links = self.cursor.fetchone()['count']
        
        self.cursor.execute('SELECT account, COUNT(*) as count FROM objects GROUP BY account')
        by_account = {row['account']: row['count'] for row in self.cursor.fetchall()}
        
        self.cursor.execute('SELECT type, COUNT(*) as count FROM objects GROUP BY type')
        by_type = {row['type']: row['count'] for row in self.cursor.fetchall()}
        
        return {
            'total_objects': total_objects,
            'total_embeddings': total_embeddings,
            'total_links': total_links,
            'by_account': by_account,
            'by_type': by_type
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()


# Backward compatibility alias
PtImporter = PtMultiAccountImporter
