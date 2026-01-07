#!/usr/bin/env python3
"""
Mindmap index store abstraction layer.

Design principles:
- Storage backend and caching are orthogonal concerns
- Any storage format can be a "database" (text, JSON, SQLite, etc.)
- CachedStore wraps any backend with in-memory caching

Storage backends:
- JSONStore: JSON file
- SQLiteStore: SQLite database
- TSVStore: Tab-separated text file (awk-friendly)

Caching:
- CachedStore: Wraps any store with in-memory cache

Usage:
    from index_store import create_index_store

    # Auto-selects backend based on file extension
    store = create_index_store("index.json")           # JSON
    store = create_index_store("index.db")             # SQLite
    store = create_index_store("index.tsv")            # TSV (awk-friendly)
    store = create_index_store("index.db", cache=True) # SQLite with cache

    # Lookup
    path = store.get("75009241")
    abs_path = store.resolve_path("75009241")
"""

import json
import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple


class IndexStore(ABC):
    """Abstract base class for mindmap index storage."""

    @abstractmethod
    def get(self, tree_id: str) -> Optional[str]:
        """Get mindmap path by tree ID."""
        pass

    @abstractmethod
    def contains(self, tree_id: str) -> bool:
        """Check if tree ID exists in index."""
        pass

    @abstractmethod
    def set(self, tree_id: str, path: str) -> None:
        """Set/update a tree ID -> path mapping."""
        pass

    @abstractmethod
    def delete(self, tree_id: str) -> bool:
        """Delete a mapping. Returns True if existed."""
        pass

    @abstractmethod
    def items(self) -> Iterator[Tuple[str, str]]:
        """Iterate over all (tree_id, path) pairs."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return number of entries."""
        pass

    @property
    @abstractmethod
    def base_dir(self) -> Optional[str]:
        """Return base directory for relative paths."""
        pass

    def resolve_path(self, tree_id: str) -> Optional[str]:
        """Get absolute path for a tree ID."""
        rel_path = self.get(tree_id)
        if rel_path and self.base_dir:
            return os.path.join(self.base_dir, rel_path)
        return rel_path


class CachedStore(IndexStore):
    """Wrapper that adds in-memory caching to any store."""

    def __init__(self, backend: IndexStore, preload: bool = True):
        """
        Args:
            backend: Underlying storage backend
            preload: If True, load entire index into cache on init
        """
        self._backend = backend
        self._cache: Dict[str, str] = {}
        self._fully_loaded = False

        if preload:
            self._load_all()

    def _load_all(self) -> None:
        """Load entire index into cache."""
        self._cache = dict(self._backend.items())
        self._fully_loaded = True

    def get(self, tree_id: str) -> Optional[str]:
        if tree_id in self._cache:
            return self._cache[tree_id]
        if self._fully_loaded:
            return None
        # Cache miss - fetch from backend
        path = self._backend.get(tree_id)
        if path:
            self._cache[tree_id] = path
        return path

    def contains(self, tree_id: str) -> bool:
        if tree_id in self._cache:
            return True
        if self._fully_loaded:
            return False
        return self._backend.contains(tree_id)

    def set(self, tree_id: str, path: str) -> None:
        self._cache[tree_id] = path
        self._backend.set(tree_id, path)

    def delete(self, tree_id: str) -> bool:
        self._cache.pop(tree_id, None)
        return self._backend.delete(tree_id)

    def items(self) -> Iterator[Tuple[str, str]]:
        if self._fully_loaded:
            return iter(self._cache.items())
        return self._backend.items()

    def count(self) -> int:
        if self._fully_loaded:
            return len(self._cache)
        return self._backend.count()

    @property
    def base_dir(self) -> Optional[str]:
        return self._backend.base_dir

    def invalidate(self, tree_id: str = None) -> None:
        """Invalidate cache entry or entire cache."""
        if tree_id:
            self._cache.pop(tree_id, None)
        else:
            self._cache.clear()
            self._fully_loaded = False


class JSONStore(IndexStore):
    """JSON file storage."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._base_dir: Optional[str] = None
        self._index: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                self._base_dir = data.get("base_dir")
                self._index = data.get("index", {})

    def _save(self) -> None:
        data = {
            "base_dir": self._base_dir,
            "count": len(self._index),
            "index": self._index
        }
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def get(self, tree_id: str) -> Optional[str]:
        return self._index.get(tree_id)

    def contains(self, tree_id: str) -> bool:
        return tree_id in self._index

    def set(self, tree_id: str, path: str) -> None:
        self._index[tree_id] = path
        self._save()

    def delete(self, tree_id: str) -> bool:
        if tree_id in self._index:
            del self._index[tree_id]
            self._save()
            return True
        return False

    def items(self) -> Iterator[Tuple[str, str]]:
        return iter(self._index.items())

    def count(self) -> int:
        return len(self._index)

    @property
    def base_dir(self) -> Optional[str]:
        return self._base_dir

    @base_dir.setter
    def base_dir(self, value: str) -> None:
        self._base_dir = value
        self._save()


class TSVStore(IndexStore):
    """Tab-separated text file storage (awk-friendly).

    Format:
        # base_dir: /path/to/mindmaps
        tree_id<TAB>path
        75009241<TAB>id75009241.smmx
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._base_dir: Optional[str] = None
        self._index: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("# base_dir:"):
                        self._base_dir = line.split(":", 1)[1].strip()
                    elif line and not line.startswith("#"):
                        parts = line.split("\t", 1)
                        if len(parts) == 2:
                            self._index[parts[0]] = parts[1]

    def _save(self) -> None:
        with open(self.filepath, 'w') as f:
            if self._base_dir:
                f.write(f"# base_dir: {self._base_dir}\n")
            for tree_id, path in sorted(self._index.items()):
                f.write(f"{tree_id}\t{path}\n")

    def get(self, tree_id: str) -> Optional[str]:
        return self._index.get(tree_id)

    def contains(self, tree_id: str) -> bool:
        return tree_id in self._index

    def set(self, tree_id: str, path: str) -> None:
        self._index[tree_id] = path
        self._save()

    def delete(self, tree_id: str) -> bool:
        if tree_id in self._index:
            del self._index[tree_id]
            self._save()
            return True
        return False

    def items(self) -> Iterator[Tuple[str, str]]:
        return iter(self._index.items())

    def count(self) -> int:
        return len(self._index)

    @property
    def base_dir(self) -> Optional[str]:
        return self._base_dir

    @base_dir.setter
    def base_dir(self, value: str) -> None:
        self._base_dir = value
        self._save()


class SQLiteStore(IndexStore):
    """SQLite database storage."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._conn = sqlite3.connect(filepath)
        self._init_db()

    def _init_db(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mindmap_index (
                tree_id TEXT PRIMARY KEY,
                path TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        self._conn.commit()

    def get(self, tree_id: str) -> Optional[str]:
        cursor = self._conn.cursor()
        cursor.execute(
            'SELECT path FROM mindmap_index WHERE tree_id = ?',
            (tree_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def contains(self, tree_id: str) -> bool:
        cursor = self._conn.cursor()
        cursor.execute(
            'SELECT 1 FROM mindmap_index WHERE tree_id = ?',
            (tree_id,)
        )
        return cursor.fetchone() is not None

    def set(self, tree_id: str, path: str) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO mindmap_index (tree_id, path) VALUES (?, ?)',
            (tree_id, path)
        )
        self._conn.commit()

    def delete(self, tree_id: str) -> bool:
        cursor = self._conn.cursor()
        cursor.execute(
            'DELETE FROM mindmap_index WHERE tree_id = ?',
            (tree_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def items(self) -> Iterator[Tuple[str, str]]:
        cursor = self._conn.cursor()
        cursor.execute('SELECT tree_id, path FROM mindmap_index')
        return iter(cursor.fetchall())

    def count(self) -> int:
        cursor = self._conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM mindmap_index')
        return cursor.fetchone()[0]

    @property
    def base_dir(self) -> Optional[str]:
        cursor = self._conn.cursor()
        cursor.execute(
            'SELECT value FROM metadata WHERE key = ?',
            ('base_dir',)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    @base_dir.setter
    def base_dir(self, value: str) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)',
            ('base_dir', value)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_index_store(filepath: str, cache: bool = False) -> IndexStore:
    """Factory function to create appropriate index store.

    Selects backend based on file extension:
    - .json -> JSONStore
    - .tsv, .txt -> TSVStore (awk-friendly)
    - .db, .sqlite -> SQLiteStore

    Args:
        filepath: Path to index file
        cache: If True, wrap with CachedStore

    Returns:
        IndexStore instance
    """
    ext = Path(filepath).suffix.lower()

    if ext == '.json':
        store = JSONStore(filepath)
    elif ext in ('.tsv', '.txt'):
        store = TSVStore(filepath)
    elif ext in ('.db', '.sqlite', '.sqlite3'):
        store = SQLiteStore(filepath)
    else:
        # Default to JSON
        store = JSONStore(filepath)

    if cache:
        store = CachedStore(store)

    return store


class ReverseIndex:
    """Tracks which mindmaps link to each mindmap (backlinks).

    Usage:
        reverse = ReverseIndex()
        reverse.add_link("75009241", "10388356")  # 75009241 links to 10388356
        backlinks = reverse.get_backlinks("10388356")  # -> ["75009241"]
    """

    def __init__(self):
        self._links: Dict[str, set] = {}  # target -> set of sources

    def add_link(self, source_id: str, target_id: str) -> None:
        """Record that source links to target."""
        if target_id not in self._links:
            self._links[target_id] = set()
        self._links[target_id].add(source_id)

    def get_backlinks(self, tree_id: str) -> list:
        """Get all mindmaps that link to this tree."""
        return list(self._links.get(tree_id, set()))

    def count_backlinks(self, tree_id: str) -> int:
        """Count how many mindmaps link to this tree."""
        return len(self._links.get(tree_id, set()))

    def items(self) -> Iterator[Tuple[str, list]]:
        """Iterate over (target_id, [source_ids])."""
        for target, sources in self._links.items():
            yield (target, list(sources))

    def save_json(self, filepath: str) -> None:
        """Save reverse index to JSON file."""
        data = {
            target: list(sources)
            for target, sources in self._links.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_json(self, filepath: str) -> None:
        """Load reverse index from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self._links = {k: set(v) for k, v in data.items()}

    def save_tsv(self, filepath: str) -> None:
        """Save reverse index to TSV file (awk-friendly)."""
        with open(filepath, 'w') as f:
            for target, sources in sorted(self._links.items()):
                # Format: target_id<TAB>source1,source2,source3
                f.write(f"{target}\t{','.join(sorted(sources))}\n")

    def load_tsv(self, filepath: str) -> None:
        """Load reverse index from TSV file."""
        self._links = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        target, sources = parts
                        self._links[target] = set(sources.split(','))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: index_store.py <index_file> [tree_id]")
        print("       index_store.py index.json 75009241")
        print("       index_store.py index.tsv   # awk-friendly format")
        sys.exit(1)

    filepath = sys.argv[1]
    store = create_index_store(filepath)

    if len(sys.argv) > 2:
        tree_id = sys.argv[2]
        path = store.get(tree_id)
        if path:
            print(f"{tree_id} -> {path}")
            abs_path = store.resolve_path(tree_id)
            if abs_path:
                print(f"Absolute: {abs_path}")
        else:
            print(f"Tree {tree_id} not found")
            sys.exit(1)
    else:
        print(f"Index: {filepath}")
        print(f"Count: {store.count()}")
        print(f"Base dir: {store.base_dir}")
