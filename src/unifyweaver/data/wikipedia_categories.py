"""
Wikipedia Category Bridge for Organizational Metric Training

Provides integration between Wikipedia's category hierarchy and Pearltrees
organizational structure. Used to generate training pairs that bridge
Wikipedia articles to the appropriate Pearltrees folders.

Usage:
    from unifyweaver.data import get_category_bridge

    bridge = get_category_bridge()

    # Find connection point for an article
    folder, hops = bridge.find_folder_connection(
        "David Lee (physicist)",
        pearltrees_folders={"Physicists", "Nobel laureates", ...}
    )

    # Generate training pair
    training_pair = bridge.generate_training_pair(
        article_title="David Lee (physicist)",
        pearltrees_data=pearltrees_jsonl_records
    )
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DB_PATH = Path("data/wikipedia_categories.db")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CategoryMatch:
    """Result of finding a folder connection."""
    folder_name: str
    folder_path: str  # Full path in Pearltrees hierarchy
    hops: int         # Category hierarchy hops to reach match
    via_category: str # Which category led to the match


@dataclass
class TrainingPair:
    """A training pair generated from Wikipedia category bridge."""
    query: str        # Wikipedia article title
    target: str       # Pearltrees target text
    distance: float   # Estimated organizational distance
    source: str       # "wikipedia_category_bridge"
    metadata: Dict    # Extra info (categories, hops, etc.)


# =============================================================================
# Wikipedia Category Bridge
# =============================================================================

class WikipediaCategoryBridge:
    """
    Bridge Wikipedia articles to Pearltrees organizational structure.

    Uses Wikipedia's category hierarchy to find connections between
    articles and Pearltrees folders, then generates training pairs.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        """
        Initialize the category bridge.

        Args:
            db_path: Path to SQLite database with category data
        """
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._parent_cache: Dict[str, List[str]] = {}

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy connection to database."""
        if self._conn is None:
            if not self.db_path.exists():
                raise FileNotFoundError(
                    f"Wikipedia categories database not found: {self.db_path}\n"
                    "Run: python scripts/fetch_wikipedia_categories.py --download --parse"
                )
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def is_available(self) -> bool:
        """Check if the database is available."""
        return self.db_path.exists()

    def get_categories(self, page_id: int) -> List[str]:
        """Get categories for a page by ID."""
        cursor = self.conn.execute(
            "SELECT cl_to FROM categorylinks WHERE cl_from = ?",
            (page_id,)
        )
        return [row['cl_to'] for row in cursor]

    def get_categories_by_title(self, title: str) -> List[str]:
        """Get categories for a page by title."""
        # First get page_id from title
        cursor = self.conn.execute(
            "SELECT page_id FROM page WHERE page_title = ?",
            (title.replace(' ', '_'),)
        )
        row = cursor.fetchone()
        if row:
            return self.get_categories(row['page_id'])
        return []

    def get_parent_categories(self, category: str) -> List[str]:
        """
        Get parent categories of a category.

        Categories are also pages in Wikipedia, so we look up their categories.
        """
        if category in self._parent_cache:
            return self._parent_cache[category]

        cursor = self.conn.execute(
            """SELECT cl_to FROM categorylinks
               WHERE cl_from = (SELECT page_id FROM page WHERE page_title = ? AND page_namespace = 14)""",
            (category,)
        )
        parents = [row['cl_to'] for row in cursor]
        self._parent_cache[category] = parents
        return parents

    def walk_hierarchy(
        self,
        categories: List[str],
        max_depth: int = 10
    ) -> Dict[str, Tuple[int, str]]:
        """
        Walk up the category hierarchy from starting categories.

        Returns: {category: (depth, via_category)} for all reachable categories
        """
        visited = {}
        queue = [(cat, 0, cat) for cat in categories]  # (category, depth, source)

        while queue:
            category, depth, via = queue.pop(0)

            if category in visited:
                continue
            if depth > max_depth:
                continue

            visited[category] = (depth, via)

            # Get parent categories
            parents = self.get_parent_categories(category)
            for parent in parents:
                if parent not in visited:
                    queue.append((parent, depth + 1, via))

        return visited

    def find_folder_connection(
        self,
        title: str,
        pearltrees_folders: Set[str],
        max_depth: int = 10
    ) -> Optional[CategoryMatch]:
        """
        Find closest matching Pearltrees folder by walking up category hierarchy.

        Args:
            title: Wikipedia article title
            pearltrees_folders: Set of Pearltrees folder names to match
            max_depth: Maximum category hierarchy depth to search

        Returns:
            CategoryMatch if found, None otherwise
        """
        categories = self.get_categories_by_title(title)
        if not categories:
            return None

        hierarchy = self.walk_hierarchy(categories, max_depth)

        best_match = None
        best_depth = float('inf')
        best_via = None

        # Build normalized lookup for folder matching
        folder_lookup = {}
        for folder in pearltrees_folders:
            folder_lookup[folder.lower()] = folder

        for category, (depth, via) in hierarchy.items():
            # Normalize category name for matching
            normalized = category.replace('_', ' ').lower()

            # Direct match
            if normalized in folder_lookup:
                if depth < best_depth:
                    best_match = folder_lookup[normalized]
                    best_depth = depth
                    best_via = via

            # Partial match (category ends with folder name)
            for folder_lower, folder in folder_lookup.items():
                if normalized.endswith(folder_lower):
                    if depth < best_depth:
                        best_match = folder
                        best_depth = depth
                        best_via = via

        if best_match:
            return CategoryMatch(
                folder_name=best_match,
                folder_path="",  # Will be filled in by caller with full path
                hops=best_depth,
                via_category=best_via
            )
        return None

    def generate_training_pair(
        self,
        article_title: str,
        pearltrees_data: List[Dict],
        max_depth: int = 10
    ) -> Optional[TrainingPair]:
        """
        Generate a training pair connecting a Wikipedia article to Pearltrees.

        Args:
            article_title: Wikipedia article title
            pearltrees_data: List of Pearltrees JSONL records
            max_depth: Maximum category hierarchy depth

        Returns:
            TrainingPair if connection found, None otherwise
        """
        # Extract folder names and build lookup
        folders = {}  # folder_name -> record
        for rec in pearltrees_data:
            if rec.get('type') == 'Tree':
                title = rec.get('raw_title', rec.get('query', ''))
                if title:
                    folders[title] = rec

                # Also extract folder names from target_text hierarchy
                target = rec.get('target_text', '')
                for part in target.split('\n'):
                    part = part.strip().lstrip('- ')
                    if part and not part.startswith('/'):
                        if part not in folders:
                            folders[part] = rec

        # Find connection
        match = self.find_folder_connection(
            article_title,
            set(folders.keys()),
            max_depth
        )

        if not match:
            return None

        # Get the matched record
        folder_record = folders[match.folder_name]
        target_text = folder_record.get('target_text', match.folder_name)

        # Compute distance: more hops = greater distance
        # Base distance is 1.0 for direct category match, increases with hops
        distance = 1.0 + (match.hops * 0.2)

        return TrainingPair(
            query=article_title,
            target=target_text,
            distance=distance,
            source="wikipedia_category_bridge",
            metadata={
                'matched_folder': match.folder_name,
                'category_hops': match.hops,
                'via_category': match.via_category,
                'categories': self.get_categories_by_title(article_title)[:10],
            }
        )

    def generate_batch_training_pairs(
        self,
        article_titles: List[str],
        pearltrees_data: List[Dict],
        max_depth: int = 10
    ) -> List[TrainingPair]:
        """
        Generate training pairs for multiple articles.

        Args:
            article_titles: List of Wikipedia article titles
            pearltrees_data: List of Pearltrees JSONL records
            max_depth: Maximum category hierarchy depth

        Returns:
            List of TrainingPair objects (may be shorter than input if some fail)
        """
        pairs = []
        for title in article_titles:
            try:
                pair = self.generate_training_pair(title, pearltrees_data, max_depth)
                if pair:
                    pairs.append(pair)
            except Exception as e:
                # Log but continue
                print(f"Warning: Failed to generate pair for '{title}': {e}")
        return pairs


# =============================================================================
# Convenience Functions
# =============================================================================

_default_bridge: Optional[WikipediaCategoryBridge] = None


def get_category_bridge(db_path: Path = DEFAULT_DB_PATH) -> WikipediaCategoryBridge:
    """Get or create default category bridge."""
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = WikipediaCategoryBridge(db_path)
    return _default_bridge


def load_pearltrees_folders(jsonl_path: str) -> Set[str]:
    """Load folder names from Pearltrees JSONL."""
    folders = set()

    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('type') == 'Tree':
                title = rec.get('raw_title', rec.get('query', ''))
                if title:
                    folders.add(title)

                # Also extract folder names from target_text hierarchy
                target = rec.get('target_text', '')
                for part in target.split('\n'):
                    part = part.strip().lstrip('- ')
                    if part and not part.startswith('/'):
                        folders.add(part)

    return folders
