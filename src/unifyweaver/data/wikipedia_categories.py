"""
Wikipedia Category Bridge for Organizational Metric Training

Provides integration between Wikipedia's category hierarchy and Pearltrees
organizational structure. Used to generate training pairs that bridge
Wikipedia articles to the appropriate Pearltrees folders.

Uses effective distance with dimension parameter n (default=5) to combine
multiple paths. This accounts for Wikipedia's graph structure where many
paths exist between nodes.

    d_eff = (Σ dᵢ^(-n))^(-1/n)

With n=5:
- Shortest paths dominate but don't completely override longer paths
- Balances Wikipedia's dimension (~6) and Pearltrees' dimension (~4)

Usage:
    from unifyweaver.data import get_category_bridge

    bridge = get_category_bridge()

    # Find all connections with effective distance
    matches = bridge.find_all_folder_connections(
        page_id=12345,
        pearltrees_folders={"Physics", "Science", ...},
        n=5
    )

    # Generate training pairs with weighted distances
    pairs = bridge.generate_weighted_training_pairs(
        page_ids=[12345, 67890],
        pearltrees_data=records,
        n=5
    )
"""

import json
import sqlite3
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DB_PATH = Path("data/wikipedia_categories.db")
DEFAULT_DIMENSION_N = 5  # Effective dimension parameter


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
class MultiPathMatch:
    """Result of finding all paths to a folder connection."""
    folder_name: str
    folder_path: str              # Full path in Pearltrees hierarchy
    path_distances: List[int]     # All path lengths found
    effective_distance: float     # Combined distance with dimension n
    via_categories: List[str]     # Categories that led to matches
    n: float                      # Dimension parameter used


@dataclass
class TrainingPair:
    """A training pair generated from Wikipedia category bridge."""
    query: str        # Wikipedia article title or page_id
    target: str       # Pearltrees target text
    distance: float   # Estimated organizational distance
    source: str       # "wikipedia_category_bridge"
    metadata: Dict = field(default_factory=dict)  # Extra info


def compute_effective_distance(path_distances: List[int], n: float = DEFAULT_DIMENSION_N) -> float:
    """
    Compute effective distance from multiple paths using dimension parameter n.

    Formula: d_eff = (Σ dᵢ^(-n))^(-1/n)

    With high n, shortest path dominates.
    With low n, all paths contribute more equally.

    Args:
        path_distances: List of path lengths (hops)
        n: Dimension parameter (default 5)

    Returns:
        Effective combined distance
    """
    if not path_distances:
        return float('inf')

    # Filter out zero distances (direct matches count as 1)
    distances = [max(d, 1) for d in path_distances]

    # Compute: (Σ d^(-n))^(-1/n)
    weight_sum = sum(d ** (-n) for d in distances)

    if weight_sum <= 0:
        return float('inf')

    return weight_sum ** (-1/n)


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

    def walk_hierarchy_all_paths(
        self,
        categories: List[str],
        max_depth: int = 10
    ) -> Dict[str, List[Tuple[int, str]]]:
        """
        Walk up the category hierarchy, collecting ALL paths (not just shortest).

        Returns: {category: [(depth1, via1), (depth2, via2), ...]}
        """
        all_paths = defaultdict(list)
        visited_at_depth = {}  # track min depth visited to avoid cycles
        queue = [(cat, 0, cat) for cat in categories]

        while queue:
            category, depth, via = queue.pop(0)

            if depth > max_depth:
                continue

            # Allow revisiting at same or greater depth from different paths
            prev_depth = visited_at_depth.get(category, float('inf'))
            if depth > prev_depth + 2:  # Allow some slack for alternative paths
                continue

            all_paths[category].append((depth, via))
            visited_at_depth[category] = min(prev_depth, depth)

            # Get parent categories
            parents = self.get_parent_categories(category)
            for parent in parents:
                queue.append((parent, depth + 1, via))

        return dict(all_paths)

    def get_categories_for_page(self, page_id: int) -> List[str]:
        """Get category names for a page by ID (using cl_to field)."""
        cursor = self.conn.execute(
            "SELECT cl_to FROM categorylinks WHERE cl_from = ? AND cl_to != ''",
            (page_id,)
        )
        return [row[0] for row in cursor]

    def find_all_folder_connections(
        self,
        page_id: int,
        pearltrees_folders: Set[str],
        max_depth: int = 10,
        n: float = DEFAULT_DIMENSION_N
    ) -> List[MultiPathMatch]:
        """
        Find ALL matching Pearltrees folders with effective distances.

        Uses all paths to compute effective distance with dimension n.

        Args:
            page_id: Wikipedia page ID
            pearltrees_folders: Set of Pearltrees folder names to match
            max_depth: Maximum category hierarchy depth to search
            n: Dimension parameter for effective distance

        Returns:
            List of MultiPathMatch objects, sorted by effective distance
        """
        categories = self.get_categories_for_page(page_id)
        if not categories:
            return []

        # Get all paths through hierarchy
        all_paths = self.walk_hierarchy_all_paths(categories, max_depth)

        # Build normalized lookup for folder matching
        folder_lookup = {}
        for folder in pearltrees_folders:
            folder_lookup[folder.lower()] = folder

        # Find all matches
        matches_by_folder = defaultdict(lambda: {'distances': [], 'via': []})

        for category, paths in all_paths.items():
            normalized = category.replace('_', ' ').lower()

            matched_folder = None
            # Direct match
            if normalized in folder_lookup:
                matched_folder = folder_lookup[normalized]
            else:
                # Partial match (category ends with folder name)
                for folder_lower, folder in folder_lookup.items():
                    if len(folder_lower) > 3 and normalized.endswith(folder_lower):
                        matched_folder = folder
                        break

            if matched_folder:
                for depth, via in paths:
                    matches_by_folder[matched_folder]['distances'].append(depth)
                    if via not in matches_by_folder[matched_folder]['via']:
                        matches_by_folder[matched_folder]['via'].append(via)

        # Convert to MultiPathMatch objects
        results = []
        for folder_name, data in matches_by_folder.items():
            distances = data['distances']
            eff_dist = compute_effective_distance(distances, n)

            results.append(MultiPathMatch(
                folder_name=folder_name,
                folder_path="",  # Filled in later
                path_distances=distances,
                effective_distance=eff_dist,
                via_categories=data['via'][:5],  # Limit for readability
                n=n
            ))

        # Sort by effective distance
        results.sort(key=lambda m: m.effective_distance)
        return results

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

    def generate_weighted_training_pairs(
        self,
        page_ids: List[int],
        pearltrees_data: List[Dict],
        max_depth: int = 10,
        n: float = DEFAULT_DIMENSION_N,
        max_matches_per_page: int = 3
    ) -> List[TrainingPair]:
        """
        Generate training pairs using weighted multi-path effective distances.

        Args:
            page_ids: List of Wikipedia page IDs
            pearltrees_data: List of Pearltrees JSONL records
            max_depth: Maximum category hierarchy depth
            n: Dimension parameter for effective distance (default 5)
            max_matches_per_page: Max number of folder matches per page

        Returns:
            List of TrainingPair objects with effective distances
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
                    if part and not part.startswith('/') and len(part) > 3:
                        if part not in folders:
                            folders[part] = rec

        folder_names = set(folders.keys())

        pairs = []
        for page_id in page_ids:
            try:
                matches = self.find_all_folder_connections(
                    page_id, folder_names, max_depth, n
                )

                for match in matches[:max_matches_per_page]:
                    folder_record = folders.get(match.folder_name, {})
                    target_text = folder_record.get('target_text', match.folder_name)

                    pairs.append(TrainingPair(
                        query=str(page_id),
                        target=target_text,
                        distance=match.effective_distance,
                        source="wikipedia_category_bridge",
                        metadata={
                            'page_id': page_id,
                            'matched_folder': match.folder_name,
                            'path_count': len(match.path_distances),
                            'min_hops': min(match.path_distances),
                            'max_hops': max(match.path_distances),
                            'via_categories': match.via_categories,
                            'n': n,
                        }
                    ))
            except Exception as e:
                print(f"Warning: Failed to generate pairs for page {page_id}: {e}")

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
