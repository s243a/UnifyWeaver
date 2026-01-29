#!/usr/bin/env python3
"""
Federated Pearltrees Inference.

Given a query (bookmark title/URL), returns top-k candidate folders
for filing the bookmark. Uses query-level routing with cluster-shared
Procrustes transforms.

See docs/design/FEDERATED_MODEL_FORMAT.md for the model file format specification.

Usage:
    python3 scripts/infer_pearltrees_federated.py \
        --model models/pearltrees_federated_single.pkl \
        --query "New bookmark about quantum computing" \
        --top-k 5

    # Interactive mode:
    python3 scripts/infer_pearltrees_federated.py \
        --model models/pearltrees_federated_single.pkl \
        --interactive

    # JSON output for piping to other tools:
    python3 scripts/infer_pearltrees_federated.py \
        --model models/pearltrees_federated_single.pkl \
        --query "quantum physics bookmark" \
        --json
"""

# Auto-environment: switch to numpy 2.x compatible Python if needed
# This must be before importing numpy. Set AUTO_ENV_DEBUG=1 for debug output.
import sys
import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, 'src'))
try:
    from unifyweaver.config.model_registry import auto_environment
    if not auto_environment():
        print("Warning: No compatible environment found for numpy 2.x", file=sys.stderr)
except ImportError:
    pass  # Registry not available, continue with current environment

import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A candidate folder for filing a bookmark."""
    rank: int
    score: float
    tree_id: str
    title: str
    path: str
    cluster_id: str
    dataset_index: int = -1
    account: str = ""


class FederatedInferenceEngine:
    """
    Inference engine for federated Pearltrees model.

    Uses query-level routing: finds most similar training queries,
    then uses their cluster's W matrix to project the new query.
    """

    def __init__(self, model_path: Path, embedder_name: str = "nomic-ai/nomic-embed-text-v1.5",
                 data_path: Optional[Path] = None):
        self.model_path = Path(model_path)
        self.embedder_name = embedder_name

        # Load model metadata
        logger.info(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            self.meta = pickle.load(f)

        self.cluster_ids = self.meta["cluster_ids"]
        self.cluster_centroids = self.meta["cluster_centroids"]
        self.global_target_ids = self.meta.get("global_target_ids", [])
        self.global_target_titles = self.meta.get("global_target_titles", [])
        self.global_target_accounts = self.meta.get("global_target_accounts", [])
        self.temperature = self.meta.get("temperature", 0.1)

        # Determine cluster directory
        if "cluster_dir" in self.meta:
            self.cluster_dir = Path(self.meta["cluster_dir"])
        else:
            self.cluster_dir = self.model_path.with_suffix('')

        logger.info(f"Model has {len(self.cluster_ids)} clusters")

        # Load routing data
        self._load_routing_data()

        # Load cluster W matrices
        self._load_clusters()

        # Load account lookup from JSONL if not in model and data_path provided
        if not self.global_target_accounts and data_path:
            self._load_account_lookup(data_path)

        # Load embedder (lazy)
        self._embedder = None

    def _load_account_lookup(self, data_path: Path):
        """Load account information from JSONL file, keyed by tree_id."""
        data_path = Path(data_path)
        if not data_path.exists():
            logger.warning(f"Data path {data_path} not found, account filtering disabled")
            return

        logger.info(f"Loading account lookup from {data_path}...")

        # Build tree_id -> account mapping from JSONL
        tree_id_to_account = {}
        with open(data_path) as f:
            for line in f:
                record = json.loads(line)
                tree_id = record.get("tree_id", "")
                account = record.get("account", "")
                if tree_id and account:
                    tree_id_to_account[tree_id] = account

        # Map to global_target_ids order
        self.global_target_accounts = [
            tree_id_to_account.get(tid, "") for tid in self.global_target_ids
        ]

        # Count accounts
        from collections import Counter
        account_counts = Counter(self.global_target_accounts)
        logger.info(f"Loaded accounts: {dict(account_counts)}")
    
    def _load_routing_data(self):
        """Load query embeddings and index-to-cluster mapping for routing."""
        routing_path = self.cluster_dir / "routing_data.npz"
        
        if routing_path.exists():
            logger.info(f"Loading routing data from {routing_path}...")
            data = np.load(routing_path)
            self.query_embeddings = data["query_embeddings"]
            self.target_embeddings = data["target_embeddings"]
            
            # Reconstruct index-to-cluster mapping
            keys = data["idx_to_cluster_keys"]
            values = data["idx_to_cluster_values"]
            self.idx_to_cluster = {int(k): str(v) for k, v in zip(keys, values)}
            
            logger.info(f"Loaded {len(self.query_embeddings)} query embeddings for routing")
        else:
            logger.warning(f"Routing data not found at {routing_path}, using cluster centroids")
            self.query_embeddings = self.cluster_centroids
            self.target_embeddings = None
            self.idx_to_cluster = {i: cid for i, cid in enumerate(self.cluster_ids)}
    
    def _load_clusters(self):
        """Load W matrices from each cluster."""
        self.clusters = {}
        
        for cid in self.cluster_ids:
            cluster_path = self.cluster_dir / f"{cid}.npz"
            if cluster_path.exists():
                data = np.load(cluster_path)
                self.clusters[cid] = {
                    "W": data["W_stack"][0],  # Single W per cluster
                    "target_embeddings": data["target_embeddings"],
                    "indices": data["indices"]
                }
        
        logger.info(f"Loaded {len(self.clusters)} cluster W matrices")
    
    @property
    def embedder(self):
        """Lazy load embedder on first use."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedder: {self.embedder_name}")
            self._embedder = SentenceTransformer(self.embedder_name, trust_remote_code=True)
        return self._embedder
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.embedder.encode([query], show_progress_bar=False)[0].astype(np.float32)
    
    def project_query(self, q_emb: np.ndarray, top_k_routing: int = 10) -> np.ndarray:
        """
        Project query using query-level routing.
        
        1. Find top-k most similar training queries
        2. For each, look up its cluster's W
        3. Compute weighted projection
        """
        # Compute similarities to all training queries
        sims = q_emb @ self.query_embeddings.T
        
        # Softmax weights
        sims_shifted = (sims - np.max(sims)) / self.temperature
        weights = np.exp(sims_shifted)
        weights /= weights.sum()
        
        # Get top-k training queries
        top_indices = np.argsort(weights)[-top_k_routing:]
        
        # Weighted projection using their clusters' W
        proj = np.zeros_like(q_emb)
        for idx in top_indices:
            cid = self.idx_to_cluster.get(int(idx))
            if cid and cid in self.clusters:
                W = self.clusters[cid]["W"]
                proj += weights[idx] * (q_emb @ W)
        
        return proj
    
    def search(self, query: str, top_k: int = 5, top_k_routing: int = 10,
               account_filter: Optional[Union[str, List[str]]] = None) -> List[Candidate]:
        """
        Search for best folders to file a bookmark.

        Args:
            query: Bookmark title, URL, or description
            top_k: Number of candidates to return
            top_k_routing: Number of training queries to use for routing
            account_filter: If set, only return candidates from this account (str)
                           or accounts (list of str)

        Returns:
            List of Candidate objects with scores
        """
        # Embed query
        q_emb = self.embed_query(query)

        # Project using federated model
        q_proj = self.project_query(q_emb, top_k_routing)

        # Normalize
        q_proj_norm = q_proj / (np.linalg.norm(q_proj) + 1e-8)

        # Compare to all targets
        if self.target_embeddings is not None:
            A_norm = self.target_embeddings / (np.linalg.norm(self.target_embeddings, axis=1, keepdims=True) + 1e-8)
            scores = q_proj_norm @ A_norm.T
        else:
            # Fall back to cluster centroids
            scores = q_proj_norm @ self.cluster_centroids.T

        # Get sorted indices (fetch more if filtering by account)
        # Need larger multiplier since filtered accounts may be sparse in top results
        fetch_k = top_k * 50 if account_filter else top_k
        all_indices = np.argsort(scores)[::-1][:fetch_k]

        # Build candidates
        candidates = []
        for idx in all_indices:
            idx = int(idx)

            # Get account for this target
            account = ""
            if idx < len(self.global_target_accounts):
                account = self.global_target_accounts[idx]

            # Skip if account filter is set and doesn't match
            if account_filter:
                if isinstance(account_filter, list):
                    if account not in account_filter:
                        continue
                elif account != account_filter:
                    continue

            # Get cluster ID for this target
            cluster_id = self.idx_to_cluster.get(idx, "unknown")

            # Get title from stored titles
            if idx < len(self.global_target_titles):
                title = self.global_target_titles[idx]
            else:
                title = f"Target {idx}"

            # Get tree ID
            if idx < len(self.global_target_ids):
                tree_id = self.global_target_ids[idx]
            else:
                tree_id = str(idx)

            # For now, path is same as title (could be enhanced)
            path = title

            candidates.append(Candidate(
                rank=len(candidates) + 1,  # Re-rank after filtering
                score=float(scores[idx]),
                tree_id=tree_id,
                title=title,
                path=path,
                cluster_id=cluster_id,
                dataset_index=idx,
                account=account
            ))

            # Stop when we have enough
            if len(candidates) >= top_k:
                break

        return candidates
    
    def search_batch(self, queries: List[str], top_k: int = 5) -> List[List[Candidate]]:
        """Search for multiple queries."""
        return [self.search(q, top_k) for q in queries]


def format_candidates(candidates: List[Candidate], json_output: bool = False) -> str:
    """Format candidates for output."""
    if json_output:
        return json.dumps([{
            "rank": c.rank,
            "score": c.score,
            "tree_id": c.tree_id,
            "title": c.title,
            "path": c.path,
            "cluster_id": c.cluster_id,
            "account": c.account
        } for c in candidates], indent=2)

    lines = []
    for c in candidates:
        account_str = f" @{c.account}" if c.account else ""
        lines.append(f"{c.rank}. [{c.score:.4f}] {c.title}{account_str}")
        lines.append(f"   ID: {c.tree_id} | Cluster: {c.cluster_id}")
    return "\n".join(lines)


def enrich_tree_data_with_rdf(tree_data_map: Dict[str, dict], rdf_path: Path,
                               api_trees_dir: Optional[Path] = None,
                               queue_missing: bool = False,
                               harvest_queue_path: Optional[Path] = None) -> int:
    """Enrich tree_data_map entries with parent relationships from RDF and API.

    For entries with incomplete paths (not starting from account root),
    uses RDF RefPearl/AliasPearl relationships to build complete paths.
    Falls back to API tree data for parent info when RDF doesn't have it.

    Args:
        tree_data_map: Dict mapping tree_id -> entry dict (modified in place)
        rdf_path: Path to Pearltrees RDF export file
        api_trees_dir: Optional path to API trees directory for fallback parent info
        queue_missing: If True, queue trees with missing parent info for harvesting
        harvest_queue_path: Path to harvest queue JSON file (required if queue_missing=True)

    Returns:
        Number of entries enriched
    """
    import re

    if not rdf_path.exists():
        logger.warning(f"RDF file not found: {rdf_path}")
        return 0

    with open(rdf_path, 'r') as f:
        content = f.read()

    # Extract tree titles from pt:Tree elements
    tree_pattern = re.compile(
        r'<pt:Tree rdf:about="[^"]*id(\d+)">\s*'
        r'<dcterms:title><!\[CDATA\[([^\]]+)\]\]></dcterms:title>',
        re.MULTILINE | re.DOTALL
    )
    tree_titles = {}
    for tree_id, title in tree_pattern.findall(content):
        tree_titles[tree_id] = title

    def extract_parent_from_url(parent_url: str) -> Optional[Tuple[str, str]]:
        """Extract parent_id and parent_title from a parentTree URL."""
        parent_id_match = re.search(r'id(\d+)$', parent_url)
        if parent_id_match:
            parent_id = parent_id_match.group(1)
            parent_title = tree_titles.get(parent_id, '')
            return (parent_id, parent_title)
        else:
            # Account root URL
            account_match = re.search(r'pearltrees\.com/([^/#]+)$', parent_url)
            if account_match:
                return (f"account:{account_match.group(1)}", account_match.group(1))
        return None

    def make_pearl_pattern_with_date(pearl_type: str):
        """Pattern that captures title, child_id, parent_url, and inTreeSinceDate."""
        return re.compile(
            rf'<pt:{pearl_type}[^>]*>\s*'
            r'<dcterms:title><!\[CDATA\[([^\]]+)\]\]></dcterms:title>\s*'
            r'<rdfs:seeAlso rdf:resource="[^"]*id(\d+)"[^/]*/>\s*'
            r'<pt:parentTree rdf:resource="([^"]*)"[^/]*/>\s*'
            r'<pt:inTreeSinceDate>([^<]*)</pt:inTreeSinceDate>',
            re.MULTILINE | re.DOTALL
        )

    # Build parent map: child_id -> (parent_id, parent_title)
    # Priority: 1) API data (explicit parentTree), 2) RDF RefPearl, 3) RDF AliasPearl
    # Use oldest inTreeSinceDate for RDF entries to find canonical parent
    parent_map = {}
    parent_dates = {}  # Track dates to prefer oldest
    api_parents = set()  # Track which entries came from API (higher priority)

    # Load API parent info FIRST (most reliable - explicit parentTree field)
    if api_trees_dir and api_trees_dir.exists():
        api_count = 0
        for tree_file in api_trees_dir.glob('*.json'):
            try:
                with open(tree_file) as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue
                resp = data.get('api_response', data.get('response', {}))
                if not isinstance(resp, dict):
                    continue
                info = resp.get('info', {})
                tree_id = str(info.get('id', tree_file.stem))
                parent = info.get('parentTree', {})
                if parent and isinstance(parent, dict) and 'id' in parent:
                    parent_id = str(parent['id'])
                    parent_title = parent.get('title', '')
                    parent_map[tree_id] = (parent_id, parent_title)
                    api_parents.add(tree_id)
                    # Also store title if we don't have it
                    if parent_id not in tree_titles and parent_title:
                        tree_titles[parent_id] = parent_title
                    api_count += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        if api_count:
            logger.info(f"Loaded {api_count} parent relationships from API")

    # Then load RDF RefPearl (canonical hierarchy), then AliasPearl as fallback
    # Only add if not already from API (API is more reliable)
    for pearl_type in ['RefPearl', 'AliasPearl']:
        pattern = make_pearl_pattern_with_date(pearl_type)
        for title, child_id, parent_url, date_str in pattern.findall(content):
            # Skip if we have API data for this tree
            if child_id in api_parents:
                continue
            result = extract_parent_from_url(parent_url)
            if result:
                # Keep entry with oldest date (canonical/original location)
                if child_id not in parent_map or date_str < parent_dates.get(child_id, ''):
                    parent_map[child_id] = result
                    parent_dates[child_id] = date_str

    logger.info(f"Loaded {len(parent_map)} total parent relationships (API + RDF)")

    def get_path_to_root(tree_id: str, visited: set = None) -> List[Tuple[str, str]]:
        """Build path from tree_id to root as list of (id, title) tuples."""
        if visited is None:
            visited = set()
        if tree_id in visited:
            return []  # Cycle detected
        visited.add(tree_id)

        path = []
        current = tree_id
        while current and current in parent_map:
            parent_id, parent_title = parent_map[current]
            if parent_id is None or parent_id in visited:
                break
            visited.add(parent_id)
            path.append((parent_id, parent_title or tree_titles.get(parent_id, '')))
            if parent_id.startswith('account:'):
                break
            current = parent_id
        return list(reversed(path))

    # Enrich entries that need fuller paths
    # But preserve existing good paths from training data
    enriched_count = 0
    for tree_id, entry in tree_data_map.items():
        path_ids = entry.get('path_ids', [])

        # Extract path_ids from target_text first line if not present as field
        if not path_ids:
            target_text = entry.get('target_text', '')
            if target_text:
                first_line = target_text.split('\n')[0]
                if first_line.startswith('/'):
                    path_ids = first_line.strip('/').split('/')

        # Check if existing path is already good (has reasonable depth)
        # Training data paths are typically 5+ levels deep and correct
        existing_depth = len(path_ids) if path_ids else 0
        if existing_depth >= 5:
            # Path is already good, don't overwrite
            # Just add path_ids field if missing
            if 'path_ids' not in entry and path_ids:
                entry['path_ids'] = path_ids
            continue

        # Try to build/extend path from parent_map
        rdf_path_to_root = get_path_to_root(tree_id)
        if rdf_path_to_root:
            new_depth = len(rdf_path_to_root) + 1  # +1 for the tree itself

            # Only use new path if it's better (longer) than existing
            if new_depth > existing_depth:
                # Build new path_ids and path_nodes
                new_path_ids = [pid for pid, _ in rdf_path_to_root] + [tree_id]
                new_path_nodes = [title for _, title in rdf_path_to_root]
                tree_title = entry.get('raw_title', entry.get('query', tree_titles.get(tree_id, '')))
                new_path_nodes.append(tree_title)

                # Update entry
                entry['path_ids'] = new_path_ids
                entry['path_depth'] = len(new_path_ids)

                # Rebuild target_text with new path
                id_line = '/' + '/'.join(str(i).replace('account:', '') for i in new_path_ids)
                path_lines = ['- ' + node for node in new_path_nodes]
                entry['target_text'] = id_line + '\n' + '\n'.join(path_lines)

                # Set account from path
                if new_path_ids[0].startswith('account:'):
                    entry['account'] = new_path_ids[0].replace('account:', '')

                enriched_count += 1

    # Queue trees with missing/incomplete parent info for harvesting
    if queue_missing and harvest_queue_path:
        missing_trees = []
        for tree_id, entry in tree_data_map.items():
            # Check if tree still has incomplete path after enrichment
            path_ids = entry.get('path_ids', [])
            if not path_ids:
                target_text = entry.get('target_text', '')
                if target_text:
                    first_line = target_text.split('\n')[0]
                    if first_line.startswith('/'):
                        path_ids = first_line.strip('/').split('/')

            # If path is still short and tree not in parent_map, it needs fetching
            if len(path_ids) < 5 and tree_id not in parent_map:
                # Check if we already have API data for this tree
                api_file = api_trees_dir / f"{tree_id}.json" if api_trees_dir else None
                if api_file and api_file.exists():
                    continue  # Already have data, just missing parent chain

                missing_trees.append({
                    'tree_id': tree_id,
                    'title': entry.get('raw_title', entry.get('query', 'Unknown')),
                    'account': entry.get('account', 's243a'),
                    'uri': f"https://www.pearltrees.com/{entry.get('account', 's243a')}/id{tree_id}",
                    'queued_by': 'inference',
                    'reason': 'missing_parent_info'
                })

        if missing_trees:
            # Load existing queue or create new
            queue_data = {'count': 0, 'maps': []}
            if harvest_queue_path.exists():
                try:
                    with open(harvest_queue_path) as f:
                        queue_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass

            # Add new trees (avoid duplicates by tree_id)
            existing_ids = {m['tree_id'] for m in queue_data.get('maps', [])}
            new_trees = [t for t in missing_trees if t['tree_id'] not in existing_ids]

            if new_trees:
                queue_data['maps'].extend(new_trees)
                queue_data['count'] = len(queue_data['maps'])

                with open(harvest_queue_path, 'w') as f:
                    json.dump(queue_data, f, indent=2)

                logger.info(f"Queued {len(new_trees)} trees with missing parent info to {harvest_queue_path}")

    return enriched_count


def build_merged_tree(candidates: List[Candidate], tree_data_map: Dict[str, dict]) -> 'TreeNode':
    """Build a merged tree from candidates with full paths.

    Args:
        candidates: List of search result candidates
        tree_data_map: Dict mapping tree_id -> entry dict with target_text paths

    Uses supplementary data to bridge/complete truncated paths when possible.
    ID-based bridging is preferred over title-based for reliability.
    """

    class TreeNode:
        def __init__(self, name):
            self.name = name
            self.children = {}
            self.is_result = False
            self.score = 0.0
            self.rank = 0

    root = TreeNode('ROOT')

    # Build tree_id -> full path info lookup from tree_data_map
    # This allows ID-based bridging of truncated paths
    id_to_path_info = {}
    for entry in tree_data_map.values():
        path_ids = entry.get('path_ids', [])
        if path_ids:
            target_text = entry.get('target_text', '')
            lines = target_text.split('\n')[1:]  # Skip ID line
            path_nodes = [line.lstrip('- ') for line in lines if line.lstrip('- ')]
            # Map each tree_id in the path to the full path info
            for i, tid in enumerate(path_ids):
                existing = id_to_path_info.get(tid)
                # Keep the entry with the longest path to root (more context)
                if not existing or len(path_nodes) > len(existing['path_nodes']):
                    id_to_path_info[tid] = {
                        'path_nodes': path_nodes,
                        'path_ids': path_ids,
                        'account': entry.get('account', ''),
                        'index_in_path': i
                    }

    for c in candidates:
        # Look up entry by tree_id
        d = tree_data_map.get(c.tree_id)

        if d is None:
            # No path data - show as root item with account prefix if available
            if c.account:
                label = f"{c.account} .../  {c.title}"
            else:
                label = c.title
            if label not in root.children:
                root.children[label] = TreeNode(label)
            node = root.children[label]
            node.is_result = True
            node.score = c.score
            node.rank = c.rank
            continue

        path_text = d.get('target_text', '')
        lines = path_text.split('\n')

        # Extract path IDs from first line (format: /id1 or /id1/id2/id3...)
        path_ids = d.get('path_ids', [])
        if not path_ids and lines and lines[0].startswith('/'):
            # Parse path_ids from target_text first line
            id_line = lines[0].lstrip('/')
            if id_line:
                path_ids = id_line.split('/')

        # Extract path nodes from remaining lines
        path_nodes = []
        for line in lines[1:]:
            stripped = line.lstrip('- ')
            if stripped:
                path_nodes.append(stripped)

        account = d.get('account', c.account) or c.account

        # Try to bridge/complete truncated paths using ID-based lookup
        # If the first node's tree_id is found with parent nodes above it, prepend them
        if path_ids and len(path_ids) > 0 and path_nodes and path_nodes[0] != account:
            first_id = path_ids[0]
            bridged = id_to_path_info.get(first_id)
            if bridged:
                bridge_idx = bridged['index_in_path']
                # If bridge_idx > 0, there are parent nodes above this node
                if bridge_idx > 0:
                    bridge_path = bridged['path_nodes']
                    parent_path = bridge_path[:bridge_idx]
                    path_nodes = parent_path + path_nodes
                    if bridged['account']:
                        account = bridged['account']

        # Check if path starts with account root - if not, prefix with "account ..."
        if account and path_nodes and path_nodes[0] != account:
            # Path is truncated - doesn't start from account root
            path_nodes.insert(0, f"{account} ...")

        # Build tree from path
        current = root
        for node_name in path_nodes:
            if node_name not in current.children:
                current.children[node_name] = TreeNode(node_name)
            current = current.children[node_name]

        current.is_result = True
        current.score = c.score
        current.rank = c.rank

    return root


def format_tree(node, depth=0, prefix='') -> str:
    """Format tree as string with box-drawing characters."""
    lines = []
    items = sorted(node.children.items())
    
    for i, (name, child) in enumerate(items):
        is_last = i == len(items) - 1
        connector = '└── ' if is_last else '├── '
        
        if child.is_result:
            result_str = f' ★ #{child.rank} [{child.score:.3f}]'
        else:
            result_str = ''
        
        lines.append(f'{prefix}{connector}{name}{result_str}')
        
        new_prefix = prefix + ('    ' if is_last else '│   ')
        lines.append(format_tree(child, depth + 1, new_prefix))
    
    return '\n'.join(filter(None, lines))


def interactive_mode(engine: 'FederatedInferenceEngine', top_k: int = 5,
                     account_filter: Optional[str] = None):
    """Run in interactive mode."""
    print("\n=== Pearltrees Bookmark Filing Assistant ===")
    print(f"Model: {engine.model_path}")
    print(f"Clusters: {len(engine.cluster_ids)}")
    print(f"Targets: {len(engine.query_embeddings)}")
    if account_filter:
        print(f"Account filter: {account_filter}")
    print("\nEnter bookmark titles/URLs to find best folders.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Query> ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break

            candidates = engine.search(query, top_k, account_filter=account_filter)
            print(f"\nTop {top_k} candidates:")
            print(format_candidates(candidates))
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Inference for federated Pearltrees projection model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # Model selection - can use --model (path) or --infer (name from registry)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=Path, default=None,
                       help="Path to model .pkl file")
    model_group.add_argument("--infer", type=str, metavar="MODEL_NAME", default=None,
                       help="Model name from registry (e.g., pearltrees_federated_nomic)")
    parser.add_argument("--task", type=str, default="bookmark_filing",
                       help="Task for default model lookup (default: bookmark_filing)")
    parser.add_argument("--query", type=str, default=None,
                       help="Query to search for")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of candidates to return")
    parser.add_argument("--top-k-routing", type=int, default=10,
                       help="Number of training queries for routing")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON")
    parser.add_argument("--tree", action="store_true",
                       help="Output results as merged hierarchical tree")
    parser.add_argument("--no-tree", action="store_true", dest="no_tree",
                       help="Disable tree output (overrides config default)")
    parser.add_argument("--data", type=Path, default=None,
                       help="Path to original JSONL data (for account lookup and --tree mode)")
    parser.add_argument("--tree-data", type=str, default=None, dest="tree_data",
                       help="Comma-separated fallback JSONL files for tree display (checked in order)")
    parser.add_argument("--tree-data-only", action="store_true", dest="tree_data_only",
                       help="Exclude primary --data from tree lookup, use only --tree-data sources")
    parser.add_argument("--rdf", type=str, default=None,
                       help="RDF export file to enrich tree paths with parent relationships")
    parser.add_argument("--api-trees", type=str, default=None, dest="api_trees",
                       help="API trees directory for fallback parent info (used with --rdf)")
    parser.add_argument("--embedder", type=str, default=None,
                       help="Embedding model to use (default: from registry or nomic)")
    parser.add_argument("--account", type=str, default=None,
                       help="Filter results to single account (e.g., s243a)")
    parser.add_argument("--accounts", type=str, default=None,
                       help="Filter to multiple accounts, comma-separated (e.g., s243a,s243a_groups)")
    parser.add_argument("--accounts-tree", type=str, default=None, dest="accounts_tree",
                       help="Filter to accounts AND show hierarchical tree (shorthand for --accounts + --tree)")
    parser.add_argument("--queue-missing", action="store_true", dest="queue_missing",
                       help="Queue trees with missing parent info for harvesting (requires --api-trees)")
    parser.add_argument("--harvest-queue", type=str, default=None, dest="harvest_queue",
                       help="Path to harvest queue JSON file (default: {api-trees}/../harvest_queue.json)")

    args = parser.parse_args()

    # Validate --queue-missing requires --api-trees
    if args.queue_missing and not args.api_trees:
        parser.error("--queue-missing requires --api-trees to be specified")

    # Handle --accounts-tree shorthand
    if args.accounts_tree:
        args.accounts = args.accounts_tree
        args.tree = True

    # Handle account filtering (--account or --accounts)
    account_filter = None
    if args.accounts:
        # Comma-separated list of accounts
        account_filter = [a.strip() for a in args.accounts.split(',')]
    elif args.account:
        # Single account (backwards compatible)
        account_filter = args.account

    # Check if --accounts should imply --tree (from config)
    hierarchical_default = False
    if args.accounts and not args.tree and not args.no_tree:
        try:
            from unifyweaver.config.model_registry import ModelRegistry
            registry = ModelRegistry()
            hierarchical_default = registry.is_hierarchical_accounts_default()
            if hierarchical_default:
                args.tree = True
                logger.info("Hierarchical display enabled (config: hierarchical_accounts_default)")
        except ImportError:
            pass

    # --no-tree overrides everything
    if args.no_tree:
        args.tree = False

    # Resolve model path from registry if needed
    model_path = args.model
    embedder = args.embedder

    if args.infer or not model_path:
        # Use registry to resolve model
        try:
            from unifyweaver.config.model_registry import ModelRegistry
            registry = ModelRegistry()

            if args.infer:
                # Lookup by model name
                model_path = registry.get_model_path(args.infer)
                if not model_path:
                    print(f"Error: Model '{args.infer}' not found in registry", file=sys.stderr)
                    sys.exit(1)
                # Get embedder from model metadata
                model_meta = registry.get_model(args.infer)
                if model_meta and model_meta.embedding_model and not embedder:
                    embedder = model_meta.embedding_model
                    if not embedder.startswith(('/', '~')):
                        # Convert short name to HuggingFace path if needed
                        hf_map = {
                            'nomic-embed-text-v1.5': 'nomic-ai/nomic-embed-text-v1.5',
                            'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
                            'bge-small-en-v1.5': 'BAAI/bge-small-en-v1.5',
                        }
                        embedder = hf_map.get(embedder, embedder)
            else:
                # Use default model for task
                task_models = registry.get_for_task(args.task)
                if 'projection' in task_models:
                    model_meta = task_models['projection']
                    model_name = model_meta.name if hasattr(model_meta, 'name') else str(model_meta)
                    model_path = registry.get_model_path(model_name)
                    if not model_path:
                        print(f"Error: Default model '{model_name}' for task '{args.task}' not found", file=sys.stderr)
                        sys.exit(1)
                    logger.info(f"Using default model for {args.task}: {model_name}")
                    # Get embedder from model metadata
                    if model_meta.embedding_model and not embedder:
                        embedder = model_meta.embedding_model
                        hf_map = {
                            'nomic-embed-text-v1.5': 'nomic-ai/nomic-embed-text-v1.5',
                            'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
                            'bge-small-en-v1.5': 'BAAI/bge-small-en-v1.5',
                        }
                        embedder = hf_map.get(embedder, embedder)
                else:
                    print(f"Error: No projection model configured for task '{args.task}'", file=sys.stderr)
                    print("Use --model to specify a model path, or --infer to specify a model name", file=sys.stderr)
                    sys.exit(1)
        except ImportError:
            print("Error: Model registry not available and --model not specified", file=sys.stderr)
            print("Either install unifyweaver or provide --model PATH", file=sys.stderr)
            sys.exit(1)

    # Default embedder if still not set
    if not embedder:
        embedder = "nomic-ai/nomic-embed-text-v1.5"

    # Determine data path - prefer: CLI arg > model metadata > default
    if args.data:
        data_path = args.data
    else:
        # Check if model has stored data_path
        with open(model_path, "rb") as f:
            meta = pickle.load(f)
        stored_path = meta.get("data_path")
        if stored_path and Path(stored_path).exists():
            data_path = Path(stored_path)
            logger.info(f"Using data path from model: {data_path}")
        else:
            data_path = Path("reports/pearltrees_targets_full_multi_account.jsonl")

    # Load engine (pass data_path for account lookup)
    engine = FederatedInferenceEngine(model_path, embedder, data_path=data_path)

    # Load data if tree mode requested
    # Build tree_id -> entry mapping from multiple sources (first match wins)
    tree_data_map = None
    if args.tree:
        tree_data_map = {}
        sources_loaded = 0

        # Collect data sources in priority order
        tree_sources = []
        if not args.tree_data_only and data_path and data_path.exists():
            tree_sources.append(('primary', data_path))
        if args.tree_data:
            for path_str in args.tree_data.split(','):
                path = Path(path_str.strip())
                if path.exists():
                    tree_sources.append(('fallback', path))
                else:
                    logger.warning(f"Tree data file not found: {path}")

        # Load each source, prefer deeper paths for same tree_id
        for source_type, source_path in tree_sources:
            count_before = len(tree_data_map)
            count_updated = 0
            with open(source_path) as f:
                for line in f:
                    entry = json.loads(line)
                    tid = entry.get('tree_id', '')
                    if not tid:
                        continue
                    new_depth = entry.get('path_depth', 1)
                    if tid not in tree_data_map:
                        tree_data_map[tid] = entry
                    else:
                        # Prefer deeper path (API data typically has fuller paths)
                        existing_depth = tree_data_map[tid].get('path_depth', 1)
                        if new_depth > existing_depth:
                            tree_data_map[tid] = entry
                            count_updated += 1
            count_added = len(tree_data_map) - count_before
            if count_updated:
                logger.info(f"Loaded {count_added} new + {count_updated} updated entries from {source_type}: {source_path}")
            else:
                logger.info(f"Loaded {count_added} new entries from {source_type}: {source_path}")
            sources_loaded += 1

        if not tree_data_map:
            logger.warning("No tree data sources available, tree mode disabled")
            args.tree = False
        else:
            logger.info(f"Total tree entries: {len(tree_data_map)} from {sources_loaded} source(s)")

        # Enrich with RDF parent relationships if --rdf specified
        if args.rdf and tree_data_map:
            rdf_path = Path(args.rdf)
            api_trees_dir = Path(args.api_trees) if args.api_trees else None

            # Determine harvest queue path
            harvest_queue_path = None
            if args.queue_missing:
                if args.harvest_queue:
                    harvest_queue_path = Path(args.harvest_queue)
                elif api_trees_dir:
                    # Default: sibling of api-trees directory
                    harvest_queue_path = api_trees_dir.parent / "harvest_queue.json"

            enriched = enrich_tree_data_with_rdf(
                tree_data_map, rdf_path, api_trees_dir,
                queue_missing=args.queue_missing,
                harvest_queue_path=harvest_queue_path
            )
            if enriched:
                logger.info(f"Enriched {enriched} entries with RDF/API parent relationships")

    if args.interactive:
        interactive_mode(engine, args.top_k, account_filter=account_filter)
    elif args.query:
        candidates = engine.search(args.query, args.top_k, args.top_k_routing,
                                   account_filter=account_filter)
        
        if args.tree and tree_data_map:
            tree = build_merged_tree(candidates, tree_data_map)
            print(f"Query: {args.query}")
            print(f"\nMerged tree of top {args.top_k} results:")
            print("=" * 60)
            print(format_tree(tree))
        else:
            print(format_candidates(candidates, args.json))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
