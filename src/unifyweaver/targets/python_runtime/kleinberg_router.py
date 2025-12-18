# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Kleinberg small-world routing for distributed KG topology.

"""
Kleinberg small-world routing for distributed KG topology (Phase 3).

Implements greedy forwarding based on semantic distance (embedding similarity)
with HTL limits and path folding for shortcut creation.

Key concepts:
- Greedy forwarding: Route to node with closest interface centroid
- HTL (Hops-To-Live): Limit query propagation depth
- Path folding: Create shortcuts based on successful query paths
- Parallel paths: Query multiple nodes concurrently (optional)

See: docs/proposals/ROADMAP_KG_TOPOLOGY.md (Phase 3)
     docs/proposals/SMALL_WORLD_ROUTING.md
"""

import base64
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
import urllib.request
import urllib.error
import json

import numpy as np

from .discovery_clients import DiscoveryClient, ServiceInstance


@dataclass
class KGNode:
    """Represents a discovered KG node in the network."""
    node_id: str
    endpoint: str
    centroid: np.ndarray
    topics: List[str]
    embedding_model: str
    similarity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'endpoint': self.endpoint,
            'centroid': self.centroid.tolist() if self.centroid is not None else None,
            'topics': self.topics,
            'embedding_model': self.embedding_model,
            'similarity': self.similarity
        }


@dataclass
class RoutingEnvelope:
    """
    Routing information for inter-node queries.

    Tracks the query path through the network and enforces HTL limits.
    """
    origin_node: str
    htl: int
    visited: Set[str] = field(default_factory=set)
    path_folding_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'origin_node': self.origin_node,
            'htl': self.htl,
            'visited': list(self.visited),
            'path_folding_enabled': self.path_folding_enabled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoutingEnvelope':
        """Create from dictionary."""
        return cls(
            origin_node=data.get('origin_node', 'unknown'),
            htl=data.get('htl', 0),
            visited=set(data.get('visited', [])),
            path_folding_enabled=data.get('path_folding_enabled', True)
        )


@dataclass
class QueryResult:
    """Result from a distributed query."""
    answer_id: int
    score: float
    text: str
    source_node: str
    interface_similarity: float = 0.0
    record_id: str = ''
    source_file: str = ''
    hops: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'answer_id': self.answer_id,
            'score': self.score,
            'text': self.text,
            'source_node': self.source_node,
            'interface_similarity': self.interface_similarity,
            'record_id': self.record_id,
            'source_file': self.source_file,
            'hops': self.hops
        }


class KleinbergRouter:
    """
    Kleinberg small-world routing for distributed KG topology.

    Implements greedy forwarding to semantically closest interface centroids
    with HTL limits and optional path folding.

    Attributes:
        local_node_id: ID of the local node
        discovery_client: Service discovery client
        alpha: Link distribution exponent (default 2.0)
        max_hops: Maximum routing hops (HTL, default 10)
        parallel_paths: Number of parallel query paths (default 1)
        similarity_threshold: Minimum similarity for forwarding (default 0.5)
        path_folding_enabled: Whether to create shortcuts (default True)
    """

    def __init__(
        self,
        local_node_id: str,
        discovery_client: DiscoveryClient,
        alpha: float = 2.0,
        max_hops: int = 10,
        parallel_paths: int = 1,
        similarity_threshold: float = 0.5,
        path_folding_enabled: bool = True,
        request_timeout: float = 30.0
    ):
        """
        Initialize Kleinberg router.

        Args:
            local_node_id: Unique ID for this node
            discovery_client: Discovery client for finding nodes
            alpha: Kleinberg exponent (higher = more local routing)
            max_hops: Maximum hops before query termination
            parallel_paths: Number of parallel query paths
            similarity_threshold: Minimum similarity to forward
            path_folding_enabled: Create shortcuts from successful paths
            request_timeout: HTTP request timeout in seconds
        """
        self.local_node_id = local_node_id
        self.discovery_client = discovery_client
        self.alpha = alpha
        self.max_hops = max_hops
        self.parallel_paths = parallel_paths
        self.similarity_threshold = similarity_threshold
        self.path_folding_enabled = path_folding_enabled
        self.request_timeout = request_timeout

        # Cache of discovered nodes
        self._node_cache: Dict[str, KGNode] = {}
        self._cache_timestamp: float = 0.0
        self._cache_ttl: float = 60.0  # Refresh cache every 60 seconds

        # Shortcut links from path folding
        self._shortcuts: Dict[str, str] = {}  # query_hash -> node_id

    def discover_nodes(
        self,
        tags: List[str] = None,
        force_refresh: bool = False
    ) -> List[KGNode]:
        """
        Discover KG nodes from service registry.

        Args:
            tags: Filter tags (default: ['kg_node'])
            force_refresh: Force cache refresh

        Returns:
            List of discovered KG nodes with centroids
        """
        # Check cache
        current_time = time.time()
        if not force_refresh and (current_time - self._cache_timestamp) < self._cache_ttl:
            return list(self._node_cache.values())

        tags = tags or ['kg_node']
        instances = self.discovery_client.discover('kg_topology', tags)

        nodes = []
        for instance in instances:
            metadata = instance.metadata or {}

            # Decode centroid from base64
            centroid_b64 = metadata.get('semantic_centroid')
            if not centroid_b64:
                continue

            try:
                if isinstance(centroid_b64, str):
                    centroid_bytes = base64.b64decode(centroid_b64)
                    centroid = np.frombuffer(centroid_bytes, dtype=np.float32).copy()
                elif isinstance(centroid_b64, list):
                    centroid = np.array(centroid_b64, dtype=np.float32)
                else:
                    continue
            except Exception:
                continue

            # Extract topics
            topics = metadata.get('interface_topics', [])
            if isinstance(topics, str):
                try:
                    topics = json.loads(topics)
                except json.JSONDecodeError:
                    topics = [topics]

            node = KGNode(
                node_id=instance.service_id,
                endpoint=f"http://{instance.host}:{instance.port}",
                centroid=centroid,
                topics=topics,
                embedding_model=metadata.get('embedding_model', 'unknown')
            )
            nodes.append(node)
            self._node_cache[node.node_id] = node

        self._cache_timestamp = current_time
        return nodes

    def route_query(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        envelope: RoutingEnvelope = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Route a query through the small-world network.

        Uses greedy forwarding to the closest interface centroid,
        with backtracking on failure.

        Args:
            query_embedding: Query embedding vector
            query_text: Original query text
            envelope: Routing envelope (created if None)
            top_k: Number of results to return

        Returns:
            List of query results from network
        """
        # Initialize envelope
        if envelope is None:
            envelope = RoutingEnvelope(
                origin_node=self.local_node_id,
                htl=self.max_hops,
                path_folding_enabled=self.path_folding_enabled
            )

        # Check HTL
        if envelope.htl <= 0:
            return []

        envelope.visited.add(self.local_node_id)

        # Check for shortcut
        shortcut_node = self.check_shortcut(query_text)
        if shortcut_node and shortcut_node not in envelope.visited:
            # Try shortcut first
            if shortcut_node in self._node_cache:
                node = self._node_cache[shortcut_node]
                shortcut_results = self._forward_to_node(
                    node, query_embedding, query_text, envelope, top_k
                )
                if shortcut_results:
                    return shortcut_results

        # Get available nodes (excluding visited)
        nodes = self.discover_nodes()
        candidates = [n for n in nodes if n.node_id not in envelope.visited]

        if not candidates:
            return []

        # Compute similarities to all candidate centroids
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

        for node in candidates:
            centroid_norm = node.centroid / (np.linalg.norm(node.centroid) + 1e-9)
            node.similarity = float(np.dot(query_norm, centroid_norm))

        # Sort by similarity (greedy)
        candidates.sort(key=lambda n: n.similarity, reverse=True)

        # Select paths
        selected = candidates[:self.parallel_paths]

        results = []
        if self.parallel_paths > 1:
            # Parallel query execution
            with ThreadPoolExecutor(max_workers=self.parallel_paths) as executor:
                futures = {
                    executor.submit(
                        self._forward_to_node,
                        node, query_embedding, query_text, envelope, top_k
                    ): node for node in selected
                }
                for future in as_completed(futures):
                    try:
                        node_results = future.result()
                        results.extend(node_results)
                    except Exception:
                        pass  # Node failed, try others
        else:
            # Sequential greedy with backtracking
            for node in selected:
                if node.similarity >= self.similarity_threshold:
                    node_results = self._forward_to_node(
                        node, query_embedding, query_text, envelope, top_k
                    )
                    if node_results:
                        results.extend(node_results)
                        break  # Greedy: stop on first success

        # Path folding: create shortcut if successful
        if results and envelope.path_folding_enabled:
            source_node = results[0].get('source_node', selected[0].node_id if selected else None)
            if source_node:
                self.create_shortcut(query_text, source_node)

        return results

    def _forward_to_node(
        self,
        node: KGNode,
        query_embedding: np.ndarray,
        query_text: str,
        envelope: RoutingEnvelope,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Forward query to a remote node via HTTP.

        Args:
            node: Target node
            query_embedding: Query embedding vector
            query_text: Original query text
            envelope: Current routing envelope
            top_k: Number of results

        Returns:
            List of results from remote node
        """
        # Prepare routing envelope for next hop
        next_envelope = RoutingEnvelope(
            origin_node=envelope.origin_node,
            htl=envelope.htl - 1,
            visited=envelope.visited.copy(),
            path_folding_enabled=envelope.path_folding_enabled
        )

        request_body = {
            '__type': 'kg_query',
            '__id': hashlib.sha256(f'{query_text}{time.time()}'.encode()).hexdigest()[:16],
            '__routing': next_envelope.to_dict(),
            '__embedding': {
                'model': node.embedding_model,
                'vector': query_embedding.tolist()
            },
            'payload': {
                'query_text': query_text,
                'top_k': top_k
            }
        }

        try:
            req = urllib.request.Request(
                f"{node.endpoint}/kg/query",
                data=json.dumps(request_body).encode(),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            response = urllib.request.urlopen(req, timeout=self.request_timeout)

            if response.status == 200:
                result = json.loads(response.read())
                results = result.get('results', [])

                # Add source node info and increment hop count
                for r in results:
                    if 'source_node' not in r:
                        r['source_node'] = node.node_id
                    r['hops'] = r.get('hops', 0) + 1

                return results

        except urllib.error.URLError:
            pass  # Node unreachable
        except json.JSONDecodeError:
            pass  # Invalid response
        except Exception:
            pass  # Other errors

        return []

    def check_shortcut(self, query_text: str) -> Optional[str]:
        """
        Check if a shortcut exists for this query.

        Args:
            query_text: Query to check

        Returns:
            Node ID if shortcut exists, None otherwise
        """
        query_hash = self._hash_query(query_text)
        return self._shortcuts.get(query_hash)

    def create_shortcut(self, query_text: str, target_node_id: str):
        """
        Create a path-folded shortcut.

        Args:
            query_text: Query that was successful
            target_node_id: Node that provided good results
        """
        query_hash = self._hash_query(query_text)
        self._shortcuts[query_hash] = target_node_id

    def remove_shortcut(self, query_text: str):
        """Remove a shortcut."""
        query_hash = self._hash_query(query_text)
        self._shortcuts.pop(query_hash, None)

    def clear_shortcuts(self):
        """Clear all shortcuts."""
        self._shortcuts.clear()

    def get_shortcuts(self) -> Dict[str, str]:
        """Get all shortcuts (for persistence)."""
        return dict(self._shortcuts)

    def load_shortcuts(self, shortcuts: Dict[str, str]):
        """Load shortcuts from persistence."""
        self._shortcuts.update(shortcuts)

    def _hash_query(self, query_text: str) -> str:
        """Hash query for shortcut lookup."""
        return hashlib.sha256(query_text.encode()).hexdigest()[:16]

    def compute_routing_probability(
        self,
        nodes: List[KGNode],
        query_embedding: np.ndarray,
        temperature: float = 1.0
    ) -> List[Tuple[KGNode, float]]:
        """
        Compute routing probabilities using softmax over similarities.

        This is similar to the local map_query_to_interface() but for
        distributed routing decisions.

        Args:
            nodes: List of candidate nodes
            query_embedding: Query embedding vector
            temperature: Softmax temperature (lower = sharper)

        Returns:
            List of (node, probability) tuples, sorted by probability
        """
        if not nodes:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

        similarities = []
        for node in nodes:
            centroid_norm = node.centroid / (np.linalg.norm(node.centroid) + 1e-9)
            sim = float(np.dot(query_norm, centroid_norm))
            node.similarity = sim
            similarities.append(sim)

        # Softmax
        similarities = np.array(similarities)
        exp_sims = np.exp(similarities / temperature)
        probs = exp_sims / exp_sims.sum()

        # Create sorted list
        result = [(nodes[i], float(probs[i])) for i in range(len(nodes))]
        result.sort(key=lambda x: x[1], reverse=True)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            'local_node_id': self.local_node_id,
            'cached_nodes': len(self._node_cache),
            'shortcuts': len(self._shortcuts),
            'config': {
                'alpha': self.alpha,
                'max_hops': self.max_hops,
                'parallel_paths': self.parallel_paths,
                'similarity_threshold': self.similarity_threshold,
                'path_folding_enabled': self.path_folding_enabled
            }
        }
