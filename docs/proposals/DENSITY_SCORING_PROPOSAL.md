# Density-Based Confidence Scoring for Federated KG Queries

## Status: Phase 4d-i Complete (Basic Density Scoring)

## Overview

This proposal extends Phase 4 Federation with **density-based confidence scoring** - using the semantic clustering of results to measure consensus strength beyond exact deduplication.

**Key insight:** Tight clusters in embedding space indicate stronger agreement than scattered results, even when answers aren't exact duplicates. A smooth density estimate over result embeddings provides a principled confidence signal.

## Motivation

Phase 4b tracks diversity via `corpus_id` for exact duplicates:
- Same answer from different corpora → boost (independent confirmation)
- Same answer from same corpus → no boost (echo)

But this misses **semantic consensus**:
```
Query: "How do I parse CSV in Python?"

Node A: "Use csv.reader() to read CSV files"
Node B: "csv.reader() handles CSV parsing"
Node C: "For CSV, import csv and use reader()"
```

These are semantically equivalent but hash differently. A density-based approach would detect the tight cluster and boost confidence.

## Mathematical Foundation

### Kernel Density Estimation

Given result embeddings {e₁, e₂, ..., eₙ} in ℝᵈ, estimate density at point x:

```
p̂(x) = (1/n) Σᵢ K_h(x - eᵢ)
```

Where K_h is a kernel with bandwidth h. For embeddings, use a **Gaussian kernel**:

```
K_h(u) = (1/(2πh²)^(d/2)) exp(-||u||² / 2h²)
```

### Smoothness Assumption

We assume the "true" answer distribution is **smooth in semantic space**:

1. **Lipschitz continuity**: Small perturbations in embedding → small changes in density
2. **Unimodal clusters**: True consensus forms connected high-density regions
3. **Noise is diffuse**: Irrelevant/wrong answers scatter uniformly

This is formalized as a prior on the density function:

```
p(f) ∝ exp(-λ ∫ ||∇²f(x)||² dx)
```

Penalizing high curvature enforces smoothness. The bandwidth h controls the smoothness-fidelity tradeoff.

### Density Score for a Result

For result rᵢ with embedding eᵢ, the **density score** is:

```
density_score(rᵢ) = p̂(eᵢ) = (1/n) Σⱼ K_h(eᵢ - eⱼ)
```

This measures "how many other results are semantically nearby?"

### Cluster Confidence

For a cluster of results C, the **cluster confidence** is:

```
confidence(C) = (1/|C|) Σᵢ∈C density_score(rᵢ) × diversity_score(rᵢ)
```

Combining density (semantic clustering) with diversity (source independence).

## Bandwidth Selection

The bandwidth h is critical:
- Too small → overfitting, every point is its own cluster
- Too large → everything merges into one blob

### Silverman's Rule (baseline)

```
h = 0.9 × min(σ, IQR/1.34) × n^(-1/5)
```

Where σ is standard deviation of embedding norms, IQR is interquartile range.

### Adaptive Bandwidth

Better: vary bandwidth based on local density (balloon estimator):

```
h(x) = h₀ × (p̂_pilot(x))^(-α)
```

Where p̂_pilot is a pilot density estimate and α ∈ [0, 0.5] controls adaptation.

### Cross-Validation

For production: use leave-one-out cross-validation to select h:

```
h* = argmax_h Σᵢ log p̂_{-i,h}(eᵢ)
```

## Integration with Federation

### Extended Result Provenance

```python
@dataclass
class ResultProvenance:
    node_id: str
    exp_score: float
    corpus_id: Optional[str] = None
    data_sources: List[str] = field(default_factory=list)
    embedding_model: Optional[str] = None
    # Phase 4d additions
    embedding: Optional[np.ndarray] = None  # For density computation
    density_score: float = 0.0
    cluster_id: Optional[int] = None
```

### Extended Aggregated Result

```python
@dataclass
class AggregatedResult:
    answer_text: str
    answer_hash: str
    combined_score: float
    source_nodes: List[str]
    provenance: List[ResultProvenance]
    # Phase 4d additions
    density_score: float = 0.0
    cluster_confidence: float = 0.0
    semantic_centroid: Optional[np.ndarray] = None
```

### Density-Weighted Aggregation

New aggregation strategy that combines exp_score with density:

```python
class DensityWeightedAggregator(Aggregator):
    """Weight scores by semantic density."""

    def __init__(self, density_weight: float = 0.3):
        self.density_weight = density_weight

    def merge_with_density(
        self,
        existing_score: float,
        new_score: float,
        existing_density: float,
        new_density: float
    ) -> float:
        # Weighted combination
        base_merge = existing_score + new_score  # SUM for consensus
        density_bonus = (existing_density + new_density) / 2

        return base_merge * (1 + self.density_weight * density_bonus)
```

### Federated Density Estimation

Challenge: embeddings live on different nodes, and clustering at a single coordinator doesn't scale.

**Original options (insufficient):**
1. ~~Centralized~~: Collect all embeddings at coordinator - bottleneck
2. ~~Sketched~~: Random projections - loses cluster structure
3. ~~Local + Global~~: Local density only - misses cross-node consensus

### Distributed Cluster Aggregators

**Better approach:** Spin up cluster aggregator services dynamically based on incoming results.

```
                    ┌─────────────────┐
                    │   Coordinator   │
                    │  (query origin) │
                    └────────┬────────┘
                             │ routes results by similarity
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ Cluster Agg A │ │ Cluster Agg B │ │ Cluster Agg C │
    │ (text files)  │ │ (JSON parsing)│ │ (CSV handling)│
    │               │ │               │ │               │
    │ results: 5    │ │ results: 2    │ │ results: 3    │
    │ density: 0.92 │ │ density: 0.45 │ │ density: 0.78 │
    └───────────────┘ └───────────────┘ └───────────────┘
```

**How it works:**

1. **First result arrives** → Creates initial cluster aggregator with that result's centroid
2. **Subsequent results** → Route to nearest existing aggregator (if similarity > threshold) OR spawn new aggregator
3. **Aggregators compute** → Density within their cluster, report back to coordinator
4. **Coordinator merges** → Final ranking across clusters

### Cluster Aggregator Service

```python
@dataclass
class ClusterAggregator:
    """Dynamic service for aggregating a semantic cluster."""
    cluster_id: str
    centroid: np.ndarray
    results: List[AggregatedResult] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)

    def should_accept(self, embedding: np.ndarray, threshold: float = 0.7) -> bool:
        """Check if result belongs to this cluster."""
        similarity = cosine_similarity(self.centroid, embedding)
        return similarity >= threshold

    def add_result(self, result: AggregatedResult):
        """Add result and update centroid."""
        self.results.append(result)
        # Running average centroid
        embeddings = [r.semantic_centroid for r in self.results]
        self.centroid = np.mean(embeddings, axis=0)

    def compute_density(self) -> Dict[str, float]:
        """Compute density scores for all results in cluster."""
        if len(self.results) < 2:
            return {r.answer_hash: 1.0 for r in self.results}

        embeddings = np.array([r.semantic_centroid for r in self.results])
        # Intra-cluster KDE
        ...
        return density_scores

    def get_cluster_stats(self) -> Dict:
        return {
            'cluster_id': self.cluster_id,
            'size': len(self.results),
            'centroid': self.centroid.tolist(),
            'density_scores': self.compute_density(),
            'avg_density': np.mean(list(self.compute_density().values()))
        }
```

### Dynamic Aggregator Spawning

```python
class ClusterAggregatorPool:
    """Manages dynamic pool of cluster aggregators."""

    def __init__(self, similarity_threshold: float = 0.7):
        self.aggregators: Dict[str, ClusterAggregator] = {}
        self.threshold = similarity_threshold

    def route_result(self, result: AggregatedResult) -> str:
        """Route result to existing aggregator or spawn new one."""
        embedding = result.semantic_centroid

        # Find best matching aggregator
        best_match = None
        best_similarity = 0.0

        for agg_id, agg in self.aggregators.items():
            similarity = cosine_similarity(embedding, agg.centroid)
            if similarity > best_similarity and similarity >= self.threshold:
                best_match = agg_id
                best_similarity = similarity

        if best_match:
            # Route to existing aggregator
            self.aggregators[best_match].add_result(result)
            return best_match
        else:
            # Spawn new aggregator
            new_id = f"cluster_{len(self.aggregators)}_{int(time.time())}"
            self.aggregators[new_id] = ClusterAggregator(
                cluster_id=new_id,
                centroid=embedding.copy(),
                results=[result]
            )
            return new_id

    def finalize(self) -> List[Dict]:
        """Compute final density scores across all clusters."""
        return [agg.get_cluster_stats() for agg in self.aggregators.values()]
```

### Service Discovery for Aggregators

Cluster aggregators register with discovery (like Phase 3 nodes):

```python
def spawn_aggregator_service(cluster_id: str, centroid: np.ndarray):
    """Spawn a cluster aggregator as a discoverable service."""

    # Register with discovery
    discovery.register(
        service_name='kg_cluster_aggregator',
        service_id=cluster_id,
        tags=['cluster_aggregator', 'ephemeral'],
        metadata={
            'centroid': base64.b64encode(centroid.tobytes()).decode(),
            'created_at': datetime.now().isoformat(),
            'ttl': 300  # 5 minute lifetime
        }
    )

    # Start HTTP endpoint for result ingestion
    # ...
```

### Aggregator Coordination (Semantic DHT)

**Critical:** Aggregators for a transaction must know about each other to route optimally. This is analogous to how Freenet decides where to insert files - routing requires global awareness of the key space.

**Analogy to Freenet/DHT:**

| DHT Concept | Semantic Clustering Equivalent |
|-------------|-------------------------------|
| Key hash | Result embedding |
| Node ID / key space | Aggregator centroid / semantic region |
| XOR distance | Cosine distance |
| Insert routing | Result routing to best cluster |
| Key space partitioning | Semantic space partitioning |

### Transaction-Scoped Aggregator Registry

Each federated query (transaction) has its own aggregator namespace:

```python
@dataclass
class AggregatorRegistry:
    """Transaction-scoped registry of cluster aggregators."""
    transaction_id: str
    aggregators: Dict[str, ClusterAggregator] = field(default_factory=dict)
    centroid_index: Optional[np.ndarray] = None  # For fast nearest lookup

    def register_aggregator(self, aggregator: ClusterAggregator):
        """Register aggregator and broadcast to peers."""
        self.aggregators[aggregator.cluster_id] = aggregator

        # Update centroid index for fast routing
        self._rebuild_index()

        # Broadcast to all other aggregators in this transaction
        for peer_id, peer in self.aggregators.items():
            if peer_id != aggregator.cluster_id:
                peer.notify_new_peer(aggregator.cluster_id, aggregator.centroid)

    def find_best_aggregator(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Find closest aggregator by cosine similarity."""
        if not self.aggregators:
            return None, 0.0

        best_id = None
        best_sim = -1.0

        for agg_id, agg in self.aggregators.items():
            sim = cosine_similarity(embedding, agg.centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = agg_id

        return best_id, best_sim

    def _rebuild_index(self):
        """Rebuild centroid index for efficient lookup."""
        if len(self.aggregators) > 10:  # Use index for larger registries
            centroids = np.array([a.centroid for a in self.aggregators.values()])
            # Could use FAISS or Annoy for approximate nearest neighbor
            self.centroid_index = centroids
```

### Gossip Protocol for Aggregator Discovery

Aggregators discover each other via gossip within the transaction scope:

```python
class ClusterAggregator:
    def __init__(self, cluster_id: str, centroid: np.ndarray, transaction_id: str):
        self.cluster_id = cluster_id
        self.centroid = centroid
        self.transaction_id = transaction_id
        self.known_peers: Dict[str, np.ndarray] = {}  # peer_id -> centroid

    def notify_new_peer(self, peer_id: str, peer_centroid: np.ndarray):
        """Called when a new aggregator joins the transaction."""
        self.known_peers[peer_id] = peer_centroid

    def should_redirect(self, embedding: np.ndarray) -> Optional[str]:
        """Check if another aggregator is better suited for this result."""
        my_sim = cosine_similarity(embedding, self.centroid)

        for peer_id, peer_centroid in self.known_peers.items():
            peer_sim = cosine_similarity(embedding, peer_centroid)
            if peer_sim > my_sim + 0.1:  # Significant improvement threshold
                return peer_id

        return None  # Keep locally

    def route_or_accept(self, result: AggregatedResult) -> Tuple[str, bool]:
        """Route to better peer or accept locally."""
        redirect_to = self.should_redirect(result.semantic_centroid)

        if redirect_to:
            return redirect_to, False  # Redirect
        else:
            self.add_result(result)
            return self.cluster_id, True  # Accepted
```

### Freenet-Style Routing

Like Freenet's insert routing, we use greedy forwarding with backtracking:

```python
def route_result_freenet_style(
    result: AggregatedResult,
    entry_aggregator: ClusterAggregator,
    max_hops: int = 5
) -> str:
    """Route result to best aggregator using Freenet-style greedy routing."""

    current = entry_aggregator
    visited = {current.cluster_id}
    embedding = result.semantic_centroid

    for hop in range(max_hops):
        # Check if current should redirect
        redirect_to = current.should_redirect(embedding)

        if redirect_to is None:
            # Current is best known location
            current.add_result(result)
            return current.cluster_id

        if redirect_to in visited:
            # Cycle detected, stay at current
            current.add_result(result)
            return current.cluster_id

        # Forward to better aggregator
        visited.add(redirect_to)
        current = get_aggregator(redirect_to)  # Fetch from registry

    # Max hops reached, accept at current
    current.add_result(result)
    return current.cluster_id
```

### Splitting and Merging Aggregators

As results accumulate, aggregators may need to split (too diverse) or merge (too similar):

```python
class ClusterAggregator:
    def check_split(self, variance_threshold: float = 0.3) -> Optional[List['ClusterAggregator']]:
        """Check if cluster should split due to high internal variance."""
        if len(self.results) < 4:
            return None

        embeddings = np.array([r.semantic_centroid for r in self.results])
        variance = np.var(embeddings, axis=0).mean()

        if variance > variance_threshold:
            # Split using k-means with k=2
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, n_init=3)
            labels = kmeans.fit_predict(embeddings)

            # Create two new aggregators
            new_aggs = []
            for label in [0, 1]:
                mask = labels == label
                new_results = [r for r, m in zip(self.results, mask) if m]
                new_centroid = kmeans.cluster_centers_[label]
                new_aggs.append(ClusterAggregator(
                    cluster_id=f"{self.cluster_id}_split{label}",
                    centroid=new_centroid,
                    transaction_id=self.transaction_id,
                    results=new_results
                ))
            return new_aggs

        return None

    def should_merge_with(self, other: 'ClusterAggregator', threshold: float = 0.9) -> bool:
        """Check if two aggregators should merge."""
        sim = cosine_similarity(self.centroid, other.centroid)
        return sim >= threshold
```

### Transaction Management

Aggregators are scoped to a transaction (federated query) and require explicit lifecycle management:

```python
@dataclass
class TransactionConfig:
    """Configuration for a federated query transaction."""
    transaction_id: str
    timeout_ms: int = 30000  # 30 second default
    max_aggregators: int = 50  # Prevent runaway spawning
    cleanup_on_complete: bool = True
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        elapsed = (time.time() - self.created_at) * 1000
        return elapsed > self.timeout_ms


class TransactionManager:
    """Manages lifecycle of aggregator transactions."""

    def __init__(self):
        self.transactions: Dict[str, AggregatorRegistry] = {}
        self.configs: Dict[str, TransactionConfig] = {}
        self._cleanup_task = None

    def begin_transaction(
        self,
        transaction_id: str,
        timeout_ms: int = 30000,
        max_aggregators: int = 50
    ) -> AggregatorRegistry:
        """Start a new aggregator transaction."""
        config = TransactionConfig(
            transaction_id=transaction_id,
            timeout_ms=timeout_ms,
            max_aggregators=max_aggregators
        )
        registry = AggregatorRegistry(transaction_id=transaction_id)

        self.transactions[transaction_id] = registry
        self.configs[transaction_id] = config

        # Start timeout watchdog if not running
        self._ensure_cleanup_task()

        return registry

    def close_transaction(self, transaction_id: str) -> Optional[List[Dict]]:
        """
        Explicitly close a transaction and collect results.

        Returns cluster stats before cleanup.
        """
        if transaction_id not in self.transactions:
            return None

        registry = self.transactions[transaction_id]

        # Collect final stats from all aggregators
        results = registry.finalize()

        # Kill all aggregators
        for agg in registry.aggregators.values():
            agg.shutdown()

        # Remove from tracking
        del self.transactions[transaction_id]
        del self.configs[transaction_id]

        return results

    def kill_transaction(self, transaction_id: str):
        """Force-kill a transaction without collecting results."""
        if transaction_id not in self.transactions:
            return

        registry = self.transactions[transaction_id]

        # Immediate shutdown, no finalization
        for agg in registry.aggregators.values():
            agg.shutdown(force=True)

        del self.transactions[transaction_id]
        del self.configs[transaction_id]

    def _ensure_cleanup_task(self):
        """Start background task to clean up expired transactions."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Periodically check for and clean up expired transactions."""
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds

            expired = [
                tid for tid, config in self.configs.items()
                if config.is_expired()
            ]

            for tid in expired:
                logger.warning(f"Transaction {tid} expired, forcing cleanup")
                self.kill_transaction(tid)

            if not self.transactions:
                self._cleanup_task = None
                break


@dataclass
class ClusterAggregator:
    """Cluster aggregator with transaction awareness."""
    cluster_id: str
    centroid: np.ndarray
    transaction_id: str
    results: List[AggregatedResult] = field(default_factory=list)
    known_peers: Dict[str, np.ndarray] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    _shutdown: bool = False

    def shutdown(self, force: bool = False):
        """Shutdown this aggregator."""
        self._shutdown = True

        if not force:
            # Graceful: compute final density before shutdown
            self.compute_density()

        # Deregister from discovery
        discovery.deregister(self.cluster_id)

        # Clear results to free memory
        self.results.clear()
        self.known_peers.clear()

    def is_active(self) -> bool:
        return not self._shutdown

    def add_result(self, result: AggregatedResult) -> bool:
        """Add result, returns False if shutdown."""
        if self._shutdown:
            return False
        self.results.append(result)
        self._update_centroid()
        return True
```

### Transaction Protocol

Coordinator includes transaction management in federated queries:

**Begin Transaction (implicit with first query):**
```json
{
    "__type": "kg_federated_query",
    "__id": "query-123",
    "__transaction": {
        "id": "txn-456",
        "timeout_ms": 30000,
        "max_aggregators": 50
    },
    "payload": { "query_text": "...", "top_k": 5 }
}
```

**Close Transaction (explicit):**
```json
{
    "__type": "kg_transaction_close",
    "__transaction_id": "txn-456",
    "collect_results": true
}
```

**Kill Transaction (force):**
```json
{
    "__type": "kg_transaction_kill",
    "__transaction_id": "txn-456"
}
```

**Aggregator Heartbeat:**
```json
{
    "__type": "kg_aggregator_heartbeat",
    "__transaction_id": "txn-456",
    "__aggregator_id": "cluster_0_1234567890",
    "result_count": 12,
    "centroid_drift": 0.05
}
```

### Transaction Lifecycle

```
1. Query arrives at coordinator
   ├── Generate transaction_id
   ├── TransactionManager.begin_transaction()
   └── Create empty AggregatorRegistry with timeout

2. First result arrives
   ├── Spawn first ClusterAggregator with transaction_id
   ├── Register in transaction-scoped registry
   └── Aggregator registers with discovery (ephemeral, tagged with txn_id)

3. Subsequent results
   ├── Registry finds best aggregator
   ├── Route to aggregator (Freenet-style if needed)
   ├── Aggregator accepts or redirects
   └── Check max_aggregators limit before spawning new

4. Aggregator spawning
   ├── If no aggregator within threshold AND under limit, spawn new
   ├── New aggregator registers with registry
   ├── Gossip notifies existing aggregators
   └── New aggregator tagged with transaction_id

5. Periodic maintenance (while transaction active)
   ├── Check for splits (high variance)
   ├── Check for merges (high similarity)
   ├── Aggregators send heartbeats
   └── TransactionManager checks timeout

6. Query completion (explicit close)
   ├── Coordinator calls close_transaction()
   ├── Each aggregator computes final density
   ├── Coordinator collects cluster stats
   ├── Final ranking across clusters
   └── All aggregators shutdown and deregister

7. Timeout (implicit kill)
   ├── TransactionManager detects expired transaction
   ├── kill_transaction() force-closes all aggregators
   └── Resources freed, partial results lost
```

### Timeout Hierarchy

Multiple timeout levels for robustness:

```
Transaction timeout (30s default)
├── Aggregator idle timeout (10s) - no new results
├── Heartbeat timeout (5s) - aggregator presumed dead
└── Discovery TTL (60s) - final cleanup via discovery service
```

```python
@dataclass
class AggregatorTimeouts:
    """Timeout configuration for aggregators."""
    transaction_timeout_ms: int = 30000  # Overall transaction limit
    aggregator_idle_ms: int = 10000      # No activity timeout
    heartbeat_interval_ms: int = 2000    # Heartbeat frequency
    heartbeat_timeout_ms: int = 5000     # Missed heartbeat = dead
    discovery_ttl_s: int = 60            # Final cleanup via discovery
```

### Streaming Architecture

For high-throughput scenarios, use streaming aggregation:

```
Results Stream → Router → Aggregator Pool → Density Computation → Merge
                   │
                   ├── Find best aggregator (semantic DHT lookup)
                   │
                   ├── Route with Freenet-style forwarding
                   │
                   └── Spawn new if no good match
```

This integrates with Phase 3's discovery infrastructure - aggregators are ephemeral services that:
1. Register their centroid with transaction-scoped registry
2. Gossip with peers to maintain global awareness
3. Accept results routed by semantic similarity
4. Redirect to better peers when appropriate
5. Compute intra-cluster density
6. Report back to coordinator
7. Deregister after query completes (TTL-based cleanup)

```python
def federated_density_estimate(responses: List[NodeResponse]) -> Dict[str, float]:
    """Estimate density across federated results."""

    # Collect all embeddings
    all_embeddings = []
    embedding_to_result = {}

    for resp in responses:
        for result in resp.results:
            if result.embedding is not None:
                all_embeddings.append(result.embedding)
                embedding_to_result[id(result.embedding)] = result

    if len(all_embeddings) < 2:
        return {r.answer_hash: 1.0 for r in embedding_to_result.values()}

    # Compute pairwise distances
    embeddings = np.array(all_embeddings)

    # Bandwidth selection (Silverman's rule)
    dists = pdist(embeddings, metric='cosine')
    h = 0.9 * np.std(dists) * len(dists) ** (-0.2)

    # KDE for each point
    density_scores = {}
    for i, emb in enumerate(embeddings):
        # Gaussian kernel with cosine distance
        similarities = 1 - cdist([emb], embeddings, metric='cosine')[0]
        kernel_values = np.exp(-(1 - similarities) ** 2 / (2 * h ** 2))
        density = kernel_values.mean()

        result = embedding_to_result[id(all_embeddings[i])]
        density_scores[result.answer_hash] = density

    # Normalize to [0, 1]
    max_density = max(density_scores.values())
    if max_density > 0:
        density_scores = {k: v / max_density for k, v in density_scores.items()}

    return density_scores
```

## Two-Stage Pipeline: Cluster Then Density

**Critical insight:** Density must be computed *within* semantic clusters, not across all results. Otherwise unrelated answers dilute the signal.

### Stage 1: Route Results to Clusters

First, partition results into semantic neighborhoods:

```
All Results → Clustering → { Cluster₁, Cluster₂, ..., Clusterₖ, Noise }
```

This is analogous to Phase 3 routing - results are "routed" to their semantic home before local analysis.

**Connection to Phase 3:** Just as queries are routed to the most relevant node via Kleinberg routing, results are routed to the most relevant cluster via semantic similarity. The same infrastructure (embeddings, cosine distance, centroid matching) applies at both levels.

### Stage 2: Density Within Each Cluster

Then compute density scores relative to cluster membership:

```
For each cluster Cⱼ:
    density_score(rᵢ) = (1/|Cⱼ|) Σ_{r∈Cⱼ} K_h(eᵢ - eᵣ)
```

This measures "how central is this result within its semantic neighborhood?"

### Why Two Stages?

Without clustering first:
```
Query: "How to read files in Python?"

Results:
  - "Use open() and read()" [text files]
  - "Use json.load()" [JSON files]
  - "Use csv.reader()" [CSV files]
  - "open('file.txt').read()" [text files]

Single-stage density: All results seem sparse (different topics mixed)
Two-stage density:
  - Cluster "text files": 2 results, tight → high density
  - Cluster "JSON": 1 result → low density (singleton)
  - Cluster "CSV": 1 result → low density (singleton)
```

The two-stage approach correctly identifies the text file cluster as having consensus.

### Pipeline Integration

```python
def density_scoring_pipeline(results: List[AggregatedResult]) -> List[AggregatedResult]:
    """Two-stage density scoring: cluster then measure."""

    # Extract embeddings
    embeddings = np.array([r.semantic_centroid for r in results])

    # Stage 1: Route to clusters
    cluster_labels = cluster_results(embeddings)

    # Stage 2: Density within each cluster
    for cluster_id in set(cluster_labels):
        if cluster_id < 0:  # Skip noise
            continue

        mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_results = [r for r, m in zip(results, mask) if m]

        # Compute intra-cluster density
        densities = compute_cluster_density(cluster_embeddings)

        for result, density in zip(cluster_results, densities):
            result.density_score = density
            result.cluster_id = cluster_id

    return results
```

## Clustering Methods for Stage 1

Beyond density scoring, clustering partitions results into semantic equivalence classes:

### HDBSCAN Clustering

```python
from hdbscan import HDBSCAN

def cluster_results(embeddings: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:
    """Cluster results by semantic similarity."""

    # Use cosine distance
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='cosine',
        cluster_selection_method='eom'  # Excess of mass
    )

    labels = clusterer.fit_predict(embeddings)
    return labels
```

### Cluster-Based Aggregation

```python
def aggregate_by_cluster(
    results: List[AggregatedResult],
    cluster_labels: np.ndarray
) -> List[AggregatedResult]:
    """Merge results within same semantic cluster."""

    clusters = defaultdict(list)
    for result, label in zip(results, cluster_labels):
        if label >= 0:  # -1 is noise
            clusters[label].append(result)
        else:
            clusters[f"noise_{id(result)}"].append(result)

    merged = []
    for cluster_id, cluster_results in clusters.items():
        if len(cluster_results) == 1:
            merged.append(cluster_results[0])
        else:
            # Merge cluster into single result
            merged.append(merge_cluster(cluster_results))

    return merged

def merge_cluster(results: List[AggregatedResult]) -> AggregatedResult:
    """Merge semantically equivalent results."""

    # Use highest-scoring result as representative
    best = max(results, key=lambda r: r.combined_score)

    # Aggregate scores (SUM for consensus boost)
    total_score = sum(r.combined_score for r in results)

    # Collect all provenance
    all_provenance = []
    all_nodes = set()
    for r in results:
        all_provenance.extend(r.provenance)
        all_nodes.update(r.source_nodes)

    # Compute cluster confidence
    density_scores = [r.density_score for r in results if r.density_score > 0]
    cluster_confidence = np.mean(density_scores) if density_scores else 0.0

    return AggregatedResult(
        answer_text=best.answer_text,
        answer_hash=best.answer_hash,
        combined_score=total_score,
        source_nodes=list(all_nodes),
        provenance=all_provenance,
        density_score=np.mean([r.density_score for r in results]),
        cluster_confidence=cluster_confidence * len(results),  # Boost by cluster size
        semantic_centroid=np.mean([r.semantic_centroid for r in results if r.semantic_centroid is not None], axis=0)
    )
```

## Protocol Extensions

### Request: Enable Density Scoring

```json
{
    "__type": "kg_federated_query",
    "__routing": {
        "federation_k": 5,
        "aggregation": {
            "score_function": "density_weighted",
            "density_config": {
                "bandwidth": "auto",
                "clustering": true,
                "min_cluster_size": 2
            }
        }
    },
    "payload": {
        "query_text": "How do I parse CSV?",
        "top_k": 5,
        "return_embeddings": true
    }
}
```

### Response: Include Density Metrics

```json
{
    "__type": "kg_federated_response",
    "results": [
        {
            "answer_id": 42,
            "answer_text": "Use csv.reader()...",
            "exp_score": 2.718,
            "embedding": [0.1, 0.2, ...],
            "local_density": 0.85
        }
    ],
    "node_metadata": {
        "corpus_id": "stackoverflow",
        "embedding_model": "all-MiniLM-L6-v2"
    }
}
```

### Aggregated Response: Cluster Info

```json
{
    "results": [
        {
            "answer_text": "Use csv.reader()...",
            "combined_score": 8.5,
            "normalized_prob": 0.45,
            "density_score": 0.92,
            "cluster_confidence": 2.76,
            "cluster_size": 3,
            "unique_corpora": 3,
            "diversity_score": 1.0
        }
    ],
    "clustering_stats": {
        "num_clusters": 4,
        "noise_points": 2,
        "avg_cluster_size": 2.5,
        "bandwidth_used": 0.15
    }
}
```

## Prolog Validation Extensions

```prolog
%% Density scoring options
is_valid_density_option(bandwidth(auto)).
is_valid_density_option(bandwidth(B)) :- number(B), B > 0.
is_valid_density_option(clustering(Bool)) :- (Bool = true ; Bool = false).
is_valid_density_option(min_cluster_size(N)) :- integer(N), N >= 2.
is_valid_density_option(density_weight(W)) :- number(W), W >= 0, W =< 1.

%% Extended aggregation strategy
is_valid_aggregation_strategy(density_weighted).
is_valid_aggregation_strategy(density_weighted(Opts)) :-
    is_list(Opts), maplist(is_valid_density_option, Opts).
```

## Implementation Phases

### Phase 4d-i: Basic Density Scoring ✅ COMPLETE
- [x] Add `embedding` field to result protocol (`NodeResult.embedding`, `AggregatedResult.semantic_centroid`)
- [x] Implement KDE with `compute_density_scores()` and `two_stage_density_pipeline()`
- [x] Add `density_score`, `cluster_id`, `cluster_confidence` to `AggregatedResult`
- [x] Silverman's rule bandwidth selection (`silverman_bandwidth()`)
- [x] Flux-softmax: `P(i) = exp(sᵢ/τ) * (1 + w * dᵢ) / Z`
- [x] `DensityAwareFederatedEngine` with `DENSITY_FLUX` strategy
- [x] Greedy centroid-based clustering (`cluster_by_similarity()`)
- [x] Transaction management (`TransactionManager`, `ClusterAggregator`, `AggregatorRegistry`)
- [x] Prolog validation for density options
- [x] 35 unit tests

**Implementation:**
- `density_scoring.py` (~800 lines)
- `federated_query.py` extended (~1100 lines)
- `service_validation.pl` extended

### Phase 4d-ii: Semantic Clustering
- [ ] HDBSCAN integration for hierarchical clustering
- [ ] `merge_cluster()` for semantic dedup
- [ ] Cluster visualization utilities

### Phase 4d-iii: Adaptive Methods
- [ ] Cross-validation bandwidth selection
- [ ] Adaptive (balloon) kernel density estimation
- [ ] Query-dependent bandwidth (specificity-aware)

### Phase 4d-iv: Efficiency
- [ ] Sketched density estimation (random projections)
- [ ] Approximate nearest neighbor for large result sets
- [ ] Caching of pairwise distances

## Example: Density in Action

```
Query: "How to read JSON in Python?"

Results from 3 nodes:

Node A (StackOverflow):
  - "Use json.load() for files" [embedding: e1]
  - "json.loads() parses strings" [embedding: e2]

Node B (Python Docs):
  - "The json module provides load()" [embedding: e3]

Node C (Tutorial Site):
  - "import json; data = json.load(f)" [embedding: e4]
  - "Use pandas.read_json() for DataFrames" [embedding: e5]

Embedding space visualization:

         e5 (pandas)
           •

    e1 • • e3    ← Tight cluster: json.load()
      • e4

        • e2     ← Nearby: json.loads()

Density scores:
  e1: 0.92 (in main cluster)
  e3: 0.95 (cluster center)
  e4: 0.88 (cluster edge)
  e2: 0.65 (nearby but distinct)
  e5: 0.20 (outlier - different topic)

Result after density-weighted aggregation:
  1. "Use json.load() for files"
     - combined_score: 8.5 (boosted by cluster)
     - cluster_confidence: 2.8
     - density_score: 0.92

  2. "json.loads() parses strings"
     - combined_score: 2.1
     - density_score: 0.65

  3. "Use pandas.read_json()..."
     - combined_score: 0.8 (penalized as outlier)
     - density_score: 0.20
```

## Relation to Existing Work

- **Phase 4b diversity**: Corpus-level independence
- **Density scoring**: Semantic-level consensus
- Together: "Different sources agreeing on semantically similar answers"

The smoothness assumption connects to:
- **Manifold hypothesis**: Embeddings lie on low-dim manifold
- **Kernel methods**: Smoothness via RKHS norm
- **Gaussian processes**: Smooth function priors

## Open Questions

1. **Bandwidth for high-dim embeddings**: Curse of dimensionality affects KDE. Should we reduce dimensionality first (PCA/UMAP)?

2. **Cross-model density**: Different nodes may use different embedding models. How to compare densities across incompatible spaces?

3. **Streaming updates**: Can we incrementally update density estimates as results arrive?

4. **Adversarial robustness**: Malicious nodes could inject results to inflate cluster density.

## References

- Silverman, B.W. (1986). Density Estimation for Statistics and Data Analysis
- Campello et al. (2013). HDBSCAN: Density-Based Clustering
- Scott, D.W. (2015). Multivariate Density Estimation
- Phase 4: Federated Query Algebra (FEDERATED_QUERY_ALGEBRA.md)
