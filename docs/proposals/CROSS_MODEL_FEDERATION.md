# Cross-Model Federation Proposal

**Status:** Proposed
**Date:** 2024-12-19
**Prerequisites:** Phase 1-5 Complete, Phase 6a Complete
**Related:** [KG_TOPOLOGY_FUTURE_WORK.md](KG_TOPOLOGY_FUTURE_WORK.md), [DENSITY_SCORING_PROPOSAL.md](DENSITY_SCORING_PROPOSAL.md)

## Problem Statement

Current KG Topology federation assumes all nodes use the same embedding model. In practice, production deployments may have:

- **Legacy nodes** using older models (e.g., `all-MiniLM-L6-v2`)
- **Specialized nodes** using domain-specific fine-tuned models
- **Upgraded nodes** using newer, larger models (e.g., `E5-large-v2`, `BGE-large`)
- **Multi-vendor deployments** where different teams chose different models

Cosine similarity between vectors from different embedding models is **meaningless** - the vector spaces are incompatible. However, we can still achieve meaningful cross-model federation by exploiting a key insight.

## Core Insight: Probabilities Are Comparable

While raw embeddings are incompatible across models, **normalized scores are comparable**:

```
P(answer_i | query, model_m) = exp(sim_i) / Σ_j exp(sim_j)
```

After softmax normalization, cosine similarities become probabilities. These probabilities represent "how much does this model believe this answer is relevant?" - a question that's meaningful regardless of the underlying embedding space.

## Two-Phase Architecture

The key architectural insight is that density-based methods work **within** each model pool, and their outputs (density-adjusted scores) can then be combined **across** pools.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Query                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│     Model Pool A      │ │     Model Pool B      │ │     Model Pool C      │
│    (all-MiniLM-L6)    │ │     (E5-large-v2)     │ │     (BGE-base-en)     │
│                       │ │                       │ │                       │
│  ┌─────┐ ┌─────┐     │ │  ┌─────┐ ┌─────┐     │ │  ┌─────┐ ┌─────┐     │
│  │Node1│ │Node2│ ... │ │  │Node4│ │Node5│ ... │ │  │Node7│ │Node8│ ... │
│  └─────┘ └─────┘     │ │  └─────┘ └─────┘     │ │  └─────┘ └─────┘     │
│                       │ │                       │ │                       │
│  Phase 1: Intra-Pool  │ │  Phase 1: Intra-Pool  │ │  Phase 1: Intra-Pool  │
│  ├─ HDBSCAN cluster   │ │  ├─ HDBSCAN cluster   │ │  ├─ HDBSCAN cluster   │
│  ├─ KDE density       │ │  ├─ KDE density       │ │  ├─ KDE density       │
│  ├─ Density flux      │ │  ├─ Density flux      │ │  ├─ Density flux      │
│  └─ Adaptive bandwidth│ │  └─ Adaptive bandwidth│ │  └─ Adaptive bandwidth│
│                       │ │                       │ │                       │
│  Output:              │ │  Output:              │ │  Output:              │
│  - density_score      │ │  - density_score      │ │  - density_score      │
│  - cluster_id         │ │  - cluster_id         │ │  - cluster_id         │
│  - cluster_confidence │ │  - cluster_confidence │ │  - cluster_confidence │
│  - adjusted_score     │ │  - adjusted_score     │ │  - adjusted_score     │
└───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
            │                         │                         │
            │    PoolResult[]         │    PoolResult[]         │
            │    (scores, not         │    (scores, not         │
            │     embeddings)         │     embeddings)         │
            ▼                         ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Phase 2: Cross-Model Fusion                          │
│                                                                         │
│  Fusion Methods:                                                        │
│  ├─ Weighted Sum: Σ w_m * P(answer | model_m)                          │
│  ├─ Reciprocal Rank Fusion: Σ 1/(k + rank_m)                           │
│  ├─ Consensus Boost: geometric_mean(densities) if multi-pool           │
│  └─ Learned Weights: gradient descent on relevance feedback            │
│                                                                         │
│  Cross-Pool Signals:                                                    │
│  ├─ Agreement: Answer in top-k of multiple pools → boost               │
│  ├─ Density Consistency: High density across pools → high confidence   │
│  └─ Cluster Size Correlation: Large clusters agree → strong signal     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          Final Ranked Results
```

## What Works vs. What Doesn't

### Within Each Model Pool (Embeddings Compatible)

| Method | Status | Notes |
|--------|--------|-------|
| Cosine similarity | ✓ | Same embedding space |
| HDBSCAN clustering | ✓ | Cluster similar answers |
| KDE density estimation | ✓ | Measure answer centrality |
| Density flux scoring | ✓ | Combine similarity + density |
| Adaptive bandwidth | ✓ | Cross-validation within pool |
| Kleinberg routing | ✓ | Route to nearest centroid |
| Centroid computation | ✓ | Average embeddings |

### Across Model Pools (Only Scores, Not Embeddings)

| Method | Status | Notes |
|--------|--------|-------|
| Raw embedding comparison | ✗ | Incompatible vector spaces |
| Cross-pool clustering | ✗ | Can't cluster across spaces |
| Cross-pool density | ✗ | KDE requires same space |
| **Score fusion** | ✓ | Normalized probabilities comparable |
| **Rank fusion** | ✓ | Ranks are model-agnostic |
| **Consensus detection** | ✓ | Agreement across pools |
| **Density score propagation** | ✓ | Use pool's density output |

## Data Structures

### Pool Result

Each model pool returns density-enriched results:

```python
@dataclass
class PoolResult:
    """Result from a single model pool with density information."""
    answer_id: str
    answer_text: str

    # Raw scores
    raw_score: float              # Cosine similarity
    exp_score: float              # exp(raw_score) for partition

    # Density information (computed within pool)
    density_score: float          # KDE density at this point
    density_adjusted_score: float # flux_softmax output

    # Cluster information
    cluster_id: int               # HDBSCAN cluster label
    cluster_size: int             # Number of results in cluster
    cluster_confidence: float     # Cluster membership probability

    # Provenance
    source_nodes: List[str]       # Which nodes contributed
    model_name: str               # Embedding model used


@dataclass
class PoolResponse:
    """Aggregated response from a model pool."""
    model_name: str
    results: List[PoolResult]

    # Pool-level statistics
    total_results: int
    num_clusters: int
    avg_density: float
    partition_sum: float          # For probability normalization

    # Timing
    query_time_ms: float
    nodes_queried: int
    nodes_responded: int
```

### Cross-Model Configuration

```python
@dataclass
class ModelPoolConfig:
    """Configuration for a single model pool."""
    model_name: str
    weight: float = 1.0           # Relative weight in fusion
    node_filter: Optional[Callable] = None  # Filter nodes for this pool

    # Intra-pool settings (use existing Phase 4-5 settings)
    federation_k: int = 5
    aggregation_strategy: AggregationStrategy = AggregationStrategy.DENSITY_FLUX
    use_adaptive_k: bool = True
    use_hierarchical: bool = False


@dataclass
class CrossModelConfig:
    """Configuration for cross-model federation."""
    pools: List[ModelPoolConfig]

    # Fusion method
    fusion_method: FusionMethod = FusionMethod.WEIGHTED_SUM

    # Fusion parameters
    rrf_k: int = 60                        # RRF smoothing constant
    consensus_threshold: float = 0.1       # Min probability for consensus
    min_pools_for_consensus: int = 2       # Pools needed for boost
    consensus_boost_factor: float = 1.5    # Multiplier for consensus

    # Weight learning
    learn_weights: bool = False
    weight_learning_rate: float = 0.01

    # Query routing
    query_all_pools: bool = True           # vs. adaptive pool selection
    pool_timeout_ms: int = 30000


class FusionMethod(Enum):
    """Methods for combining results across model pools."""
    WEIGHTED_SUM = "weighted_sum"          # Σ w_m * score_m
    RECIPROCAL_RANK = "rrf"                # Σ 1/(k + rank_m)
    CONSENSUS = "consensus"                 # Boost multi-pool agreement
    GEOMETRIC_MEAN = "geometric_mean"       # (Π score_m)^(1/n)
    MAX = "max"                             # max(score_m)
    LEARNED = "learned"                     # Neural combination
```

## Fusion Algorithms

### 1. Weighted Sum Fusion

The simplest approach - weighted average of normalized probabilities:

```python
def weighted_sum_fusion(
    pool_responses: Dict[str, PoolResponse],
    config: CrossModelConfig,
    top_k: int
) -> List[FusedResult]:
    """Weighted sum of density-adjusted probabilities."""

    answer_scores = defaultdict(float)
    answer_metadata = {}

    for pool_config in config.pools:
        model = pool_config.model_name
        response = pool_responses.get(model)
        if not response:
            continue

        weight = pool_config.weight

        # Normalize to probabilities within pool
        total = sum(r.density_adjusted_score for r in response.results)

        for r in response.results:
            prob = r.density_adjusted_score / total if total > 0 else 0
            answer_scores[r.answer_id] += weight * prob

            # Track metadata for later
            if r.answer_id not in answer_metadata:
                answer_metadata[r.answer_id] = {
                    'text': r.answer_text,
                    'pool_contributions': {}
                }
            answer_metadata[r.answer_id]['pool_contributions'][model] = {
                'probability': prob,
                'density': r.density_score,
                'cluster_size': r.cluster_size
            }

    # Build final results
    ranked = sorted(answer_scores.items(), key=lambda x: -x[1])

    return [
        FusedResult(
            answer_id=aid,
            answer_text=answer_metadata[aid]['text'],
            fused_score=score,
            pool_contributions=answer_metadata[aid]['pool_contributions'],
            num_pools=len(answer_metadata[aid]['pool_contributions'])
        )
        for aid, score in ranked[:top_k]
    ]
```

### 2. Reciprocal Rank Fusion (RRF)

Rank-based fusion that doesn't depend on score calibration:

```python
def rrf_fusion(
    pool_responses: Dict[str, PoolResponse],
    config: CrossModelConfig,
    top_k: int
) -> List[FusedResult]:
    """Reciprocal Rank Fusion - combines rankings not scores."""

    answer_rrf_scores = defaultdict(float)
    answer_ranks = defaultdict(dict)

    for pool_config in config.pools:
        model = pool_config.model_name
        response = pool_responses.get(model)
        if not response:
            continue

        # Sort by density-adjusted score to get ranks
        sorted_results = sorted(
            response.results,
            key=lambda r: -r.density_adjusted_score
        )

        for rank, r in enumerate(sorted_results, start=1):
            rrf_contribution = 1.0 / (config.rrf_k + rank)
            answer_rrf_scores[r.answer_id] += rrf_contribution
            answer_ranks[r.answer_id][model] = rank

    ranked = sorted(answer_rrf_scores.items(), key=lambda x: -x[1])

    return [
        FusedResult(
            answer_id=aid,
            fused_score=score,
            ranks_by_pool=answer_ranks[aid],
            num_pools=len(answer_ranks[aid])
        )
        for aid, score in ranked[:top_k]
    ]
```

### 3. Consensus Fusion with Density Propagation

Rewards answers that appear with high density in multiple pools:

```python
def consensus_fusion(
    pool_responses: Dict[str, PoolResponse],
    config: CrossModelConfig,
    top_k: int
) -> List[FusedResult]:
    """Boost answers with high density across multiple pools."""

    answer_evidence = defaultdict(list)

    # Collect evidence from each pool
    for pool_config in config.pools:
        model = pool_config.model_name
        response = pool_responses.get(model)
        if not response:
            continue

        # Normalize probabilities
        total = sum(r.density_adjusted_score for r in response.results)

        for r in response.results:
            prob = r.density_adjusted_score / total if total > 0 else 0

            if prob >= config.consensus_threshold:
                answer_evidence[r.answer_id].append({
                    'model': model,
                    'probability': prob,
                    'density': r.density_score,
                    'cluster_size': r.cluster_size,
                    'cluster_confidence': r.cluster_confidence,
                    'weight': pool_config.weight
                })

    # Compute fused scores with consensus boost
    final_scores = {}

    for answer_id, evidences in answer_evidence.items():
        # Base score: weighted sum of probabilities
        base_score = sum(
            e['probability'] * e['weight']
            for e in evidences
        )

        # Consensus boost: reward multi-pool agreement
        num_pools = len(evidences)

        if num_pools >= config.min_pools_for_consensus:
            # Geometric mean of densities (rewards consistent high density)
            densities = [e['density'] for e in evidences]
            density_agreement = geometric_mean(densities)

            # Boost proportional to agreement strength
            boost = 1 + (config.consensus_boost_factor - 1) * density_agreement
            base_score *= boost

            # Additional boost for cluster size agreement
            cluster_sizes = [e['cluster_size'] for e in evidences]
            if min(cluster_sizes) >= 3:  # All pools have substantial clusters
                base_score *= 1.1

        final_scores[answer_id] = {
            'score': base_score,
            'num_pools': num_pools,
            'evidences': evidences
        }

    ranked = sorted(final_scores.items(), key=lambda x: -x[1]['score'])

    return [
        FusedResult(
            answer_id=aid,
            fused_score=data['score'],
            num_pools=data['num_pools'],
            consensus_strength=data['num_pools'] / len(config.pools),
            pool_evidences=data['evidences']
        )
        for aid, data in ranked[:top_k]
    ]


def geometric_mean(values: List[float]) -> float:
    """Geometric mean - rewards consistency."""
    if not values:
        return 0.0
    product = 1.0
    for v in values:
        product *= max(v, 1e-10)  # Avoid zero
    return product ** (1.0 / len(values))
```

## Cross-Model Engine

```python
class CrossModelFederatedEngine:
    """Federate queries across nodes with different embedding models."""

    def __init__(
        self,
        router: KleinbergRouter,
        config: CrossModelConfig
    ):
        self.router = router
        self.config = config
        self.pool_engines: Dict[str, FederatedQueryEngine] = {}
        self.embedding_cache: Dict[str, Callable] = {}

        # Initialize per-pool engines
        self._init_pool_engines()

    def _init_pool_engines(self):
        """Create a FederatedQueryEngine for each model pool."""
        all_nodes = self.router.discover_nodes()

        for pool_config in self.config.pools:
            model = pool_config.model_name

            # Filter nodes for this model
            pool_nodes = [
                n for n in all_nodes
                if n.metadata.get('embedding_model') == model
            ]

            if not pool_nodes:
                continue

            # Create pool-specific router
            pool_router = self._create_pool_router(pool_nodes)

            # Create engine with Phase 4-5 features
            agg_config = AggregationConfig(
                strategy=pool_config.aggregation_strategy,
                dedup_key='answer_id'
            )

            if pool_config.use_adaptive_k:
                engine = AdaptiveFederatedEngine(
                    router=pool_router,
                    aggregation_config=agg_config
                )
            elif pool_config.use_hierarchical:
                engine = HierarchicalFederatedEngine(
                    router=pool_router,
                    aggregation_config=agg_config
                )
            else:
                engine = FederatedQueryEngine(
                    router=pool_router,
                    aggregation_config=agg_config
                )

            self.pool_engines[model] = engine

    def federated_query(
        self,
        query_text: str,
        top_k: int = 10,
        timeout_ms: int = None
    ) -> CrossModelResponse:
        """Execute cross-model federated query."""

        timeout_ms = timeout_ms or self.config.pool_timeout_ms
        start_time = time.time()

        # Phase 1: Query each pool in parallel
        pool_responses = self._query_all_pools(query_text, top_k, timeout_ms)

        # Phase 2: Fuse results
        if self.config.fusion_method == FusionMethod.WEIGHTED_SUM:
            fused = weighted_sum_fusion(pool_responses, self.config, top_k)
        elif self.config.fusion_method == FusionMethod.RECIPROCAL_RANK:
            fused = rrf_fusion(pool_responses, self.config, top_k)
        elif self.config.fusion_method == FusionMethod.CONSENSUS:
            fused = consensus_fusion(pool_responses, self.config, top_k)
        else:
            fused = weighted_sum_fusion(pool_responses, self.config, top_k)

        elapsed_ms = (time.time() - start_time) * 1000

        return CrossModelResponse(
            results=fused,
            pools_queried=list(pool_responses.keys()),
            total_time_ms=elapsed_ms,
            fusion_method=self.config.fusion_method.value
        )

    def _query_all_pools(
        self,
        query_text: str,
        top_k: int,
        timeout_ms: int
    ) -> Dict[str, PoolResponse]:
        """Query all model pools in parallel."""

        results = {}

        with ThreadPoolExecutor(max_workers=len(self.pool_engines)) as executor:
            futures = {}

            for model, engine in self.pool_engines.items():
                # Get embedding for this model
                embedding = self._get_embedding(query_text, model)

                future = executor.submit(
                    self._query_pool,
                    engine, model, query_text, embedding, top_k, timeout_ms
                )
                futures[future] = model

            # Collect results with timeout
            for future in as_completed(futures, timeout=timeout_ms/1000):
                model = futures[future]
                try:
                    results[model] = future.result()
                except Exception as e:
                    logger.warning(f"Pool {model} failed: {e}")

        return results

    def _query_pool(
        self,
        engine: FederatedQueryEngine,
        model: str,
        query_text: str,
        embedding: np.ndarray,
        top_k: int,
        timeout_ms: int
    ) -> PoolResponse:
        """Query a single model pool."""

        response = engine.federated_query(
            query_text=query_text,
            embedding=embedding,
            top_k=top_k,
            timeout_ms=timeout_ms
        )

        # Convert to PoolResult format with density info
        pool_results = [
            PoolResult(
                answer_id=r.answer_id,
                answer_text=r.answer_text,
                raw_score=r.raw_score,
                exp_score=r.exp_score,
                density_score=r.density_score,
                density_adjusted_score=r.aggregated_score,
                cluster_id=r.cluster_id,
                cluster_size=r.cluster_size,
                cluster_confidence=r.cluster_confidence,
                source_nodes=r.source_nodes,
                model_name=model
            )
            for r in response.results
        ]

        return PoolResponse(
            model_name=model,
            results=pool_results,
            total_results=len(pool_results),
            num_clusters=len(set(r.cluster_id for r in pool_results)),
            avg_density=np.mean([r.density_score for r in pool_results]),
            partition_sum=response.total_partition_sum,
            query_time_ms=response.total_time_ms,
            nodes_queried=response.nodes_queried,
            nodes_responded=response.nodes_responded
        )

    def _get_embedding(self, text: str, model: str) -> np.ndarray:
        """Get embedding for text using specified model."""
        if model not in self.embedding_cache:
            # Lazy load embedding function
            self.embedding_cache[model] = load_embedding_model(model)

        return self.embedding_cache[model](text)
```

## Model Weight Learning

Weights can be learned from relevance feedback:

```python
class AdaptiveModelWeights:
    """Learn optimal model weights from user feedback."""

    def __init__(self, models: List[str], learning_rate: float = 0.01):
        # Start with uniform weights
        n = len(models)
        self.weights = {m: 1.0 / n for m in models}
        self.learning_rate = learning_rate
        self.feedback_count = 0

    def update(
        self,
        query: str,
        chosen_answer: str,
        pool_rankings: Dict[str, List[str]]  # model -> [answer_ids in rank order]
    ):
        """Update weights based on which models ranked chosen answer highest."""

        rewards = {}

        for model, ranking in pool_rankings.items():
            if chosen_answer in ranking:
                rank = ranking.index(chosen_answer) + 1
                # Higher reward for better rank (inverse rank)
                rewards[model] = 1.0 / rank
            else:
                rewards[model] = 0.0

        # Normalize rewards
        total_reward = sum(rewards.values())
        if total_reward > 0:
            rewards = {m: r / total_reward for m, r in rewards.items()}

        # Gradient update toward rewarded models
        for model in self.weights:
            target = rewards.get(model, 0.0)
            current = self.weights[model]
            self.weights[model] += self.learning_rate * (target - current)

        # Renormalize weights
        total = sum(self.weights.values())
        self.weights = {m: w / total for m, w in self.weights.items()}

        self.feedback_count += 1

    def get_weights(self) -> Dict[str, float]:
        return self.weights.copy()
```

## Prolog Integration

### Validation Predicates

```prolog
% Cross-model federation options
is_valid_federation_option(cross_model(true)).
is_valid_federation_option(cross_model(false)).
is_valid_federation_option(cross_model(Opts)) :-
    is_list(Opts), maplist(is_valid_cross_model_option, Opts).

is_valid_cross_model_option(fusion_method(M)) :-
    member(M, [weighted_sum, rrf, consensus, geometric_mean, max, learned]).
is_valid_cross_model_option(rrf_k(K)) :- integer(K), K > 0.
is_valid_cross_model_option(consensus_threshold(T)) :- number(T), T >= 0, T =< 1.
is_valid_cross_model_option(consensus_boost(B)) :- number(B), B >= 1.
is_valid_cross_model_option(min_pools(N)) :- integer(N), N >= 1.
is_valid_cross_model_option(learn_weights(Bool)) :- (Bool = true ; Bool = false).
is_valid_cross_model_option(pool_timeout_ms(T)) :- integer(T), T > 0.

% Model pool configuration
is_valid_cross_model_option(pools(Pools)) :-
    is_list(Pools), maplist(is_valid_pool_config, Pools).

is_valid_pool_config(pool(Model, Opts)) :-
    atom(Model),
    is_list(Opts),
    maplist(is_valid_pool_option, Opts).

is_valid_pool_option(weight(W)) :- number(W), W > 0.
is_valid_pool_option(federation_k(K)) :- integer(K), K > 0.
is_valid_pool_option(strategy(S)) :- is_valid_aggregation_strategy(S).
is_valid_pool_option(adaptive_k(Bool)) :- (Bool = true ; Bool = false).
is_valid_pool_option(hierarchical(Bool)) :- (Bool = true ; Bool = false).
```

### Example Service Definition

```prolog
service(cross_model_kg, [
    transport(http('/kg', [port(8080)])),

    % Enable cross-model federation
    cross_model([
        fusion_method(consensus),
        consensus_threshold(0.1),
        consensus_boost(1.5),
        min_pools(2),
        learn_weights(true),
        pool_timeout_ms(30000),

        pools([
            pool('all-MiniLM-L6-v2', [
                weight(0.3),
                federation_k(5),
                strategy(density_flux),
                adaptive_k(true)
            ]),
            pool('E5-large-v2', [
                weight(0.5),
                federation_k(3),
                strategy(density_flux),
                hierarchical(true)
            ]),
            pool('BGE-base-en-v1.5', [
                weight(0.2),
                federation_k(5),
                strategy(sum)
            ])
        ])
    ])
], [
    receive(Query),
    handle_cross_model_query(Query, Response),
    respond(Response)
]).
```

## HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/kg/cross-model` | POST | Cross-model federated query |
| `/kg/cross-model/pools` | GET | List configured model pools |
| `/kg/cross-model/weights` | GET | Current model weights |
| `/kg/cross-model/feedback` | POST | Submit relevance feedback |

### Query Request

```json
{
    "query_text": "How do I parse CSV files?",
    "top_k": 10,
    "fusion_method": "consensus",
    "pool_timeout_ms": 30000
}
```

### Query Response

```json
{
    "results": [
        {
            "answer_id": "ans_123",
            "answer_text": "Use csv.reader() in Python...",
            "fused_score": 0.847,
            "num_pools": 3,
            "consensus_strength": 1.0,
            "pool_contributions": {
                "all-MiniLM-L6-v2": {
                    "probability": 0.82,
                    "density": 0.75,
                    "cluster_size": 12,
                    "rank": 1
                },
                "E5-large-v2": {
                    "probability": 0.79,
                    "density": 0.81,
                    "cluster_size": 8,
                    "rank": 2
                },
                "BGE-base-en-v1.5": {
                    "probability": 0.71,
                    "density": 0.68,
                    "cluster_size": 15,
                    "rank": 1
                }
            }
        }
    ],
    "pools_queried": ["all-MiniLM-L6-v2", "E5-large-v2", "BGE-base-en-v1.5"],
    "pools_responded": 3,
    "total_time_ms": 245.3,
    "fusion_method": "consensus"
}
```

## Implementation Phases

### Phase 6e-i: Core Cross-Model Engine

- [ ] `CrossModelConfig` and `ModelPoolConfig` data classes
- [ ] `PoolResult` and `PoolResponse` data classes
- [ ] Pool discovery and grouping by `embedding_model` metadata
- [ ] Parallel pool querying with per-model embeddings
- [ ] Basic `weighted_sum_fusion`

### Phase 6e-ii: Fusion Methods

- [ ] `rrf_fusion` - Reciprocal Rank Fusion
- [ ] `consensus_fusion` with density propagation
- [ ] `geometric_mean_fusion`
- [ ] Fusion method selection in config

### Phase 6e-iii: Prolog Integration

- [ ] Validation predicates for cross-model options
- [ ] `compile_cross_model_engine_python/2`
- [ ] HTTP endpoint generation
- [ ] Integration tests

### Phase 6e-iv: Weight Learning

- [ ] `AdaptiveModelWeights` class
- [ ] Feedback endpoint
- [ ] Weight persistence
- [ ] Online learning loop

## Testing Strategy

### Unit Tests

```python
class TestCrossModelFusion(unittest.TestCase):
    def test_weighted_sum_fusion(self):
        """Weighted sum produces expected scores."""

    def test_rrf_fusion(self):
        """RRF is robust to score miscalibration."""

    def test_consensus_boost(self):
        """Multi-pool answers get boosted."""

    def test_density_propagation(self):
        """Density scores affect final ranking."""

    def test_single_pool_fallback(self):
        """Works when only one pool responds."""


class TestCrossModelEngine(unittest.TestCase):
    def test_pool_discovery(self):
        """Nodes grouped correctly by embedding_model."""

    def test_parallel_query(self):
        """All pools queried in parallel."""

    def test_timeout_handling(self):
        """Slow pools don't block results."""

    def test_weight_learning(self):
        """Weights update from feedback."""
```

### Integration Tests

```python
def test_cross_model_e2e():
    """End-to-end with multiple embedding models."""
    # Start 3 nodes with different models
    # Query cross-model endpoint
    # Verify fusion produces meaningful ranking
```

## Performance Considerations

1. **Embedding Generation**: Must generate query embedding for each model - cache models
2. **Parallel Queries**: Query all pools concurrently to minimize latency
3. **Memory**: Each pool engine maintains its own state - be mindful of memory
4. **Timeout**: Per-pool timeout prevents slow pools from blocking

## Open Questions

1. **Embedding Model Loading**: Should the cross-model engine load all embedding models, or delegate to pool nodes?

2. **Adaptive Pool Selection**: Can we predict which pools are useful for a query type and skip others?

3. **Cross-Model Density**: Is there a way to approximate density across models using only scores?

4. **Model Quality Normalization**: Should we normalize scores by model "quality" (e.g., benchmark scores)?

## References

- [FEDERATED_QUERY_ALGEBRA.md](FEDERATED_QUERY_ALGEBRA.md) - Base federation algebra
- [DENSITY_SCORING_PROPOSAL.md](DENSITY_SCORING_PROPOSAL.md) - Density scoring within pools
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - Cormack et al.
- [Learning to Rank](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/) - Cao et al.
