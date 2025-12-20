# Adversarial Robustness for Federated Semantic Search

**Status:** Proposed
**Date:** 2024-12-19
**Priority:** Low (Phase 6f)
**Prerequisites:** Phase 1-5 Complete
**Related:** [DENSITY_SCORING_PROPOSAL.md](DENSITY_SCORING_PROPOSAL.md), [CROSS_MODEL_FEDERATION.md](CROSS_MODEL_FEDERATION.md)

## Problem Statement

The current federated KG topology assumes honest nodes. In open or semi-trusted environments, malicious nodes could manipulate results through:

1. **Density Inflation:** Inject similar embeddings to create artificial high-density clusters
2. **Consensus Manipulation:** Echo results from other nodes to boost their scores
3. **Sybil Attacks:** Create many fake nodes to dominate voting
4. **Latency Attacks:** Slow responses to bias adaptive-k calculations
5. **Embedding Poisoning:** Return subtly wrong embeddings that pass similarity checks

This document explores defense mechanisms, acknowledging that **robust approaches trade performance for security**. These techniques are more valuable for high-stakes information aggregation than for efficiency-focused deployments.

## Approach 1: Output Smoothing (Outlier Rejection)

### Concept

Use statistical methods to reject results that deviate significantly from the consensus. Similar to robust statistics (trimmed means, Winsorized estimators).

### Implementation

```python
from scipy import stats
import numpy as np

def smooth_outliers(
    results: List[PoolResult],
    method: str = "zscore",
    threshold: float = 2.5
) -> List[PoolResult]:
    """Reject results that are statistical outliers."""

    scores = np.array([r.density_adjusted_score for r in results])

    if method == "zscore":
        # Z-score based rejection
        z_scores = np.abs(stats.zscore(scores))
        mask = z_scores < threshold

    elif method == "iqr":
        # Interquartile range based rejection
        q1, q3 = np.percentile(scores, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (scores >= lower) & (scores <= upper)

    elif method == "mad":
        # Median Absolute Deviation (more robust to outliers)
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        modified_z = 0.6745 * (scores - median) / mad
        mask = np.abs(modified_z) < threshold

    return [r for r, m in zip(results, mask) if m]


def robust_aggregate(
    node_responses: List[NodeResponse],
    trim_fraction: float = 0.1
) -> float:
    """Trimmed mean aggregation - discard extreme values."""
    scores = [r.exp_score for resp in node_responses for r in resp.results]
    return stats.trim_mean(scores, trim_fraction)
```

### Embedding-Space Smoothing

For density-based methods, we can also smooth in embedding space:

```python
def embedding_outlier_detection(
    embeddings: np.ndarray,
    method: str = "lof"
) -> np.ndarray:
    """Detect outlier embeddings using Local Outlier Factor."""
    from sklearn.neighbors import LocalOutlierFactor

    lof = LocalOutlierFactor(n_neighbors=min(20, len(embeddings) - 1))
    outlier_labels = lof.fit_predict(embeddings)

    # -1 = outlier, 1 = inlier
    return outlier_labels == 1
```

### Pros/Cons

| Pros | Cons |
|------|------|
| Simple to implement | Attacker can stay within bounds |
| Low computational overhead | May reject legitimate edge cases |
| No trust infrastructure needed | Threshold tuning is tricky |

---

## Approach 2: Voluntary Collision Detection (Freenet KSK-style)

### Concept

In Freenet, **Keyword-Signed Keys (KSKs)** derive both the public and private keys from a human-readable keyword. Anyone who knows the keyword can derive the keypair and insert content. The protection mechanism is **voluntary collision detection**: once content is inserted at a KSK location, nodes refuse to overwrite it with different content.

From the [Freenet wiki](https://github.com/hyphanet/wiki/wiki/Keyword-Signed-Key):
> "There is voluntary collision detection in fred, which tries to prevent overwriting of a once-inserted page."

### How Freenet Implements This

The mechanism is **fetch-before-insert with optional conflict handling**:

1. When inserting to a KSK, fred first fetches existing data at that key
2. If data exists, the insert can optionally fail rather than overwrite
3. With Freenet's redundancy, multiple nodes may hold different versions
4. Consensus among nodes determines which version "wins"

Key source files in [hyphanet/fred](https://github.com/hyphanet/fred):
- `ClientKSK.java` - KSK key generation and insertion logic
- `SingleBlockInserter.java` - Collision detection before overwriting
- `SSKInsertSender.java` - Network protocol for SSK/KSK inserts
- `ClientPutMessage.java` - FCP interface exposing insert options

### Semantic Search Analogy

For federated semantic search, we can apply the same principle: **once an answer is established with high confidence in a semantic region, new conflicting results shouldn't easily displace it**.

This is "semantic collision detection" - protecting established consensus from late-arriving adversarial results.

### Consensus with Redundancy

Like Freenet's redundancy model where multiple nodes may hold different versions and consensus determines the winner, our semantic collision detector can:

1. **Track version counts**: How many nodes report each answer in a region
2. **Require quorum for locking**: Only lock when N nodes agree
3. **Allow version superseding**: Higher-consensus version can displace lower

```python
@dataclass
class VersionedAnswer:
    """Answer with version tracking for consensus."""
    answer_hash: str
    answer_text: str
    node_votes: Set[str]  # Nodes that returned this answer
    first_seen: float

    @property
    def vote_count(self) -> int:
        return len(self.node_votes)


class ConsensusCollisionDetector(SemanticCollisionDetector):
    """Collision detection with consensus-based locking."""

    def __init__(self, quorum: int = 3, supersede_margin: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.quorum = quorum
        self.supersede_margin = supersede_margin
        # Track all versions per region
        self.versions: Dict[str, Dict[str, VersionedAnswer]] = {}

    def register_vote(
        self,
        result: AggregatedResult,
        embedding: np.ndarray,
        node_id: str
    ):
        """Register a node's vote for an answer."""
        region = self._compute_region(embedding)

        if region not in self.versions:
            self.versions[region] = {}

        versions = self.versions[region]
        ahash = result.answer_hash

        if ahash not in versions:
            versions[ahash] = VersionedAnswer(
                answer_hash=ahash,
                answer_text=result.answer_text,
                node_votes={node_id},
                first_seen=time.time()
            )
        else:
            versions[ahash].node_votes.add(node_id)

        # Check if this version should become established
        self._update_established(region)

    def _update_established(self, region: str):
        """Update established answer based on consensus."""
        versions = self.versions.get(region, {})
        if not versions:
            return

        # Find version with most votes
        best = max(versions.values(), key=lambda v: v.vote_count)

        if best.vote_count < self.quorum:
            return  # No quorum yet

        current = self.established.get(region)

        if current is None:
            # First to reach quorum
            self.established[region] = EstablishedAnswer(
                answer_hash=best.answer_hash,
                answer_text=best.answer_text,
                semantic_region=region,
                confidence=best.vote_count / self.quorum,
                first_seen=best.first_seen,
                node_count=best.vote_count,
                locked=True
            )
        elif best.answer_hash != current.answer_hash:
            # Different answer - check if it supersedes
            current_votes = versions.get(current.answer_hash, VersionedAnswer("", "", set(), 0)).vote_count
            if best.vote_count >= current_votes + self.supersede_margin:
                # New consensus supersedes old
                self.established[region] = EstablishedAnswer(
                    answer_hash=best.answer_hash,
                    answer_text=best.answer_text,
                    semantic_region=region,
                    confidence=best.vote_count / self.quorum,
                    first_seen=best.first_seen,
                    node_count=best.vote_count,
                    locked=True
                )
```

### Combining Consensus with Trust Rankings

With network redundancy and trust rankings from Approach 3/4, votes can be **trust-weighted**:

```python
class TrustWeightedConsensusDetector(ConsensusCollisionDetector):
    """Consensus voting weighted by node trust scores."""

    def __init__(self, trust_manager: 'DirectTrustManager', **kwargs):
        super().__init__(**kwargs)
        self.trust_manager = trust_manager

    def register_vote(
        self,
        result: AggregatedResult,
        embedding: np.ndarray,
        node_id: str
    ):
        """Register trust-weighted vote."""
        region = self._compute_region(embedding)
        trust = self.trust_manager.get_trust(node_id)

        if region not in self.versions:
            self.versions[region] = {}

        versions = self.versions[region]
        ahash = result.answer_hash

        if ahash not in versions:
            versions[ahash] = VersionedAnswer(
                answer_hash=ahash,
                answer_text=result.answer_text,
                node_votes={node_id},
                first_seen=time.time()
            )
            versions[ahash].trust_sum = trust  # Track weighted sum
        else:
            versions[ahash].node_votes.add(node_id)
            versions[ahash].trust_sum += trust

        self._update_established_by_trust(region)

    def _update_established_by_trust(self, region: str):
        """Lock based on trust-weighted votes, not raw count."""
        versions = self.versions.get(region, {})
        if not versions:
            return

        # Find version with highest trust-weighted score
        best = max(versions.values(),
                   key=lambda v: getattr(v, 'trust_sum', v.vote_count))

        trust_quorum = self.quorum * 0.5  # Trust score threshold
        if getattr(best, 'trust_sum', 0) >= trust_quorum:
            # Highly trusted nodes agreeing = lock sooner
            self._lock_answer(region, best)
```

This synergy means:
- **High-trust nodes** can establish consensus faster (trust_quorum instead of raw count)
- **Low-trust nodes** need more agreement to overcome trusted consensus
- **Sybil resistance**: Many untrusted nodes can't outvote few trusted ones

### Implementation

```python
from dataclasses import dataclass, field
from typing import Dict, Optional
import hashlib

@dataclass
class EstablishedAnswer:
    """An answer that has been established with sufficient confidence."""
    answer_hash: str
    answer_text: str
    semantic_region: str  # Hash of quantized centroid
    confidence: float
    first_seen: float
    node_count: int
    locked: bool = False  # Once locked, cannot be displaced

    def should_lock(self, min_confidence: float = 0.8, min_nodes: int = 3) -> bool:
        """Determine if this answer should be locked."""
        return self.confidence >= min_confidence and self.node_count >= min_nodes


class SemanticCollisionDetector:
    """
    Voluntary collision detection for semantic search.

    Once an answer achieves high confidence in a semantic region,
    it becomes "established" and new conflicting answers are rejected.
    """

    def __init__(
        self,
        lock_threshold: float = 0.8,
        min_nodes_to_lock: int = 3,
        region_granularity: int = 100  # Quantization level
    ):
        self.lock_threshold = lock_threshold
        self.min_nodes_to_lock = min_nodes_to_lock
        self.region_granularity = region_granularity

        # Established answers by semantic region
        self.established: Dict[str, EstablishedAnswer] = {}

    def _compute_region(self, embedding: np.ndarray) -> str:
        """Compute semantic region hash from embedding."""
        # Quantize to region
        quantized = np.round(
            embedding * self.region_granularity
        ).astype(np.int32)
        return hashlib.sha256(quantized.tobytes()).hexdigest()[:16]

    def check_collision(
        self,
        new_result: AggregatedResult,
        embedding: np.ndarray
    ) -> tuple[bool, Optional[str]]:
        """
        Check if new result collides with established answer.

        Returns:
            (allowed, reason) - allowed=True if result should be accepted
        """
        region = self._compute_region(embedding)

        if region not in self.established:
            return True, None

        established = self.established[region]

        if not established.locked:
            return True, None

        # Region is locked - check if same answer or collision
        if new_result.answer_hash == established.answer_hash:
            # Same answer, allow (reinforces consensus)
            return True, None
        else:
            # Different answer trying to enter locked region
            return False, f"Collision with established answer in region {region}"

    def register_result(
        self,
        result: AggregatedResult,
        embedding: np.ndarray,
        confidence: float,
        node_count: int
    ):
        """Register a result, potentially establishing it."""
        region = self._compute_region(embedding)

        if region not in self.established:
            # First answer in this region
            answer = EstablishedAnswer(
                answer_hash=result.answer_hash,
                answer_text=result.answer_text,
                semantic_region=region,
                confidence=confidence,
                first_seen=time.time(),
                node_count=node_count
            )
            if answer.should_lock(self.lock_threshold, self.min_nodes_to_lock):
                answer.locked = True
            self.established[region] = answer

        else:
            established = self.established[region]
            if result.answer_hash == established.answer_hash:
                # Same answer - update confidence
                established.confidence = max(established.confidence, confidence)
                established.node_count = max(established.node_count, node_count)
                if not established.locked and established.should_lock(
                    self.lock_threshold, self.min_nodes_to_lock
                ):
                    established.locked = True


def collision_aware_aggregate(
    responses: List[NodeResponse],
    detector: SemanticCollisionDetector
) -> List[AggregatedResult]:
    """Aggregate with collision detection."""

    results = standard_aggregate(responses)
    filtered = []

    for result in results:
        embedding = result.semantic_centroid
        if embedding is None:
            filtered.append(result)
            continue

        allowed, reason = detector.check_collision(result, embedding)
        if allowed:
            detector.register_result(
                result, embedding,
                confidence=result.density_score,
                node_count=len(result.source_nodes)
            )
            filtered.append(result)
        else:
            log_collision_rejection(result, reason)

    return filtered
```

### Key Differences from My Original Description

| Aspect | Wrong (my original) | Correct (KSK-style) |
|--------|---------------------|---------------------|
| Protection mechanism | Computational difficulty | Voluntary collision detection |
| Who can insert | Only content creator | Anyone who knows keyword |
| What prevents attacks | Hash binding | Nodes refuse to overwrite |
| Analogy for us | Make adversarial content expensive | Protect established consensus |

### Pros/Cons

| Pros | Cons |
|------|------|
| Simple to implement | First-mover advantage |
| Protects established consensus | Legitimate updates blocked |
| Low computational overhead | Region granularity tuning |
| Matches Freenet's proven approach | Requires coordination on locking policy |

---

## Approach 3: Web of Trust

### Concept

Nodes build reputation over time. Trust is transitive but decays with distance. Similar to PGP's web of trust, but adapted for automated semantic search.

### Trust Models

#### 3a. Direct Trust (Simple)

```python
@dataclass
class NodeTrust:
    """Direct trust score for a node."""
    node_id: str
    trust_score: float  # 0.0 to 1.0
    successful_queries: int
    failed_verifications: int
    last_updated: datetime

    def update(self, success: bool, weight: float = 0.1):
        """Exponential moving average update."""
        outcome = 1.0 if success else 0.0
        self.trust_score = (1 - weight) * self.trust_score + weight * outcome
        if success:
            self.successful_queries += 1
        else:
            self.failed_verifications += 1
        self.last_updated = datetime.now()


class DirectTrustManager:
    """Simple direct trust without transitivity."""

    def __init__(self, default_trust: float = 0.5):
        self.trust_scores: Dict[str, NodeTrust] = {}
        self.default_trust = default_trust

    def get_trust(self, node_id: str) -> float:
        if node_id in self.trust_scores:
            return self.trust_scores[node_id].trust_score
        return self.default_trust

    def weight_by_trust(
        self,
        responses: List[NodeResponse]
    ) -> List[Tuple[NodeResponse, float]]:
        """Weight responses by node trust."""
        return [
            (r, self.get_trust(r.source_node))
            for r in responses
        ]
```

#### 3b. Transitive Trust (PGP-style)

```python
class TransitiveTrustManager:
    """Web of trust with transitive relationships."""

    def __init__(self, max_depth: int = 3, decay: float = 0.7):
        self.direct_trust: Dict[str, Dict[str, float]] = {}  # truster -> trustee -> score
        self.max_depth = max_depth
        self.decay = decay  # Trust decays with each hop

    def set_trust(self, truster: str, trustee: str, score: float):
        """Set direct trust relationship."""
        if truster not in self.direct_trust:
            self.direct_trust[truster] = {}
        self.direct_trust[truster][trustee] = score

    def compute_trust(self, from_node: str, to_node: str) -> float:
        """Compute transitive trust using BFS with decay."""
        if from_node == to_node:
            return 1.0

        # BFS with trust accumulation
        visited = {from_node}
        frontier = [(from_node, 1.0)]  # (node, accumulated_trust)

        for depth in range(self.max_depth):
            next_frontier = []

            for current, current_trust in frontier:
                if current not in self.direct_trust:
                    continue

                for neighbor, edge_trust in self.direct_trust[current].items():
                    if neighbor in visited:
                        continue

                    # Trust decays with each hop
                    new_trust = current_trust * edge_trust * self.decay

                    if neighbor == to_node:
                        return new_trust

                    visited.add(neighbor)
                    next_frontier.append((neighbor, new_trust))

            frontier = next_frontier

        return 0.0  # No trust path found
```

#### 3c. FMS-style Trust (Efficient Web of Trust)

The Freenet Messaging System (FMS) used an efficient trust propagation approach:

```python
class FMSTrustManager:
    """
    FMS-style trust management.

    Key insights from FMS:
    1. Trust is message-based, not identity-based
    2. Positive trust requires explicit action
    3. Negative trust (distrust) propagates faster
    4. Trust scores are integers (-100 to +100) for efficiency
    """

    def __init__(self):
        # Trust lists: node -> (trustee -> score)
        self.trust_lists: Dict[str, Dict[str, int]] = {}
        # Cached computed trust
        self.trust_cache: Dict[Tuple[str, str], int] = {}

    def publish_trust_list(self, node_id: str, trust_list: Dict[str, int]):
        """Node publishes its trust list (like FMS trust messages)."""
        # Validate scores in range
        validated = {
            k: max(-100, min(100, v))
            for k, v in trust_list.items()
        }
        self.trust_lists[node_id] = validated
        # Invalidate cache
        self.trust_cache = {}

    def compute_trust(
        self,
        my_node: str,
        target_node: str,
        trusted_introducers: Optional[List[str]] = None
    ) -> int:
        """
        Compute trust score for target.

        Uses weighted average of trusted introducers' opinions.
        """
        cache_key = (my_node, target_node)
        if cache_key in self.trust_cache:
            return self.trust_cache[cache_key]

        if my_node == target_node:
            return 100

        # Direct trust
        if my_node in self.trust_lists:
            if target_node in self.trust_lists[my_node]:
                score = self.trust_lists[my_node][target_node]
                self.trust_cache[cache_key] = score
                return score

        # Transitive trust through introducers
        if trusted_introducers is None:
            # Use nodes I directly trust as introducers
            trusted_introducers = [
                n for n, s in self.trust_lists.get(my_node, {}).items()
                if s > 0
            ]

        if not trusted_introducers:
            return 0

        # Weighted average of introducer opinions
        weighted_sum = 0
        weight_total = 0

        for introducer in trusted_introducers:
            # How much do I trust this introducer?
            introducer_trust = self.trust_lists.get(my_node, {}).get(introducer, 0)
            if introducer_trust <= 0:
                continue

            # What does introducer think of target?
            target_opinion = self.trust_lists.get(introducer, {}).get(target_node, 0)

            # Weight by introducer trust
            weighted_sum += introducer_trust * target_opinion
            weight_total += introducer_trust

        if weight_total == 0:
            return 0

        score = weighted_sum // weight_total
        self.trust_cache[cache_key] = score
        return score

    def filter_by_trust(
        self,
        my_node: str,
        responses: List[NodeResponse],
        min_trust: int = 0
    ) -> List[NodeResponse]:
        """Filter responses to only include trusted nodes."""
        return [
            r for r in responses
            if self.compute_trust(my_node, r.source_node) >= min_trust
        ]
```

### Pros/Cons

| Approach | Pros | Cons |
|----------|------|------|
| Direct Trust | Simple, fast | No newcomer bootstrapping |
| Transitive (PGP) | Handles sparse trust | Complex, slow BFS |
| FMS-style | Efficient, practical | Requires trust list publishing |

---

## Approach 4: Reputation with Stake

### Concept

Nodes stake resources (compute time, storage, or tokens) that can be slashed for misbehavior. Similar to proof-of-stake systems.

### Implementation Sketch

```python
@dataclass
class NodeStake:
    """Node's staked resources."""
    node_id: str
    staked_compute_hours: float
    staked_storage_gb: float
    slashed_amount: float = 0.0

    @property
    def effective_stake(self) -> float:
        return self.staked_compute_hours + self.staked_storage_gb - self.slashed_amount


class StakeWeightedAggregation:
    """Weight node contributions by stake."""

    def __init__(self, stakes: Dict[str, NodeStake]):
        self.stakes = stakes

    def aggregate(self, responses: List[NodeResponse]) -> List[AggregatedResult]:
        """Stake-weighted aggregation."""
        # Weight each response by node's effective stake
        weights = {
            r.source_node: self.stakes.get(r.source_node, NodeStake(r.source_node, 0, 0)).effective_stake
            for r in responses
        }

        total_stake = sum(weights.values())
        if total_stake == 0:
            # Fall back to uniform
            return standard_aggregate(responses)

        # Weighted merge
        return stake_weighted_merge(responses, weights, total_stake)

    def slash(self, node_id: str, amount: float, reason: str):
        """Slash a node's stake for misbehavior."""
        if node_id in self.stakes:
            self.stakes[node_id].slashed_amount += amount
            log_slash_event(node_id, amount, reason)
```

### Pros/Cons

| Pros | Cons |
|------|------|
| Economic incentive for good behavior | Barrier to entry for new nodes |
| Clear punishment mechanism | Requires stake management |
| Sybil-resistant (stake is limited) | May favor wealthy actors |

---

## Approach 5: Cryptographic Attestation

### Concept

Nodes sign their results with private keys. Aggregators can verify signatures and track signing key reputation separately from node identity.

```python
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

@dataclass
class SignedResult:
    """Result with cryptographic signature."""
    result: NodeResult
    public_key: bytes
    signature: bytes
    timestamp: float

    def verify(self) -> bool:
        """Verify the signature."""
        try:
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(self.public_key)
            message = self._serialize_for_signing()
            public_key.verify(self.signature, message)
            return True
        except Exception:
            return False

    def _serialize_for_signing(self) -> bytes:
        """Deterministic serialization for signing."""
        data = {
            'answer_hash': self.result.answer_hash,
            'exp_score': self.result.exp_score,
            'timestamp': self.timestamp
        }
        return json.dumps(data, sort_keys=True).encode()


class SigningNode:
    """Node that signs its results."""

    def __init__(self):
        self.private_key = ed25519.Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

    def sign_result(self, result: NodeResult) -> SignedResult:
        timestamp = time.time()
        signed = SignedResult(
            result=result,
            public_key=self.public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            ),
            signature=b'',  # Placeholder
            timestamp=timestamp
        )
        message = signed._serialize_for_signing()
        signed.signature = self.private_key.sign(message)
        return signed
```

---

## Recommended Phased Implementation

Given the performance costs, we recommend a phased approach:

### Phase 6f-i: Output Smoothing (Low Cost)

- Implement Z-score and MAD outlier rejection
- Add to aggregation pipeline as optional filter
- Minimal performance impact

### Phase 6f-ii: Direct Trust (Medium Cost)

- Track per-node success/failure rates
- Weight responses by trust score
- Optional verification sampling

### Phase 6f-iii: FMS-style Web of Trust (Higher Cost)

- Trust list publication via discovery metadata
- Transitive trust computation
- Trust-filtered federation

### Phase 6f-iv: Full Attestation (Highest Cost)

- Cryptographic signatures on results
- Content-addressable verification
- Key-based reputation tracking

## Prolog Configuration

```prolog
service(secure_kg_node, [
    transport(http('/kg', [port(8080)])),

    % Phase 6f: Adversarial robustness
    adversarial_protection([
        % Output smoothing
        outlier_rejection(enabled),
        outlier_method(mad),  % zscore | iqr | mad
        outlier_threshold(2.5),

        % Trust
        trust_model(fms),  % none | direct | transitive | fms
        min_trust_score(10),  % -100 to 100

        % Verification
        verification_sampling(0.1),  % 10% of results
        require_signatures(false)
    ])
], Handler).
```

## Performance Considerations

| Feature | Latency Impact | Memory Impact | When to Use |
|---------|---------------|---------------|-------------|
| Output smoothing | ~1ms | O(n) | Always (cheap) |
| Direct trust | ~1ms | O(nodes) | Production |
| Transitive trust | ~10-100ms | O(nodes²) | High-stakes |
| Verification | ~50-200ms/result | O(1) | Sensitive data |
| Signatures | ~5ms/sign+verify | O(1) | Audit requirements |

## Open Questions

1. **Bootstrap Problem:** How do new nodes gain initial trust?
2. **Trust Decay:** Should old trust scores decay over time?
3. **Collusion Detection:** How to detect coordinated malicious nodes?
4. **Threshold Selection:** Optimal outlier rejection thresholds?
5. **Hybrid Approaches:** Can we combine cheap+expensive methods adaptively?

## References

- Freenet KSK design: https://freenetproject.org/
- FMS trust system: Historical Freenet forums
- PGP Web of Trust: RFC 4880
- Robust Statistics: Huber, P.J. "Robust Statistics" (1981)
- Local Outlier Factor: Breunig et al. (2000)

## Next Steps

After this proposal is reviewed:
1. Implement Phase 6f-i (output smoothing) as proof of concept
2. Benchmark performance impact
3. Decide on trust model complexity based on deployment requirements
