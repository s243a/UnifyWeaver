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

## Approach 1: Output Smoothing (Soft Collision Detection)

### Concept

Use statistical methods to reject results that deviate significantly from the consensus. Similar to robust statistics (trimmed means, Winsorized estimators).

**Key Insight:** Outliers are **soft collisions** with the established consensus. This connects directly to the KSK-style hard collision detection in Approach 2:

| Collision Type | Mechanism | Rejection |
|---------------|-----------|-----------|
| **Soft** (outliers) | Statistical deviation from consensus | Gradual (Z-score, MAD) |
| **Hard** (KSK-style) | Different answer in locked region | Binary (reject/accept) |

Both protect established consensus - outlier smoothing is the "soft" first line of defense, while collision detection is the "hard" lock once consensus is strong enough.

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

The Freenet Messaging System (FMS) used an efficient trust propagation approach with a **two-dimensional trust model**:

**Two Trust Dimensions (from FMS design):**

| Dimension | Purpose | Semantic Search Analogy |
|-----------|---------|------------------------|
| **Local Message Trust** | Trust content quality (not spam/malicious) | Trust node's results are accurate |
| **Local Trust List Trust** | Trust judgment about other users | Trust node's trust ratings of peers |

This separation is crucial: a node might return good results (high message trust) but have poor judgment about which other nodes to trust (low trust list trust). Or vice versa.

**Efficiency from FMS:**
- You only download messages from: (1) directly trusted identities, (2) identities trusted by people you trust
- Spam/malicious actors naturally excluded from your view
- No centralized moderation needed

```python
@dataclass
class TwoDimensionalTrust:
    """FMS-style two-dimensional trust score."""
    message_trust: int      # -100 to +100: trust for content quality
    trust_list_trust: int   # -100 to +100: trust for judgment of others

    def effective_trust(self) -> int:
        """Combined trust score for simple operations."""
        return (self.message_trust + self.trust_list_trust) // 2


class FMSTrustManager:
    """
    FMS-style trust management with two-dimensional trust.

    Key insights from FMS:
    1. Two separate trust dimensions (message vs trust list)
    2. Positive trust requires explicit action
    3. Negative trust (distrust) propagates faster
    4. Trust scores are integers (-100 to +100) for efficiency
    5. Download from trusted + trusted-by-trusted (transitive)

    References:
    - https://github.com/SeekingFor/FMS
    - https://freesocial.draketo.de/fms_en.html
    """

    def __init__(self):
        # Trust lists: node -> (trustee -> TwoDimensionalTrust)
        self.trust_lists: Dict[str, Dict[str, TwoDimensionalTrust]] = {}
        # Cached computed trust
        self.trust_cache: Dict[Tuple[str, str], TwoDimensionalTrust] = {}

    def publish_trust_list(
        self,
        node_id: str,
        trust_list: Dict[str, Tuple[int, int]]
    ):
        """
        Node publishes its trust list (like FMS trust messages).

        Args:
            node_id: Publishing node
            trust_list: Dict of trustee -> (message_trust, trust_list_trust)
        """
        validated = {}
        for trustee, (msg_trust, list_trust) in trust_list.items():
            validated[trustee] = TwoDimensionalTrust(
                message_trust=max(-100, min(100, msg_trust)),
                trust_list_trust=max(-100, min(100, list_trust))
            )
        self.trust_lists[node_id] = validated
        # Invalidate cache
        self.trust_cache = {}

    def compute_trust(
        self,
        my_node: str,
        target_node: str,
        trusted_introducers: Optional[List[str]] = None
    ) -> TwoDimensionalTrust:
        """
        Compute two-dimensional trust score for target.

        Uses weighted average of trusted introducers' opinions,
        weighted by introducers' trust_list_trust (their judgment quality).
        """
        cache_key = (my_node, target_node)
        if cache_key in self.trust_cache:
            return self.trust_cache[cache_key]

        if my_node == target_node:
            return TwoDimensionalTrust(100, 100)

        # Direct trust
        if my_node in self.trust_lists:
            if target_node in self.trust_lists[my_node]:
                trust = self.trust_lists[my_node][target_node]
                self.trust_cache[cache_key] = trust
                return trust

        # Transitive trust through introducers
        if trusted_introducers is None:
            # Use nodes I directly trust (positive trust_list_trust) as introducers
            trusted_introducers = [
                n for n, t in self.trust_lists.get(my_node, {}).items()
                if t.trust_list_trust > 0  # Trust their judgment
            ]

        if not trusted_introducers:
            return TwoDimensionalTrust(0, 0)

        # Weighted average of introducer opinions
        # Key: weight by trust_list_trust (how much I trust their judgment)
        msg_weighted_sum = 0
        list_weighted_sum = 0
        weight_total = 0

        for introducer in trusted_introducers:
            my_trust_of_introducer = self.trust_lists.get(my_node, {}).get(introducer)
            if my_trust_of_introducer is None:
                continue

            # Weight by how much I trust introducer's JUDGMENT
            introducer_weight = my_trust_of_introducer.trust_list_trust
            if introducer_weight <= 0:
                continue

            # What does introducer think of target?
            target_trust = self.trust_lists.get(introducer, {}).get(target_node)
            if target_trust is None:
                continue

            # Weighted contribution
            msg_weighted_sum += introducer_weight * target_trust.message_trust
            list_weighted_sum += introducer_weight * target_trust.trust_list_trust
            weight_total += introducer_weight

        if weight_total == 0:
            return TwoDimensionalTrust(0, 0)

        result = TwoDimensionalTrust(
            message_trust=msg_weighted_sum // weight_total,
            trust_list_trust=list_weighted_sum // weight_total
        )
        self.trust_cache[cache_key] = result
        return result

    def filter_by_trust(
        self,
        my_node: str,
        responses: List[NodeResponse],
        min_message_trust: int = 0
    ) -> List[NodeResponse]:
        """Filter responses to only include nodes with sufficient message trust."""
        return [
            r for r in responses
            if self.compute_trust(my_node, r.source_node).message_trust >= min_message_trust
        ]

    def weight_by_trust(
        self,
        my_node: str,
        responses: List[NodeResponse]
    ) -> List[Tuple[NodeResponse, float]]:
        """Weight responses by message trust (result quality)."""
        weighted = []
        for r in responses:
            trust = self.compute_trust(my_node, r.source_node)
            # Normalize -100..+100 to 0..1
            weight = (trust.message_trust + 100) / 200.0
            weighted.append((r, weight))
        return weighted
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

## Approach 6: Cryptocurrency/Blockchain Insights

Cryptocurrency networks have developed robust mechanisms for achieving consensus among untrusted parties. Key insights applicable to semantic search:

### Proof-of-Stake Parallels

| Crypto Concept | Semantic Search Analog |
|---------------|----------------------|
| **Stake as collateral** | Node invests resources (storage, compute) |
| **Slashing** | Reduce trust score for bad behavior |
| **Validator selection** | Weight node voting by stake/trust |
| **Finality** | Lock answer after sufficient consensus |

### Nakamoto Consensus Insights

- **Longest chain wins**: In semantic search, "most nodes agree" wins
- **51% attack**: Need majority to corrupt consensus → our quorum mechanisms
- **Economic incentive**: Make attacks expensive → stake-weighted voting

### Delegated Proof of Stake (DPoS) Parallel

DPoS elects "witnesses" who validate - similar to FMS's trust list trust:
- Nodes with high `trust_list_trust` are like elected witnesses
- Their opinions on other nodes carry more weight
- Reduces computation by focusing on trusted introducers

### Byzantine Fault Tolerance Insights

- BFT tolerates up to 1/3 malicious nodes → our `quorum` and `supersede_margin`
- Practical BFT uses pre-prepare/prepare/commit phases → our soft collision → hard lock progression
- View change protocol → our `supersede_margin` allowing new consensus to override old

### Applicable Mechanisms

```python
class StakeAwareConsensus:
    """Combine stake-weighted voting with collision detection."""

    def __init__(self, stakes: Dict[str, NodeStake], collision_detector: ConsensusCollisionDetector):
        self.stakes = stakes
        self.detector = collision_detector

    def register_stake_weighted_vote(
        self,
        result: AggregatedResult,
        embedding: np.ndarray,
        node_id: str
    ):
        """Vote weight = sqrt(effective_stake) to reduce plutocracy."""
        stake = self.stakes.get(node_id, NodeStake(node_id, 0, 0))
        vote_weight = max(1, int(math.sqrt(stake.effective_stake)))

        # Register multiple "virtual votes" based on stake
        for _ in range(vote_weight):
            self.detector.register_vote(result, embedding, f"{node_id}_{_}")
```

---

## Recommended Phased Implementation

Given the performance costs and user preference for consensus-based approaches first, we recommend:

### Phase 6f-i: Output Smoothing (Low Cost) - Soft Collisions

- Implement Z-score and MAD outlier rejection
- Add to aggregation pipeline as optional filter
- Minimal performance impact
- **Conceptual foundation**: Outliers = soft collisions with consensus

### Phase 6f-ii: Consensus Collision Detection (Low Cost) - Hard Collisions

**IMPLEMENT FIRST** - Core KSK-style mechanism:
- `SemanticCollisionDetector` with region locking
- `ConsensusCollisionDetector` with quorum-based voting
- `TrustWeightedConsensusDetector` for integration with trust
- No external dependencies, pure consensus

### Phase 6f-iii: Direct Trust (Medium Cost)

- Track per-node success/failure rates
- Weight responses by trust score
- Optional verification sampling

### Phase 6f-iv: FMS-style Web of Trust (Higher Cost)

- Two-dimensional trust (message trust + trust list trust)
- Trust list publication via discovery metadata
- Transitive trust computation weighted by trust_list_trust
- Trust-filtered federation

### Phase 6f-v: Full Attestation (Highest Cost)

- Cryptographic signatures on results (optional for KSK-style)
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
- KSK Wiki: https://github.com/hyphanet/wiki/wiki/Keyword-Signed-Key
- Fred source code: https://github.com/hyphanet/fred
  - `ClientKSK.java` - KSK key handling
  - `SingleBlockInserter.java` - Collision detection
  - `SSKInsertSender.java` - Insert protocol
- FMS source: https://github.com/SeekingFor/FMS
- FMS documentation: https://freesocial.draketo.de/fms_en.html
- Web of Trust plugin: https://github.com/hyphanet/plugin-WebOfTrust
- PGP Web of Trust: RFC 4880
- Robust Statistics: Huber, P.J. "Robust Statistics" (1981)
- Local Outlier Factor: Breunig et al. (2000)
- Practical BFT: Castro & Liskov (1999)
- Proof of Stake: Ethereum 2.0 specification

## Implementation Plan

Starting with the easiest methods first:

### Step 1: Core Data Structures (30 min)

Create `src/unifyweaver/targets/python_runtime/adversarial_robustness.py`:

```python
# Basic dataclasses
- EstablishedAnswer
- VersionedAnswer
- TwoDimensionalTrust (for later)
```

### Step 2: Output Smoothing - Soft Collisions (1 hour)

Easiest to implement, no state management:

```python
# Functions to implement:
- smooth_outliers(results, method='zscore', threshold=2.5)
- robust_aggregate(responses, trim_fraction=0.1)
- embedding_outlier_detection(embeddings, method='lof')  # Optional, needs sklearn
```

Tests:
- `test_zscore_outlier_rejection`
- `test_mad_outlier_rejection`
- `test_iqr_outlier_rejection`
- `test_trimmed_mean_aggregation`

### Step 3: Semantic Collision Detector - Hard Collisions (1.5 hours)

Core KSK-style mechanism:

```python
# Classes to implement:
- SemanticCollisionDetector
  - _compute_region(embedding) -> str
  - check_collision(result, embedding) -> (allowed, reason)
  - register_result(result, embedding, confidence, node_count)

- collision_aware_aggregate(responses, detector) -> List[AggregatedResult]
```

Tests:
- `test_region_computation`
- `test_first_answer_establishes`
- `test_same_answer_reinforces`
- `test_different_answer_rejected_in_locked_region`
- `test_unlocked_region_accepts_different`

### Step 4: Consensus Collision Detector (1 hour)

Multi-node voting:

```python
# Classes to implement:
- ConsensusCollisionDetector(SemanticCollisionDetector)
  - register_vote(result, embedding, node_id)
  - _update_established(region)
```

Tests:
- `test_quorum_required_to_lock`
- `test_supersede_margin_required`
- `test_multiple_versions_tracked`

### Step 5: Direct Trust Manager (45 min)

Simple per-node trust:

```python
# Classes to implement:
- NodeTrust
- DirectTrustManager
  - get_trust(node_id) -> float
  - update_trust(node_id, success: bool)
  - weight_by_trust(responses) -> List[(response, weight)]
```

Tests:
- `test_default_trust`
- `test_trust_update_ema`
- `test_weight_by_trust`

### Step 6: Trust-Weighted Consensus (30 min)

Combine Steps 4 + 5:

```python
# Classes to implement:
- TrustWeightedConsensusDetector(ConsensusCollisionDetector)
  - register_vote with trust weighting
  - _update_established_by_trust
```

Tests:
- `test_high_trust_locks_faster`
- `test_low_trust_cant_outvote_trusted`

### Step 7: Prolog Validation Predicates (30 min)

Add to `service_validation.pl`:

```prolog
- is_valid_adversarial_option/1
- is_valid_outlier_method/1
- is_valid_trust_model/1
```

### Step 8: Integration with Federated Query (1 hour)

Wire into `federated_query.py`:

```python
# Extend FederatedQueryEngine:
- Optional adversarial_config parameter
- Apply smoothing in aggregation pipeline
- Track collision detector state
```

### Total Estimate: ~7 hours

| Step | Difficulty | Dependencies |
|------|------------|--------------|
| 1. Data structures | Easy | None |
| 2. Output smoothing | Easy | Step 1 |
| 3. Semantic collision | Medium | Steps 1, 2 |
| 4. Consensus collision | Medium | Step 3 |
| 5. Direct trust | Easy | Step 1 |
| 6. Trust-weighted | Medium | Steps 4, 5 |
| 7. Prolog validation | Easy | None |
| 8. Integration | Medium | Steps 2-6 |

## Next Steps

After this proposal is reviewed:
1. Create feature branch `feat/kg-topology-phase6f-adversarial`
2. Implement Steps 1-4 (core collision detection)
3. Run tests, benchmark performance
4. Implement Steps 5-6 (trust integration)
5. Add Prolog validation and integration
