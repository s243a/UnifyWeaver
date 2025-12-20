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

## Approach 2: Content-Addressable Keys (Freenet KSK-style)

### Concept

In Freenet, **Keyword-Signed Keys (KSKs)** derive the storage location from the content itself. This makes it computationally expensive to create content that maps to a specific location while saying something different.

For semantic search, we can apply a similar principle: **derive a verification hash from the semantic content**, making it expensive to create adversarial content that passes verification.

### Implementation

```python
import hashlib
from dataclasses import dataclass

@dataclass
class VerifiedResult:
    """Result with content-addressable verification."""
    answer_text: str
    answer_hash: str
    embedding: np.ndarray

    # Verification fields
    content_hash: str  # SHA-256 of normalized text
    embedding_hash: str  # Hash of quantized embedding
    combined_key: str  # Derived from both

    @classmethod
    def create(cls, text: str, embedding: np.ndarray) -> 'VerifiedResult':
        # Normalize text (lowercase, strip, collapse whitespace)
        normalized = ' '.join(text.lower().split())
        content_hash = hashlib.sha256(normalized.encode()).hexdigest()

        # Quantize embedding to reduce floating point variance
        quantized = np.round(embedding * 1000).astype(np.int32)
        embedding_hash = hashlib.sha256(quantized.tobytes()).hexdigest()

        # Combined key binds text to embedding
        combined = hashlib.sha256(
            (content_hash + embedding_hash).encode()
        ).hexdigest()

        return cls(
            answer_text=text,
            answer_hash=content_hash[:16],
            embedding=embedding,
            content_hash=content_hash,
            embedding_hash=embedding_hash,
            combined_key=combined
        )

    def verify(self, embedding_model) -> bool:
        """Verify that the embedding matches the content."""
        recomputed = embedding_model.encode(self.answer_text)
        quantized = np.round(recomputed * 1000).astype(np.int32)
        expected_hash = hashlib.sha256(quantized.tobytes()).hexdigest()
        return expected_hash == self.embedding_hash
```

### Verification at Aggregation

```python
def verified_aggregate(
    responses: List[NodeResponse],
    embedding_model,
    verification_sample_rate: float = 0.1
) -> List[AggregatedResult]:
    """Aggregate with probabilistic verification."""

    verified_results = []

    for response in responses:
        for result in response.results:
            # Probabilistically verify
            if random.random() < verification_sample_rate:
                if not result.verify(embedding_model):
                    # Log and reject
                    log_verification_failure(response.source_node, result)
                    continue

            verified_results.append(result)

    return standard_aggregate(verified_results)
```

### Pros/Cons

| Pros | Cons |
|------|------|
| Cryptographic binding of content to embedding | Requires re-embedding to verify |
| Makes embedding poisoning expensive | Verification is slow |
| Deterministic verification | Different models = different hashes |

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
