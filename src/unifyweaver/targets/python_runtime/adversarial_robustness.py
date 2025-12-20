"""
Adversarial Robustness for Federated Semantic Search

Phase 6f implementation providing:
- Output smoothing (soft collision detection)
- Semantic collision detection (hard collision detection, KSK-style)
- Consensus-based collision detection with quorum voting
- Trust management (direct and FMS-style two-dimensional)

Key concepts:
- Soft collisions: Statistical outliers rejected gradually
- Hard collisions: Different answers in locked semantic regions rejected outright
- Consensus: Multiple nodes must agree before locking a region

References:
- Freenet KSK: https://github.com/hyphanet/wiki/wiki/Keyword-Signed-Key
- FMS trust: https://github.com/SeekingFor/FMS
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
from abc import ABC, abstractmethod
import hashlib
import time
import math

import numpy as np
from scipy import stats


# =============================================================================
# STEP 1: CORE DATA STRUCTURES
# =============================================================================

@dataclass
class EstablishedAnswer:
    """
    An answer that has been established with sufficient confidence in a semantic region.

    Once locked, this answer is protected from being displaced by conflicting results
    (similar to Freenet's KSK collision detection).
    """
    answer_hash: str
    answer_text: str
    semantic_region: str  # Hash of quantized centroid
    confidence: float
    first_seen: float
    node_count: int
    locked: bool = False

    def should_lock(self, min_confidence: float = 0.8, min_nodes: int = 3) -> bool:
        """Determine if this answer should be locked."""
        return self.confidence >= min_confidence and self.node_count >= min_nodes


@dataclass
class VersionedAnswer:
    """
    Answer with version tracking for consensus voting.

    Tracks which nodes have voted for this answer in a semantic region.
    """
    answer_hash: str
    answer_text: str
    node_votes: Set[str] = field(default_factory=set)
    first_seen: float = field(default_factory=time.time)
    trust_sum: float = 0.0  # For trust-weighted voting

    @property
    def vote_count(self) -> int:
        return len(self.node_votes)


@dataclass
class TwoDimensionalTrust:
    """
    FMS-style two-dimensional trust score.

    Separates trust for content quality from trust for judgment about others.
    This distinction is crucial: a node might return good results but have
    poor judgment about which other nodes to trust.
    """
    message_trust: int      # -100 to +100: trust for content quality
    trust_list_trust: int   # -100 to +100: trust for judgment of others

    def effective_trust(self) -> int:
        """Combined trust score for simple operations."""
        return (self.message_trust + self.trust_list_trust) // 2

    def normalized_message_trust(self) -> float:
        """Normalize message trust to 0..1 range."""
        return (self.message_trust + 100) / 200.0

    def normalized_trust_list_trust(self) -> float:
        """Normalize trust list trust to 0..1 range."""
        return (self.trust_list_trust + 100) / 200.0


@dataclass
class NodeTrust:
    """Direct trust score for a node."""
    node_id: str
    trust_score: float  # 0.0 to 1.0
    successful_queries: int = 0
    failed_verifications: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def update(self, success: bool, weight: float = 0.1):
        """Exponential moving average update."""
        outcome = 1.0 if success else 0.0
        self.trust_score = (1 - weight) * self.trust_score + weight * outcome
        if success:
            self.successful_queries += 1
        else:
            self.failed_verifications += 1
        self.last_updated = datetime.now()


# =============================================================================
# STEP 2: OUTPUT SMOOTHING (SOFT COLLISION DETECTION)
# =============================================================================

def smooth_outliers(
    scores: np.ndarray,
    method: str = "zscore",
    threshold: float = 2.5
) -> np.ndarray:
    """
    Compute mask for non-outlier values using statistical methods.

    Outliers are "soft collisions" with the consensus - they deviate significantly
    from the established distribution and should be rejected.

    Args:
        scores: Array of scores to check
        method: Detection method - "zscore", "iqr", or "mad"
        threshold: Rejection threshold (meaning varies by method)

    Returns:
        Boolean mask where True = keep, False = reject as outlier
    """
    if len(scores) < 3:
        return np.ones(len(scores), dtype=bool)

    if method == "zscore":
        # Z-score based rejection
        # Standard but sensitive to outliers in small samples
        z_scores = np.abs(stats.zscore(scores))
        return z_scores < threshold

    elif method == "iqr":
        # Interquartile range based rejection
        # More robust to outliers than z-score
        q1, q3 = np.percentile(scores, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return np.ones(len(scores), dtype=bool)
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (scores >= lower) & (scores <= upper)

    elif method == "mad":
        # Median Absolute Deviation (most robust to outliers)
        # Recommended for adversarial scenarios
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        if mad == 0:
            return np.ones(len(scores), dtype=bool)
        modified_z = 0.6745 * (scores - median) / mad
        return np.abs(modified_z) < threshold

    else:
        raise ValueError(f"Unknown outlier method: {method}")


def robust_aggregate(
    scores: np.ndarray,
    trim_fraction: float = 0.1
) -> float:
    """
    Trimmed mean aggregation - discard extreme values before averaging.

    This is a robust estimator that resists manipulation by outliers.

    Args:
        scores: Array of scores to aggregate
        trim_fraction: Fraction to trim from each end (0.1 = 10% each side)

    Returns:
        Trimmed mean of scores
    """
    if len(scores) == 0:
        return 0.0
    return float(stats.trim_mean(scores, trim_fraction))


def winsorize_scores(
    scores: np.ndarray,
    limits: Tuple[float, float] = (0.1, 0.1)
) -> np.ndarray:
    """
    Winsorize scores - clip extreme values to percentile thresholds.

    Unlike trimming, this keeps all values but limits extremes.

    Args:
        scores: Array of scores
        limits: (lower_fraction, upper_fraction) to clip

    Returns:
        Winsorized scores
    """
    return stats.mstats.winsorize(scores, limits=limits)


class OutlierSmoother:
    """
    Configurable outlier smoothing for result filtering.

    Applies soft collision detection to reject results that deviate
    significantly from the consensus.
    """

    def __init__(
        self,
        method: str = "mad",
        threshold: float = 2.5,
        min_samples: int = 3
    ):
        """
        Initialize outlier smoother.

        Args:
            method: Detection method ("zscore", "iqr", "mad")
            threshold: Rejection threshold
            min_samples: Minimum samples needed for outlier detection
        """
        self.method = method
        self.threshold = threshold
        self.min_samples = min_samples

    def filter_results(
        self,
        results: List[Any],
        score_key: str = "density_adjusted_score"
    ) -> List[Any]:
        """
        Filter results, removing outliers.

        Args:
            results: List of result objects
            score_key: Attribute name for score to check

        Returns:
            Filtered list with outliers removed
        """
        if len(results) < self.min_samples:
            return results

        scores = np.array([getattr(r, score_key) for r in results])
        mask = smooth_outliers(scores, self.method, self.threshold)

        return [r for r, keep in zip(results, mask) if keep]

    def smooth_and_aggregate(
        self,
        scores: np.ndarray,
        aggregation: str = "trimmed_mean"
    ) -> Tuple[float, np.ndarray]:
        """
        Smooth outliers and aggregate remaining values.

        Args:
            scores: Array of scores
            aggregation: Aggregation method ("trimmed_mean", "median", "mean")

        Returns:
            (aggregated_value, mask_of_kept_values)
        """
        mask = smooth_outliers(scores, self.method, self.threshold)
        kept = scores[mask]

        if len(kept) == 0:
            return 0.0, mask

        if aggregation == "trimmed_mean":
            agg = robust_aggregate(kept)
        elif aggregation == "median":
            agg = float(np.median(kept))
        else:
            agg = float(np.mean(kept))

        return agg, mask


# =============================================================================
# STEP 3: SEMANTIC COLLISION DETECTOR (HARD COLLISIONS)
# =============================================================================

class SemanticCollisionDetector:
    """
    Voluntary collision detection for semantic search.

    Once an answer achieves high confidence in a semantic region,
    it becomes "established" and new conflicting answers are rejected.

    This is the "hard collision" mechanism - binary reject/accept
    (compared to outlier smoothing which is "soft" gradual rejection).

    Inspired by Freenet's KSK collision detection where nodes refuse
    to overwrite once-inserted content.
    """

    def __init__(
        self,
        lock_threshold: float = 0.8,
        min_nodes_to_lock: int = 3,
        region_granularity: int = 100
    ):
        """
        Initialize collision detector.

        Args:
            lock_threshold: Confidence threshold to lock a region
            min_nodes_to_lock: Minimum nodes required to lock
            region_granularity: Quantization level for region computation
        """
        self.lock_threshold = lock_threshold
        self.min_nodes_to_lock = min_nodes_to_lock
        self.region_granularity = region_granularity

        # Established answers by semantic region
        self.established: Dict[str, EstablishedAnswer] = {}

    def _compute_region(self, embedding: np.ndarray) -> str:
        """
        Compute semantic region hash from embedding.

        Quantizes the embedding and hashes it to create a region identifier.
        Results in the same region map to the same hash.
        """
        # Quantize to region (reduces continuous space to discrete regions)
        quantized = np.round(
            embedding * self.region_granularity
        ).astype(np.int32)
        return hashlib.sha256(quantized.tobytes()).hexdigest()[:16]

    def check_collision(
        self,
        answer_hash: str,
        embedding: np.ndarray
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if new result collides with established answer.

        Args:
            answer_hash: Hash of the new answer
            embedding: Embedding vector for the answer

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
        if answer_hash == established.answer_hash:
            # Same answer, allow (reinforces consensus)
            return True, None
        else:
            # Different answer trying to enter locked region
            return False, f"Collision with established answer in region {region}"

    def register_result(
        self,
        answer_hash: str,
        answer_text: str,
        embedding: np.ndarray,
        confidence: float,
        node_count: int
    ):
        """
        Register a result, potentially establishing it in the region.

        Args:
            answer_hash: Hash of the answer
            answer_text: Text of the answer
            embedding: Embedding vector
            confidence: Confidence score
            node_count: Number of nodes reporting this answer
        """
        region = self._compute_region(embedding)

        if region not in self.established:
            # First answer in this region
            answer = EstablishedAnswer(
                answer_hash=answer_hash,
                answer_text=answer_text,
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
            if answer_hash == established.answer_hash:
                # Same answer - update confidence
                established.confidence = max(established.confidence, confidence)
                established.node_count = max(established.node_count, node_count)
                if not established.locked and established.should_lock(
                    self.lock_threshold, self.min_nodes_to_lock
                ):
                    established.locked = True

    def get_established(self, embedding: np.ndarray) -> Optional[EstablishedAnswer]:
        """Get established answer for a region, if any."""
        region = self._compute_region(embedding)
        return self.established.get(region)

    def is_locked(self, embedding: np.ndarray) -> bool:
        """Check if a region is locked."""
        region = self._compute_region(embedding)
        established = self.established.get(region)
        return established is not None and established.locked

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about established regions."""
        total = len(self.established)
        locked = sum(1 for e in self.established.values() if e.locked)
        return {
            "total_regions": total,
            "locked_regions": locked,
            "unlocked_regions": total - locked
        }


# =============================================================================
# STEP 4: CONSENSUS COLLISION DETECTOR (QUORUM VOTING)
# =============================================================================

class ConsensusCollisionDetector(SemanticCollisionDetector):
    """
    Collision detection with consensus-based locking.

    Extends SemanticCollisionDetector with multi-node voting:
    - Tracks all versions (answers) proposed in each region
    - Requires quorum (minimum votes) to lock a region
    - Allows superseding if new consensus is significantly stronger

    Like Freenet's redundancy model where multiple nodes may hold
    different versions and consensus determines the winner.
    """

    def __init__(
        self,
        quorum: int = 3,
        supersede_margin: int = 2,
        **kwargs
    ):
        """
        Initialize consensus collision detector.

        Args:
            quorum: Minimum votes to lock a region
            supersede_margin: Extra votes needed to override existing lock
            **kwargs: Passed to SemanticCollisionDetector
        """
        super().__init__(**kwargs)
        self.quorum = quorum
        self.supersede_margin = supersede_margin
        # Track all versions per region
        self.versions: Dict[str, Dict[str, VersionedAnswer]] = {}

    def register_vote(
        self,
        answer_hash: str,
        answer_text: str,
        embedding: np.ndarray,
        node_id: str
    ):
        """
        Register a node's vote for an answer.

        Args:
            answer_hash: Hash of the answer
            answer_text: Text of the answer
            embedding: Embedding vector
            node_id: ID of the voting node
        """
        region = self._compute_region(embedding)

        if region not in self.versions:
            self.versions[region] = {}

        versions = self.versions[region]

        if answer_hash not in versions:
            versions[answer_hash] = VersionedAnswer(
                answer_hash=answer_hash,
                answer_text=answer_text,
                node_votes={node_id},
                first_seen=time.time()
            )
        else:
            versions[answer_hash].node_votes.add(node_id)

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
            current_version = versions.get(current.answer_hash)
            current_votes = current_version.vote_count if current_version else 0

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
        else:
            # Same answer - update confidence
            current.confidence = max(current.confidence, best.vote_count / self.quorum)
            current.node_count = best.vote_count

    def get_version_stats(self, embedding: np.ndarray) -> Dict[str, int]:
        """Get vote counts for all versions in a region."""
        region = self._compute_region(embedding)
        versions = self.versions.get(region, {})
        return {v.answer_hash: v.vote_count for v in versions.values()}


# =============================================================================
# STEP 5: DIRECT TRUST MANAGER
# =============================================================================

class DirectTrustManager:
    """
    Simple direct trust without transitivity.

    Tracks per-node trust scores updated via exponential moving average
    based on success/failure of their results.
    """

    def __init__(self, default_trust: float = 0.5, ema_weight: float = 0.1):
        """
        Initialize trust manager.

        Args:
            default_trust: Initial trust for unknown nodes
            ema_weight: Weight for exponential moving average updates
        """
        self.trust_scores: Dict[str, NodeTrust] = {}
        self.default_trust = default_trust
        self.ema_weight = ema_weight

    def get_trust(self, node_id: str) -> float:
        """Get trust score for a node."""
        if node_id in self.trust_scores:
            return self.trust_scores[node_id].trust_score
        return self.default_trust

    def update_trust(self, node_id: str, success: bool):
        """Update trust based on verification outcome."""
        if node_id not in self.trust_scores:
            self.trust_scores[node_id] = NodeTrust(
                node_id=node_id,
                trust_score=self.default_trust
            )
        self.trust_scores[node_id].update(success, self.ema_weight)

    def weight_responses(
        self,
        responses: List[Tuple[str, Any]]
    ) -> List[Tuple[Any, float]]:
        """
        Weight responses by node trust.

        Args:
            responses: List of (node_id, response) tuples

        Returns:
            List of (response, weight) tuples
        """
        return [
            (response, self.get_trust(node_id))
            for node_id, response in responses
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get trust statistics."""
        if not self.trust_scores:
            return {"node_count": 0}

        trusts = [n.trust_score for n in self.trust_scores.values()]
        return {
            "node_count": len(self.trust_scores),
            "avg_trust": sum(trusts) / len(trusts),
            "min_trust": min(trusts),
            "max_trust": max(trusts)
        }


# =============================================================================
# STEP 6: TRUST-WEIGHTED CONSENSUS DETECTOR
# =============================================================================

class TrustWeightedConsensusDetector(ConsensusCollisionDetector):
    """
    Consensus voting weighted by node trust scores.

    Combines consensus collision detection with trust management:
    - High-trust nodes can establish consensus faster
    - Low-trust nodes need more agreement to overcome trusted consensus
    - Sybil resistance: Many untrusted nodes can't outvote few trusted ones
    """

    def __init__(
        self,
        trust_manager: DirectTrustManager,
        trust_quorum_factor: float = 0.5,
        **kwargs
    ):
        """
        Initialize trust-weighted consensus detector.

        Args:
            trust_manager: Trust manager for getting node trust scores
            trust_quorum_factor: Multiply quorum by this for trust-based locking
            **kwargs: Passed to ConsensusCollisionDetector
        """
        super().__init__(**kwargs)
        self.trust_manager = trust_manager
        self.trust_quorum_factor = trust_quorum_factor

    def register_vote(
        self,
        answer_hash: str,
        answer_text: str,
        embedding: np.ndarray,
        node_id: str
    ):
        """Register trust-weighted vote."""
        region = self._compute_region(embedding)
        trust = self.trust_manager.get_trust(node_id)

        if region not in self.versions:
            self.versions[region] = {}

        versions = self.versions[region]

        if answer_hash not in versions:
            versions[answer_hash] = VersionedAnswer(
                answer_hash=answer_hash,
                answer_text=answer_text,
                node_votes={node_id},
                first_seen=time.time(),
                trust_sum=trust
            )
        else:
            versions[answer_hash].node_votes.add(node_id)
            versions[answer_hash].trust_sum += trust

        self._update_established_by_trust(region)

    def _update_established_by_trust(self, region: str):
        """Lock based on trust-weighted votes, not raw count."""
        versions = self.versions.get(region, {})
        if not versions:
            return

        # Find version with highest trust-weighted score
        best = max(versions.values(), key=lambda v: v.trust_sum)

        # Trust-based quorum (lower threshold for trusted nodes)
        trust_quorum = self.quorum * self.trust_quorum_factor

        if best.trust_sum < trust_quorum:
            return  # No quorum yet

        current = self.established.get(region)

        if current is None:
            # First to reach quorum
            self._lock_answer(region, best)
        elif best.answer_hash != current.answer_hash:
            # Check if supersedes
            current_version = versions.get(current.answer_hash)
            current_trust = current_version.trust_sum if current_version else 0

            # Supersede requires significant trust margin
            if best.trust_sum >= current_trust + (self.supersede_margin * 0.5):
                self._lock_answer(region, best)

    def _lock_answer(self, region: str, version: VersionedAnswer):
        """Lock an answer in a region."""
        self.established[region] = EstablishedAnswer(
            answer_hash=version.answer_hash,
            answer_text=version.answer_text,
            semantic_region=region,
            confidence=version.trust_sum / (self.quorum * self.trust_quorum_factor),
            first_seen=version.first_seen,
            node_count=version.vote_count,
            locked=True
        )


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_adversarial_pipeline(
    outlier_method: str = "mad",
    outlier_threshold: float = 2.5,
    collision_detector: Optional[SemanticCollisionDetector] = None,
    trust_manager: Optional[DirectTrustManager] = None
) -> 'AdversarialPipeline':
    """
    Create a configured adversarial protection pipeline.

    Args:
        outlier_method: Method for soft collision detection
        outlier_threshold: Threshold for outlier rejection
        collision_detector: For hard collision detection (optional)
        trust_manager: For trust-weighted operations (optional)

    Returns:
        Configured AdversarialPipeline
    """
    return AdversarialPipeline(
        smoother=OutlierSmoother(outlier_method, outlier_threshold),
        collision_detector=collision_detector,
        trust_manager=trust_manager
    )


class AdversarialPipeline:
    """
    Combined adversarial protection pipeline.

    Applies multiple protection mechanisms in sequence:
    1. Output smoothing (soft collisions)
    2. Collision detection (hard collisions)
    3. Trust weighting (optional)
    """

    def __init__(
        self,
        smoother: Optional[OutlierSmoother] = None,
        collision_detector: Optional[SemanticCollisionDetector] = None,
        trust_manager: Optional[DirectTrustManager] = None
    ):
        self.smoother = smoother or OutlierSmoother()
        self.collision_detector = collision_detector
        self.trust_manager = trust_manager

    def process_results(
        self,
        results: List[Any],
        embeddings: Optional[List[np.ndarray]] = None,
        node_ids: Optional[List[str]] = None,
        score_key: str = "density_adjusted_score",
        hash_key: str = "answer_hash",
        text_key: str = "answer_text"
    ) -> List[Any]:
        """
        Process results through the adversarial pipeline.

        Args:
            results: List of result objects
            embeddings: Embeddings for each result (for collision detection)
            node_ids: Node IDs for each result (for trust weighting)
            score_key: Attribute name for score
            hash_key: Attribute name for answer hash
            text_key: Attribute name for answer text

        Returns:
            Filtered and processed results
        """
        if not results:
            return results

        # Step 1: Output smoothing (soft collisions)
        filtered = self.smoother.filter_results(results, score_key)

        # Step 2: Collision detection (hard collisions)
        if self.collision_detector and embeddings:
            collision_filtered = []
            for i, result in enumerate(filtered):
                if i >= len(embeddings):
                    collision_filtered.append(result)
                    continue

                embedding = embeddings[i]
                answer_hash = getattr(result, hash_key)

                allowed, reason = self.collision_detector.check_collision(
                    answer_hash, embedding
                )

                if allowed:
                    # Register this result
                    self.collision_detector.register_result(
                        answer_hash=answer_hash,
                        answer_text=getattr(result, text_key, ""),
                        embedding=embedding,
                        confidence=getattr(result, score_key, 0.5),
                        node_count=1
                    )
                    collision_filtered.append(result)

            filtered = collision_filtered

        return filtered

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all components."""
        stats = {"smoother": {"method": self.smoother.method}}

        if self.collision_detector:
            stats["collision_detector"] = self.collision_detector.get_stats()

        if self.trust_manager:
            stats["trust_manager"] = self.trust_manager.get_stats()

        return stats
