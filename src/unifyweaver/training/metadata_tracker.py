"""
Training Metadata Tracker

Tracks when each datapoint was last trained to avoid overtraining.
Supports balanced sampling across data sources.

Usage:
    tracker = TrainingMetadataTracker("data/training_metadata.json")

    # Check what needs training
    stale_ids = tracker.get_stale_datapoints(current_iter, threshold=100)

    # Sample balanced batch
    batch = tracker.sample_balanced_batch(available_ids, batch_size=64)

    # Update after training
    tracker.update_trained(batch_ids, current_iter)
    tracker.save()
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, asdict, field


@dataclass
class DatapointMetadata:
    """Metadata for a single training datapoint."""
    source: str  # "pearltrees", "wikipedia", "synthetic"
    last_trained_iter: int = 0
    times_sampled: int = 0
    created_iter: int = 0

    def staleness(self, current_iter: int) -> int:
        """How many iterations since last trained."""
        return current_iter - self.last_trained_iter


class TrainingMetadataTracker:
    """
    Tracks training metadata for balanced, non-redundant training.
    """

    def __init__(self, path: Optional[str] = None):
        """
        Initialize tracker.

        Args:
            path: Path to JSON file for persistence (optional)
        """
        self.path = Path(path) if path else None
        self.metadata: Dict[str, DatapointMetadata] = {}
        self.current_iter = 0

        if self.path and self.path.exists():
            self.load()

    def load(self) -> None:
        """Load metadata from JSON file."""
        if not self.path or not self.path.exists():
            return

        with open(self.path) as f:
            data = json.load(f)

        self.current_iter = data.get('current_iter', 0)

        for dp_id, meta in data.get('datapoints', {}).items():
            self.metadata[dp_id] = DatapointMetadata(**meta)

        print(f"Loaded {len(self.metadata)} datapoints, iter={self.current_iter}")

    def save(self) -> None:
        """Save metadata to JSON file."""
        if not self.path:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'current_iter': self.current_iter,
            'datapoints': {
                dp_id: asdict(meta)
                for dp_id, meta in self.metadata.items()
            }
        }

        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        datapoint_id: str,
        source: str,
        created_iter: Optional[int] = None
    ) -> None:
        """
        Register a new datapoint.

        Args:
            datapoint_id: Unique identifier for the datapoint
            source: Data source ("pearltrees", "wikipedia", "synthetic")
            created_iter: When the datapoint was created (default: current_iter)
        """
        if datapoint_id not in self.metadata:
            self.metadata[datapoint_id] = DatapointMetadata(
                source=source,
                created_iter=created_iter if created_iter is not None else self.current_iter
            )

    def register_batch(
        self,
        datapoint_ids: List[str],
        source: str
    ) -> None:
        """Register multiple datapoints from same source."""
        for dp_id in datapoint_ids:
            self.register(dp_id, source)

    def needs_training(
        self,
        datapoint_id: str,
        staleness_threshold: int = 100
    ) -> bool:
        """
        Check if a datapoint needs training.

        Args:
            datapoint_id: Datapoint to check
            staleness_threshold: How stale before needing retrain

        Returns:
            True if datapoint is stale or new
        """
        if datapoint_id not in self.metadata:
            return True  # New datapoint

        meta = self.metadata[datapoint_id]
        return meta.staleness(self.current_iter) >= staleness_threshold

    def get_stale_datapoints(
        self,
        available_ids: Optional[List[str]] = None,
        staleness_threshold: int = 100
    ) -> List[str]:
        """
        Get datapoints that need training.

        Args:
            available_ids: Subset to consider (None for all)
            staleness_threshold: Staleness threshold

        Returns:
            List of stale datapoint IDs
        """
        ids_to_check = available_ids if available_ids else list(self.metadata.keys())

        return [
            dp_id for dp_id in ids_to_check
            if self.needs_training(dp_id, staleness_threshold)
        ]

    def compute_sampling_weights(
        self,
        datapoint_ids: List[str]
    ) -> Dict[str, float]:
        """
        Compute sampling weights for balanced training.

        Weights are based on:
        - Staleness (higher = more weight)
        - Inverse of times_sampled (less trained = more weight)
        - Source balancing (underrepresented sources get more weight)

        Args:
            datapoint_ids: Datapoints to compute weights for

        Returns:
            {datapoint_id: weight}
        """
        if not datapoint_ids:
            return {}

        # Count sources
        source_counts = defaultdict(int)
        for dp_id in datapoint_ids:
            if dp_id in self.metadata:
                source_counts[self.metadata[dp_id].source] += 1
            else:
                source_counts['unknown'] += 1

        total_count = len(datapoint_ids)

        weights = {}
        for dp_id in datapoint_ids:
            if dp_id in self.metadata:
                meta = self.metadata[dp_id]
                staleness = meta.staleness(self.current_iter)
                times_sampled = meta.times_sampled
                source = meta.source
            else:
                # New datapoint - high priority
                staleness = self.current_iter
                times_sampled = 0
                source = 'unknown'

            # Staleness weight (more stale = higher weight)
            staleness_weight = staleness + 1

            # Frequency weight (less sampled = higher weight)
            freq_weight = 1.0 / (1 + times_sampled)

            # Source balance weight (underrepresented = higher weight)
            source_weight = total_count / (source_counts[source] * len(source_counts))

            weights[dp_id] = staleness_weight * freq_weight * source_weight

        return weights

    def sample_balanced_batch(
        self,
        available_ids: List[str],
        batch_size: int,
        staleness_threshold: int = 100
    ) -> List[str]:
        """
        Sample a balanced batch prioritizing stale/undersampled data.

        Args:
            available_ids: Pool of datapoints to sample from
            batch_size: Number of datapoints to sample
            staleness_threshold: Only sample if stale enough

        Returns:
            List of sampled datapoint IDs
        """
        # Filter to stale datapoints
        stale_ids = [
            dp_id for dp_id in available_ids
            if self.needs_training(dp_id, staleness_threshold)
        ]

        if not stale_ids:
            return []

        if len(stale_ids) <= batch_size:
            return stale_ids

        # Compute weights
        weights = self.compute_sampling_weights(stale_ids)

        # Weighted sampling without replacement
        weight_sum = sum(weights.values())
        probs = [weights[dp_id] / weight_sum for dp_id in stale_ids]

        # Sample
        sampled = []
        remaining = list(zip(stale_ids, probs))

        for _ in range(min(batch_size, len(stale_ids))):
            # Normalize remaining probabilities
            total = sum(p for _, p in remaining)
            if total == 0:
                break

            r = random.random() * total
            cumsum = 0
            for i, (dp_id, prob) in enumerate(remaining):
                cumsum += prob
                if r <= cumsum:
                    sampled.append(dp_id)
                    remaining.pop(i)
                    break

        return sampled

    def update_trained(
        self,
        datapoint_ids: List[str],
        iteration: Optional[int] = None
    ) -> None:
        """
        Update metadata after training a batch.

        Args:
            datapoint_ids: Datapoints that were trained
            iteration: Current iteration (default: self.current_iter)
        """
        iter_num = iteration if iteration is not None else self.current_iter

        for dp_id in datapoint_ids:
            if dp_id in self.metadata:
                self.metadata[dp_id].last_trained_iter = iter_num
                self.metadata[dp_id].times_sampled += 1

    def increment_iteration(self) -> int:
        """Increment and return current iteration."""
        self.current_iter += 1
        return self.current_iter

    def get_stats(self) -> Dict:
        """Get summary statistics."""
        if not self.metadata:
            return {'total': 0}

        source_counts = defaultdict(int)
        total_staleness = 0
        never_trained = 0

        for meta in self.metadata.values():
            source_counts[meta.source] += 1
            total_staleness += meta.staleness(self.current_iter)
            if meta.times_sampled == 0:
                never_trained += 1

        return {
            'total': len(self.metadata),
            'current_iter': self.current_iter,
            'by_source': dict(source_counts),
            'avg_staleness': total_staleness / len(self.metadata),
            'never_trained': never_trained,
        }


# =============================================================================
# Convenience functions
# =============================================================================

_default_tracker: Optional[TrainingMetadataTracker] = None


def get_tracker(path: str = "data/training_metadata.json") -> TrainingMetadataTracker:
    """Get or create default tracker."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = TrainingMetadataTracker(path)
    return _default_tracker
