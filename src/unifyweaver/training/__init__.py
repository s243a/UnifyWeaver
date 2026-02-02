"""
Training utilities for UnifyWeaver.

Modules:
    metadata_tracker: Track training iterations to avoid overtraining
"""

from .metadata_tracker import (
    DatapointMetadata,
    TrainingMetadataTracker,
    get_tracker,
)

__all__ = [
    'DatapointMetadata',
    'TrainingMetadataTracker',
    'get_tracker',
]
