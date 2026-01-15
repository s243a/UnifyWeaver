"""
Shared data format for density manifold visualization.

This module defines the JSON schema that both Flask API and Pyodide
backends use to communicate with the frontend.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import json


@dataclass
class DensityGrid:
    """2D density grid from KDE."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    grid_size: int
    values: List[List[float]]  # 2D array of density values
    bandwidth: float


@dataclass
class TreeNode:
    """Node in the hierarchy tree."""
    id: int
    title: str
    parent_id: Optional[int]
    depth: int
    x: float  # 2D projected coordinate
    y: float


@dataclass
class TreeEdge:
    """Edge in the hierarchy tree."""
    source_id: int
    target_id: int
    weight: float


@dataclass
class TreeStructure:
    """Complete tree structure for overlay."""
    nodes: List[TreeNode]
    edges: List[TreeEdge]
    root_id: int
    tree_type: str  # 'mst' or 'j-guided'


@dataclass
class DensityPeak:
    """Local maximum in density field."""
    x: float
    y: float
    density: float
    nearest_node_id: int
    title: str


@dataclass
class ProjectionInfo:
    """SVD projection metadata."""
    variance_explained: List[float]  # [pc1_var, pc2_var]
    singular_values: List[float]


@dataclass
class DensityManifoldData:
    """
    Complete data package for density manifold visualization.

    This is the main data structure passed from backend to frontend.
    Both Flask API and Pyodide compute this, frontend consumes it.
    """
    # Points in 2D projected space
    points: List[Dict[str, Any]]  # [{id, title, x, y}, ...]

    # Density field
    density_grid: DensityGrid

    # Tree overlay (optional)
    tree: Optional[TreeStructure]

    # Peaks (optional)
    peaks: Optional[List[DensityPeak]]

    # Metadata
    projection: ProjectionInfo
    n_points: int

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'DensityManifoldData':
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(
            points=data['points'],
            density_grid=DensityGrid(**data['density_grid']),
            tree=TreeStructure(
                nodes=[TreeNode(**n) for n in data['tree']['nodes']],
                edges=[TreeEdge(**e) for e in data['tree']['edges']],
                root_id=data['tree']['root_id'],
                tree_type=data['tree']['tree_type']
            ) if data.get('tree') else None,
            peaks=[DensityPeak(**p) for p in data['peaks']] if data.get('peaks') else None,
            projection=ProjectionInfo(**data['projection']),
            n_points=data['n_points']
        )


# JSON Schema for validation (can be used by frontend)
JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["points", "density_grid", "projection", "n_points"],
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "x", "y"],
                "properties": {
                    "id": {"type": "integer"},
                    "title": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"}
                }
            }
        },
        "density_grid": {
            "type": "object",
            "required": ["x_min", "x_max", "y_min", "y_max", "grid_size", "values", "bandwidth"],
            "properties": {
                "x_min": {"type": "number"},
                "x_max": {"type": "number"},
                "y_min": {"type": "number"},
                "y_max": {"type": "number"},
                "grid_size": {"type": "integer"},
                "values": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}}
                },
                "bandwidth": {"type": "number"}
            }
        },
        "tree": {
            "type": ["object", "null"],
            "properties": {
                "nodes": {"type": "array"},
                "edges": {"type": "array"},
                "root_id": {"type": "integer"},
                "tree_type": {"type": "string"}
            }
        },
        "peaks": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "density": {"type": "number"},
                    "nearest_node_id": {"type": "integer"},
                    "title": {"type": "string"}
                }
            }
        },
        "projection": {
            "type": "object",
            "required": ["variance_explained", "singular_values"],
            "properties": {
                "variance_explained": {"type": "array", "items": {"type": "number"}},
                "singular_values": {"type": "array", "items": {"type": "number"}}
            }
        },
        "n_points": {"type": "integer"}
    }
}
