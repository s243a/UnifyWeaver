# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# JSON Lines I/O for Mind Map Objects
#
# Provides streaming I/O for mindmap data using JSON Lines format,
# enabling cross-target communication with Prolog via pipes.

"""
JSON Lines I/O for mindmap objects.

This module provides readers and writers for mindmap data in JSON Lines format,
supporting streaming communication with the Prolog DSL via Unix pipes.

Usage:
    from unifyweaver.mindmap.io import read_nodes, write_positions

    # Read nodes from stdin
    for node in read_nodes(sys.stdin):
        process(node)

    # Write positions to stdout
    write_positions(positions, sys.stdout)
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from typing import (
    Dict, List, Tuple, Optional, Any, Iterator,
    TextIO, Union, Protocol, TypeVar, Generic
)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MindmapNode:
    """A mind map node."""
    id: str
    label: str = ""
    node_type: str = "default"
    parent: Optional[str] = None
    link: Optional[str] = None
    cluster: Optional[str] = None
    props: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "type": "node",
            "id": self.id,
            "label": self.label,
            "node_type": self.node_type,
            "parent": self.parent,
            "link": self.link,
            "cluster": self.cluster,
            "props": self.props
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MindmapNode':
        """Create from JSON dict."""
        return cls(
            id=d["id"],
            label=d.get("label", ""),
            node_type=d.get("node_type", "default"),
            parent=d.get("parent"),
            link=d.get("link"),
            cluster=d.get("cluster"),
            props=d.get("props", {})
        )


@dataclass
class MindmapEdge:
    """A mind map edge."""
    from_id: str
    to_id: str
    edge_type: str = "default"
    weight: float = 1.0
    label: Optional[str] = None
    props: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "type": "edge",
            "from": self.from_id,
            "to": self.to_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "label": self.label,
            "props": self.props
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MindmapEdge':
        """Create from JSON dict."""
        return cls(
            from_id=d["from"],
            to_id=d["to"],
            edge_type=d.get("edge_type", "default"),
            weight=d.get("weight", 1.0),
            label=d.get("label"),
            props=d.get("props", {})
        )


@dataclass
class MindmapPosition:
    """A node position."""
    id: str
    x: float
    y: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "type": "position",
            "id": self.id,
            "x": self.x,
            "y": self.y
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MindmapPosition':
        """Create from JSON dict."""
        return cls(
            id=d["id"],
            x=d["x"],
            y=d["y"]
        )


@dataclass
class MindmapGraph:
    """A complete mind map graph."""
    id: str
    nodes: List[MindmapNode] = field(default_factory=list)
    edges: List[MindmapEdge] = field(default_factory=list)
    positions: List[MindmapPosition] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "type": "graph",
            "id": self.id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "positions": [p.to_dict() for p in self.positions]
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MindmapGraph':
        """Create from JSON dict."""
        return cls(
            id=d["id"],
            nodes=[MindmapNode.from_dict(n) for n in d.get("nodes", [])],
            edges=[MindmapEdge.from_dict(e) for e in d.get("edges", [])],
            positions=[MindmapPosition.from_dict(p) for p in d.get("positions", [])]
        )


# Type for any mindmap object
MindmapObject = Union[MindmapNode, MindmapEdge, MindmapPosition, MindmapGraph]


# ============================================================================
# PROTOCOL ABSTRACTION
# ============================================================================

class MindmapProtocol(Protocol):
    """Protocol interface for mindmap data serialization."""

    def read(self, stream: TextIO) -> Iterator[MindmapObject]:
        """Read objects from stream."""
        ...

    def write(self, obj: MindmapObject, stream: TextIO) -> None:
        """Write object to stream."""
        ...


class JsonLinesProtocol:
    """JSON Lines protocol implementation."""

    def read(self, stream: TextIO) -> Iterator[MindmapObject]:
        """Read mindmap objects from JSON Lines stream."""
        for line in stream:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                obj_type = obj.get("type")

                if obj_type == "node":
                    yield MindmapNode.from_dict(obj)
                elif obj_type == "edge":
                    yield MindmapEdge.from_dict(obj)
                elif obj_type == "position":
                    yield MindmapPosition.from_dict(obj)
                elif obj_type == "graph":
                    yield MindmapGraph.from_dict(obj)
            except json.JSONDecodeError:
                continue

    def write(self, obj: MindmapObject, stream: TextIO) -> None:
        """Write mindmap object as JSON line."""
        json_str = json.dumps(obj.to_dict(), ensure_ascii=False)
        print(json_str, file=stream)


# Default protocol instance
DEFAULT_PROTOCOL = JsonLinesProtocol()


# ============================================================================
# READER FUNCTIONS
# ============================================================================

def read_jsonl(stream: TextIO = sys.stdin) -> Iterator[Dict[str, Any]]:
    """Read raw JSON objects from JSON Lines stream."""
    for line in stream:
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_objects(stream: TextIO = sys.stdin) -> Iterator[MindmapObject]:
    """Read typed mindmap objects from JSON Lines stream."""
    yield from DEFAULT_PROTOCOL.read(stream)


def read_nodes(stream: TextIO = sys.stdin) -> Iterator[MindmapNode]:
    """Read only node objects from stream."""
    for obj in read_objects(stream):
        if isinstance(obj, MindmapNode):
            yield obj


def read_edges(stream: TextIO = sys.stdin) -> Iterator[MindmapEdge]:
    """Read only edge objects from stream."""
    for obj in read_objects(stream):
        if isinstance(obj, MindmapEdge):
            yield obj


def read_positions(stream: TextIO = sys.stdin) -> Iterator[MindmapPosition]:
    """Read only position objects from stream."""
    for obj in read_objects(stream):
        if isinstance(obj, MindmapPosition):
            yield obj


def read_positions_dict(stream: TextIO = sys.stdin) -> Dict[str, Tuple[float, float]]:
    """Read positions as dict {id: (x, y)}."""
    return {pos.id: (pos.x, pos.y) for pos in read_positions(stream)}


def read_graph(stream: TextIO = sys.stdin) -> Optional[MindmapGraph]:
    """Read a complete graph object from stream."""
    for obj in read_objects(stream):
        if isinstance(obj, MindmapGraph):
            return obj
    return None


def read_all(stream: TextIO = sys.stdin) -> Tuple[
    List[MindmapNode], List[MindmapEdge], List[MindmapPosition]
]:
    """Read all objects, grouped by type."""
    nodes = []
    edges = []
    positions = []

    for obj in read_objects(stream):
        if isinstance(obj, MindmapNode):
            nodes.append(obj)
        elif isinstance(obj, MindmapEdge):
            edges.append(obj)
        elif isinstance(obj, MindmapPosition):
            positions.append(obj)

    return nodes, edges, positions


# ============================================================================
# WRITER FUNCTIONS
# ============================================================================

def write_jsonl(obj: Dict[str, Any], stream: TextIO = sys.stdout) -> None:
    """Write a JSON object as a single line."""
    print(json.dumps(obj, ensure_ascii=False), file=stream)


def write_object(obj: MindmapObject, stream: TextIO = sys.stdout) -> None:
    """Write a mindmap object as JSON line."""
    DEFAULT_PROTOCOL.write(obj, stream)


def write_node(node: MindmapNode, stream: TextIO = sys.stdout) -> None:
    """Write a node object."""
    write_object(node, stream)


def write_edge(edge: MindmapEdge, stream: TextIO = sys.stdout) -> None:
    """Write an edge object."""
    write_object(edge, stream)


def write_position(
    node_id: str, x: float, y: float, stream: TextIO = sys.stdout
) -> None:
    """Write a position object."""
    pos = MindmapPosition(id=node_id, x=x, y=y)
    write_object(pos, stream)


def write_positions(
    positions: Dict[str, Tuple[float, float]],
    stream: TextIO = sys.stdout
) -> None:
    """Write positions dict as JSON Lines."""
    for node_id, (x, y) in positions.items():
        write_position(node_id, x, y, stream)


def write_graph(graph: MindmapGraph, stream: TextIO = sys.stdout) -> None:
    """Write a complete graph object."""
    write_object(graph, stream)


# ============================================================================
# CONVENIENCE FUNCTIONS FOR LAYOUT/OPTIMIZER INTEGRATION
# ============================================================================

def positions_from_dict(d: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """Convert JSON dict to positions dict."""
    positions = d.get("positions", {})
    return {k: tuple(v) for k, v in positions.items()}


def positions_to_list(
    positions: Dict[str, Tuple[float, float]]
) -> List[Dict[str, Any]]:
    """Convert positions dict to list of position dicts."""
    return [
        {"type": "position", "id": k, "x": v[0], "y": v[1]}
        for k, v in positions.items()
    ]


def graph_from_dict(d: Dict[str, Any]) -> Tuple[
    Dict[str, Dict[str, Any]],  # nodes: {id: props}
    List[Tuple[str, str]],      # edges: [(from, to)]
    Dict[str, Any]              # options
]:
    """
    Convert JSON graph dict to layout-friendly format.

    Returns:
        nodes: {id: {label, type, ...}}
        edges: [(from_id, to_id)]
        options: layout options
    """
    nodes = {}
    for n in d.get("nodes", []):
        node_id = n.get("id")
        if node_id:
            nodes[node_id] = n.get("props", {})
            nodes[node_id]["label"] = n.get("label", "")
            nodes[node_id]["type"] = n.get("node_type", "default")

    edges = []
    for e in d.get("edges", []):
        from_id = e.get("from")
        to_id = e.get("to")
        if from_id and to_id:
            edges.append((from_id, to_id))

    options = d.get("options", {})

    return nodes, edges, options


# ============================================================================
# MAIN ENTRY POINT (for testing)
# ============================================================================

def main():
    """Test JSON Lines I/O by echoing input."""
    import sys

    print("Reading mindmap objects from stdin...", file=sys.stderr)

    nodes, edges, positions = read_all(sys.stdin)

    print(f"Read {len(nodes)} nodes, {len(edges)} edges, {len(positions)} positions",
          file=sys.stderr)

    # Echo back
    for node in nodes:
        write_object(node)
    for edge in edges:
        write_object(edge)
    for pos in positions:
        write_object(pos)


if __name__ == "__main__":
    main()
