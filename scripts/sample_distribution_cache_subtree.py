#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Sample a rooted category subtree for distribution-cache benchmarks."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.distribution_cache_support import load_parent_edges_tsv  # noqa: E402


SIMPLEWIKI_ARTICLES_ROOT = "Category:Articles"

DEFAULT_EXCLUDE_PATTERNS = [
    r"^Category:Categories$",
    r"^Category:Container_categories$",
    r"^Category:Wikipedia",
    r"^Category:Template",
    r"^Category:User",
    r"^Category:Hidden_categories$",
    r"^Category:Tracking_categories$",
    r"(?i)maintenance",
    r"(?i)admin",
    r"(?i)template",
]


def compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def is_excluded(node: str, root: str, exclude_patterns: list[re.Pattern[str]]) -> bool:
    if node == root:
        return False
    return any(pattern.search(node) for pattern in exclude_patterns)


def build_children_index(parents: dict[str, list[str]], exclude_patterns: list[re.Pattern[str]], root: str):
    children = defaultdict(list)
    for child, parent_nodes in parents.items():
        if is_excluded(child, root, exclude_patterns):
            continue
        for parent in parent_nodes:
            if is_excluded(parent, root, exclude_patterns):
                continue
            children[parent].append(child)
    return {parent: sorted(set(child_nodes)) for parent, child_nodes in children.items()}


def select_subtree_nodes(children: dict[str, list[str]], root: str, max_depth: int, node_limit: int | None = None):
    selected = set()
    distances = {}
    queue = deque([(root, 0)])
    while queue:
        node, depth = queue.popleft()
        if node in selected:
            continue
        selected.add(node)
        distances[node] = depth
        if node_limit is not None and len(selected) >= node_limit:
            break
        if depth >= max_depth:
            continue
        for child in children.get(node, []):
            queue.append((child, depth + 1))
    return selected, distances


def sample_edges(parents, selected_nodes: set[str]):
    rows = []
    for child in sorted(selected_nodes):
        for parent in parents.get(child, []):
            if parent in selected_nodes:
                rows.append((child, parent))
    return rows


def write_edges(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("child\tparent\n")
        for child, parent in rows:
            handle.write(f"{child}\t{parent}\n")


def write_targets(path: Path, distances: dict[str, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for node, _distance in sorted(distances.items(), key=lambda item: (item[1], item[0])):
            handle.write(f"{node}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-file", required=True, type=Path, help="Input TSV with child<TAB>parent category edges.")
    parser.add_argument("--output", required=True, type=Path, help="Output sampled TSV with child<TAB>parent rows.")
    parser.add_argument("--targets-output", type=Path, help="Optional newline-delimited selected target nodes.")
    parser.add_argument("--root", default=SIMPLEWIKI_ARTICLES_ROOT, help="Benchmark subtree root.")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum child distance from the root to include.")
    parser.add_argument("--node-limit", type=int, help="Optional cap on selected nodes after BFS ordering.")
    parser.add_argument("--exclude-regex", action="append", default=[], help="Additional regex for nodes to exclude.")
    parser.add_argument("--no-default-excludes", action="store_true", help="Disable built-in admin/container filters.")
    args = parser.parse_args(argv)

    if args.max_depth < 0:
        raise SystemExit("--max-depth must be non-negative")

    patterns = [] if args.no_default_excludes else list(DEFAULT_EXCLUDE_PATTERNS)
    patterns.extend(args.exclude_regex)
    exclude_patterns = compile_patterns(patterns)

    parents = load_parent_edges_tsv(args.edge_file)
    children = build_children_index(parents, exclude_patterns, args.root)
    selected_nodes, distances = select_subtree_nodes(children, args.root, args.max_depth, args.node_limit)
    rows = sample_edges(parents, selected_nodes)
    write_edges(args.output, rows)
    if args.targets_output:
        write_targets(args.targets_output, distances)

    print(f"root={args.root}")
    print(f"selected_nodes={len(selected_nodes)}")
    print(f"sampled_edges={len(rows)}")
    print(f"max_depth={args.max_depth}")
    print(f"wrote_edges={args.output}")
    if args.targets_output:
        print(f"wrote_targets={args.targets_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
