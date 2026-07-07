#!/usr/bin/env python3
"""Prepare fresh-corpus pairs for the preregistered Sigma(hop) confirmation.

This is the pre-scoring companion to `PREREG_sigma_hop_confirmatory.md`: it removes every node seen in the
exploratory graph, selects a category slice, and samples balanced exact-hop descendant/ancestor pairs using the
`transitive_h{hop}` label expected by `sigma_hop_confirmatory.py`.

The output is a score-input TSV, not a scored dataset. Running this script does not create labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter, deque


ADMIN = re.compile(
    r"(Wikipedia|Articles?_|All_|Hidden_|CS1|Pages_|Webarchive|Commons|_stubs?$|Stub|"
    r"Redirects|Short_desc|Use_|Templates?|Track|_by_|_in_\d|established_in|introductions|"
    r"disambiguation|_people$|_journals$|_awards$|Wikipedians)"
)


class FreshCorpusError(ValueError):
    pass


def load_edges(path):
    edges = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            cols = line.rstrip("\n").split("\t")
            if i == 0 and len(cols) >= 2 and cols[0] == "child" and cols[1] == "parent":
                continue
            if len(cols) >= 2 and cols[0] and cols[1]:
                edges.append((cols[0], cols[1]))
    return edges


def node_block(edges):
    return {n for edge in edges for n in edge}


def build_maps(edges):
    parents, children = {}, {}
    for child, parent in edges:
        parents.setdefault(child, set()).add(parent)
        parents.setdefault(parent, set())
        children.setdefault(parent, set()).add(child)
        children.setdefault(child, set())
    return parents, children


def filter_candidate_edges(candidate_edges, exploratory_nodes):
    filtered = []
    removed_overlap = 0
    removed_admin = 0
    for child, parent in candidate_edges:
        if child in exploratory_nodes or parent in exploratory_nodes:
            removed_overlap += 1
            continue
        if ADMIN.search(child) or ADMIN.search(parent):
            removed_admin += 1
            continue
        filtered.append((child, parent))
    return filtered, {"overlap_edges": removed_overlap, "admin_edges": removed_admin}


def descendants_within(children, roots, max_depth=None):
    kept = set(roots)
    depth = {root: 0 for root in roots}
    queue = deque(roots)
    while queue:
        node = queue.popleft()
        if max_depth is not None and depth[node] >= max_depth:
            continue
        for child in sorted(children.get(node) or []):
            if child not in kept:
                kept.add(child)
                depth[child] = depth[node] + 1
                queue.append(child)
    return kept


def restrict_parents(parents, nodes):
    return {node: {parent for parent in parents.get(node, set()) if parent in nodes} for node in nodes}


def ancestors_by_hop(parents, start, hmax):
    seen, queue, by_hop = {start: 0}, deque([start]), {}
    while queue:
        node = queue.popleft()
        hop = seen[node]
        if hop >= hmax:
            continue
        for parent in sorted(parents.get(node) or []):
            if parent not in seen:
                seen[parent] = hop + 1
                by_hop.setdefault(hop + 1, []).append(parent)
                queue.append(parent)
    return by_hop


def pair_pool_for_roots(parents, children, roots, hmax, slice_depth=None):
    slice_nodes = descendants_within(children, roots, max_depth=slice_depth)
    slice_parents = restrict_parents(parents, slice_nodes)
    pool = {hop: [] for hop in range(1, hmax + 1)}
    for desc in sorted(slice_nodes):
        by_hop = ancestors_by_hop(slice_parents, desc, hmax)
        for hop in range(1, hmax + 1):
            for anc in by_hop.get(hop, []):
                if desc != anc:
                    pool[hop].append((desc, anc))
    return slice_nodes, pool


def first_eligible_roots(parents, children, roots, excluded_roots, hmax, target_per_hop, min_descendants, slice_depth):
    candidates = sorted(roots or children.keys())
    for root in candidates:
        if root in excluded_roots:
            continue
        if root not in children or ADMIN.search(root):
            continue
        slice_nodes, pool = pair_pool_for_roots(parents, children, (root,), hmax, slice_depth=slice_depth)
        if len(slice_nodes) < min_descendants:
            continue
        if all(len(pool[h]) >= target_per_hop for h in range(1, hmax + 1)):
            return (root,), slice_nodes, pool
    raise FreshCorpusError("no eligible root slice supplied enough no-overlap pairs for every hop")


def sample_balanced_pairs(pool, total_pairs, hmax, seed):
    per_hop = math.ceil(total_pairs / hmax)
    rng = random.Random(seed)
    rows = []
    seen_unordered = set()
    counts = Counter()
    for hop in range(1, hmax + 1):
        candidates = list(pool[hop])
        rng.shuffle(candidates)
        for desc, anc in candidates:
            key = tuple(sorted((desc, anc)))
            if key in seen_unordered:
                continue
            seen_unordered.add(key)
            rows.append((desc, anc, hop))
            counts[hop] += 1
            if counts[hop] >= per_hop:
                break
        if counts[hop] < per_hop:
            raise FreshCorpusError(f"hop {hop} has only {counts[hop]} unique pairs; need {per_hop}")
    rng.shuffle(rows)
    if len(rows) > total_pairs:
        rows = rows[:total_pairs]
        counts = Counter(hop for _, _, hop in rows)
    return rows, counts


def write_score_in(rows, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# node_title\troot_title\tcur_relation\tconf\tneighborhood\tnode_type\troot_type\traw\n")
        for desc, anc, hop in rows:
            f.write(f"{desc}\t{anc}\tsubcategory\t1.0\ttransitive_h{hop}\tcategory\tcategory\t\n")


def write_manifest(path, manifest):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-graph", required=True, help="fresh/later Wikipedia category child<TAB>parent graph")
    ap.add_argument("--exploratory-graph", required=True, help="PR #3517 exploratory 100k_cats/category_parent.tsv")
    ap.add_argument("--root", action="append", default=[], help="candidate root slice to validate/sample")
    ap.add_argument("--exclude-root", action="append", default=[], help="exploratory seed/root category to disallow")
    ap.add_argument("--pairs", type=int, default=250)
    ap.add_argument("--hmax", type=int, default=5)
    ap.add_argument("--min-descendants", type=int, default=300)
    ap.add_argument("--slice-depth", type=int, default=None, help="optional downward closure cap from selected root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--allow-small-sample", action="store_true", help="permit non-confirmatory toy/dry-run sizes")
    args = ap.parse_args()

    if args.hmax != 5:
        raise FreshCorpusError("the preregistered confirmatory sampler requires --hmax 5")
    if not args.allow_small_sample and args.pairs < 250:
        raise FreshCorpusError("the preregistered confirmatory sampler requires at least 250 pairs")
    if not args.allow_small_sample and args.min_descendants < 300:
        raise FreshCorpusError("the preregistered confirmatory sampler requires --min-descendants >= 300")

    exploratory_edges = load_edges(args.exploratory_graph)
    exploratory_nodes = node_block(exploratory_edges)
    candidate_edges = load_edges(args.candidate_graph)
    filtered_edges, removed = filter_candidate_edges(candidate_edges, exploratory_nodes)
    parents, children = build_maps(filtered_edges)
    target_per_hop = math.ceil(args.pairs / args.hmax)
    excluded_roots = set(args.exclude_root)
    roots = tuple(root for root in args.root if root not in excluded_roots)
    if args.root and not roots:
        raise FreshCorpusError("all supplied --root values were excluded by --exclude-root")
    selected_roots, slice_nodes, pool = first_eligible_roots(
        parents,
        children,
        roots,
        excluded_roots,
        args.hmax,
        target_per_hop,
        args.min_descendants,
        args.slice_depth,
    )
    rows, counts = sample_balanced_pairs(pool, args.pairs, args.hmax, args.seed)
    overlap = node_block((desc, anc) for desc, anc, _ in rows) & exploratory_nodes
    if overlap:
        raise FreshCorpusError(f"sampled pairs overlap exploratory nodes: {sorted(overlap)[:10]}")
    write_score_in(rows, args.out)
    manifest = {
        "candidate_graph": args.candidate_graph,
        "exploratory_graph": args.exploratory_graph,
        "selected_roots": list(selected_roots),
        "excluded_roots": sorted(args.exclude_root),
        "selection_rule": "first alphabetically eligible root after no-overlap/admin filtering" if not args.root else "user-supplied root validation",
        "seed": args.seed,
        "hmax": args.hmax,
        "allow_small_sample": args.allow_small_sample,
        "requested_pairs": args.pairs,
        "written_pairs": len(rows),
        "hop_counts": {str(h): counts[h] for h in range(1, args.hmax + 1)},
        "slice_nodes": len(slice_nodes),
        "candidate_edges": len(candidate_edges),
        "filtered_edges": len(filtered_edges),
        "removed_edges": removed,
        "node_overlap_with_exploratory": 0,
        "output": args.out,
    }
    write_manifest(args.manifest, manifest)
    print(f"selected root(s): {', '.join(selected_roots)}")
    print(f"wrote {len(rows)} no-overlap score-in pairs -> {args.out}")
    print("hop counts:", dict(sorted(counts.items())))
    print(f"manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
