#!/usr/bin/env python3
"""Prepare fresh-corpus pairs for the preregistered Sigma(hop) confirmation.

This is the pre-scoring companion to `PREREG_sigma_hop_confirmatory.md`: it removes every node seen in the
exploratory graph, selects a category slice, and samples balanced shortest-hop descendant/ancestor pairs using the
`transitive_h{hop}` label expected by `sigma_hop_confirmatory.py`. Here "hop" means the minimum upward graph
distance found by BFS inside the retained slice, not every possible path length through a DAG.

The output is a comment-header score-input TSV matching existing score_in readers, not a scored dataset. Running
this script does not create labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import tempfile
from collections import Counter, deque


ADMIN = re.compile(
    r"(Wikipedia|Articles?_|All_|Hidden_|CS1|Pages_|Webarchive|Commons|_stubs?$|Stub|"
    r"Redirects|Short_desc|Use_|Templates?|Track|_by_|_in_\d|established_in|introductions|"
    r"disambiguation|_people$|_journals$|_awards$|Wikipedians)"
)


PREREGISTERED_SEED = 0


class FreshCorpusError(ValueError):
    pass


def load_edges(path, stats=None):
    edges = []
    stats = stats if stats is not None else {}
    stats.setdefault("malformed_rows", 0)
    stats.setdefault("wide_rows", 0)
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        if first:
            cols = first.rstrip("\n").split("\t")
            if not (len(cols) >= 2 and cols[0] == "child" and cols[1] == "parent"):
                if len(cols) >= 2 and cols[0] and cols[1]:
                    if len(cols) > 2:
                        stats["wide_rows"] += 1
                    edges.append((cols[0], cols[1]))
                else:
                    stats["malformed_rows"] += 1
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 2 and cols[0] and cols[1]:
                if len(cols) > 2:
                    stats["wide_rows"] += 1
                edges.append((cols[0], cols[1]))
            else:
                stats["malformed_rows"] += 1
    return edges


def sort_key(title):
    return (title.casefold(), title)


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
    blocked_roots = set()
    for child, parent in candidate_edges:
        if child in exploratory_nodes or parent in exploratory_nodes:
            removed_overlap += 1
            blocked_roots.add(child)
            continue
        if ADMIN.search(child) or ADMIN.search(parent):
            removed_admin += 1
            blocked_roots.add(child)
            continue
        filtered.append((child, parent))
    return filtered, {
        "overlap_edges": removed_overlap,
        "admin_edges": removed_admin,
        "blocked_root_candidates": len(blocked_roots),
    }, blocked_roots


def descendants_within(children, roots, max_depth=None):
    kept = set(roots)
    depth = {root: 0 for root in roots}
    queue = deque(roots)
    while queue:
        node = queue.popleft()
        if max_depth is not None and depth[node] >= max_depth:
            continue
        for child in sorted(children.get(node, ())):
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
        for parent in sorted(parents.get(node, ())):
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


def hop_targets(total_pairs, hmax):
    base, rem = divmod(total_pairs, hmax)
    return {hop: base + (1 if hop <= rem else 0) for hop in range(1, hmax + 1)}


def can_satisfy_targets(pool, targets):
    seen = set()
    available = {}
    for hop in sorted(targets):
        count = 0
        for desc, anc in pool[hop]:
            key = tuple(sorted((desc, anc)))
            if key in seen:
                continue
            seen.add(key)
            count += 1
        available[hop] = count
    return all(available[h] >= targets[h] for h in targets), available


def first_eligible_roots(parents, children, roots, excluded_roots, blocked_roots, hmax, targets, min_descendants, slice_depth):
    if slice_depth is not None and slice_depth < hmax:
        raise FreshCorpusError(f"--slice-depth {slice_depth} < --hmax {hmax}: hop-{hmax} pairs are impossible")
    candidates = sorted(roots or children.keys(), key=sort_key)
    for root in candidates:
        if root in excluded_roots or root in blocked_roots:
            continue
        if root not in children or ADMIN.search(root):
            continue
        slice_nodes, pool = pair_pool_for_roots(parents, children, (root,), hmax, slice_depth=slice_depth)
        if len(slice_nodes) < min_descendants:
            continue
        ok, _ = can_satisfy_targets(pool, targets)
        if ok:
            return (root,), slice_nodes, pool
    raise FreshCorpusError("no eligible root slice supplied enough no-overlap pairs for every hop")


def sample_balanced_pairs(pool, total_pairs, hmax, seed):
    targets = hop_targets(total_pairs, hmax)
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
            if counts[hop] >= targets[hop]:
                break
        if counts[hop] < targets[hop]:
            raise FreshCorpusError(f"hop {hop} has only {counts[hop]} unique pairs; need {targets[hop]}")
    random.Random(seed ^ 0x5EED5EED).shuffle(rows)
    return rows, counts


def write_score_in(rows, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        # Comment-prefixed header matches existing score_in readers, which skip metadata lines beginning with #.
        f.write("# node_title\troot_title\tcur_relation\tconf\tneighborhood\tnode_type\troot_type\traw\n")
        for desc, anc, hop in rows:
            # conf=1.0 records that the structural hop label is known; LLM relation labels are scored later.
            f.write(f"{desc}\t{anc}\tsubcategory\t1.0\ttransitive_h{hop}\tcategory\tcategory\t\n")


def write_manifest(path, manifest):
    if not path:
        return
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


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

    if os.path.realpath(args.candidate_graph) == os.path.realpath(args.exploratory_graph):
        raise FreshCorpusError("--candidate-graph and --exploratory-graph must be different files")
    if args.hmax != 5:
        raise FreshCorpusError("the preregistered confirmatory sampler requires --hmax 5")
    if args.slice_depth is not None and args.slice_depth < args.hmax:
        raise FreshCorpusError(
            f"--slice-depth {args.slice_depth} < --hmax {args.hmax}: hop-{args.hmax} pairs are impossible"
        )
    if not args.allow_small_sample and args.pairs < 250:
        raise FreshCorpusError("the preregistered confirmatory sampler requires at least 250 pairs")
    if not args.allow_small_sample and args.min_descendants < 300:
        raise FreshCorpusError("the preregistered confirmatory sampler requires --min-descendants >= 300")
    if not args.allow_small_sample and args.seed != PREREGISTERED_SEED:
        raise FreshCorpusError(f"the preregistered confirmatory sampler requires --seed {PREREGISTERED_SEED}")

    exploratory_stats, candidate_stats = {}, {}
    exploratory_edges = load_edges(args.exploratory_graph, stats=exploratory_stats)
    exploratory_nodes = node_block(exploratory_edges)
    candidate_edges = load_edges(args.candidate_graph, stats=candidate_stats)
    filtered_edges, removed, blocked_roots = filter_candidate_edges(candidate_edges, exploratory_nodes)
    parents, children = build_maps(filtered_edges)
    targets = hop_targets(args.pairs, args.hmax)
    excluded_roots = set(args.exclude_root)
    roots = tuple(root for root in args.root if root not in excluded_roots)
    if args.root and not roots:
        raise FreshCorpusError("all supplied --root values were excluded by --exclude-root")
    selected_roots, slice_nodes, pool = first_eligible_roots(
        parents,
        children,
        roots,
        excluded_roots,
        blocked_roots,
        args.hmax,
        targets,
        args.min_descendants,
        args.slice_depth,
    )
    rows, counts = sample_balanced_pairs(pool, args.pairs, args.hmax, args.seed)
    overlap = node_block((desc, anc) for desc, anc, _ in rows) & exploratory_nodes
    overlap_count = len(overlap)
    if overlap:
        raise FreshCorpusError(f"sampled pairs overlap exploratory nodes: {sorted(overlap)[:10]}")
    write_score_in(rows, args.out)
    manifest = {
        "candidate_graph": args.candidate_graph,
        "exploratory_graph": args.exploratory_graph,
        "selected_roots": list(selected_roots),
        "excluded_roots": sorted(args.exclude_root),
        "selection_rule": "casefold-lexicographically smallest eligible root after no-overlap/admin filtering" if not args.root else "user-supplied root validation",
        "seed": args.seed,
        "hmax": args.hmax,
        "allow_small_sample": args.allow_small_sample,
        "requested_pairs": args.pairs,
        "written_pairs": len(rows),
        "target_hop_counts": {str(h): targets[h] for h in range(1, args.hmax + 1)},
        "hop_counts": {str(h): counts[h] for h in range(1, args.hmax + 1)},
        "hop_semantics": "shortest upward graph distance within retained slice",
        "row_shuffle_seed": args.seed ^ 0x5EED5EED,
        "slice_nodes": len(slice_nodes),
        "candidate_edges": len(candidate_edges),
        "filtered_edges": len(filtered_edges),
        "removed_edges": removed,
        "blocked_root_candidates": len(blocked_roots),
        "edge_file_stats": {"candidate": candidate_stats, "exploratory": exploratory_stats},
        "node_overlap_with_exploratory": overlap_count,
        "output": args.out,
    }
    write_manifest(args.manifest, manifest)
    print(f"selected root(s): {', '.join(selected_roots)}")
    print(f"wrote {len(rows)} no-overlap score-in pairs -> {args.out}")
    print("hop counts:", dict(sorted(counts.items())))
    print(f"manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
