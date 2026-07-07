#!/usr/bin/env python3
"""Prepare fresh-corpus pairs for the preregistered Sigma(hop) confirmation.

This is the pre-scoring companion to `PREREG_sigma_hop_confirmatory.md`: it removes every node seen in the
exploratory graph, selects a category slice, and samples balanced shortest-hop descendant/ancestor pairs using the
`transitive_h{hop}` label expected by `sigma_hop_confirmatory.py`. Here "hop" means the minimum upward graph
distance found by BFS inside the retained slice, not every possible path length through a DAG.

Candidate graph structure can come from a child<TAB>parent TSV or from the repository's Phase-1 category LMDB layout.
The LMDB path keeps graph traversal on unsigned int32 IDs and uses a separate real-title layer only at the score-input boundary.
Running this script does not create labels.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
from collections import Counter, deque

from lmdb_id import dec_id, enc_id, looks_int


ADMIN = re.compile(
    r"(Wikipedia|Articles?_|All_|Hidden_|CS1|Pages_|Webarchive|Commons|_stubs?$|Stub|"
    r"Redirects|Short_desc|Use_|Templates?|Track|_by_|_in_\d|established_in|introductions|"
    r"disambiguation|_people$|_journals$|_awards$|Wikipedians)"
)


PREREGISTERED_SEED = 0


class FreshCorpusError(ValueError):
    pass


def is_identity_numeric_title(node_id, title):
    return looks_int(title) and int(title) == int(node_id)


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
        for child in sorted(children.get(node, ()), key=sort_key):
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
        for parent in sorted(parents.get(node, ()), key=sort_key):
            if parent not in seen:
                seen[parent] = hop + 1
                by_hop.setdefault(hop + 1, []).append(parent)
                queue.append(parent)
    return by_hop


def pair_pool_for_slice_nodes(parents, slice_nodes, hmax):
    slice_parents = restrict_parents(parents, slice_nodes)
    pool = {hop: [] for hop in range(1, hmax + 1)}
    for desc in sorted(slice_nodes, key=sort_key):
        by_hop = ancestors_by_hop(slice_parents, desc, hmax)
        for hop in range(1, hmax + 1):
            for anc in by_hop.get(hop, []):
                if desc != anc:
                    pool[hop].append((desc, anc))
    return pool


def pair_pool_for_roots(parents, children, roots, hmax, slice_depth=None):
    slice_nodes = descendants_within(children, roots, max_depth=slice_depth)
    return slice_nodes, pair_pool_for_slice_nodes(parents, slice_nodes, hmax)


def hop_targets(total_pairs, hmax):
    # Remainders are assigned to lower hops deterministically; the manifest records this frozen policy.
    base, rem = divmod(total_pairs, hmax)
    return {hop: base + (1 if hop <= rem else 0) for hop in range(1, hmax + 1)}


def string_key_counts(counts):
    return {str(key): counts[key] for key in sorted(counts)}


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
    attempts = []
    for root in candidates:
        if root in excluded_roots or root in blocked_roots:
            continue
        if root not in children or ADMIN.search(root):
            continue
        slice_nodes = descendants_within(children, (root,), max_depth=slice_depth)
        if len(slice_nodes) < min_descendants:
            attempts.append({
                "root": root,
                "slice_nodes": len(slice_nodes),
                "available_hop_counts": string_key_counts({hop: 0 for hop in targets}),
            })
            continue
        pool = pair_pool_for_slice_nodes(parents, slice_nodes, hmax)
        ok, available = can_satisfy_targets(pool, targets)
        attempts.append({
            "root": root,
            "slice_nodes": len(slice_nodes),
            "available_hop_counts": string_key_counts(available),
        })
        if ok:
            return (root,), slice_nodes, pool, {"candidate_root_attempts": attempts}
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


def iter_dups(txn, db, key):
    cursor = txn.cursor(db=db)
    try:
        if not cursor.set_key(enc_id(key)):
            return []
        return [dec_id(value) for value in cursor.iternext_dup(keys=False, values=True)]
    finally:
        close = getattr(cursor, "close", None)
        if close is not None:
            close()


class LmdbTitleGraph:
    """Read Phase-1 graph edges plus a separate real-title layer.

    The sampler keeps one read transaction open for a short-lived CLI run so all graph and title reads share one
    snapshot. Run the title materializer before this sampler, not concurrently with it; use --lmdb-no-lock only for
    immutable fixtures where stale lock files are a bigger operational risk than concurrent writers.
    """

    def __init__(self, lmdb_dir, title_i2s_db="title_i2s", title_s2i_db="title_s2i", lock=True):
        try:
            import lmdb
        except ImportError as exc:
            raise FreshCorpusError("python-lmdb is required for --candidate-lmdb") from exc

        self.lmdb = lmdb
        self.lmdb_dir = lmdb_dir
        self.env = lmdb.open(str(lmdb_dir), readonly=True, lock=lock, max_dbs=32, subdir=True)
        self.txn = self.env.begin(buffers=True)
        self.category_parent = self.env.open_db(b"category_parent", txn=self.txn, create=False)
        self.category_child = self.env.open_db(b"category_child", txn=self.txn, create=False)
        self.meta = self._open_optional_db("meta")
        self.title_i2s_name, self.title_i2s = self._open_title_db(title_i2s_db, "title_i2s_db", "i2s")
        self.title_s2i_name, self.title_s2i = self._open_title_db(title_s2i_db, "title_s2i_db", "s2i")
        self._title_cache = {}
        self._id_cache = {}
        self._parents = {}
        self._children = {}

    def _open_optional_db(self, name):
        try:
            return self.env.open_db(name.encode("utf-8"), txn=self.txn, create=False)
        except self.lmdb.NotFoundError:
            return None

    def _open_title_db(self, preferred, meta_key, fallback):
        for name in (preferred, self.meta_text(meta_key), fallback):
            if not name:
                continue
            db = self._open_optional_db(name)
            if db is not None:
                return name, db
        raise FreshCorpusError(
            f"candidate LMDB has no real title layer: missing `{preferred}`, meta `{meta_key}`, and legacy `{fallback}` sub-dbs"
        )

    def close(self):
        try:
            txn = getattr(self, "txn", None)
            if txn is not None:
                try:
                    txn.abort()
                except Exception:
                    pass
                self.txn = None
        finally:
            self.env.close()

    def meta_text(self, key):
        if self.meta is None:
            return None
        raw = self.txn.get(key.encode("utf-8"), db=self.meta)
        if raw is None:
            return None
        return bytes(raw).decode("utf-8")

    def meta_int(self, key):
        text = self.meta_text(key)
        if text is None or not looks_int(text):
            return None
        return int(text)

    def title(self, node_id):
        node_id = int(node_id)
        if node_id in self._title_cache:
            return self._title_cache[node_id]
        raw = self.txn.get(enc_id(node_id), db=self.title_i2s)
        if raw is None:
            raise FreshCorpusError(f"candidate LMDB title layer lacks node id {node_id}")
        title = bytes(raw).decode("utf-8")
        if is_identity_numeric_title(node_id, title):
            raise FreshCorpusError(
                "candidate LMDB title layer is identity numeric; materialize real titles into "
                "separate `title_i2s`/`title_s2i` sub-dbs before producing score-input rows"
            )
        self._title_cache[node_id] = title
        self._id_cache.setdefault(title, node_id)
        return title

    def node_id(self, title_or_id):
        if title_or_id in self._id_cache:
            return self._id_cache[title_or_id]
        candidates = [title_or_id]
        if title_or_id.startswith("Category:"):
            candidates.append(title_or_id[len("Category:"):])
        for title in candidates:
            raw = self.txn.get(title.encode("utf-8"), db=self.title_s2i)
            if raw is not None:
                node_id = dec_id(raw)
                self._id_cache[title_or_id] = node_id
                self._id_cache.setdefault(title, node_id)
                return node_id
        if looks_int(title_or_id):
            node_id = int(title_or_id)
            self.title(node_id)
            return node_id
        raise FreshCorpusError(f"candidate LMDB title layer cannot resolve category title `{title_or_id}`")

    def parents(self, node_id):
        node_id = int(node_id)
        if node_id not in self._parents:
            self._parents[node_id] = iter_dups(self.txn, self.category_parent, node_id)
        return self._parents[node_id]

    def children(self, node_id):
        node_id = int(node_id)
        if node_id not in self._children:
            self._children[node_id] = iter_dups(self.txn, self.category_child, node_id)
        return self._children[node_id]


def title_block_reason(title, exploratory_nodes):
    if title in exploratory_nodes:
        return "overlap_node"
    if ADMIN.search(title):
        return "admin_node"
    return None


def load_lmdb_slice_maps(graph, root_id, exploratory_nodes, slice_depth=None):
    stats = Counter()

    def title_for(node_id):
        stats["title_lookups"] += 1
        return graph.title(node_id)

    root_title = title_for(root_id)
    reason = title_block_reason(root_title, exploratory_nodes)
    if reason:
        stats[reason] += 1
        return root_title, set(), {}, {}, stats

    kept_ids = {root_id}
    depth = {root_id: 0}
    queue = deque([root_id])
    while queue:
        node = queue.popleft()
        if slice_depth is not None and depth[node] >= slice_depth:
            continue
        child_ids = graph.children(node)
        stats["child_edges_examined"] += len(child_ids)
        child_records = []
        for child_id in child_ids:
            child_title = title_for(child_id)
            reason = title_block_reason(child_title, exploratory_nodes)
            if reason:
                stats[reason] += 1
                continue
            child_records.append((sort_key(child_title), child_id))
        for _key, child_id in sorted(child_records):
            if child_id not in kept_ids:
                kept_ids.add(child_id)
                depth[child_id] = depth[node] + 1
                queue.append(child_id)

    parents, children = {}, {}
    for node_id in kept_ids:
        title = title_for(node_id)
        parents.setdefault(title, set())
        children.setdefault(title, set())

    for child_id in sorted(kept_ids, key=lambda nid: sort_key(title_for(nid))):
        child_title = title_for(child_id)
        parent_ids = graph.parents(child_id)
        stats["parent_edges_examined"] += len(parent_ids)
        for parent_id in parent_ids:
            if parent_id not in kept_ids:
                continue
            parent_title = title_for(parent_id)
            parents[child_title].add(parent_title)
            children[parent_title].add(child_title)
            stats["retained_edges"] += 1
    return root_title, {title_for(node_id) for node_id in kept_ids}, parents, children, stats


def lmdb_candidate_root_ids(graph, roots, scope_root):
    if roots:
        return [graph.node_id(root) for root in roots], "user-supplied root validation", None

    scope_id = None
    scope_title = scope_root
    if scope_root:
        scope_id = graph.node_id(scope_root)
        scope_title = graph.title(scope_id)
    else:
        scope_id = graph.meta_int("scoped_root")
        if scope_id is None:
            scope_id = graph.meta_int("metric_min_dist_to_root.root")
        if scope_id is not None:
            scope_title = graph.title(scope_id)
    if scope_id is None:
        raise FreshCorpusError("--candidate-lmdb requires --root, --scope-root, or LMDB meta.scoped_root")

    children = graph.children(scope_id)
    if not children:
        raise FreshCorpusError(f"scope root `{scope_title}` has no candidate child roots in LMDB")
    return children, "casefold-lexicographically smallest eligible direct child of scope root", {
        "scope_root": scope_title,
        "scope_root_id": scope_id,
        "scope_child_roots": len(children),
    }


def first_eligible_lmdb_roots(graph, root_ids, exploratory_nodes, excluded_roots, hmax, targets, min_descendants, slice_depth):
    candidates = sorted(root_ids, key=lambda nid: sort_key(graph.title(nid)))
    attempts = []
    for root_id in candidates:
        root_title = graph.title(root_id)
        if root_title in excluded_roots or ADMIN.search(root_title):
            continue
        root_title, slice_nodes, parents, children, stats = load_lmdb_slice_maps(
            graph, root_id, exploratory_nodes, slice_depth=slice_depth
        )
        pool = {hop: [] for hop in range(1, hmax + 1)}
        available = {hop: 0 for hop in range(1, hmax + 1)}
        ok = False
        if len(slice_nodes) >= min_descendants:
            # `slice_nodes` was already BFS-limited by the LMDB traversal above; do not reapply a second depth cap.
            pool = pair_pool_for_slice_nodes(parents, slice_nodes, hmax)
            ok, available = can_satisfy_targets(pool, targets)
        attempts.append({
            "root": root_title,
            "root_id": root_id,
            "slice_nodes": len(slice_nodes),
            "available_hop_counts": string_key_counts(available),
            "lmdb_slice_stats": dict(sorted(stats.items())),
        })
        if len(slice_nodes) < min_descendants:
            continue
        if ok:
            return (root_title,), slice_nodes, pool, {
                "candidate_root_attempts": attempts,
                "selected_root_id": root_id,
                "selected_lmdb_slice_stats": dict(sorted(stats.items())),
            }
    raise FreshCorpusError("no eligible LMDB root slice supplied enough no-overlap pairs for every hop")


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


def validate_args(args):
    if args.candidate_graph and os.path.realpath(args.candidate_graph) == os.path.realpath(args.exploratory_graph):
        raise FreshCorpusError("--candidate-graph and --exploratory-graph must be different files")
    if args.pairs <= 0:
        raise FreshCorpusError("--pairs must be positive")
    if args.hmax <= 0:
        raise FreshCorpusError("--hmax must be positive")
    if args.min_descendants <= 0:
        raise FreshCorpusError("--min-descendants must be positive")
    if args.root and args.scope_root:
        raise FreshCorpusError("--root and --scope-root are mutually exclusive; pass one concrete root or one broad scope")
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


def sample_from_tsv(args, exploratory_nodes, candidate_stats, targets):
    candidate_edges = load_edges(args.candidate_graph, stats=candidate_stats)
    filtered_edges, removed, blocked_roots = filter_candidate_edges(candidate_edges, exploratory_nodes)
    parents, children = build_maps(filtered_edges)
    excluded_roots = set(args.exclude_root)
    roots = tuple(root for root in args.root if root not in excluded_roots)
    if args.root and not roots:
        raise FreshCorpusError("all supplied --root values were excluded by --exclude-root")
    selection_rule = "user-supplied root validation" if args.root else "casefold-lexicographically smallest eligible root after no-overlap/admin filtering"
    if args.scope_root and not args.root:
        if args.scope_root not in children:
            raise FreshCorpusError(f"--scope-root `{args.scope_root}` is absent after TSV filtering")
        roots = tuple(sorted(children[args.scope_root], key=sort_key))
        selection_rule = "casefold-lexicographically smallest eligible direct child of scope root"
    selected_roots, slice_nodes, pool, extra = first_eligible_roots(
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
    source_manifest = {
        "candidate_source_kind": "tsv",
        "candidate_graph": args.candidate_graph,
        "candidate_lmdb": None,
        "candidate_edges": len(candidate_edges),
        "filtered_edges": len(filtered_edges),
        "removed_edges": removed,
        "blocked_root_candidates": len(blocked_roots),
        "edge_file_stats": {"candidate": candidate_stats},
        "selection_rule": selection_rule,
        "node_filter_semantics": "drop every retained node whose title is exploratory-overlap or admin; retain only edges with both endpoints kept",
        "traversal_order": "casefold-title order for candidate roots, descendants, and ancestors",
    }
    source_manifest.update(extra)
    return selected_roots, slice_nodes, pool, source_manifest


def sample_from_lmdb(args, exploratory_nodes, targets):
    excluded_roots = set(args.exclude_root)
    graph = LmdbTitleGraph(args.candidate_lmdb, args.title_i2s_db, args.title_s2i_db, lock=not args.lmdb_no_lock)
    try:
        root_ids, selection_rule, scope_info = lmdb_candidate_root_ids(graph, args.root, args.scope_root)
        selected_roots, slice_nodes, pool, extra = first_eligible_lmdb_roots(
            graph,
            root_ids,
            exploratory_nodes,
            excluded_roots,
            args.hmax,
            targets,
            args.min_descendants,
            args.slice_depth,
        )
        source_manifest = {
            "candidate_source_kind": "lmdb",
            "candidate_graph": None,
            "candidate_lmdb": args.candidate_lmdb,
            "candidate_edges": None,
            "filtered_edges": None,
            "removed_edges": None,
            "blocked_root_candidates": None,
            "edge_file_stats": {"candidate": {}},
            "selection_rule": selection_rule,
            "node_filter_semantics": "drop every retained node whose title is exploratory-overlap or admin; retain only edges with both endpoints kept",
            "traversal_order": "casefold-title order for candidate roots, descendants, and ancestors",
            "lmdb_lock": not args.lmdb_no_lock,
            "title_i2s_db": graph.title_i2s_name,
            "title_s2i_db": graph.title_s2i_name,
            "title_layer_kind": graph.meta_text("title_layer_kind"),
        }
        if scope_info:
            source_manifest.update(scope_info)
        source_manifest.update(extra)
        return selected_roots, slice_nodes, pool, source_manifest
    finally:
        graph.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate-graph", help="fresh/later Wikipedia category child<TAB>parent graph")
    source.add_argument("--candidate-lmdb", help="fresh/later Phase-1 category LMDB with category_parent/category_child")
    ap.add_argument("--exploratory-graph", required=True, help="PR #3517 exploratory 100k_cats/category_parent.tsv")
    ap.add_argument("--root", action="append", default=[], help="candidate root slice to validate/sample")
    ap.add_argument("--scope-root", help="broad LMDB/TSV root whose direct children are candidate root slices")
    ap.add_argument("--exclude-root", action="append", default=[], help="exploratory seed/root category to disallow")
    ap.add_argument("--title-i2s-db", default="title_i2s", help="LMDB uint32 id -> real category title sub-db; meta.title_i2s_db is also honored")
    ap.add_argument("--title-s2i-db", default="title_s2i", help="LMDB real category title -> uint32 id sub-db; meta.title_s2i_db is also honored")
    ap.add_argument("--lmdb-no-lock", action="store_true", help="open candidate LMDB with lock=False; use only for immutable fixtures")
    ap.add_argument("--pairs", type=int, default=250)
    ap.add_argument("--hmax", type=int, default=5)
    ap.add_argument("--min-descendants", type=int, default=300)
    ap.add_argument("--slice-depth", type=int, default=None, help="optional downward closure cap from selected root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--allow-small-sample", action="store_true", help="permit non-confirmatory toy/dry-run sizes")
    args = ap.parse_args()

    validate_args(args)

    targets = hop_targets(args.pairs, args.hmax)
    exploratory_stats, candidate_stats = {}, {}
    exploratory_edges = load_edges(args.exploratory_graph, stats=exploratory_stats)
    exploratory_nodes = node_block(exploratory_edges)
    if args.candidate_lmdb:
        selected_roots, slice_nodes, pool, source_manifest = sample_from_lmdb(args, exploratory_nodes, targets)
    else:
        selected_roots, slice_nodes, pool, source_manifest = sample_from_tsv(args, exploratory_nodes, candidate_stats, targets)

    rows, counts = sample_balanced_pairs(pool, args.pairs, args.hmax, args.seed)
    overlap = node_block((desc, anc) for desc, anc, _ in rows) & exploratory_nodes
    overlap_count = len(overlap)
    if overlap:
        raise FreshCorpusError(f"sampled pairs overlap exploratory nodes: {sorted(overlap)[:10]}")
    write_score_in(rows, args.out)
    manifest = {
        "exploratory_graph": args.exploratory_graph,
        "selected_roots": list(selected_roots),
        "excluded_roots": sorted(args.exclude_root),
        "scope_root": args.scope_root,
        "seed": args.seed,
        "hmax": args.hmax,
        "allow_small_sample": args.allow_small_sample,
        "requested_pairs": args.pairs,
        "written_pairs": len(rows),
        "target_hop_counts": {str(h): targets[h] for h in range(1, args.hmax + 1)},
        "hop_targeting": "deterministic balanced counts; any remainder is assigned to lower hop numbers",
        "hop_counts": {str(h): counts[h] for h in range(1, args.hmax + 1)},
        "hop_semantics": "shortest upward graph distance within retained slice",
        "row_shuffle_seed": args.seed ^ 0x5EED5EED,
        "slice_nodes": len(slice_nodes),
        "edge_file_stats": {"exploratory": exploratory_stats},
        "node_overlap_with_exploratory": overlap_count,
        "output": args.out,
    }
    manifest.update(source_manifest)
    if source_manifest.get("edge_file_stats"):
        merged_stats = dict(source_manifest["edge_file_stats"])
        merged_stats["exploratory"] = exploratory_stats
        manifest["edge_file_stats"] = merged_stats
    write_manifest(args.manifest, manifest)
    print(f"selected root(s): {', '.join(selected_roots)}")
    print(f"wrote {len(rows)} no-overlap score-in pairs -> {args.out}")
    print("hop counts:", dict(sorted(counts.items())))
    print(f"manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
