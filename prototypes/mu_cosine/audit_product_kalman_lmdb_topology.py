#!/usr/bin/env python3
"""Stream a descriptive topology audit from numeric category LMDB tables.

The graph scan touches only uint32 keys and duplicate counts. A title lookup is
performed only for the requested scope root so graph-scale work stays numeric.
Degree statistics describe corpus structure; they are not confidence features.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

from lmdb_id import dec_id, enc_id, looks_int


SCHEMA_VERSION = 1
QUANTILE_METHOD = "nearest-rank over all graph nodes, including degree zero"


class TopologyAuditError(ValueError):
    pass


def _open_optional_db(env, txn, lmdb_module, name):
    try:
        return env.open_db(name.encode("utf-8"), txn=txn, create=False)
    except lmdb_module.NotFoundError:
        return None


def _meta_text(txn, meta, key):
    if meta is None:
        return None
    raw = txn.get(key.encode("utf-8"), db=meta)
    return None if raw is None else bytes(raw).decode("utf-8")


def _named_db(env, txn, lmdb_module, preferred, meta, meta_key, required):
    candidates = []
    for name in (preferred, _meta_text(txn, meta, meta_key)):
        if name and name not in candidates:
            candidates.append(name)
    for name in candidates:
        db = _open_optional_db(env, txn, lmdb_module, name)
        if db is not None:
            return name, db
    if required:
        raise TopologyAuditError(f"LMDB is missing required sub-db `{preferred}`")
    return None, None


def _iter_raw_key_counts(txn, db):
    cursor = txn.cursor(db=db)
    try:
        present = cursor.first()
        while present:
            yield bytes(cursor.key()), cursor.count()
            present = cursor.next_nodup()
    finally:
        cursor.close()


def _next_or_none(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


def merged_degree_histograms(txn, category_parent, category_child):
    """Return exact all-node parent/child degree histograms without ID sets."""
    parent_items = iter(_iter_raw_key_counts(txn, category_parent))
    child_items = iter(_iter_raw_key_counts(txn, category_child))
    parent_item = _next_or_none(parent_items)
    child_item = _next_or_none(child_items)
    parent_hist = Counter()
    child_hist = Counter()
    node_count = 0

    while parent_item is not None or child_item is not None:
        if child_item is None or (parent_item is not None and parent_item[0] < child_item[0]):
            parent_degree, child_degree = parent_item[1], 0
            parent_item = _next_or_none(parent_items)
        elif parent_item is None or child_item[0] < parent_item[0]:
            parent_degree, child_degree = 0, child_item[1]
            child_item = _next_or_none(child_items)
        else:
            parent_degree, child_degree = parent_item[1], child_item[1]
            parent_item = _next_or_none(parent_items)
            child_item = _next_or_none(child_items)
        parent_hist[parent_degree] += 1
        child_hist[child_degree] += 1
        node_count += 1

    return node_count, parent_hist, child_hist


def nearest_rank(histogram, quantile):
    count = sum(histogram.values())
    if count == 0:
        return None
    rank = max(1, math.ceil(float(quantile) * count))
    cumulative = 0
    for value, frequency in sorted(histogram.items()):
        cumulative += frequency
        if cumulative >= rank:
            return value
    raise AssertionError("histogram rank exceeded total count")


def summarize_degree(histogram, node_count, relation):
    if sum(histogram.values()) != node_count:
        raise TopologyAuditError(f"{relation} histogram does not cover every graph node")
    edge_ends = sum(degree * count for degree, count in histogram.items())
    multiple = sum(count for degree, count in histogram.items() if degree > 1)
    return {
        "relation": relation,
        "denominator": "all graph nodes in the union of numeric adjacency keys",
        "node_count": node_count,
        "keyed_node_count": node_count - histogram.get(0, 0),
        "zero_degree_nodes": histogram.get(0, 0),
        "degree_gt_one_nodes": multiple,
        "degree_gt_one_fraction": multiple / node_count if node_count else None,
        "edge_end_count": edge_ends,
        "mean": edge_ends / node_count if node_count else None,
        "median": nearest_rank(histogram, 0.5),
        "p95": nearest_rank(histogram, 0.95),
        "p99": nearest_rank(histogram, 0.99),
        "max": max(histogram, default=None),
        "histogram": {str(degree): histogram[degree] for degree in sorted(histogram)},
        "quantile_method": QUANTILE_METHOD,
    }


def _duplicate_count(txn, db, key):
    cursor = txn.cursor(db=db)
    try:
        return cursor.count() if cursor.set_key(key) else 0
    finally:
        cursor.close()


def _resolve_root(txn, scope_root, category_child, title_i2s, title_s2i):
    if scope_root is None:
        return None
    if looks_int(scope_root):
        root_id = int(scope_root)
    else:
        if title_s2i is None:
            raise TopologyAuditError("a non-numeric --scope-root requires a title_s2i table")
        raw = txn.get(scope_root.encode("utf-8"), db=title_s2i)
        if raw is None and scope_root.startswith("Category:"):
            raw = txn.get(scope_root[len("Category:"):].encode("utf-8"), db=title_s2i)
        if raw is None:
            raise TopologyAuditError(f"title_s2i cannot resolve scope root `{scope_root}`")
        root_id = dec_id(raw)
    raw_title = None if title_i2s is None else txn.get(enc_id(root_id), db=title_i2s)
    return {
        "requested": scope_root,
        "id": root_id,
        "title": None if raw_title is None else bytes(raw_title).decode("utf-8"),
        "direct_child_count": _duplicate_count(txn, category_child, enc_id(root_id)),
    }


def audit_lmdb_topology(
    lmdb_dir,
    scope_root=None,
    corpus=None,
    graph_view=None,
    category_parent_db="category_parent",
    category_child_db="category_child",
    title_i2s_db="title_i2s",
    title_s2i_db="title_s2i",
    lock=True,
):
    try:
        import lmdb
    except ImportError as exc:
        raise TopologyAuditError("python-lmdb is required") from exc

    lmdb_dir = Path(lmdb_dir)
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=lock, max_dbs=32, subdir=True)
    try:
        with env.begin(buffers=True) as txn:
            meta = _open_optional_db(env, txn, lmdb, "meta")
            parent_name, category_parent = _named_db(
                env, txn, lmdb, category_parent_db, meta, "category_parent_db", required=True
            )
            child_name, category_child = _named_db(
                env, txn, lmdb, category_child_db, meta, "category_child_db", required=True
            )
            title_i2s_name, title_i2s = _named_db(
                env, txn, lmdb, title_i2s_db, meta, "title_i2s_db", required=False
            )
            title_s2i_name, title_s2i = _named_db(
                env, txn, lmdb, title_s2i_db, meta, "title_s2i_db", required=False
            )
            parent_entries = txn.stat(category_parent)["entries"]
            child_entries = txn.stat(category_child)["entries"]
            node_count, parent_hist, child_hist = merged_degree_histograms(
                txn, category_parent, category_child
            )
            root = _resolve_root(txn, scope_root, category_child, title_i2s, title_s2i)
            title_entries = None if title_i2s is None else txn.stat(title_i2s)["entries"]

            return {
                "schema_version": SCHEMA_VERSION,
                "audit_kind": "numeric_lmdb_topology",
                "corpus": corpus,
                "graph_view": graph_view,
                "source": {
                    "lmdb_dir": str(lmdb_dir),
                    "data_size": (lmdb_dir / "data.mdb").stat().st_size,
                    "data_mtime_ns": (lmdb_dir / "data.mdb").stat().st_mtime_ns,
                    "category_parent_db": parent_name,
                    "category_child_db": child_name,
                    "title_i2s_db": title_i2s_name,
                    "title_s2i_db": title_s2i_name,
                    "id_encoding": "unsigned uint32 little-endian",
                    "scan_boundary": "numeric adjacency keys and duplicate counts",
                    "title_lookup_boundary": "optional scope-root provenance only",
                    "title_layer_kind": _meta_text(txn, meta, "title_layer_kind"),
                },
                "root": root,
                "integrity": {
                    "category_parent_entries": parent_entries,
                    "category_child_entries": child_entries,
                    "reciprocal_entry_counts_match": parent_entries == child_entries,
                    "note": "entry-count equality is checked; individual reciprocal edges are not re-materialized",
                },
                "nodes": {
                    "graph_node_count": node_count,
                    "title_i2s_entries": title_entries,
                    "title_entries_per_graph_node": (
                        title_entries / node_count if title_entries is not None and node_count else None
                    ),
                },
                "parent_degree": summarize_degree(parent_hist, node_count, "number of parents per node"),
                "child_degree": summarize_degree(child_hist, node_count, "number of children per node"),
                "interpretation": {
                    "purpose": "describe and compare corpus topology before calibration",
                    "tree_likeness_proxy": "fraction of all graph nodes with more than one parent",
                    "confidence_use": "forbidden: node degree is not trained-neighbor density or calibrated confidence",
                    "causal_claims": "forbidden without within-corpus held-out ablations",
                    "cycle_status": "not evaluated by this streaming degree audit",
                },
            }
    finally:
        env.close()


def write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lmdb-dir", required=True, type=Path)
    ap.add_argument("--scope-root", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--graph-view", default=None)
    ap.add_argument("--category-parent-db", default="category_parent")
    ap.add_argument("--category-child-db", default="category_child")
    ap.add_argument("--title-i2s-db", default="title_i2s")
    ap.add_argument("--title-s2i-db", default="title_s2i")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--lmdb-no-lock", action="store_true")
    args = ap.parse_args(argv)

    report = audit_lmdb_topology(
        args.lmdb_dir,
        scope_root=args.scope_root,
        corpus=args.corpus,
        graph_view=args.graph_view,
        category_parent_db=args.category_parent_db,
        category_child_db=args.category_child_db,
        title_i2s_db=args.title_i2s_db,
        title_s2i_db=args.title_s2i_db,
        lock=not args.lmdb_no_lock,
    )
    write_json_atomic(args.output, report)
    print(f"audited {report['nodes']['graph_node_count']} numeric graph nodes")
    print(f"report -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
