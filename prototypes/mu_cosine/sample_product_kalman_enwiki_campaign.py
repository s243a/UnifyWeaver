#!/usr/bin/env python3
"""Sample a branch-stratified enwiki campaign from a titled category LMDB.

Graph traversal stays on uint32 IDs. Titles are joined only for the scope root,
direct branch inventory, and structurally accepted pair endpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from sample_sigma_hop_fresh_corpus import ADMIN, FreshCorpusError, LmdbTitleGraph, hop_targets, load_edges, node_block


SCHEMA_VERSION = 1
CORPUS = "enwiki"
GRAPH_VIEW = "category_dag"
CONFLICTED_COPY = re.compile(r"\b(?:conflicted|conflicting) copy\b", re.IGNORECASE)
REPEATED_SEPARATOR = re.compile(r"([_\-.,:;])\1{2,}")


def normalize_title(title):
    text = unicodedata.normalize("NFKC", str(title)).replace("_", " ")
    return " ".join(text.split()).casefold()


def title_quality_flags(title):
    flags = []
    if not title:
        flags.append("empty")
    if title != title.strip():
        flags.append("outer_whitespace")
    if "\ufffd" in title:
        flags.append("replacement_character")
    if CONFLICTED_COPY.search(title):
        flags.append("conflicted_copy")
    if REPEATED_SEPARATOR.search(title):
        flags.append("repeated_separator")
    return flags


def shortest_upward_distance(graph, descendant_id, ancestor_id, max_hop):
    descendant_id = int(descendant_id)
    ancestor_id = int(ancestor_id)
    if descendant_id == ancestor_id:
        return 0
    seen = {descendant_id}
    frontier = [descendant_id]
    for hop in range(1, max_hop + 1):
        nxt = set()
        for node_id in frontier:
            for parent_id in graph.parents(node_id):
                if parent_id == ancestor_id:
                    return hop
                if parent_id not in seen:
                    seen.add(parent_id)
                    nxt.add(parent_id)
        if not nxt:
            break
        frontier = sorted(nxt)
    return None


def deterministic_rng(seed, hop, branch_id, attempt):
    payload = f"{seed}:{hop}:{branch_id}:{attempt}".encode("ascii")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return random.Random(value)


def random_downward_path(graph, start_id, length, rng):
    path = [int(start_id)]
    seen = {int(start_id)}
    for _ in range(length):
        children = [child for child in sorted(set(graph.children(path[-1]))) if child not in seen]
        if not children:
            return None
        child = children[rng.randrange(len(children))]
        path.append(child)
        seen.add(child)
    return path


def load_excluded_titles(path):
    if not path:
        return set()
    return node_block(load_edges(path))


def branch_inventory(graph, scope_id, excluded_branches):
    records = []
    skipped = Counter()
    for branch_id in sorted(set(graph.children(scope_id))):
        title = graph.title(branch_id, missing_ok=True)
        if title is None:
            skipped["missing_title"] += 1
            continue
        if title in excluded_branches:
            skipped["explicitly_excluded"] += 1
            continue
        if ADMIN.search(title):
            skipped["admin_title"] += 1
            continue
        records.append({"branch_id": branch_id, "branch_title": title})
    records.sort(key=lambda item: (item["branch_title"].casefold(), item["branch_title"], item["branch_id"]))
    if not records:
        raise FreshCorpusError("scope root has no eligible titled direct branches")
    return records, dict(sorted(skipped.items()))


def endpoint_record(graph, node_id, excluded_normalized_titles, stats):
    stats["endpoint_title_lookups"] += 1
    title = graph.title(node_id, missing_ok=True)
    if title is None:
        stats["missing_endpoint_title"] += 1
        return None
    normalized_title = normalize_title(title)
    if normalized_title in excluded_normalized_titles:
        stats["excluded_title"] += 1
        return None
    if ADMIN.search(title):
        stats["admin_endpoint_title"] += 1
        return None
    return {
        "id": int(node_id),
        "title": title,
        "normalized_title": normalized_title,
        "quality_flags": title_quality_flags(title),
    }


def sample_campaign_pairs(
    graph,
    scope_title="Main_topic_classifications",
    pairs=250,
    hmax=5,
    seed=0,
    max_prefix_depth=5,
    attempts_per_branch_visit=8,
    max_attempts_per_hop=20000,
    excluded_titles=(),
    excluded_branches=(),
):
    if pairs <= 0 or hmax <= 0:
        raise FreshCorpusError("pairs and hmax must be positive")
    if max_prefix_depth < 0 or attempts_per_branch_visit <= 0 or max_attempts_per_hop <= 0:
        raise FreshCorpusError("sampling limits must be positive, with max_prefix_depth >= 0")

    scope_id = graph.node_id(scope_title)
    resolved_scope_title = graph.title(scope_id)
    branches, skipped_branches = branch_inventory(graph, scope_id, set(excluded_branches))
    targets = hop_targets(pairs, hmax)
    excluded_titles = {normalize_title(title) for title in excluded_titles}
    rows = []
    seen_pairs = set()
    attempts = Counter()
    accepted_by_hop_branch = defaultdict(Counter)

    for hop in range(1, hmax + 1):
        accepted = 0
        branch_attempt_index = Counter()
        while accepted < targets[hop] and attempts[f"hop_{hop}_total"] < max_attempts_per_hop:
            cycle_progress = False
            for branch in branches:
                if accepted >= targets[hop]:
                    break
                branch_id = branch["branch_id"]
                candidate = None
                for _ in range(attempts_per_branch_visit):
                    if attempts[f"hop_{hop}_total"] >= max_attempts_per_hop:
                        break
                    attempt = branch_attempt_index[branch_id]
                    branch_attempt_index[branch_id] += 1
                    attempts[f"hop_{hop}_total"] += 1
                    attempts["total"] += 1
                    rng = deterministic_rng(seed, hop, branch_id, attempt)
                    prefix = rng.randrange(max_prefix_depth + 1)
                    path = random_downward_path(graph, branch_id, prefix + hop, rng)
                    if path is None:
                        attempts["dead_end"] += 1
                        continue
                    ancestor_id = path[prefix]
                    descendant_id = path[-1]
                    distance = shortest_upward_distance(graph, descendant_id, ancestor_id, hop)
                    if distance != hop:
                        attempts["shorter_dag_path"] += 1
                        continue
                    pair_key = tuple(sorted((descendant_id, ancestor_id)))
                    if pair_key in seen_pairs:
                        attempts["duplicate_pair"] += 1
                        continue
                    descendant = endpoint_record(graph, descendant_id, excluded_titles, attempts)
                    ancestor = endpoint_record(graph, ancestor_id, excluded_titles, attempts)
                    if descendant is None or ancestor is None:
                        continue
                    candidate = {
                        "pair_id": f"enwiki-h{hop}-{accepted:04d}",
                        "corpus": CORPUS,
                        "graph_view": GRAPH_VIEW,
                        "branch_id": branch_id,
                        "branch_title": branch["branch_title"],
                        "descendant": descendant,
                        "ancestor": ancestor,
                        "hop": hop,
                    }
                    seen_pairs.add(pair_key)
                    break
                if candidate is not None:
                    rows.append(candidate)
                    accepted += 1
                    cycle_progress = True
                    accepted_by_hop_branch[hop][branch["branch_title"]] += 1
            if not cycle_progress and attempts[f"hop_{hop}_total"] >= max_attempts_per_hop:
                break
        if accepted < targets[hop]:
            raise FreshCorpusError(
                f"hop {hop} produced {accepted} pairs; need {targets[hop]} after "
                f"{attempts[f'hop_{hop}_total']} attempts"
            )

    return rows, {
        "scope_root_id": scope_id,
        "scope_root_title": resolved_scope_title,
        "eligible_branch_count": len(branches),
        "eligible_branches": branches,
        "skipped_branches": skipped_branches,
        "target_hop_counts": {str(hop): targets[hop] for hop in sorted(targets)},
        "hop_counts": {str(hop): sum(1 for row in rows if row["hop"] == hop) for hop in sorted(targets)},
        "accepted_by_hop_branch": {
            str(hop): dict(sorted(accepted_by_hop_branch[hop].items())) for hop in sorted(targets)
        },
        "sampling_stats": dict(sorted(attempts.items())),
    }


def title_audit(rows):
    titles_by_id = {}
    for row in rows:
        for endpoint in (row["descendant"], row["ancestor"]):
            titles_by_id.setdefault(endpoint["id"], endpoint)
    normalized = defaultdict(list)
    flags = Counter()
    for endpoint in titles_by_id.values():
        normalized[endpoint["normalized_title"]].append(endpoint["id"])
        flags.update(endpoint["quality_flags"])
    duplicates = {title: ids for title, ids in normalized.items() if len(ids) > 1}
    return {
        "unique_endpoint_ids": len(titles_by_id),
        "unique_raw_titles": len({item["title"] for item in titles_by_id.values()}),
        "unique_normalized_titles": len(normalized),
        "duplicate_normalized_title_groups": len(duplicates),
        "duplicate_normalized_title_ids": sum(len(ids) for ids in duplicates.values()),
        "quality_flag_counts": dict(sorted(flags.items())),
        "normalization": "NFKC; underscore-to-space; whitespace-collapse; casefold",
        "semantic_corrections_applied": False,
    }


def write_pairs_tsv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id", "corpus", "graph_view", "branch_id", "branch_title", "descendant_id",
        "descendant_title", "descendant_normalized_title", "ancestor_id", "ancestor_title",
        "ancestor_normalized_title", "hop",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "pair_id": row["pair_id"],
                "corpus": row["corpus"],
                "graph_view": row["graph_view"],
                "branch_id": row["branch_id"],
                "branch_title": row["branch_title"],
                "descendant_id": row["descendant"]["id"],
                "descendant_title": row["descendant"]["title"],
                "descendant_normalized_title": row["descendant"]["normalized_title"],
                "ancestor_id": row["ancestor"]["id"],
                "ancestor_title": row["ancestor"]["title"],
                "ancestor_normalized_title": row["ancestor"]["normalized_title"],
                "hop": row["hop"],
            })


def write_score_in(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# node_title\troot_title\tcur_relation\tconf\tneighborhood\tnode_type\troot_type\traw\n")
        for row in rows:
            f.write(
                f"{row['descendant']['title']}\t{row['ancestor']['title']}\tsubcategory\t1.0\t"
                f"transitive_h{row['hop']}\tcategory\tcategory\t\n"
            )


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
    ap.add_argument("--scope-root", default="Main_topic_classifications")
    ap.add_argument("--pairs", type=int, default=250)
    ap.add_argument("--hmax", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prefix-depth", type=int, default=5)
    ap.add_argument("--attempts-per-branch-visit", type=int, default=8)
    ap.add_argument("--max-attempts-per-hop", type=int, default=20000)
    ap.add_argument("--exclude-graph", help="optional child<TAB>parent title graph whose nodes are excluded")
    ap.add_argument("--exclude-branch", action="append", default=[])
    ap.add_argument("--pairs-tsv", required=True, type=Path)
    ap.add_argument("--score-in", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--lmdb-no-lock", action="store_true")
    ap.add_argument("--allow-small-sample", action="store_true")
    args = ap.parse_args(argv)
    if not args.allow_small_sample and args.pairs < 250:
        ap.error("campaign sampling requires at least 250 pairs; use --allow-small-sample for tests")

    excluded_titles = load_excluded_titles(args.exclude_graph)
    graph = LmdbTitleGraph(args.lmdb_dir, lock=not args.lmdb_no_lock)
    try:
        rows, summary = sample_campaign_pairs(
            graph,
            scope_title=args.scope_root,
            pairs=args.pairs,
            hmax=args.hmax,
            seed=args.seed,
            max_prefix_depth=args.max_prefix_depth,
            attempts_per_branch_visit=args.attempts_per_branch_visit,
            max_attempts_per_hop=args.max_attempts_per_hop,
            excluded_titles=excluded_titles,
            excluded_branches=args.exclude_branch,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "corpus": CORPUS,
            "graph_view": GRAPH_VIEW,
            "lmdb_dir": str(args.lmdb_dir),
            "lmdb_data_size": (args.lmdb_dir / "data.mdb").stat().st_size,
            "lmdb_data_mtime_ns": (args.lmdb_dir / "data.mdb").stat().st_mtime_ns,
            "category_parent_entries": graph.txn.stat(graph.category_parent)["entries"],
            "category_child_entries": graph.txn.stat(graph.category_child)["entries"],
            "title_i2s_db": graph.title_i2s_name,
            "title_s2i_db": graph.title_s2i_name,
            "title_i2s_entries": graph.txn.stat(graph.title_i2s)["entries"],
            "title_s2i_entries": graph.txn.stat(graph.title_s2i)["entries"],
            "title_layer_kind": graph.meta_text("title_layer_kind"),
            "title_layer_count": graph.meta_int("title_layer_count"),
            "sampling_method": "branch-round-robin deterministic numeric downward walks; exact shortest upward hop verified",
            "title_lookup_phase": "scope/branch inventory and structurally accepted endpoint boundary only",
            "pairs": len(rows),
            "hmax": args.hmax,
            "seed": args.seed,
            "max_prefix_depth": args.max_prefix_depth,
            "attempts_per_branch_visit": args.attempts_per_branch_visit,
            "max_attempts_per_hop": args.max_attempts_per_hop,
            "exclude_graph": args.exclude_graph,
            "excluded_title_count": len(excluded_titles),
            "excluded_title_matching": "normalized title: NFKC; underscore-to-space; whitespace-collapse; casefold",
            "excluded_branches": sorted(args.exclude_branch),
            "pairs_tsv": str(args.pairs_tsv),
            "score_in": str(args.score_in),
            "title_audit": title_audit(rows),
        }
        manifest.update(summary)
    finally:
        graph.close()

    write_pairs_tsv(args.pairs_tsv, rows)
    write_score_in(args.score_in, rows)
    write_json_atomic(args.manifest, manifest)
    print(f"sampled {len(rows)} enwiki pairs across {manifest['eligible_branch_count']} direct branches")
    print(f"pairs -> {args.pairs_tsv}")
    print(f"score input -> {args.score_in}")
    print(f"manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
