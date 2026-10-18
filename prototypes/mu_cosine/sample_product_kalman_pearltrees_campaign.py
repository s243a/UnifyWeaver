#!/usr/bin/env python3
"""Sample a hop-balanced Pearltrees campaign from recorded principal paths.

`api_tree_paths` records one user-selected path per tree. Consecutive numeric
IDs define the primary principal-parent view. Alias and secondary-reference
edges from the assembled DAG are intentionally reserved for sensitivity runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from privacy import is_private_title
from sample_product_kalman_enwiki_campaign import (
    load_excluded_titles,
    normalize_title,
    title_audit,
    title_quality_flags,
    write_json_atomic,
)
from sample_sigma_hop_fresh_corpus import FreshCorpusError, hop_targets


SCHEMA_VERSION = 1
CORPUS = "pearltrees"
GRAPH_VIEW = "principal_path_lineage"
HTML_ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-f]+|[a-z][a-z0-9]+);", re.IGNORECASE)


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_title_table(path):
    titles = {}
    duplicates = Counter()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2 or not parts[0]:
                raise FreshCorpusError(f"{path}:{line_no}: expected id<TAB>title")
            node_id, title = parts
            if node_id in titles and titles[node_id] != title:
                duplicates["conflicting_duplicate_title"] += 1
                continue
            titles.setdefault(node_id, title)
    return titles, dict(sorted(duplicates.items()))


def clean_path_ids(raw_ids):
    return tuple(str(item) for item in raw_ids if not str(item).startswith("account:"))


def record_is_private(record, path_ids, titles):
    target_text = str(record.get("target_text") or "")
    if "*private*" in target_text.casefold():
        return True
    if is_private_title(str(record.get("title") or "")):
        return True
    return any(is_private_title(titles.get(node_id, "")) for node_id in path_ids)


def load_principal_paths(paths_jsonl, titles_tsv):
    titles, title_table_stats = load_title_table(titles_tsv)
    records = []
    stats = Counter()
    account_records = Counter()
    title_mismatches = 0
    title_aliases = defaultdict(set)

    with open(paths_jsonl, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            stats["path_records_total"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FreshCorpusError(f"{paths_jsonl}:{line_no}: invalid JSON: {exc}") from exc
            account = str(record.get("account") or "").strip()
            path_ids = clean_path_ids(record.get("path_ids") or ())
            if not account or not path_ids:
                stats["path_records_missing_account_or_path"] += 1
                continue
            tree_id = str(record.get("tree_id") or "")
            record_title = str(record.get("title") or "")
            if record_is_private(record, path_ids, titles):
                stats["path_records_private"] += 1
                continue
            if tree_id and record_title:
                if tree_id in titles and titles[tree_id] != record_title:
                    title_mismatches += 1
                    title_aliases[tree_id].add(record_title)
                titles.setdefault(tree_id, record_title)
            records.append({"account": account, "path_ids": path_ids, "tree_id": tree_id})
            account_records[account] += 1
            stats["path_records_retained"] += 1

    parent_candidates = defaultdict(set)
    nodes = set()
    path_edges = set()
    edge_observations = 0
    for record in records:
        account, path_ids = record["account"], record["path_ids"]
        keyed = tuple((account, node_id) for node_id in path_ids)
        nodes.update(keyed)
        for parent, child in zip(keyed, keyed[1:]):
            if parent != child:
                parent_candidates[child].add(parent)
                path_edges.add((parent, child))
                edge_observations += 1

    conflicts = sum(len(parents) > 1 for parents in parent_candidates.values())
    covered = sum(node_id in titles for _account, node_id in nodes)
    stats.update({
        "principal_path_nodes": len(nodes),
        "principal_path_unique_edges": len(path_edges),
        "principal_path_edge_observations": edge_observations,
        "merged_parent_conflict_nodes": conflicts,
        "title_covered_nodes": covered,
        "missing_title_nodes": len(nodes) - covered,
        "record_vs_table_title_mismatches": title_mismatches,
        "record_title_alias_nodes": len(title_aliases),
        "record_title_aliases": sum(len(values) for values in title_aliases.values()),
    })
    return {
        "titles": titles,
        "title_aliases": {key: tuple(sorted(values)) for key, values in sorted(title_aliases.items())},
        "records": records,
        "nodes": nodes,
        "stats": dict(sorted(stats.items())),
        "account_records": dict(sorted(account_records.items())),
        "title_table_stats": title_table_stats,
    }



def pearltrees_quality_flags(title):
    flags = list(title_quality_flags(title))
    if HTML_ENTITY.search(title):
        flags.append("html_entity")
    if html.unescape(title) != title:
        flags.append("encoded_display_text")
    return sorted(set(flags))


def endpoint(node, titles, title_aliases=None):
    account, node_id = node
    title = titles.get(node_id)
    if not title:
        return None
    return {
        "id": node_id,
        "account": account,
        "title": title,
        "normalized_title": normalize_title(title),
        "quality_flags": pearltrees_quality_flags(title),
        "title_aliases": tuple((title_aliases or {}).get(node_id, ())),
    }


def candidate_rank(seed, hop, component, descendant, ancestor):
    payload = ":".join((
        str(seed), str(hop), component[0], component[1],
        descendant[0], descendant[1], ancestor[0], ancestor[1],
    )).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()


def candidate_pools(forest, hmax, seed, excluded_titles=()):
    titles = forest["titles"]
    pools = {hop: defaultdict(list) for hop in range(1, hmax + 1)}
    rejected = Counter()
    excluded_titles = {normalize_title(title) for title in excluded_titles}
    observations = defaultdict(list)
    conflict_examples = []

    for record in forest["records"]:
        account = record["account"]
        keyed_path = tuple((account, node_id) for node_id in record["path_ids"])
        component = keyed_path[0]
        component_endpoint = endpoint(component, titles, forest["title_aliases"])
        if component_endpoint is None:
            rejected["records_missing_component_title"] += 1
            continue
        for index in range(1, len(keyed_path)):
            descendant_node = keyed_path[index]
            descendant = endpoint(descendant_node, titles, forest["title_aliases"])
            if descendant is None:
                rejected["missing_descendant_title"] += 1
                continue
            if descendant["normalized_title"] in excluded_titles:
                rejected["excluded_descendant_title"] += 1
                continue
            for hop in range(1, min(hmax, index) + 1):
                ancestor_node = keyed_path[index - hop]
                if ancestor_node == descendant_node:
                    rejected["self_pair_observations"] += 1
                    continue
                ancestor = endpoint(ancestor_node, titles, forest["title_aliases"])
                if ancestor is None:
                    rejected["missing_ancestor_title"] += 1
                    continue
                if ancestor["normalized_title"] in excluded_titles:
                    rejected["excluded_ancestor_title"] += 1
                    continue
                unordered = tuple(sorted((descendant_node, ancestor_node)))
                observations[unordered].append({
                    "component": component,
                    "component_title": component_endpoint["title"],
                    "source_tree_id": record["tree_id"] or keyed_path[-1][1],
                    "descendant_node": descendant_node,
                    "ancestor_node": ancestor_node,
                    "descendant": descendant,
                    "ancestor": ancestor,
                    "hop": hop,
                })

    for unordered in sorted(observations):
        seen = observations[unordered]
        orientations = {(row["descendant_node"], row["ancestor_node"]) for row in seen}
        hops = {row["hop"] for row in seen}
        if len(orientations) > 1:
            rejected["direction_conflict_pairs"] += 1
            rejected["direction_conflict_observations"] += len(seen)
            if len(conflict_examples) < 20:
                conflict_examples.append({
                    "kind": "direction",
                    "endpoint_ids": [unordered[0][1], unordered[1][1]],
                    "observation_count": len(seen),
                    "orientations": [
                        {
                            "descendant_id": orientation[0][1],
                            "ancestor_id": orientation[1][1],
                            "observation_count": sum(
                                (item["descendant_node"], item["ancestor_node"]) == orientation
                                for item in seen
                            ),
                            "observed_hops": sorted({
                                item["hop"]
                                for item in seen
                                if (item["descendant_node"], item["ancestor_node"]) == orientation
                            }),
                        }
                        for orientation in sorted(orientations)
                    ],
                })
            continue
        if len(hops) > 1:
            rejected["hop_conflict_pairs"] += 1
            rejected["hop_conflict_observations"] += len(seen)
            if len(conflict_examples) < 20:
                conflict_examples.append({
                    "kind": "hop",
                    "endpoint_ids": [unordered[0][1], unordered[1][1]],
                    "observed_hops": sorted(hops),
                    "observation_count": len(seen),
                })
            continue
        provenance = sorted(
            seen,
            key=lambda row: (row["component"], row["source_tree_id"], row["descendant_node"], row["ancestor_node"]),
        )
        row = dict(provenance[0])
        row["source_tree_ids"] = tuple(sorted({item["source_tree_id"] for item in provenance}))
        row["source_record_count"] = len(provenance)
        rejected["duplicate_consistent_observations"] += len(provenance) - 1
        pools[row["hop"]][row["component"]].append(row)

    for hop, by_component in pools.items():
        for component, rows in by_component.items():
            rows.sort(key=lambda row: (
                candidate_rank(
                    seed,
                    hop,
                    component,
                    row["descendant_node"],
                    row["ancestor_node"],
                ),
                row["descendant"]["id"],
                row["ancestor"]["id"],
            ))
    return pools, dict(sorted(rejected.items())), conflict_examples


def sample_campaign_pairs(forest, pairs=250, hmax=5, seed=0, excluded_titles=()):
    if pairs <= 0 or hmax <= 0:
        raise FreshCorpusError("pairs and hmax must be positive")
    targets = hop_targets(pairs, hmax)
    pools, rejected, conflict_examples = candidate_pools(forest, hmax, seed, excluded_titles)
    rows = []
    accepted_by_hop_component = defaultdict(Counter)
    pool_counts = {}

    for hop in range(1, hmax + 1):
        components = sorted(
            pools[hop],
            key=lambda item: (pools[hop][item][0]["component_title"].casefold(), item),
        )
        queues = {component: list(pools[hop][component]) for component in components}
        pool_counts[str(hop)] = sum(len(queue) for queue in queues.values())
        accepted = 0
        while accepted < targets[hop]:
            progressed = False
            for component in components:
                if accepted >= targets[hop]:
                    break
                queue = queues[component]
                if not queue:
                    continue
                row = queue.pop(0)
                row.update({
                    "pair_id": f"pearltrees-h{hop}-{accepted:04d}",
                    "corpus": CORPUS,
                    "graph_view": GRAPH_VIEW,
                    "branch_id": component[1],
                    "branch_title": row["component_title"],
                    "account": component[0],
                })
                rows.append(row)
                accepted_by_hop_component[hop][f"{component[0]}:{component[1]}"] += 1
                accepted += 1
                progressed = True
            if not progressed:
                break
        if accepted < targets[hop]:
            raise FreshCorpusError(
                f"hop {hop} has {pool_counts[str(hop)]} eligible principal-path pairs; need {targets[hop]}"
            )

    return rows, {
        "target_hop_counts": {str(hop): targets[hop] for hop in sorted(targets)},
        "hop_counts": {str(hop): sum(row["hop"] == hop for row in rows) for hop in sorted(targets)},
        "candidate_pool_counts": pool_counts,
        "candidate_rejections": rejected,
        "candidate_conflict_examples": conflict_examples,
        "accepted_by_hop_component": {
            str(hop): dict(sorted(accepted_by_hop_component[hop].items()))
            for hop in sorted(targets)
        },
    }


def write_pairs_tsv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id", "corpus", "graph_view", "account", "branch_id", "branch_title",
        "source_tree_ids", "source_record_count", "descendant_id", "descendant_title", "descendant_title_aliases", "descendant_normalized_title",
        "ancestor_id", "ancestor_title", "ancestor_title_aliases", "ancestor_normalized_title", "hop",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "pair_id": row["pair_id"],
                "corpus": row["corpus"],
                "graph_view": row["graph_view"],
                "account": row["account"],
                "branch_id": row["branch_id"],
                "branch_title": row["branch_title"],
                "source_tree_ids": ",".join(row["source_tree_ids"]),
                "source_record_count": row["source_record_count"],
                "descendant_id": row["descendant"]["id"],
                "descendant_title": row["descendant"]["title"],
                "descendant_title_aliases": json.dumps(row["descendant"]["title_aliases"], ensure_ascii=False),
                "descendant_normalized_title": row["descendant"]["normalized_title"],
                "ancestor_id": row["ancestor"]["id"],
                "ancestor_title": row["ancestor"]["title"],
                "ancestor_title_aliases": json.dumps(row["ancestor"]["title_aliases"], ensure_ascii=False),
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
                f"{row['descendant']['title']}\t{row['ancestor']['title']}\tsubtopic\t1.0\t"
                f"principal_h{row['hop']}\tpearltrees_collection\tpearltrees_collection\t\n"
            )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths-jsonl", required=True, type=Path)
    ap.add_argument("--titles-tsv", required=True, type=Path)
    ap.add_argument("--pairs", type=int, default=250)
    ap.add_argument("--hmax", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-graph")
    ap.add_argument("--pairs-tsv", required=True, type=Path)
    ap.add_argument("--score-in", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--allow-small-sample", action="store_true")
    args = ap.parse_args(argv)
    if not args.allow_small_sample and args.pairs < 250:
        ap.error("campaign sampling requires at least 250 pairs; use --allow-small-sample for tests")

    forest = load_principal_paths(args.paths_jsonl, args.titles_tsv)
    excluded_titles = load_excluded_titles(args.exclude_graph)
    rows, summary = sample_campaign_pairs(
        forest,
        pairs=args.pairs,
        hmax=args.hmax,
        seed=args.seed,
        excluded_titles=excluded_titles,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus": CORPUS,
        "graph_view": GRAPH_VIEW,
        "paths_jsonl": str(args.paths_jsonl),
        "paths_jsonl_size": args.paths_jsonl.stat().st_size,
        "paths_jsonl_mtime_ns": args.paths_jsonl.stat().st_mtime_ns,
        "paths_jsonl_sha256": sha256_path(args.paths_jsonl),
        "titles_tsv": str(args.titles_tsv),
        "titles_tsv_size": args.titles_tsv.stat().st_size,
        "titles_tsv_mtime_ns": args.titles_tsv.stat().st_mtime_ns,
        "titles_tsv_sha256": sha256_path(args.titles_tsv),
        "principal_path_rule": "sample within each recorded non-account path_ids lineage",
        "cross_record_conflict_policy": "exclude endpoint pairs whose direction or hop differs across path records",
        "secondary_edge_policy": "assembled DAG aliases and secondary references excluded from primary view",
        "privacy_rule": "drop an entire record when target_text masks private ancestry or any known title is private",
        "title_policy": "assembled raw title; retained path-record differences preserved as aliases; no semantic correction",
        "title_normalization": "NFKC; underscore-to-space; whitespace-collapse; casefold (matching only)",
        "sampling_method": "canonical component for consistent duplicate observations; deterministic hash order; component round-robin",
        "pairs": len(rows),
        "hmax": args.hmax,
        "seed": args.seed,
        "exclude_graph": args.exclude_graph,
        "excluded_title_count": len(excluded_titles),
        "account_record_counts": forest["account_records"],
        "forest_stats": forest["stats"],
        "title_table_stats": forest["title_table_stats"],
        "pairs_tsv": str(args.pairs_tsv),
        "score_in": str(args.score_in),
        "title_audit": title_audit(rows),
    }
    manifest.update(summary)
    write_pairs_tsv(args.pairs_tsv, rows)
    write_score_in(args.score_in, rows)
    write_json_atomic(args.manifest, manifest)
    print(f"sampled {len(rows)} Pearltrees principal-path pairs")
    print(f"pairs -> {args.pairs_tsv}")
    print(f"score input -> {args.score_in}")
    print(f"manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
