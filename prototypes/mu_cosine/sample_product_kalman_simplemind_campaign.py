#!/usr/bin/env python3
"""Sample a hop-balanced SimpleMind campaign from within-map principal paths.

Only plain hierarchy edges, with blank structural containers bypassed, enter the
primary view. See-also, super-category, navigation, explicit-relation, and
cross-map edges remain available to later sensitivity analyses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from parse_smmx import (
    NAV_LABEL,
    SEE_ALSO,
    SUPER,
    WIKI_LABEL,
    cloud_of,
    label,
    load_xml,
    slug_of,
    wiki_of,
)
from privacy import is_private_title, propagate
from sample_product_kalman_enwiki_campaign import (
    load_excluded_titles,
    normalize_title,
    title_audit,
    title_quality_flags,
    write_json_atomic,
)
from sample_sigma_hop_fresh_corpus import FreshCorpusError, hop_targets


SCHEMA_VERSION = 1
CORPUS = "simplemind"
GRAPH_VIEW = "within_map_principal_path"
CONFLICTED_COPY = re.compile(r"\b(?:conflicted|conflicting) copy\b", re.IGNORECASE)
SENTINEL_ROOTS = {"root node", "root_node"}


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_title(title):
    text = unicodedata.normalize("NFKC", str(title)).replace("_", " ").replace("-", " ")
    return " ".join(text.split()).casefold()


SEE_ALSO_IDENTITIES = {canonical_title(title) for title in SEE_ALSO}
SUPER_IDENTITIES = {canonical_title(title) for title in SUPER} | {"super topic"}
PASS_THROUGH_IDENTITIES = {"subtopics"}


def structural_kind(title):
    identity = canonical_title(title)
    if not identity:
        return "blank"
    if identity in SEE_ALSO_IDENTITIES:
        return "see_also"
    if identity in SUPER_IDENTITIES:
        return "super_category"
    if identity in PASS_THROUGH_IDENTITIES:
        return "pass_through"
    return None


def map_id(path, digest):
    return f"{Path(path).stem}:{digest[:12]}"


def _merge_endpoint_metadata(endpoint, others):
    titles = {endpoint["title"], *endpoint["title_aliases"]}
    topic_ids = set(endpoint["topic_ids"])
    slugs = set(endpoint["pearltrees_slugs"])
    enwiki_aliases = set(endpoint["enwiki_aliases"])
    flags = set(endpoint["quality_flags"])
    for other in others:
        titles.add(other["title"])
        titles.update(other["title_aliases"])
        topic_ids.update(other["topic_ids"])
        slugs.update(other["pearltrees_slugs"])
        enwiki_aliases.update(other["enwiki_aliases"])
        flags.update(other["quality_flags"])
    out = dict(endpoint)
    out["title_aliases"] = tuple(sorted(titles - {endpoint["title"]}, key=lambda value: (value.casefold(), value)))
    out["topic_ids"] = tuple(sorted(topic_ids))
    out["pearltrees_slugs"] = tuple(sorted(slugs))
    out["enwiki_aliases"] = tuple(sorted(enwiki_aliases))
    out["quality_flags"] = tuple(sorted(flags))
    return out


def _topic_endpoint(topic_id, topic, metadata):
    title = label(topic)
    identity = canonical_title(title)
    if not identity:
        return None
    group = metadata[identity]
    all_titles = set(group["titles"])
    return {
        "id": f"title:{identity}",
        "identity": identity,
        "title": title,
        "title_aliases": tuple(sorted(all_titles - {title}, key=lambda value: (value.casefold(), value))),
        "normalized_title": normalize_title(title),
        "topic_ids": tuple(sorted(group["topic_ids"])),
        "pearltrees_slugs": tuple(sorted(group["pearltrees_slugs"])),
        "enwiki_aliases": tuple(sorted(group["enwiki_aliases"])),
        "quality_flags": tuple(sorted({
            flag
            for raw_title in all_titles
            for flag in title_quality_flags(raw_title)
        })),
    }


def parse_primary_map(path):
    path = Path(path)
    digest = sha256_path(path)
    source = {
        "path": str(path),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": digest,
        "map_id": map_id(path, digest),
    }
    root = load_xml(str(path))
    topic_list = root.findall(".//topic")
    topics = {}
    for topic in topic_list:
        topic_id = topic.get("id")
        if not topic_id:
            continue
        if topic_id in topics:
            raise FreshCorpusError(f"{path}: duplicate topic id {topic_id}")
        topics[topic_id] = topic
    parent = {topic_id: topic.get("parent") for topic_id, topic in topics.items()}
    root_ids = sorted(topic_id for topic_id in topics if parent.get(topic_id) in (None, "-1"))
    if len(root_ids) != 1:
        raise FreshCorpusError(f"{path}: expected one map root, found {len(root_ids)}")
    root_id = root_ids[0]
    root_title = label(topics[root_id])
    source["root_topic_id"] = root_id
    source["root_title"] = root_title

    children = defaultdict(list)
    for topic_id, parent_id in parent.items():
        children[parent_id].append(topic_id)
    private_seed = {topic_id for topic_id, topic in topics.items() if is_private_title(label(topic))}
    private_ids = propagate(private_seed, children)
    if root_id in private_ids:
        source["status"] = "excluded_private_root"
        return [], source, {"map_excluded_private_root": 1}
    if CONFLICTED_COPY.search(path.name) or CONFLICTED_COPY.search(root_title):
        source["status"] = "excluded_conflicted_copy"
        return [], source, {"map_excluded_conflicted_copy": 1}
    if root_title.strip().casefold() in SENTINEL_ROOTS:
        source["status"] = "excluded_sentinel_root"
        return [], source, {"map_excluded_sentinel_root": 1}

    anchor_ids = {
        topic_id
        for topic_id, topic in topics.items()
        if WIKI_LABEL.match(label(topic)) and wiki_of(topic) and parent.get(topic_id) in topics
    }
    container_ids = {
        topic_id for topic_id, topic in topics.items()
        if structural_kind(label(topic)) is not None
    }
    navigation_ids = {topic_id for topic_id, topic in topics.items() if NAV_LABEL.match(label(topic))}
    sentinel_ids = {
        topic_id for topic_id, topic in topics.items()
        if canonical_title(label(topic)) in SENTINEL_ROOTS
    }
    real_ids = set(topics) - private_ids - anchor_ids - container_ids - navigation_ids - sentinel_ids
    if root_id not in real_ids:
        source["status"] = "excluded_noncontent_root"
        return [], source, {"map_excluded_noncontent_root": 1}

    enwiki_by_topic = defaultdict(set)
    for topic_id, topic in topics.items():
        enwiki_title = wiki_of(topic)
        if not enwiki_title:
            continue
        endpoint_id = parent.get(topic_id) if topic_id in anchor_ids else topic_id
        if endpoint_id in real_ids:
            enwiki_by_topic[endpoint_id].add(enwiki_title)

    metadata = defaultdict(lambda: {
        "titles": set(),
        "topic_ids": set(),
        "pearltrees_slugs": set(),
        "enwiki_aliases": set(),
    })
    for topic_id in real_ids:
        title = label(topics[topic_id])
        identity = canonical_title(title)
        if not identity:
            continue
        slug, _pearltrees_id = slug_of(topics[topic_id])
        metadata[identity]["titles"].add(title)
        metadata[identity]["topic_ids"].add(topic_id)
        if slug:
            metadata[identity]["pearltrees_slugs"].add(slug)
        metadata[identity]["enwiki_aliases"].update(enwiki_by_topic[topic_id])

    primary_parent = {}
    edge_rejections = Counter()
    for topic_id in sorted(real_ids):
        if topic_id == root_id:
            continue
        parent_id = parent.get(topic_id)
        reason = None
        seen = set()
        while parent_id in topics and parent_id not in real_ids:
            if parent_id in seen:
                reason = "container_cycle"
                break
            seen.add(parent_id)
            if parent_id in private_ids:
                reason = "private_ancestry"
                break
            if parent_id in sentinel_ids:
                reason = "sentinel_ancestry"
                break
            if parent_id in navigation_ids:
                reason = "navigation_ancestry"
            elif parent_id in anchor_ids:
                reason = "wiki_anchor_ancestry"
            elif parent_id in container_ids:
                kind = structural_kind(label(topics[parent_id]))
                if kind == "see_also":
                    reason = "see_also_ancestry"
                elif kind == "super_category":
                    reason = "super_category_ancestry"
            parent_id = parent.get(parent_id)
        if reason:
            edge_rejections[reason] += 1
            continue
        if parent_id in real_ids and parent_id != topic_id:
            primary_parent[topic_id] = parent_id
        else:
            edge_rejections["missing_real_parent"] += 1

    records = []
    path_rejections = Counter()
    for topic_id in sorted(real_ids):
        chain = [topic_id]
        seen = {topic_id}
        cur = topic_id
        while cur in primary_parent:
            cur = primary_parent[cur]
            if cur in seen:
                path_rejections["raw_parent_cycle"] += 1
                chain = []
                break
            seen.add(cur)
            chain.append(cur)
        if not chain or chain[-1] != root_id:
            path_rejections["non_content_rooted"] += 1
            continue
        chain.reverse()
        endpoints = []
        endpoint_identities = set()
        repeated = False
        for raw_topic_id in chain:
            endpoint = _topic_endpoint(raw_topic_id, topics[raw_topic_id], metadata)
            if endpoint is None:
                repeated = True
                break
            if endpoints and endpoint["id"] == endpoints[-1]["id"]:
                endpoints[-1] = _merge_endpoint_metadata(endpoints[-1], [endpoint])
                continue
            if endpoint["id"] in endpoint_identities:
                repeated = True
                break
            endpoint_identities.add(endpoint["id"])
            endpoints.append(endpoint)
        if repeated:
            path_rejections["repeated_nonconsecutive_identity"] += 1
            continue
        if len(endpoints) < 2:
            path_rejections["short_after_identity_collapse"] += 1
            continue
        records.append({
            "map_id": source["map_id"],
            "map_title": root_title,
            "source_topic_id": topic_id,
            "endpoints": tuple(endpoints),
        })

    source["status"] = "retained"
    source["stats"] = {
        "topics": len(topics),
        "real_topics": len(real_ids),
        "private_topics": len(private_ids),
        "structural_containers": len(container_ids),
        "wiki_anchor_topics": len(anchor_ids),
        "navigation_topics": len(navigation_ids),
        "sentinel_topics": len(sentinel_ids),
        "cross_map_link_topics": sum(bool(cloud_of(topic)[0]) for topic in topics.values()),
        "explicit_relations": len(root.findall(".//relation")),
        "primary_parent_edges": len(primary_parent),
        "path_records": len(records),
        "edge_rejections": dict(sorted(edge_rejections.items())),
        "path_rejections": dict(sorted(path_rejections.items())),
    }
    return records, source, {}


def load_primary_maps(paths):
    records = []
    sources = []
    stats = Counter()
    for path in sorted({Path(path) for path in paths}, key=lambda value: str(value)):
        map_records, source, excluded = parse_primary_map(path)
        records.extend(map_records)
        sources.append(source)
        stats.update(excluded)
    stats["maps_total"] = len(sources)
    stats["maps_retained"] = sum(source.get("status") == "retained" for source in sources)
    stats["path_records"] = len(records)
    return {"records": records, "sources": sources, "stats": dict(sorted(stats.items()))}


def candidate_rank(seed, hop, map_identifier, descendant_id, ancestor_id):
    payload = f"{seed}:{hop}:{map_identifier}:{descendant_id}:{ancestor_id}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()


def candidate_pools(dataset, hmax, seed, excluded_titles=()):
    observations = defaultdict(list)
    rejected = Counter()
    excluded_titles = {normalize_title(title) for title in excluded_titles}
    for record in dataset["records"]:
        endpoints = record["endpoints"]
        for index in range(1, len(endpoints)):
            descendant = endpoints[index]
            if descendant["normalized_title"] in excluded_titles:
                rejected["excluded_descendant_title"] += 1
                continue
            for hop in range(1, min(hmax, index) + 1):
                ancestor = endpoints[index - hop]
                if ancestor["normalized_title"] in excluded_titles:
                    rejected["excluded_ancestor_title"] += 1
                    continue
                unordered = tuple(sorted((descendant["id"], ancestor["id"])))
                observations[unordered].append({
                    "map_id": record["map_id"],
                    "map_title": record["map_title"],
                    "source_topic_id": record["source_topic_id"],
                    "descendant": descendant,
                    "ancestor": ancestor,
                    "hop": hop,
                })

    pools = {hop: defaultdict(list) for hop in range(1, hmax + 1)}
    conflict_examples = []
    for unordered in sorted(observations):
        seen = observations[unordered]
        orientations = {(row["descendant"]["id"], row["ancestor"]["id"]) for row in seen}
        hops = {row["hop"] for row in seen}
        if len(orientations) > 1:
            rejected["direction_conflict_pairs"] += 1
            rejected["direction_conflict_observations"] += len(seen)
            if len(conflict_examples) < 20:
                conflict_examples.append({
                    "kind": "direction",
                    "endpoint_ids": list(unordered),
                    "orientations": [
                        {
                            "descendant_id": direction[0],
                            "ancestor_id": direction[1],
                            "observation_count": sum(
                                (item["descendant"]["id"], item["ancestor"]["id"]) == direction
                                for item in seen
                            ),
                        }
                        for direction in sorted(orientations)
                    ],
                })
            continue
        if len(hops) > 1:
            rejected["hop_conflict_pairs"] += 1
            rejected["hop_conflict_observations"] += len(seen)
            if len(conflict_examples) < 20:
                conflict_examples.append({
                    "kind": "hop",
                    "endpoint_ids": list(unordered),
                    "observed_hops": sorted(hops),
                    "observation_count": len(seen),
                })
            continue
        provenance = sorted(
            seen,
            key=lambda row: (row["map_id"], row["source_topic_id"], row["descendant"]["id"]),
        )
        row = dict(provenance[0])
        row["source_map_ids"] = tuple(sorted({item["map_id"] for item in provenance}))
        row["source_topic_ids"] = tuple(sorted({item["source_topic_id"] for item in provenance}))
        row["source_record_count"] = len(provenance)
        row["descendant"] = _merge_endpoint_metadata(
            row["descendant"], [item["descendant"] for item in provenance[1:]]
        )
        row["ancestor"] = _merge_endpoint_metadata(
            row["ancestor"], [item["ancestor"] for item in provenance[1:]]
        )
        rejected["duplicate_consistent_observations"] += len(provenance) - 1
        pools[row["hop"]][row["map_id"]].append(row)

    for hop, by_map in pools.items():
        for map_identifier, rows in by_map.items():
            rows.sort(key=lambda row: (
                candidate_rank(seed, hop, map_identifier, row["descendant"]["id"], row["ancestor"]["id"]),
                row["descendant"]["id"],
                row["ancestor"]["id"],
            ))
    return pools, dict(sorted(rejected.items())), conflict_examples


def sample_campaign_pairs(dataset, pairs=250, hmax=5, seed=0, excluded_titles=()):
    if pairs <= 0 or hmax <= 0:
        raise FreshCorpusError("pairs and hmax must be positive")
    targets = hop_targets(pairs, hmax)
    pools, rejected, conflict_examples = candidate_pools(dataset, hmax, seed, excluded_titles)
    rows = []
    accepted_by_hop_map = defaultdict(Counter)
    pool_counts = {}
    for hop in range(1, hmax + 1):
        maps = sorted(
            pools[hop],
            key=lambda identifier: (pools[hop][identifier][0]["map_title"].casefold(), identifier),
        )
        queues = {identifier: list(pools[hop][identifier]) for identifier in maps}
        pool_counts[str(hop)] = sum(len(queue) for queue in queues.values())
        accepted = 0
        while accepted < targets[hop]:
            progressed = False
            for identifier in maps:
                if accepted >= targets[hop]:
                    break
                if not queues[identifier]:
                    continue
                row = queues[identifier].pop(0)
                row.update({
                    "pair_id": f"simplemind-h{hop}-{accepted:04d}",
                    "corpus": CORPUS,
                    "graph_view": GRAPH_VIEW,
                    "branch_id": identifier,
                    "branch_title": row["map_title"],
                })
                rows.append(row)
                accepted_by_hop_map[hop][identifier] += 1
                accepted += 1
                progressed = True
            if not progressed:
                break
        if accepted < targets[hop]:
            raise FreshCorpusError(
                f"hop {hop} has {pool_counts[str(hop)]} eligible within-map pairs; need {targets[hop]}"
            )
    return rows, {
        "target_hop_counts": {str(hop): targets[hop] for hop in sorted(targets)},
        "hop_counts": {str(hop): sum(row["hop"] == hop for row in rows) for hop in sorted(targets)},
        "candidate_pool_counts": pool_counts,
        "candidate_rejections": rejected,
        "candidate_conflict_examples": conflict_examples,
        "accepted_by_hop_map": {
            str(hop): dict(sorted(accepted_by_hop_map[hop].items()))
            for hop in sorted(targets)
        },
    }


def _json_list(values):
    return json.dumps(tuple(values), ensure_ascii=False)


def write_pairs_tsv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id", "corpus", "graph_view", "branch_id", "branch_title",
        "source_map_ids", "source_topic_ids", "source_record_count",
        "descendant_id", "descendant_title", "descendant_title_aliases",
        "descendant_topic_ids", "descendant_pearltrees_slugs", "descendant_enwiki_aliases",
        "descendant_normalized_title",
        "ancestor_id", "ancestor_title", "ancestor_title_aliases",
        "ancestor_topic_ids", "ancestor_pearltrees_slugs", "ancestor_enwiki_aliases",
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
                "source_map_ids": _json_list(row["source_map_ids"]),
                "source_topic_ids": _json_list(row["source_topic_ids"]),
                "source_record_count": row["source_record_count"],
                "descendant_id": row["descendant"]["id"],
                "descendant_title": row["descendant"]["title"],
                "descendant_title_aliases": _json_list(row["descendant"]["title_aliases"]),
                "descendant_topic_ids": _json_list(row["descendant"]["topic_ids"]),
                "descendant_pearltrees_slugs": _json_list(row["descendant"]["pearltrees_slugs"]),
                "descendant_enwiki_aliases": _json_list(row["descendant"]["enwiki_aliases"]),
                "descendant_normalized_title": row["descendant"]["normalized_title"],
                "ancestor_id": row["ancestor"]["id"],
                "ancestor_title": row["ancestor"]["title"],
                "ancestor_title_aliases": _json_list(row["ancestor"]["title_aliases"]),
                "ancestor_topic_ids": _json_list(row["ancestor"]["topic_ids"]),
                "ancestor_pearltrees_slugs": _json_list(row["ancestor"]["pearltrees_slugs"]),
                "ancestor_enwiki_aliases": _json_list(row["ancestor"]["enwiki_aliases"]),
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
                f"principal_h{row['hop']}\tmindmap_node\tmindmap_node\t\n"
            )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maps", nargs="+", required=True, type=Path)
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

    dataset = load_primary_maps(args.maps)
    excluded_titles = load_excluded_titles(args.exclude_graph)
    rows, summary = sample_campaign_pairs(
        dataset,
        pairs=args.pairs,
        hmax=args.hmax,
        seed=args.seed,
        excluded_titles=excluded_titles,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus": CORPUS,
        "graph_view": GRAPH_VIEW,
        "primary_edge_rule": "plain within-map parent hierarchy; blank containers bypassed",
        "secondary_edge_policy": "see-also, super-category, navigation, explicit relation, and cross-map edges excluded",
        "identity_rule": "NFKC; underscore/hyphen-to-space; whitespace-collapse; casefold",
        "conflict_policy": "exclude identity pairs whose direction or collapsed hop differs across path records",
        "privacy_rule": "private title removes its raw topic subtree before endpoint metadata is built",
        "title_policy": "raw title primary; duplicate-title variants preserved as aliases; no semantic correction",
        "sampling_method": "canonical map for duplicate observations; deterministic hash order; map round-robin",
        "maps": dataset["sources"],
        "dataset_stats": dataset["stats"],
        "pairs": len(rows),
        "hmax": args.hmax,
        "seed": args.seed,
        "exclude_graph": args.exclude_graph,
        "excluded_title_count": len(excluded_titles),
        "pairs_tsv": str(args.pairs_tsv),
        "score_in": str(args.score_in),
        "title_audit": title_audit(rows),
        "identity_alias_counts": {
            "raw_title_aliases": len({
                value
                for row in rows
                for side in ("descendant", "ancestor")
                for value in row[side]["title_aliases"]
            }),
            "pearltrees_slugs": len({
                value
                for row in rows
                for side in ("descendant", "ancestor")
                for value in row[side]["pearltrees_slugs"]
            }),
            "enwiki_aliases": len({
                value
                for row in rows
                for side in ("descendant", "ancestor")
                for value in row[side]["enwiki_aliases"]
            }),
        },
    }
    manifest.update(summary)
    write_pairs_tsv(args.pairs_tsv, rows)
    write_score_in(args.score_in, rows)
    write_json_atomic(args.manifest, manifest)
    print(f"sampled {len(rows)} SimpleMind within-map pairs")
    print(f"pairs -> {args.pairs_tsv}")
    print(f"score input -> {args.score_in}")
    print(f"manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
