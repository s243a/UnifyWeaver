#!/usr/bin/env python3
"""Materialize fixed public-campaign sources for identity-disjoint evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

import numpy as np
import torch

from emit_direction_blend import parse_responses
from eval_relatedness import build_model
from mu_attention import OPS, Tokenizer, build_e5_tables
from sample_product_kalman_pearltrees_campaign import load_principal_paths
from sample_sigma_hop_fresh_corpus import LmdbTitleGraph


SCHEMA_VERSION = 1
E5_MODEL = "intfloat/e5-small-v2"
DIR_RELATIONS = ("element_of", "subcategory", "subtopic", "super_category")
SYM_RELATIONS = ("see_also", "assoc")
FAMILY_ORDER = ("directional", "symmetric", "open_world")
OUTPUT_FIELDS = (
    "pair_id", "corpus", "branch_unit", "descendant_identity", "ancestor_identity",
    "descendant_id", "ancestor_id", "hop", "e5_fwd", "e5_rev", "model_D", "model_S",
    "graph_measurement", "target_D", "target_S", "operator_family", "family_tie_count",
)


class PublicFeatureError(ValueError):
    pass


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            item, self.parent[item] = self.parent[item], root
        return root

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            keep, merge = sorted((left_root, right_root))
            self.parent[merge] = keep


def load_pair_rows(path, corpus):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    required = {
        "pair_id", "corpus", "branch_id", "descendant_id", "ancestor_id", "hop",
        "descendant_title", "ancestor_title", "descendant_canonical_identity",
        "ancestor_canonical_identity",
    }
    if corpus == "pearltrees":
        required.update(("account", "source_tree_ids"))
    missing = sorted(required - fields)
    if missing:
        raise PublicFeatureError(f"pair table is missing fields: {', '.join(missing)}")
    if not rows:
        raise PublicFeatureError("pair table is empty")
    if {row["corpus"] for row in rows} != {corpus}:
        raise PublicFeatureError("pair table corpus does not match --corpus")
    pair_ids = [row["pair_id"] for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise PublicFeatureError("pair_id values must be unique")
    for row in rows:
        try:
            hop = int(row["hop"])
        except ValueError as exc:
            raise PublicFeatureError(f"{row['pair_id']}: invalid hop {row['hop']!r}") from exc
        if hop < 1:
            raise PublicFeatureError(f"{row['pair_id']}: hop must be positive")
    return rows


def endpoint_id_token(row, side):
    if row["corpus"] == "pearltrees":
        return f"id:{row['corpus']}:{row['account']}:{row[f'{side}_id']}"
    return f"id:{row['corpus']}:{row[f'{side}_id']}"


def identity_components(rows):
    union = UnionFind()
    endpoint_tokens = {}
    for row in rows:
        for side in ("descendant", "ancestor"):
            id_token = endpoint_id_token(row, side)
            canonical = row[f"{side}_canonical_identity"].strip()
            if not canonical:
                raise PublicFeatureError(f"{row['pair_id']}: empty {side} canonical identity")
            title_token = f"title:{row['corpus']}:{canonical}"
            union.union(id_token, title_token)
            endpoint_tokens[(row["pair_id"], side)] = id_token

    groups = defaultdict(list)
    for token in union.parent:
        groups[union.find(token)].append(token)
    component_id = {}
    for tokens in groups.values():
        payload = "\n".join(sorted(tokens)).encode("utf-8")
        identifier = hashlib.sha256(payload).hexdigest()[:20]
        for token in tokens:
            component_id[token] = identifier
    return {
        key: component_id[token]
        for key, token in endpoint_tokens.items()
    }, {identifier: sorted(tokens) for identifier, tokens in (
        (component_id[items[0]], items) for items in groups.values()
    )}


def branch_unit(row):
    if row["corpus"] == "pearltrees":
        return f"{row['account']}:{row['branch_id']}"
    return row["branch_id"]


def _number(obj, relation, field):
    value = (obj.get(relation) or {}).get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicFeatureError(f"response has invalid {relation}.{field}: {value!r}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise PublicFeatureError(f"response has out-of-range {relation}.{field}: {value!r}")
    return value


def response_targets(obj):
    directional = max(_number(obj, relation, "mu_fwd") for relation in DIR_RELATIONS)
    symmetric = max(_number(obj, relation, "mu") for relation in SYM_RELATIONS)
    family_scores = {
        "directional": max(_number(obj, relation, "applies") for relation in DIR_RELATIONS),
        "symmetric": max(_number(obj, relation, "applies") for relation in SYM_RELATIONS),
        "open_world": max(_number(obj, relation, "applies") for relation in ("none", "unknown")),
    }
    maximum = max(family_scores.values())
    winners = [family for family in FAMILY_ORDER if family_scores[family] == maximum]
    return directional, symmetric, winners[0], len(winners)


def load_targets(path, row_count):
    responses = parse_responses(path)
    expected = set(range(row_count))
    if set(responses) != expected:
        raise PublicFeatureError(
            f"response ids do not match rows; missing={sorted(expected - set(responses))}, "
            f"extra={sorted(set(responses) - expected)}"
        )
    return [response_targets(responses[index]) for index in range(row_count)]


def degree_from_parents(parents):
    children = defaultdict(set)
    nodes = set(parents)
    for child, values in parents.items():
        nodes.update(values)
        for parent in values:
            children[parent].add(child)
    return {node: len(set(parents.get(node, ()))) + len(children.get(node, ())) for node in nodes}


def enwiki_graph_context(rows, lmdb_dir, lock=False):
    pair_keys = [(row["descendant_id"], row["ancestor_id"]) for row in rows]
    endpoint_titles = {}
    for row in rows:
        for side in ("descendant", "ancestor"):
            key, title = row[f"{side}_id"], row[f"{side}_title"]
            if key in endpoint_titles and endpoint_titles[key] != title:
                raise PublicFeatureError(f"enwiki id {key} has conflicting audited titles")
            endpoint_titles[key] = title

    with closing(LmdbTitleGraph(Path(lmdb_dir), lock=lock)) as graph:
        parents = {}
        texts = dict(endpoint_titles)
        endpoints = sorted({value for pair in pair_keys for value in pair}, key=int)
        for key in endpoints:
            node_id = int(key)
            parent_keys = []
            for parent_id in graph.parents(node_id):
                title = graph.title(parent_id, missing_ok=True)
                if title is None:
                    continue
                parent_key = str(parent_id)
                texts.setdefault(parent_key, title)
                parent_keys.append(parent_key)
            parents[key] = tuple(sorted(set(parent_keys), key=int))

        measurements = []
        for descendant, ancestor in pair_keys:
            target_id = int(ancestor)
            memo = {}
            active = set()

            def hit(node_id):
                if node_id == target_id:
                    return 1.0
                if node_id in memo:
                    return memo[node_id]
                if node_id in active:
                    return 0.0
                active.add(node_id)
                parent_ids = graph.parents(node_id)
                value = sum(hit(parent_id) for parent_id in parent_ids) / len(parent_ids) if parent_ids else 0.0
                active.remove(node_id)
                memo[node_id] = value
                return value

            measurements.append(hit(int(descendant)))
        source = {
            "kind": "enwiki_lmdb_streamed",
            "lmdb_dir": str(lmdb_dir),
            "lmdb_data_mdb_size": (Path(lmdb_dir) / "data.mdb").stat().st_size,
        }
    return pair_keys, parents, degree_from_parents(parents), texts, np.asarray(measurements), source


def _pearltrees_title_overrides(rows):
    titles = {}
    for row in rows:
        account = row["account"]
        for side in ("descendant", "ancestor"):
            key = (account, row[f"{side}_id"])
            title = row[f"{side}_title"]
            if key in titles and titles[key] != title:
                raise PublicFeatureError(f"Pearltrees endpoint {key} has conflicting audited titles")
            titles[key] = title
    return titles


def find_pearltrees_lineage(row, records):
    source_ids = {value for value in row["source_tree_ids"].split(",") if value}
    descendant, ancestor, hop = row["descendant_id"], row["ancestor_id"], int(row["hop"])
    matches = []
    for record in records:
        if record["account"] != row["account"]:
            continue
        record_source = record["tree_id"] or record["path_ids"][-1]
        if source_ids and record_source not in source_ids:
            continue
        positions = {node_id: index for index, node_id in enumerate(record["path_ids"])}
        if descendant not in positions or ancestor not in positions:
            continue
        if positions[descendant] - positions[ancestor] != hop:
            continue
        matches.append(record)
    if not matches:
        raise PublicFeatureError(f"{row['pair_id']}: no frozen Pearltrees lineage reproduces the pair")
    return min(matches, key=lambda item: (item["account"], item["tree_id"], item["path_ids"]))


def pearltrees_graph_context(rows, paths_jsonl, titles_tsv):
    forest = load_principal_paths(paths_jsonl, titles_tsv)
    overrides = _pearltrees_title_overrides(rows)
    parents, texts, pair_keys = {}, {}, []
    measurements = []
    for row in rows:
        record = find_pearltrees_lineage(row, forest["records"])
        path = record["path_ids"]
        keys = [f"{row['pair_id']}:{index}:{node_id}" for index, node_id in enumerate(path)]
        for index, (key, node_id) in enumerate(zip(keys, path)):
            title = overrides.get((row["account"], node_id), forest["titles"].get(node_id))
            if not title:
                raise PublicFeatureError(f"{row['pair_id']}: lineage node {node_id} has no title")
            texts[key] = title
            parents[key] = (keys[index - 1],) if index else ()
        positions = {node_id: index for index, node_id in enumerate(path)}
        descendant_key = keys[positions[row["descendant_id"]]]
        ancestor_key = keys[positions[row["ancestor_id"]]]
        pair_keys.append((descendant_key, ancestor_key))
        measurements.append(1.0)
    source = {
        "kind": "pearltrees_pair_specific_principal_paths",
        "paths_jsonl": str(paths_jsonl),
        "paths_jsonl_sha256": sha256_path(paths_jsonl),
        "titles_tsv": str(titles_tsv),
        "titles_tsv_sha256": sha256_path(titles_tsv),
        "pair_specific_lineages": len(rows),
    }
    return pair_keys, parents, degree_from_parents(parents), texts, np.asarray(measurements), source


def score_fixed_sources(pair_keys, parents, deg, texts, model_path, e5_cache, device, batch_size):
    names = sorted(texts)
    query, passage, idx = build_e5_tables(
        names,
        cache_path=str(e5_cache),
        model_name=E5_MODEL,
        batch_size=batch_size,
        device=device,
        texts=texts,
    )
    tokenizer = Tokenizer(query, passage, idx, parents, deg)
    model = build_model(str(model_path), torch.device(device))

    def mu(op, pairs):
        output = []
        for offset in range(0, len(pairs), batch_size):
            items = [(left, right, OPS[op]) for left, right in pairs[offset:offset + batch_size]]
            batch = tokenizer.build(items, train=False)
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            with torch.no_grad():
                output.extend(model(**batch).cpu().tolist())
        return np.asarray(output, dtype=float)

    reverse = [(right, left) for left, right in pair_keys]
    hier_fwd, hier_rev = mu("HIER", pair_keys), mu("HIER", reverse)
    model_s = mu("SYM", pair_keys)
    e5_fwd = np.asarray([float(passage[idx[left]] @ query[idx[right]]) for left, right in pair_keys])
    e5_rev = np.asarray([float(passage[idx[right]] @ query[idx[left]]) for left, right in pair_keys])
    return {
        "e5_fwd": e5_fwd,
        "e5_rev": e5_rev,
        "model_D": np.maximum(hier_fwd, hier_rev),
        "model_S": model_s,
    }


def format_float(value):
    return f"{float(value):.12g}"


def write_feature_table(path, rows):
    lines = ["\t".join(OUTPUT_FIELDS)]
    for row in rows:
        lines.append("\t".join(str(row[field]) for field in OUTPUT_FIELDS))
    atomic_text(path, "\n".join(lines) + "\n")


def run(args):
    Path(args.e5_cache).parent.mkdir(parents=True, exist_ok=True)
    pair_rows = load_pair_rows(args.pairs, args.corpus)
    targets = load_targets(args.responses, len(pair_rows))
    identities, identity_groups = identity_components(pair_rows)
    if args.corpus == "enwiki":
        if not args.lmdb_dir:
            raise PublicFeatureError("--lmdb-dir is required for enwiki")
        context = enwiki_graph_context(pair_rows, args.lmdb_dir, lock=not args.lmdb_no_lock)
    else:
        if not args.paths_jsonl or not args.titles_tsv:
            raise PublicFeatureError("--paths-jsonl and --titles-tsv are required for Pearltrees")
        context = pearltrees_graph_context(pair_rows, args.paths_jsonl, args.titles_tsv)
    pair_keys, parents, deg, texts, graph_measurement, graph_source = context
    sources = score_fixed_sources(
        pair_keys,
        parents,
        deg,
        texts,
        args.model,
        args.e5_cache,
        args.device,
        args.batch_size,
    )
    output = []
    for index, (row, target) in enumerate(zip(pair_rows, targets)):
        target_d, target_s, family, ties = target
        output.append({
            "pair_id": row["pair_id"],
            "corpus": row["corpus"],
            "branch_unit": branch_unit(row),
            "descendant_identity": identities[(row["pair_id"], "descendant")],
            "ancestor_identity": identities[(row["pair_id"], "ancestor")],
            "descendant_id": endpoint_id_token(row, "descendant"),
            "ancestor_id": endpoint_id_token(row, "ancestor"),
            "hop": int(row["hop"]),
            "e5_fwd": format_float(sources["e5_fwd"][index]),
            "e5_rev": format_float(sources["e5_rev"][index]),
            "model_D": format_float(sources["model_D"][index]),
            "model_S": format_float(sources["model_S"][index]),
            "graph_measurement": format_float(graph_measurement[index]),
            "target_D": format_float(target_d),
            "target_S": format_float(target_s),
            "operator_family": family,
            "family_tie_count": ties,
        })
    write_feature_table(args.out, output)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus": args.corpus,
        "pairs": str(args.pairs),
        "pairs_sha256": sha256_path(args.pairs),
        "responses": str(args.responses),
        "responses_sha256": sha256_path(args.responses),
        "model": str(args.model),
        "model_sha256": sha256_path(args.model),
        "e5_model": E5_MODEL,
        "e5_cache": str(args.e5_cache),
        "e5_cache_sha256": sha256_path(args.e5_cache),
        "device": args.device,
        "batch_size": args.batch_size,
        "graph_source": graph_source,
        "row_count": len(output),
        "hop_counts": dict(sorted(Counter(row["hop"] for row in output).items())),
        "operator_family_counts": dict(sorted(Counter(row["operator_family"] for row in output).items())),
        "family_tied_rows": sum(row["family_tie_count"] > 1 for row in output),
        "identity_component_count": len(identity_groups),
        "branch_unit_count": len({row["branch_unit"] for row in output}),
        "graph_measurement": {
            "minimum": float(np.min(graph_measurement)),
            "maximum": float(np.max(graph_measurement)),
            "mean": float(np.mean(graph_measurement)),
            "unique_count": len(set(graph_measurement.tolist())),
        },
        "feature_table": str(args.out),
        "feature_table_sha256": sha256_path(args.out),
        "guardrail": "fixed sources only; no split-specific calibration or comparative result",
    }
    atomic_json(args.manifest, manifest)
    print(
        f"materialized {args.corpus} public features: {len(output)} rows, "
        f"{len(identity_groups)} identity components, {manifest['branch_unit_count']} branch units"
    )
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("enwiki", "pearltrees"), required=True)
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--responses", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--e5-cache", required=True, type=Path)
    ap.add_argument("--lmdb-dir", type=Path)
    ap.add_argument("--lmdb-no-lock", action="store_true")
    ap.add_argument("--paths-jsonl", type=Path)
    ap.add_argument("--titles-tsv", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
