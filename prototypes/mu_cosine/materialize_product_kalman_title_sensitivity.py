#!/usr/bin/env python3
"""Materialize a hash-bound audited-title sensitivity view of campaign pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from sample_product_kalman_enwiki_campaign import normalize_title, write_json_atomic


POLICY_SCHEMA_VERSION = 1
OUTPUT_SCHEMA_VERSION = 1
SCORE_SPECS = {
    "enwiki": ("subcategory", "category", "transitive_h"),
    "pearltrees": ("subtopic", "pearltrees_collection", "principal_h"),
    "simplemind": ("subtopic", "mindmap_node", "principal_h"),
}


class TitlePolicyError(ValueError):
    pass


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pairs(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fields = reader.fieldnames or []
    required = {
        "pair_id", "corpus", "descendant_id", "descendant_title",
        "ancestor_id", "ancestor_title", "hop",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise TitlePolicyError(f"pair table is missing required fields: {', '.join(missing)}")
    if not rows:
        raise TitlePolicyError("pair table is empty")
    pair_ids = [row["pair_id"] for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise TitlePolicyError("pair_id values must be unique")
    return fields, rows


def load_policy(path, source_pairs, rows):
    with open(path, encoding="utf-8") as f:
        policy = json.load(f)
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise TitlePolicyError(
            f"unsupported policy schema {policy.get('schema_version')!r}; "
            f"expected {POLICY_SCHEMA_VERSION}"
        )
    if policy.get("status") != "frozen_pre_scoring":
        raise TitlePolicyError("policy status must be frozen_pre_scoring")
    for field in ("frozen_date", "review_method"):
        if not isinstance(policy.get(field), str) or not policy[field].strip():
            raise TitlePolicyError(f"policy {field} must be a nonempty string")
    corpora = {row["corpus"] for row in rows}
    if len(corpora) != 1:
        raise TitlePolicyError(f"pair table must contain one corpus, found {sorted(corpora)}")
    corpus = next(iter(corpora))
    if corpus not in SCORE_SPECS:
        raise TitlePolicyError(f"unsupported corpus {corpus!r}")
    if policy.get("corpus") != corpus:
        raise TitlePolicyError(
            f"policy corpus {policy.get('corpus')!r} does not match pair corpus {corpus!r}"
        )
    observed_hash = sha256_path(source_pairs)
    expected_hash = policy.get("source_pairs_sha256")
    if expected_hash != observed_hash:
        raise TitlePolicyError(
            f"source pair hash mismatch: expected {expected_hash}, observed {observed_hash}"
        )

    endpoint_titles = defaultdict(set)
    for row in rows:
        for side in ("descendant", "ancestor"):
            endpoint_titles[row[f"{side}_id"]].add(row[f"{side}_title"])

    items = policy.get("corrections")
    if not isinstance(items, list):
        raise TitlePolicyError("policy corrections must be a list")
    corrections = {}
    for item in items:
        if not isinstance(item, dict):
            raise TitlePolicyError("every correction must be an object")
        endpoint_id = item.get("endpoint_id")
        if not endpoint_id:
            raise TitlePolicyError("every correction requires endpoint_id")
        if endpoint_id in corrections:
            raise TitlePolicyError(f"duplicate correction for endpoint {endpoint_id}")
        if endpoint_id not in endpoint_titles:
            raise TitlePolicyError(f"correction endpoint is absent from pair table: {endpoint_id}")
        raw_titles = item.get("raw_titles")
        if not isinstance(raw_titles, list) or not raw_titles or not all(
            isinstance(value, str) and value for value in raw_titles
        ):
            raise TitlePolicyError(f"{endpoint_id}: raw_titles must be a nonempty string list")
        if set(raw_titles) != endpoint_titles[endpoint_id]:
            raise TitlePolicyError(
                f"{endpoint_id}: reviewed raw titles {sorted(raw_titles)} do not equal "
                f"observed titles {sorted(endpoint_titles[endpoint_id])}"
            )
        audited_title = item.get("audited_title")
        if not isinstance(audited_title, str) or not audited_title.strip():
            raise TitlePolicyError(f"{endpoint_id}: audited_title must be nonempty")
        if any(char in audited_title for char in "\t\r\n"):
            raise TitlePolicyError(f"{endpoint_id}: audited_title contains a TSV control character")
        if normalize_title(audited_title) in {normalize_title(value) for value in raw_titles}:
            raise TitlePolicyError(f"{endpoint_id}: correction does not change normalized title")
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence or not all(
            isinstance(value, str) and value.strip() for value in evidence
        ):
            raise TitlePolicyError(f"{endpoint_id}: evidence must be a nonempty string list")
        corrections[endpoint_id] = item
    return policy, corrections, endpoint_titles


def materialize_rows(rows, corrections):
    output = []
    corrected_pairs = set()
    corrected_occurrences = Counter()
    canonical_ids = defaultdict(set)
    for source in rows:
        row = dict(source)
        for side in ("descendant", "ancestor"):
            endpoint_id = source[f"{side}_id"]
            raw_title = source[f"{side}_title"]
            correction = corrections.get(endpoint_id)
            audited_title = correction["audited_title"] if correction else raw_title
            action = "corrected" if correction else "raw"
            row[f"{side}_raw_title"] = raw_title
            row[f"{side}_title"] = audited_title
            row[f"{side}_title_audit_action"] = action
            row[f"{side}_canonical_identity"] = normalize_title(audited_title)
            normalized_field = f"{side}_normalized_title"
            if normalized_field in row:
                row[normalized_field] = normalize_title(audited_title)
            canonical_ids[normalize_title(audited_title)].add(endpoint_id)
            if correction:
                corrected_pairs.add(source["pair_id"])
                corrected_occurrences[endpoint_id] += 1
        output.append(row)
    collision_groups = [
        {"canonical_identity": identity, "endpoint_ids": sorted(endpoint_ids)}
        for identity, endpoint_ids in sorted(canonical_ids.items())
        if len(endpoint_ids) > 1
    ]
    return output, corrected_pairs, corrected_occurrences, collision_groups


def output_fields(source_fields):
    extra = []
    for side in ("descendant", "ancestor"):
        extra.extend([
            f"{side}_raw_title",
            f"{side}_title_audit_action",
            f"{side}_canonical_identity",
        ])
    return list(source_fields) + [field for field in extra if field not in source_fields]


def write_pairs(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_score_in(path, rows, corpus):
    relation, node_type, neighborhood_prefix = SCORE_SPECS[corpus]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# node_title\troot_title\tcur_relation\tconf\tneighborhood\tnode_type\troot_type\traw\n")
        for row in rows:
            f.write(
                f"{row['descendant_title']}\t{row['ancestor_title']}\t{relation}\t1.0\t"
                f"{neighborhood_prefix}{row['hop']}\t{node_type}\t{node_type}\t\n"
            )


def pair_id_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["pair_id"].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--audited-pairs", required=True, type=Path)
    ap.add_argument("--score-in", required=True, type=Path)
    ap.add_argument("--audit-manifest", required=True, type=Path)
    args = ap.parse_args(argv)
    output_paths = {args.audited_pairs.resolve(), args.score_in.resolve(), args.audit_manifest.resolve()}
    if args.pairs.resolve() in output_paths or args.policy.resolve() in output_paths:
        ap.error("outputs must not overwrite the source pair table or frozen policy")
    if len(output_paths) != 3:
        ap.error("output paths must be distinct")

    fields, rows = load_pairs(args.pairs)
    policy, corrections, endpoint_titles = load_policy(args.policy, args.pairs, rows)
    audited, corrected_pairs, corrected_occurrences, collision_groups = materialize_rows(rows, corrections)
    audited_fields = output_fields(fields)
    write_pairs(args.audited_pairs, audited_fields, audited)
    write_score_in(args.score_in, audited, policy["corpus"])
    manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "corpus": policy["corpus"],
        "view": "audited_title_sensitivity",
        "source_pairs": str(args.pairs),
        "source_pairs_sha256": sha256_path(args.pairs),
        "policy": str(args.policy),
        "policy_sha256": sha256_path(args.policy),
        "audited_pairs": str(args.audited_pairs),
        "audited_pairs_sha256": sha256_path(args.audited_pairs),
        "score_in": str(args.score_in),
        "score_in_sha256": sha256_path(args.score_in),
        "pair_count": len(rows),
        "pair_id_sha256": pair_id_digest(rows),
        "corrected_endpoint_count": len(corrections),
        "corrected_pair_count": len(corrected_pairs),
        "corrected_endpoint_occurrences": dict(sorted(corrected_occurrences.items())),
        "reviewed_endpoint_count": len(endpoint_titles),
        "canonical_identity_collision_groups": collision_groups,
        "guardrail": "pair IDs, hops, endpoint IDs, and graph provenance unchanged; titles only",
    }
    write_json_atomic(args.audit_manifest, manifest)
    print(
        f"materialized {policy['corpus']} audited-title view: {len(rows)} pairs, "
        f"{len(corrections)} corrected endpoints, {len(corrected_pairs)} affected pairs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
