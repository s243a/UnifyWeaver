#!/usr/bin/env python3
"""Tests for hash-bound campaign title sensitivity materialization."""

import csv
import json
import tempfile
from pathlib import Path

from materialize_product_kalman_title_sensitivity import (
    TitlePolicyError,
    load_pairs,
    load_policy,
    main,
    sha256_path,
    write_score_in,
)


FIELDS = [
    "pair_id", "corpus", "graph_view", "branch_id", "branch_title",
    "descendant_id", "descendant_title", "descendant_normalized_title",
    "ancestor_id", "ancestor_title", "ancestor_normalized_title", "hop",
]


def write_pairs(path):
    rows = [
        {
            "pair_id": "p1", "corpus": "simplemind", "graph_view": "principal",
            "branch_id": "m1", "branch_title": "Map 1", "descendant_id": "title:ball valve",
            "descendant_title": "Ball Valve", "descendant_normalized_title": "ball valve",
            "ancestor_id": "title:values", "ancestor_title": "Values",
            "ancestor_normalized_title": "values", "hop": "1",
        },
        {
            "pair_id": "p2", "corpus": "simplemind", "graph_view": "principal",
            "branch_id": "m1", "branch_title": "Map 1", "descendant_id": "title:values",
            "descendant_title": "Values", "descendant_normalized_title": "values",
            "ancestor_id": "title:valves", "ancestor_title": "Valves",
            "ancestor_normalized_title": "valves", "hop": "2",
        },
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_policy(path, pairs, **overrides):
    policy = {
        "schema_version": 1,
        "status": "frozen_pre_scoring",
        "frozen_date": "2026-07-09",
        "corpus": "simplemind",
        "source_pairs_sha256": sha256_path(pairs),
        "review_method": "fixture review before scoring",
        "corrections": [
            {
                "endpoint_id": "title:values",
                "raw_titles": ["Values"],
                "audited_title": "Valves",
                "evidence": ["fixture slug: valves"],
            }
        ],
    }
    policy.update(overrides)
    path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_policy_error(pairs, policy, match):
    fields, rows = load_pairs(pairs)
    assert fields
    try:
        load_policy(policy, pairs, rows)
    except TitlePolicyError as exc:
        assert match in str(exc), str(exc)
    else:
        raise AssertionError(f"expected TitlePolicyError containing {match!r}")


def test_materializes_matched_view_and_identity_closure():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pairs = root / "pairs.tsv"
        policy = root / "policy.json"
        audited = root / "audited.tsv"
        score = root / "score.tsv"
        manifest = root / "manifest.json"
        write_pairs(pairs)
        write_policy(policy, pairs)
        args = [
            "--pairs", str(pairs), "--policy", str(policy),
            "--audited-pairs", str(audited), "--score-in", str(score),
            "--audit-manifest", str(manifest),
        ]
        assert main(args) == 0

        with open(audited, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert [row["pair_id"] for row in rows] == ["p1", "p2"]
        assert [row["hop"] for row in rows] == ["1", "2"]
        assert rows[0]["ancestor_title"] == "Valves"
        assert rows[0]["ancestor_raw_title"] == "Values"
        assert rows[0]["ancestor_title_audit_action"] == "corrected"
        assert rows[1]["descendant_title"] == "Valves"
        assert rows[1]["ancestor_title"] == "Valves"
        assert rows[1]["ancestor_title_audit_action"] == "raw"

        score_rows = [
            line.split("\t")
            for line in score.read_text(encoding="utf-8").splitlines()
            if not line.startswith("#")
        ]
        assert score_rows[0][0:3] == ["Ball Valve", "Valves", "subtopic"]
        assert score_rows[0][4:7] == ["principal_h1", "mindmap_node", "mindmap_node"]

        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["pair_count"] == 2
        assert data["corrected_endpoint_count"] == 1
        assert data["corrected_pair_count"] == 2
        assert data["corrected_endpoint_occurrences"] == {"title:values": 2}
        assert data["canonical_identity_collision_groups"] == [
            {
                "canonical_identity": "valves",
                "endpoint_ids": ["title:values", "title:valves"],
            }
        ]

        first = (audited.read_bytes(), score.read_bytes(), manifest.read_bytes())
        assert main(args) == 0
        assert (audited.read_bytes(), score.read_bytes(), manifest.read_bytes()) == first


def test_score_specs_cover_each_corpus():
    expected = {
        "enwiki": ("subcategory", "transitive_h3", "category"),
        "pearltrees": ("subtopic", "principal_h3", "pearltrees_collection"),
        "simplemind": ("subtopic", "principal_h3", "mindmap_node"),
    }
    row = {"descendant_title": "Child", "ancestor_title": "Parent", "hop": "3"}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for corpus, contract in expected.items():
            path = root / f"{corpus}.tsv"
            write_score_in(path, [row], corpus)
            data = [
                line.split("\t")
                for line in path.read_text(encoding="utf-8").splitlines()
                if not line.startswith("#")
            ][0]
            assert (data[2], data[4], data[5]) == contract


def test_rejects_hash_drift_and_unreviewed_raw_variants():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pairs = root / "pairs.tsv"
        policy = root / "policy.json"
        write_pairs(pairs)
        write_policy(policy, pairs, source_pairs_sha256="0" * 64)
        assert_policy_error(pairs, policy, "source pair hash mismatch")

        write_policy(policy, pairs)
        text = pairs.read_text(encoding="utf-8").replace("Values\tvalues", "VALUES\tvalues", 1)
        pairs.write_text(text, encoding="utf-8")
        write_policy(policy, pairs)
        assert_policy_error(pairs, policy, "reviewed raw titles")


def test_rejects_malformed_policy_and_score_controls():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pairs = root / "pairs.tsv"
        policy = root / "policy.json"
        write_pairs(pairs)
        write_policy(policy, pairs, corrections={})
        assert_policy_error(pairs, policy, "corrections must be a list")

        write_policy(policy, pairs)
        data = json.loads(policy.read_text(encoding="utf-8"))
        data["corrections"][0]["audited_title"] = "Valves\tbroken"
        policy.write_text(json.dumps(data), encoding="utf-8")
        assert_policy_error(pairs, policy, "TSV control character")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} title sensitivity tests passed")
