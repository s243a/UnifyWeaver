#!/usr/bin/env python3
"""Privacy-contract tests for the routed filing population."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval_filing import load_filing
from filing_privacy import (
    FilingPrivacyError,
    build_pearltrees_privacy_index,
    merge_visibility_claims,
    public_catalog_title_eligible,
    visibility_claim,
)
import eval_pearltrees_filing as filing_eval


def _tree(
    tree_id,
    title,
    visibility,
    *,
    bookmarks=(),
    parent=None,
    collections=(),
    other_links=(),
):
    pearls = [
        {"id": 100000 + i, "contentType": 1, "title": bookmark}
        for i, bookmark in enumerate(bookmarks)
    ]
    for child_id, child_title, child_visibility in collections:
        pearls.append(
            {
                "id": 200000 + int(child_id),
                "contentType": 2,
                "title": child_title,
                "contentTree": {
                    "id": child_id,
                    "title": child_title,
                    "visibility": child_visibility,
                },
            }
        )
    for child_id, child_title, child_visibility in other_links:
        pearls.append(
            {
                "id": 300000 + int(child_id),
                "contentType": 5,
                "title": child_title,
                "contentTree": {
                    "id": child_id,
                    "title": child_title,
                    "visibility": child_visibility,
                },
            }
        )
    info = {}
    if parent is not None:
        parent_id, parent_title, parent_visibility = parent
        info["parentTree"] = {
            "id": parent_id,
            "title": parent_title,
            "visibility": parent_visibility,
        }
    return {
        "tree_id": tree_id,
        "api_response": {
            "info": info,
            "tree": {
                "id": tree_id,
                "title": title,
                "visibility": visibility,
                "isUserRoot": 1 if parent is None else 0,
                "pearls": pearls,
            },
        },
    }


def _write_tree(trees: Path, payload):
    tree_id = payload["api_response"]["tree"]["id"]
    (trees / f"{tree_id}.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )


def _layout(tmp_path):
    data = tmp_path / "data"
    trees = data / "pearltrees_api" / "trees"
    trees.mkdir(parents=True)
    (data / "api_tree_paths_v8.jsonl").write_text("", encoding="utf-8")
    return data, trees


def test_visibility_is_tri_state_and_private_wins():
    assert visibility_claim(0) == "public"
    assert visibility_claim("0") == "public"
    assert visibility_claim(2) == "private"
    assert visibility_claim(None) == "unknown"
    assert merge_visibility_claims(["unknown", "public"]) == "public"
    assert merge_visibility_claims(["public", "private"]) == "private"
    with pytest.raises(FilingPrivacyError, match="malformed visibility"):
        visibility_claim("friends")


def test_public_catalog_title_rule_is_outcome_blind_and_explicit():
    assert public_catalog_title_eligible("Mathematics")
    assert public_catalog_title_eligible("C++")
    assert not public_catalog_title_eligible("?")
    assert not public_catalog_title_eligible("---")
    assert not public_catalog_title_eligible("")


def test_public_only_population_propagates_private_and_unknown(tmp_path):
    _, trees = _layout(tmp_path)
    _write_tree(
        trees,
        _tree(
            1,
            "Public root",
            0,
            bookmarks=("one", "two", "three"),
            collections=((2, "Restricted branch", 2), (4, "Unknown branch", None)),
        ),
    )
    _write_tree(
        trees,
        _tree(
            2,
            "Restricted branch",
            2,
            bookmarks=("secret one", "secret two", "secret three"),
            parent=(1, "Public root", 0),
        ),
    )
    _write_tree(
        trees,
        _tree(
            3,
            "Looks public below restricted",
            0,
            bookmarks=("a", "b", "c"),
            parent=(2, "Restricted branch", 2),
        ),
    )
    _write_tree(
        trees,
        _tree(
            4,
            "Unknown branch",
            None,
            bookmarks=("unknown one", "unknown two", "unknown three"),
            parent=(1, "Public root", 0),
        ),
    )
    _write_tree(
        trees,
        _tree(
            5,
            "Looks public below unknown",
            0,
            bookmarks=("d", "e", "f"),
            parent=(4, "Unknown branch", None),
        ),
    )
    queries, candidates, privacy = load_filing(
        trees, 3, return_privacy=True, paths_jsonl=None
    )
    assert candidates == {1: "Public root"}
    assert queries == [("one", 1), ("two", 1), ("three", 1)]
    assert {"2", "3"} <= privacy.private_ids
    assert {"4", "5"} <= privacy.quarantined_ids


def test_public_claim_resolves_missing_summary_but_private_conflict_wins(tmp_path):
    _, trees = _layout(tmp_path)
    _write_tree(
        trees,
        _tree(
            1,
            "Parent",
            0,
            bookmarks=("p1", "p2", "p3"),
            collections=((2, "Public child", None), (3, "Conflicted child", 2)),
        ),
    )
    _write_tree(
        trees,
        _tree(
            2,
            "Public child",
            0,
            bookmarks=("a", "b", "c"),
            parent=(1, "Parent", 0),
        ),
    )
    _write_tree(
        trees,
        _tree(
            3,
            "Conflicted child",
            0,
            bookmarks=("x", "y", "z"),
            parent=(1, "Parent", 0),
        ),
    )
    _, candidates, privacy = load_filing(
        trees, 3, return_privacy=True, paths_jsonl=None
    )
    assert set(candidates) == {1, 2}
    assert "2" in privacy.public_ids
    assert "3" in privacy.private_ids


def test_private_collection_wrapper_title_wins_over_public_child_title(tmp_path):
    _, trees = _layout(tmp_path)
    parent = _tree(
        1,
        "Parent",
        0,
        bookmarks=("p1", "p2", "p3"),
        collections=((2, "Economics", 0),),
    )
    parent["api_response"]["tree"]["pearls"][-1]["title"] = "*private*"
    _write_tree(trees, parent)
    _write_tree(
        trees,
        _tree(
            2,
            "Economics",
            0,
            bookmarks=("a", "b", "c"),
            parent=(1, "Parent", 0),
        ),
    )
    _, candidates, privacy = load_filing(trees, 3, return_privacy=True)
    assert candidates == {1: "Parent"}
    assert "2" in privacy.private_ids


def test_private_bookmarks_removed_before_minimum_count(tmp_path):
    _, trees = _layout(tmp_path)
    _write_tree(
        trees,
        _tree(
            1,
            "Public",
            0,
            bookmarks=("public one", "public two", "Private equity"),
        ),
    )
    queries, candidates = load_filing(trees, 3, paths_jsonl=None)
    assert queries == []
    assert candidates == {}


def test_bookmark_without_visibility_inherits_certified_public_container(tmp_path):
    _, trees = _layout(tmp_path)
    _write_tree(
        trees,
        _tree(1, "Public folder", 0, bookmarks=("one", "two", "three")),
    )
    queries, candidates = load_filing(trees, 3, paths_jsonl=None)
    assert candidates == {1: "Public folder"}
    assert queries == [("one", 1), ("two", 1), ("three", 1)]


def test_noncontainment_link_does_not_propagate_privacy(tmp_path):
    _, trees = _layout(tmp_path)
    _write_tree(
        trees,
        _tree(
            1,
            "Public",
            0,
            bookmarks=("a", "b", "c"),
            other_links=((2, "Private shortcut", 2),),
        ),
    )
    queries, candidates = load_filing(trees, 3, paths_jsonl=None)
    assert candidates == {1: "Public"}
    assert len(queries) == 3


def test_private_marker_in_path_scrubs_target(tmp_path):
    data, trees = _layout(tmp_path)
    _write_tree(
        trees,
        _tree(1, "Public root", 0, bookmarks=("a", "b", "c")),
    )
    _write_tree(
        trees,
        _tree(
            2,
            "Target",
            0,
            bookmarks=("d", "e", "f"),
            parent=(1, "Public root", 0),
        ),
    )
    path_file = data / "api_tree_paths_v8.jsonl"
    path_file.write_text(
        json.dumps(
            {
                "tree_id": "2",
                "title": "Target",
                "target_text": "- Public root\n  - *private*\n    - Target",
                "path_ids": ["account:test", "1", "2"],
                "parent_tree_id": "1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _, candidates, privacy = load_filing(trees, 3, return_privacy=True)
    assert candidates == {}
    assert "2" in privacy.private_ids


def test_missing_materialized_paths_fail_closed(tmp_path):
    data, trees = _layout(tmp_path)
    _write_tree(trees, _tree(1, "Public", 0, bookmarks=("a", "b", "c")))
    (data / "api_tree_paths_v8.jsonl").unlink()
    with pytest.raises(FilingPrivacyError, match="required"):
        load_filing(trees, 3)


def test_malformed_numeric_record_and_id_mismatch_fail_closed(tmp_path):
    _, trees = _layout(tmp_path)
    (trees / "1.json").write_text("{", encoding="utf-8")
    with pytest.raises(FilingPrivacyError, match="malformed JSON"):
        build_pearltrees_privacy_index(trees, paths_jsonl=None)
    (trees / "1.json").write_text(
        json.dumps(_tree(2, "Wrong", 0)), encoding="utf-8"
    )
    with pytest.raises(FilingPrivacyError, match="mismatch"):
        build_pearltrees_privacy_index(trees, paths_jsonl=None)


def test_hosted_lineage_uses_only_certified_path_and_no_dag_fallback(
    tmp_path, monkeypatch
):
    paths = tmp_path / "paths.jsonl"
    dag = tmp_path / "dag.tsv"
    titles = tmp_path / "titles.tsv"
    paths.write_text(
        json.dumps(
            {
                "tree_id": "2",
                "title": "Child",
                "target_text": "- Parent\n  - *private*\n    - Child",
                "path_ids": ["account:test", "1", "2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dag.write_text("1\t2\n", encoding="utf-8")
    # Hosted mode must never read this visibility-free/stale title source.
    titles.write_text("1\t*private stale title*\n2\tChild\n", encoding="utf-8")
    monkeypatch.setattr(filing_eval, "PATHS_JSONL", str(paths))
    monkeypatch.setattr(filing_eval, "DAG", str(dag))
    monkeypatch.setattr(filing_eval, "TITLES", str(titles))

    parents, ancestors = filing_eval.folder_lineage(
        {2: "Child"},
        depth=3,
        public_tree_titles={"1": "Parent", "2": "Child"},
    )
    assert parents == {}
    assert ancestors == set()

    paths.write_text(
        json.dumps(
            {
                "tree_id": "2",
                "title": "Child",
                "target_text": "- Parent\n  - Child",
                "path_ids": ["account:test", "1", "2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    parents, ancestors = filing_eval.folder_lineage(
        {2: "Child"},
        depth=3,
        public_tree_titles={"1": "Parent", "2": "Child"},
    )
    assert parents == {"Child": ["Parent"]}
    assert ancestors == {"Parent"}


def test_hosted_lineage_keeps_duplicate_title_paths_distinct(tmp_path, monkeypatch):
    paths = tmp_path / "paths.jsonl"
    dag = tmp_path / "dag.tsv"
    titles = tmp_path / "titles.tsv"
    records = [
        {
            "tree_id": "2",
            "title": "Same",
            "target_text": "- Root A\n  - Same",
            "path_ids": ["account:test", "1", "2"],
        },
        {
            "tree_id": "4",
            "title": "Same",
            "target_text": "- Root B\n  - Same",
            "path_ids": ["account:test", "3", "4"],
        },
    ]
    paths.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    dag.write_text("", encoding="utf-8")
    titles.write_text("", encoding="utf-8")
    monkeypatch.setattr(filing_eval, "PATHS_JSONL", str(paths))
    monkeypatch.setattr(filing_eval, "DAG", str(dag))
    monkeypatch.setattr(filing_eval, "TITLES", str(titles))
    _, _, by_id = filing_eval.folder_lineage(
        {2: "Same", 4: "Same"},
        depth=3,
        public_tree_titles={
            "1": "Root A",
            "2": "Same",
            "3": "Root B",
            "4": "Same",
        },
        return_id_chains=True,
    )
    assert by_id == {"2": ["Root A"], "4": ["Root B"]}
