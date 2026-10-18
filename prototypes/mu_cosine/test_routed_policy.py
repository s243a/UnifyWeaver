#!/usr/bin/env python3
"""Integrity and inference tests for routed-task v2."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from routed_policy import (
    RAW_PICKS_SCHEMA,
    RoutedPolicyError,
    band_qids,
    build_pick_envelope,
    build_policy_envelope,
    build_task_envelope,
    file_content_record,
    in_band,
    make_band,
    paired_node_block_bootstrap,
    policy_tier_for_margin,
    read_pick_file,
    read_policy_file,
    read_raw_picks,
    read_task_file,
    seal_pick_file,
    validate_policy_core,
    write_task_file,
)
import routed_queries
from filing_assistant import apply_public_catalog_policy


def _rows(qids=(0, 1)):
    return [
        {
            "record_type": "task",
            "qid": qid,
            "bookmark": f"bookmark {qid}",
            "menu": [
                {"pos": 0, "folder_id": "10", "title": "first"},
                {"pos": 1, "folder_id": "11", "title": "second", "path": "root"},
            ],
        }
        for qid in qids
    ]


def _task_core():
    return {
        "source": {"members_sha256": "a" * 64},
        "privacy": {
            "policy_id": "pearltrees-public-only-v1",
            "manifest_sha256": "b" * 64,
        },
        "catalog": {"sha256": "c" * 64},
        "population": {"sha256": "d" * 64},
        "ranker": {"ranking_sha256": "e" * 64},
        "selection": {
            "tier_id": "low",
            "band": make_band(None, 0.02),
            "menu_size": 2,
            "lineage": True,
            "lineage_depth": 3,
            "required_judge_family": "sonnet",
        },
        "policy_provenance": {"evidence_status": "exploratory_transductive"},
    }


def _judge():
    return {
        "provider": "test",
        "family": "sonnet",
        "model": "test-model",
        "model_revision": "r1",
        "interface": "fixture",
        "prompt_sha256": "f" * 64,
        "temperature": 0.0,
        "run_id": "test-run",
        "provenance_status": "declared",
    }


def _raw_picks(path, task_header, values=((0, 0), (1, None))):
    path.write_text(
        json.dumps(
            {
                "schema": RAW_PICKS_SCHEMA,
                "record_type": "raw_pick_header",
                "task_id": task_header["task_id"],
            }
        )
        + "\n"
        + "".join(
            json.dumps({"qid": qid, "pick": pick}) + "\n" for qid, pick in values
        ),
        encoding="utf-8",
    )


def test_band_boundaries_are_exact_and_half_open():
    band = make_band(0.02, 0.03)
    assert not in_band(np.nextafter(0.02, 0.0), band)
    assert in_band(0.02, band)
    assert in_band(np.nextafter(0.03, 0.0), band)
    assert not in_band(0.03, band)
    assert band_qids([0.019, 0.02, 0.025, 0.03], band) == (1, 2)


def test_task_and_pick_round_trip_bind_exact_bytes(tmp_path):
    task = tmp_path / "task.jsonl"
    raw = tmp_path / "raw.jsonl"
    picks = tmp_path / "picks.jsonl"
    task_header, task_record = write_task_file(task, _task_core(), _rows())
    _raw_picks(raw, task_header)
    pick_header, _ = seal_pick_file(task, raw, picks, _judge())

    loaded_task, loaded_rows, loaded_record = read_task_file(task)
    loaded_pick, _, mapping = read_pick_file(picks, task)
    assert loaded_task == task_header
    assert loaded_rows == _rows()
    assert loaded_record == task_record
    assert loaded_pick == pick_header
    assert mapping == {0: 0, 1: None}
    assert loaded_pick["pick_core"]["task_id"] == loaded_task["task_id"]


def test_task_row_tampering_is_rejected(tmp_path):
    task = tmp_path / "task.jsonl"
    write_task_file(task, _task_core(), _rows())
    lines = task.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["menu"][0]["title"] = "tampered"
    lines[1] = json.dumps(row)
    task.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RoutedPolicyError, match="hash mismatch"):
        read_task_file(task)


def test_raw_picks_reject_missing_extra_duplicate_bool_and_legacy_header(tmp_path):
    rows = _rows()
    task_header = build_task_envelope(_task_core(), rows)
    raw = tmp_path / "raw.jsonl"

    _raw_picks(raw, task_header, ((0, 0), (2, 1)))
    with pytest.raises(RoutedPolicyError, match="exactly match"):
        read_raw_picks(raw, task_header, rows)

    _raw_picks(raw, task_header, ((0, 0), (0, 1), (1, None)))
    with pytest.raises(RoutedPolicyError, match="duplicate"):
        read_raw_picks(raw, task_header, rows)

    _raw_picks(raw, task_header, ((0, True), (1, None)))
    with pytest.raises(RoutedPolicyError, match="integer"):
        read_raw_picks(raw, task_header, rows)

    raw.write_text(
        json.dumps({"manifest": "legacy"}) + "\n"
        + json.dumps({"qid": 0, "pick": 0}) + "\n"
        + json.dumps({"qid": 1, "pick": None}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RoutedPolicyError, match="task_id header"):
        read_raw_picks(raw, task_header, rows)


def test_cross_task_pick_substitution_is_rejected(tmp_path):
    task_a = tmp_path / "a.jsonl"
    task_b = tmp_path / "b.jsonl"
    raw = tmp_path / "raw.jsonl"
    picks = tmp_path / "picks.jsonl"
    task_a_header, _ = write_task_file(task_a, _task_core(), _rows())
    changed = _task_core()
    changed["selection"] = {
        **changed["selection"],
        "menu_size": 3,
    }
    rows_b = _rows()
    for row in rows_b:
        row["menu"].append({"pos": 2, "folder_id": "12", "title": "third"})
    write_task_file(task_b, changed, rows_b)
    _raw_picks(raw, task_a_header)
    seal_pick_file(task_a, raw, picks, _judge())
    with pytest.raises(RoutedPolicyError, match="hash mismatch"):
        read_pick_file(picks, task_b)


def test_raw_response_cannot_be_rebound_before_sealing(tmp_path):
    task_a = tmp_path / "a.jsonl"
    task_b = tmp_path / "b.jsonl"
    raw = tmp_path / "raw.jsonl"
    picks = tmp_path / "picks.jsonl"
    header_a, _ = write_task_file(task_a, _task_core(), _rows())
    swapped = _rows()
    for row in swapped:
        row["menu"] = [
            {**row["menu"][1], "pos": 0},
            {**row["menu"][0], "pos": 1},
        ]
    write_task_file(task_b, _task_core(), swapped)
    _raw_picks(raw, header_a)
    with pytest.raises(RoutedPolicyError, match="task_id header"):
        seal_pick_file(task_b, raw, picks, _judge())


def test_empty_task_and_header_only_raw_picks_round_trip(tmp_path):
    task = tmp_path / "task.jsonl"
    raw = tmp_path / "raw.jsonl"
    picks = tmp_path / "picks.jsonl"
    header, _ = write_task_file(task, _task_core(), [])
    _raw_picks(raw, header, ())
    seal_pick_file(task, raw, picks, _judge())
    _, rows, mapping = read_pick_file(picks, task)
    assert rows == []
    assert mapping == {}


def _policy_core():
    return {
        "name": "three-tier-margin-v1",
        "evidence_status": "exploratory_transductive",
        "privacy_policy_id": "pearltrees-public-only-v1",
        "catalog_policy_id": "pearltrees-public-alphanumeric-title-v1",
        "primary_grading": "exact_destination_id",
        "judge_prompt": {
            "path": "prompt.md",
            "sha256": "f" * 64,
        },
        "tiers": [
            {
                "tier_id": "low",
                "band": make_band(None, 0.02),
                "action": "judge",
                "menu_size": 10,
                "lineage": True,
                "lineage_depth": 3,
                "required_judge_family": "sonnet",
            },
            {
                "tier_id": "middle",
                "band": make_band(0.02, 0.03),
                "action": "judge",
                "menu_size": 20,
                "lineage": True,
                "lineage_depth": 3,
                "required_judge_family": "sonnet",
            },
            {
                "tier_id": "auto",
                "band": make_band(0.03, None),
                "action": "auto_top1",
            },
        ],
        "bootstrap": {
            "unit": "bookmark-folder-connected-component",
            "seed": 3867001,
            "replicates": 9999,
            "interval": [0.025, 0.975],
        },
    }


def test_policy_requires_exact_nonoverlapping_complete_partition():
    core = _policy_core()
    validate_policy_core(core)
    assert policy_tier_for_margin(core, 0.019)["tier_id"] == "low"
    assert policy_tier_for_margin(core, 0.02)["tier_id"] == "middle"
    assert policy_tier_for_margin(core, 0.03)["tier_id"] == "auto"

    gapped = _policy_core()
    gapped["tiers"][1]["band"] = make_band(0.021, 0.03)
    with pytest.raises(RoutedPolicyError, match="gap"):
        validate_policy_core(gapped)

    promoted = _policy_core()
    promoted["evidence_status"] = "confirmatory"
    with pytest.raises(RoutedPolicyError, match="exploratory"):
        validate_policy_core(promoted)

    premature_infinity = _policy_core()
    premature_infinity["tiers"][0]["band"] = make_band(None, None)
    with pytest.raises(RoutedPolicyError, match="final"):
        validate_policy_core(premature_infinity)


def test_committed_policy_has_valid_content_id():
    envelope = read_policy_file(ROOT / "ROUTED_POLICY_three_tier_v1.json")
    assert envelope == build_policy_envelope(envelope["policy_core"])
    assert envelope["policy_core"]["evidence_status"] == "exploratory_transductive"
    assert routed_queries._frozen_policy() == envelope


def test_task_selection_must_match_frozen_tier_and_prompt():
    envelope = read_policy_file(ROOT / "ROUTED_POLICY_three_tier_v1.json")
    tier = envelope["policy_core"]["tiers"][0]
    selection = {
        "tier_id": tier["tier_id"],
        "band": tier["band"],
        "menu_size": tier["menu_size"],
        "lineage": tier["lineage"],
        "lineage_depth": tier["lineage_depth"],
        "required_judge": {
            "provider": "fixture",
            "family": tier["required_judge_family"],
            "model": "fixture-sonnet",
            "model_revision": "r1",
            "interface": "fixture",
            "prompt_sha256": envelope["policy_core"]["judge_prompt"]["sha256"],
            "temperature": None,
        },
    }
    assert routed_queries._validate_selection_against_tier(
        envelope, selection
    ) == tier

    wrong_prompt = {
        **selection,
        "required_judge": {
            **selection["required_judge"],
            "prompt_sha256": "0" * 64,
        },
    }
    with pytest.raises(RoutedPolicyError, match="prompt"):
        routed_queries._validate_selection_against_tier(envelope, wrong_prompt)

    wrong_band = {**selection, "band": make_band(None, 0.01)}
    with pytest.raises(RoutedPolicyError, match="band"):
        routed_queries._validate_selection_against_tier(envelope, wrong_band)


def test_node_block_bootstrap_is_deterministic_and_clusters_shared_nodes():
    policy = [True, False, True, True]
    baseline = [False, False, False, True]
    bookmarks = ["same", "same", "other", "last"]
    folders = ["A", "B", "B", "C"]
    # Rows 0-2 form one connected component (shared bookmark then shared
    # folder); row 3 is the second component.
    first = paired_node_block_bootstrap(
        policy, baseline, bookmarks, folders, replicates=200, seed=11
    )
    second = paired_node_block_bootstrap(
        policy, baseline, bookmarks, folders, replicates=200, seed=11
    )
    assert first == second
    assert first["block_count"] == 2
    assert first["point"] == pytest.approx(0.5)


def test_exact_destination_is_primary_and_title_alias_is_sensitivity_only():
    state = SimpleNamespace(
        ranks=np.array([2]),
        alias_ranks=np.array([1]),
        order=np.array([[1, 0]]),
        truepos=[[0]],
        alias_truepos=[[0, 1]],
    )
    task_rows = [{"qid": 0}]
    picks = {0: 0}  # chooses column 1: same title, wrong destination id
    exact = routed_queries._policy_correct(state, task_rows, picks)
    alias = routed_queries._policy_correct(
        state, task_rows, picks, title_equivalence=True
    )
    assert exact.tolist() == [False]
    assert alias.tolist() == [True]


def test_duplicate_title_score_ties_match_exact_rank_tie_break():
    cos = np.array([[0.9, 0.9, 0.2], [0.4, 0.8, 0.8]])
    order = routed_queries.stable_score_order(cos)
    assert order.tolist() == [[0, 1, 2], [1, 2, 0]]
    assert routed_queries.ranks_np(cos, [[0], [1]]).tolist() == [1, 1]
    assert [
        int(order[row, 0]) == truth
        for row, truth in enumerate((0, 1))
    ] == [True, True]


def test_eval_and_suggest_share_one_public_catalog_policy():
    queries, candidates = apply_public_catalog_policy(
        [("kept bookmark", 1), ("excluded bookmark", 2)],
        {1: "Mathematics", 2: "?"},
    )
    assert candidates == {1: "Mathematics"}
    assert queries == [("kept bookmark", 1)]
