#!/usr/bin/env python3
"""No-spend execution-plan, attempt, and aggregation integrity tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import routed_execution
import routed_queries
from routed_policy import (
    RAW_PICKS_SCHEMA,
    RoutedPolicyError,
    build_policy_envelope,
    canonical_json_bytes,
    make_band,
    read_pick_file,
    read_task_file,
    sha256_bytes,
    write_task_file,
)


QIDS = (7, 2, 99, 41, 5)


def _required_judge(prompt_sha256):
    return {
        "provider": "fixture-provider",
        "family": "sonnet",
        "model": "fixture-model",
        "model_revision": "fixture-revision",
        "interface": "fixture-interface",
        "prompt_sha256": prompt_sha256,
        "temperature": 0.0,
    }


def _policy_core(prompt_sha256, menu_size):
    return {
        "name": "fixture-policy",
        "evidence_status": "exploratory_transductive",
        "privacy_policy_id": "pearltrees-public-only-v1",
        "catalog_policy_id": "pearltrees-public-alphanumeric-title-v1",
        "primary_grading": "exact_destination_id",
        "judge_prompt": {
            "path": "fixture-prompt.md",
            "sha256": prompt_sha256,
        },
        "tiers": [
            {
                "tier_id": "all",
                "band": make_band(None, None),
                "action": "judge",
                "menu_size": menu_size,
                "lineage": True,
                "lineage_depth": 3,
                "required_judge_family": "sonnet",
            }
        ],
        "bootstrap": {
            "unit": "bookmark-folder-connected-component",
            "seed": 1,
            "replicates": 99,
            "interval": [0.025, 0.975],
        },
    }


def _parent_rows(menu_size=3):
    rows = []
    for qid in QIDS:
        menu = []
        for position in range(menu_size):
            menu.append(
                {
                    "pos": position,
                    "folder_id": f"folder-{qid}-{position}",
                    "title": "duplicate title" if position < 2 else "third title",
                    "path": f"STEM > topic {qid} > branch {position}",
                }
            )
        rows.append(
            {
                "record_type": "task",
                "qid": qid,
                "bookmark": f"bookmark {qid}",
                "menu": menu,
            }
        )
    return rows


def _parent_core(policy_id, prompt_sha256, menu_size):
    judge = _required_judge(prompt_sha256)
    return {
        "source": {
            "members_sha256": "a" * 64,
            "fixture": "public-only",
        },
        "privacy": {
            "policy_id": "pearltrees-public-only-v1",
            "manifest_sha256": "b" * 64,
        },
        "catalog": {"min_bm": 3, "sha256": "c" * 64},
        "population": {"max_queries": 5, "seed": 7, "sha256": "d" * 64},
        "ranker": {"top_k": 100, "ranking_sha256": "e" * 64},
        "selection": {
            "tier_id": "all",
            "band": make_band(None, None),
            "menu_size": menu_size,
            "lineage": True,
            "lineage_depth": 3,
            "required_judge": judge,
        },
        "policy_provenance": {
            "policy_id": policy_id,
            "evidence_status": "exploratory_transductive",
        },
        "judge_contract": {
            "pick_semantics": "zero-based-menu-position-or-null",
            "required_context": "title-plus-lineage",
            "required_judge": judge,
        },
        "execution_contract": {
            "single_response_artifact": "legacy-supported",
            "chunked_provider_calls": "routed-execution-plan-v1-required",
            "repeated_draw_aggregation": "strict-majority-folder-id-v1",
            "draw_count": 3,
            "missing_or_terminal_failed_call": "fail-closed-no-imputation",
            "confirmatory_inference": (
                "not-authorized-until-execution-dependence-is-modeled"
            ),
        },
    }


def _fixture_plan(tmp_path, monkeypatch, *, menu_size=3, chunk_size=2, name="run"):
    prompt = tmp_path / f"{name}-prompt.md"
    prompt.write_bytes(b"Choose one public folder or null.\n")
    prompt_sha256 = sha256_bytes(prompt.read_bytes())
    policy = build_policy_envelope(_policy_core(prompt_sha256, menu_size))
    policy_path = tmp_path / f"{name}-policy.json"
    policy_path.write_bytes(canonical_json_bytes(policy))
    parent = tmp_path / f"{name}-parent.jsonl"
    parent_header, _ = write_task_file(
        parent,
        _parent_core(policy["policy_id"], prompt_sha256, menu_size),
        _parent_rows(menu_size),
    )
    certification = {
        "current_source_rebuild_verified": True,
        "parent_task_id": parent_header["task_id"],
        "privacy_manifest_sha256": "b" * 64,
        "source_members_sha256": "a" * 64,
        "implementation": {"fixture": True},
    }
    monkeypatch.setattr(
        routed_execution,
        "certify_parent_task_current",
        lambda path: dict(certification),
    )
    out = tmp_path / name
    plan = routed_execution.create_execution_plan(
        parent,
        out,
        chunk_size=chunk_size,
        namespace="fixture-namespace",
        prompt_path=prompt,
        policy_path=policy_path,
    )
    return {
        "root": out,
        "plan_path": out / routed_execution.PLAN_FILENAME,
        "parent": parent,
        "plan": plan,
    }


def _raw_for_folders(path, child, desired):
    rows = []
    for task_row in child["task_rows"]:
        folder_id = desired[task_row["qid"]]
        if folder_id is None:
            pick = None
        else:
            matches = [
                item["pos"]
                for item in task_row["menu"]
                if item["folder_id"] == folder_id
            ]
            assert len(matches) == 1
            pick = matches[0]
        rows.append({"qid": task_row["qid"], "pick": pick})
    path.write_bytes(
        canonical_json_bytes(
            {
                "schema": RAW_PICKS_SCHEMA,
                "record_type": "raw_pick_header",
                "task_id": child["task_header"]["task_id"],
            }
        )
        + b"".join(canonical_json_bytes(row) for row in rows)
    )


def _seal_success(fixture, draw, chunk, attempt, raw, request_suffix):
    return routed_execution.seal_execution_attempt(
        fixture["plan_path"],
        draw_index=draw,
        chunk_index=chunk,
        attempt_index=attempt,
        status="success",
        raw_response_path=raw,
        provider_run_id="fixture-run",
        provider_request_id=f"request-{request_suffix}",
        provider_response_id=f"response-{request_suffix}",
        started_at_utc="2026-07-23T12:00:00Z",
        completed_at_utc="2026-07-23T12:00:01Z",
    )


def test_plan_is_deterministic_partitioned_rotated_and_exact(tmp_path, monkeypatch):
    first = _fixture_plan(tmp_path, monkeypatch, name="first")
    # Reuse exact input bytes at distinct paths; paths are locators, not identity.
    second_root = tmp_path / "second"
    plan, context = routed_execution.verify_execution_plan(first["plan_path"])
    assert plan == first["plan"]
    parent_qids = [row["qid"] for row in context["parent_rows"]]
    child_ids = set()
    for draw in range(3):
        draw_qids = []
        for chunk in range(3):
            child = context["children"][(draw, chunk)]
            draw_qids.extend(row["qid"] for row in child["task_rows"])
            child_ids.add(child["task_header"]["task_id"])
            request = (
                first["root"] / child["entry"]["request"]["relative_path"]
            ).read_bytes()
            task = (first["root"] / child["entry"]["task"]["relative_path"]).read_bytes()
            assert request == routed_execution._render_request(context["prompt_bytes"], task)
        assert draw_qids == parent_qids
        assert len(draw_qids) == len(set(draw_qids))
    assert len(child_ids) == 9

    for qid in QIDS:
        positions = {}
        for draw in range(3):
            child = next(
                child
                for (child_draw, _), child in context["children"].items()
                if child_draw == draw
                and qid in [row["qid"] for row in child["task_rows"]]
            )
            row = next(row for row in child["task_rows"] if row["qid"] == qid)
            for item in row["menu"]:
                positions.setdefault(item["folder_id"], []).append(item["pos"])
        assert all(sorted(values) == [0, 1, 2] for values in positions.values())

    # A second plan made from byte-identical inputs and the same namespace is identical.
    parent_copy = tmp_path / "second-parent.jsonl"
    policy_copy = tmp_path / "second-policy.json"
    prompt_copy = tmp_path / "second-prompt.md"
    parent_copy.write_bytes(first["parent"].read_bytes())
    policy_copy.write_bytes((tmp_path / "first-policy.json").read_bytes())
    prompt_copy.write_bytes((tmp_path / "first-prompt.md").read_bytes())
    parent_header = read_task_file(parent_copy)[0]
    monkeypatch.setattr(
        routed_execution,
        "certify_parent_task_current",
        lambda path: {
            "current_source_rebuild_verified": True,
            "parent_task_id": parent_header["task_id"],
            "privacy_manifest_sha256": "b" * 64,
            "source_members_sha256": "a" * 64,
            "implementation": {"fixture": True},
        },
    )
    second = routed_execution.create_execution_plan(
        parent_copy,
        second_root,
        chunk_size=2,
        namespace="fixture-namespace",
        prompt_path=prompt_copy,
        policy_path=policy_copy,
    )
    assert second == first["plan"]


def test_menu_size_one_still_has_draw_distinct_child_ids(tmp_path, monkeypatch):
    fixture = _fixture_plan(
        tmp_path, monkeypatch, menu_size=1, chunk_size=5, name="single"
    )
    _, context = routed_execution.verify_execution_plan(fixture["plan_path"])
    ids = [
        context["children"][(draw, 0)]["task_header"]["task_id"]
        for draw in range(3)
    ]
    assert len(set(ids)) == 3


def test_plan_tamper_and_no_clobber_fail_closed(tmp_path, monkeypatch):
    fixture = _fixture_plan(tmp_path, monkeypatch)
    with pytest.raises(RoutedPolicyError, match="overwrite"):
        routed_execution.create_execution_plan(
            fixture["parent"],
            fixture["root"],
            chunk_size=2,
            namespace="fixture-namespace",
            prompt_path=tmp_path / "run-prompt.md",
            policy_path=tmp_path / "run-policy.json",
        )
    request = fixture["root"] / "draw-000/chunk-000.request.txt"
    request.write_bytes(request.read_bytes() + b"tamper")
    with pytest.raises(RoutedPolicyError, match="request bytes"):
        routed_execution.verify_execution_plan(fixture["plan_path"])


def test_retry_state_machine_cross_child_and_provider_id_reuse(tmp_path, monkeypatch):
    fixture = _fixture_plan(tmp_path, monkeypatch)
    _, context = routed_execution.verify_execution_plan(fixture["plan_path"])
    failure = tmp_path / "failure.txt"
    failure.write_text("temporary provider error", encoding="utf-8")
    routed_execution.seal_execution_attempt(
        fixture["plan_path"],
        draw_index=0,
        chunk_index=0,
        attempt_index=0,
        status="retryable_failure",
        raw_response_path=failure,
        provider_run_id="run",
        provider_request_id="request-retry",
        provider_response_id="",
        started_at_utc="2026-07-23T12:00:00Z",
        completed_at_utc="2026-07-23T12:00:01Z",
        error_type="timeout",
    )
    child = context["children"][(0, 0)]
    raw = tmp_path / "success.jsonl"
    _raw_for_folders(
        raw,
        child,
        {row["qid"]: row["menu"][0]["folder_id"] for row in child["task_rows"]},
    )
    with pytest.raises(RoutedPolicyError, match="contiguously"):
        _seal_success(fixture, 0, 0, 2, raw, "gap")
    _seal_success(fixture, 0, 0, 1, raw, "after-retry")
    with pytest.raises(RoutedPolicyError, match="terminal"):
        _seal_success(fixture, 0, 0, 2, raw, "post-terminal")

    other_child = context["children"][(0, 1)]
    cross = tmp_path / "cross.jsonl"
    _raw_for_folders(
        cross,
        child,
        {row["qid"]: row["menu"][0]["folder_id"] for row in child["task_rows"]},
    )
    with pytest.raises(RoutedPolicyError, match="task_id header"):
        _seal_success(fixture, 0, 1, 0, cross, "cross")
    other_raw = tmp_path / "other.jsonl"
    _raw_for_folders(
        other_raw,
        other_child,
        {
            row["qid"]: row["menu"][0]["folder_id"]
            for row in other_child["task_rows"]
        },
    )
    with pytest.raises(RoutedPolicyError, match="request ID"):
        routed_execution.seal_execution_attempt(
            fixture["plan_path"],
            draw_index=0,
            chunk_index=1,
            attempt_index=0,
            status="success",
            raw_response_path=other_raw,
            provider_run_id="run",
            provider_request_id="request-after-retry",
            provider_response_id="new-response",
            started_at_utc="2026-07-23T12:00:00Z",
            completed_at_utc="2026-07-23T12:00:01Z",
        )


def test_attempt_paths_cover_four_digit_indices_and_reject_symlink_root(
    tmp_path, monkeypatch
):
    for index in (999, 1000):
        relative = (
            f"{routed_execution._attempt_relative_dir(0, index, index)}/"
            "attempt.jsonl"
        )
        match = routed_execution._ATTEMPT_FILE_RE.fullmatch(relative)
        assert match is not None
        coordinate = tuple(int(match.group(part)) for part in (1, 2, 3))
        assert (
            f"{routed_execution._attempt_relative_dir(*coordinate)}/"
            f"{match.group(4)}"
        ) == relative

    fixture = _fixture_plan(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    (fixture["root"] / "attempts").symlink_to(outside, target_is_directory=True)
    failure = tmp_path / "failure.txt"
    failure.write_text("failure", encoding="utf-8")
    with pytest.raises(RoutedPolicyError, match="symlink"):
        routed_execution.seal_execution_attempt(
            fixture["plan_path"],
            draw_index=0,
            chunk_index=0,
            attempt_index=0,
            status="retryable_failure",
            raw_response_path=failure,
            provider_run_id="run",
            provider_request_id="request",
            provider_response_id="",
            started_at_utc="2026-07-23T12:00:00Z",
            completed_at_utc="2026-07-23T12:00:01Z",
            error_type="timeout",
        )

    resumable = _fixture_plan(tmp_path, monkeypatch, name="stale-staging")
    orphan = resumable["root"] / ".staging-attempts" / "orphan"
    orphan.mkdir(parents=True)
    (orphan / "attempt.jsonl").write_text("partial", encoding="utf-8")
    routed_execution.seal_execution_attempt(
        resumable["plan_path"],
        draw_index=0,
        chunk_index=0,
        attempt_index=0,
        status="retryable_failure",
        raw_response_path=failure,
        provider_run_id="run",
        provider_request_id="stale-safe-request",
        provider_response_id="",
        started_at_utc="2026-07-23T12:00:00Z",
        completed_at_utc="2026-07-23T12:00:01Z",
        error_type="timeout",
    )

    locked = _fixture_plan(tmp_path, monkeypatch, name="lock-symlink")
    external_lock = tmp_path / "external-lock"
    external_lock.write_text("", encoding="utf-8")
    (locked["root"] / ".routed-execution-writer.lock").symlink_to(
        external_lock
    )
    with pytest.raises(RoutedPolicyError, match="writer lock"):
        routed_execution.seal_execution_attempt(
            locked["plan_path"],
            draw_index=0,
            chunk_index=0,
            attempt_index=0,
            status="retryable_failure",
            raw_response_path=failure,
            provider_run_id="run",
            provider_request_id="lock-request",
            provider_response_id="",
            started_at_utc="2026-07-23T12:00:00Z",
            completed_at_utc="2026-07-23T12:00:01Z",
            error_type="timeout",
        )


def test_bundle_strict_folder_majority_and_parent_compatibility(tmp_path, monkeypatch):
    fixture = _fixture_plan(tmp_path, monkeypatch)
    _, context = routed_execution.verify_execution_plan(fixture["plan_path"])
    folders = {
        row["qid"]: [item["folder_id"] for item in row["menu"]]
        for row in context["parent_rows"]
    }
    desired = {
        7: (folders[7][0], folders[7][0], None),
        2: (None, None, folders[2][1]),
        99: (folders[99][0], folders[99][1], None),
        41: (folders[41][2], folders[41][2], folders[41][0]),
        5: (None, None, None),
    }
    retry_body = tmp_path / "retry-body.txt"
    retry_body.write_text("retryable transport failure", encoding="utf-8")
    routed_execution.seal_execution_attempt(
        fixture["plan_path"],
        draw_index=0,
        chunk_index=0,
        attempt_index=0,
        status="retryable_failure",
        raw_response_path=retry_body,
        provider_run_id="fixture-run",
        provider_request_id="request-retry-first",
        provider_response_id="",
        started_at_utc="2026-07-23T11:59:58Z",
        completed_at_utc="2026-07-23T11:59:59Z",
        error_type="transport_timeout",
    )
    for draw in range(3):
        for chunk in range(3):
            child = context["children"][(draw, chunk)]
            raw = tmp_path / f"raw-{draw}-{chunk}.jsonl"
            _raw_for_folders(
                raw,
                child,
                {row["qid"]: desired[row["qid"]][draw] for row in child["task_rows"]},
            )
            _seal_success(
                fixture,
                draw,
                chunk,
                1 if (draw, chunk) == (0, 0) else 0,
                raw,
                f"{draw}-{chunk}",
            )
    bundle, aggregate = routed_execution.build_execution_bundle(
        fixture["plan_path"]
    )
    verified, state = routed_execution.verify_execution_bundle(
        fixture["root"]
        / routed_execution.DERIVED_DIRNAME
        / routed_execution.BUNDLE_FILENAME
    )
    assert verified == bundle
    assert aggregate == state["aggregate_header"]
    assert len(bundle["bundle_core"]["attempts"]) == 10
    outcomes = {row["qid"]: row["outcome"] for row in state["vote_rows"]}
    assert outcomes == {
        7: "folder_majority",
        2: "null_majority",
        99: "no_consensus",
        41: "folder_majority",
        5: "null_majority",
    }
    aggregate_path = (
        fixture["root"]
        / routed_execution.DERIVED_DIRNAME
        / routed_execution.AGGREGATE_FILENAME
    )
    _, _, picks = read_pick_file(aggregate_path, fixture["parent"])
    assert picks == {7: 0, 2: None, 99: None, 41: 2, 5: None}
    execution = aggregate["pick_core"]["judge"]["execution_aggregate"]
    assert routed_queries._execution_aggregate(
        aggregate["pick_core"]["judge"],
        context["policy_envelope"]["policy_id"],
    ) == execution
    verification = routed_queries._verify_execution_for_score(
        fixture["root"]
        / routed_execution.DERIVED_DIRNAME
        / routed_execution.BUNDLE_FILENAME,
        fixture["parent"],
        aggregate_path,
        execution,
    )
    assert verification["bundle_id"] == bundle["bundle_id"]
    with pytest.raises(RoutedPolicyError, match="overwrite"):
        routed_execution.build_execution_bundle(fixture["plan_path"])

    aggregate_lines = aggregate_path.read_text(encoding="utf-8").splitlines()
    altered = json.loads(aggregate_lines[1])
    altered["pick"] = None
    aggregate_lines[1] = json.dumps(altered)
    aggregate_path.write_text(
        "\n".join(aggregate_lines) + "\n", encoding="utf-8"
    )
    with pytest.raises(RoutedPolicyError, match="hash mismatch"):
        routed_execution.verify_execution_bundle(
            fixture["root"]
            / routed_execution.DERIVED_DIRNAME
            / routed_execution.BUNDLE_FILENAME
        )


def test_incomplete_and_terminal_failure_never_become_null_votes(tmp_path, monkeypatch):
    fixture = _fixture_plan(tmp_path, monkeypatch)
    with pytest.raises(RoutedPolicyError, match="missing provider call"):
        routed_execution.build_execution_bundle(fixture["plan_path"])
    failure = tmp_path / "failure.txt"
    failure.write_text("hard failure", encoding="utf-8")
    routed_execution.seal_execution_attempt(
        fixture["plan_path"],
        draw_index=0,
        chunk_index=0,
        attempt_index=0,
        status="terminal_failure",
        raw_response_path=failure,
        provider_run_id="run",
        provider_request_id="terminal-request",
        provider_response_id="terminal-response",
        started_at_utc="2026-07-23T12:00:00Z",
        completed_at_utc="2026-07-23T12:00:01Z",
        error_type="provider_rejected",
    )
    with pytest.raises(RoutedPolicyError, match="terminally failed"):
        routed_execution.build_execution_bundle(fixture["plan_path"])


def test_execution_metadata_rejects_self_inconsistent_policy_id():
    plan_id = "1" * 64
    routing_policy_id = "2" * 64
    execution = {
        "schema": "unifyweaver.routed-execution-aggregate.v1",
        "plan_id": plan_id,
        "execution_policy_id": "3" * 64,
        "routing_policy_id": routing_policy_id,
        "aggregation_id": "strict-majority-folder-id-v1",
        "draw_count": 3,
        "attempt_set_sha256": "4" * 64,
        "vote_rows_sha256": "5" * 64,
        "inference_status": (
            "integrity-only-execution-dependence-unmodeled"
        ),
    }
    with pytest.raises(RoutedPolicyError, match="does not re-derive"):
        routed_queries._execution_aggregate(
            {"execution_aggregate": execution}, routing_policy_id
        )
