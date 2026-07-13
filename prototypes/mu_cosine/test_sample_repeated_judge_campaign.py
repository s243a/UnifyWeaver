#!/usr/bin/env python3
"""CLI/materialization tests for the repeated-judge campaign sampler."""

import csv
import json
from pathlib import Path

import pytest

import sample_repeated_judge_campaign as sampler

from repeated_judge_campaign import (
    CANDIDATE_REQUIRED_COLUMNS,
    DEFAULT_AGREEMENT_CLASSES,
    DEFAULT_CORPORA,
    DEFAULT_DEGREE_QUARTILES,
    DEFAULT_HOP_TRANSITIONS,
    FROZEN_NOMIC_MODEL,
    FROZEN_NOMIC_PREFIX,
    FROZEN_NOMIC_REVISION,
    FROZEN_WALK_WEIGHTS,
    content_record,
    tsv_bytes,
)
from sample_repeated_judge_campaign import main


EXTRA_COLUMN = "nomic_similarity"


def candidate_record(candidate_id, cell, base):
    corpus, hop, degree, agreement = cell
    first_hop, second_hop = (part[1:] if part.startswith("h") else part for part in hop.split("-"))
    return {
        "candidate_id": candidate_id,
        "corpus": corpus,
        "source_component": f"source-{corpus}-{base}",
        "hop_transition": hop,
        "degree_quartile": degree,
        "agreement_class": agreement,
        "anchor_hop": first_hop,
        "adjacent_hop": second_hop,
        "distant_hop": second_hop,
        "anchor_campaign_tag": "anchor-campaign",
        "adjacent_campaign_tag": "comparator-campaign",
        "distant_campaign_tag": "comparator-campaign",
        "anchor_degree_quartile": degree,
        "adjacent_degree_quartile": "q2",
        "distant_degree_quartile": "q2",
        "anchor_adjacent_direct_edge": "true",
        "anchor_distant_distance": "3",
        "anchor_distant_disconnected": "false",
        "cumulative_anchor_adjacent_similarity": "0.9",
        "cumulative_anchor_distant_similarity": "0.1",
        "nomic_anchor_adjacent_similarity": "0.8" if agreement == "agreement" else "0.2",
        "nomic_anchor_distant_similarity": "0.2" if agreement == "agreement" else "0.8",
        "descendant_id": f"{corpus}-id-{base}-x",
        "descendant_title": f"{corpus}_Title_{base}_x",
        "anchor_id": f"{corpus}-id-{base}-a",
        "anchor_title": f"{corpus}_Title_{base}_a",
        "adjacent_id": f"{corpus}-id-{base}-b",
        "adjacent_title": f"{corpus}_Title_{base}_b",
        "distant_id": f"{corpus}-id-{base}-c",
        "distant_title": f"{corpus}_Title_{base}_c",
        EXTRA_COLUMN: f"0.{base % 10}",
    }


def write_inputs(tmp_path):
    rows = []
    value = 0
    for cell in (
        (corpus, hop, degree, agreement)
        for corpus in DEFAULT_CORPORA
        for hop in DEFAULT_HOP_TRANSITIONS
        for degree in DEFAULT_DEGREE_QUARTILES
        for agreement in DEFAULT_AGREEMENT_CLASSES
    ):
        for _ in range(10):
            rows.append(candidate_record(f"candidate-{value:04d}", cell, value))
            value += 1
    historical_candidate = candidate_record(
        "candidate-historical-extra",
        (
            DEFAULT_CORPORA[0], DEFAULT_HOP_TRANSITIONS[0],
            DEFAULT_DEGREE_QUARTILES[0], DEFAULT_AGREEMENT_CLASSES[0],
        ),
        999999,
    )
    rows.append(historical_candidate)
    pool = tmp_path / "candidate_pool.tsv"
    pool.write_bytes(tsv_bytes((*CANDIDATE_REQUIRED_COLUMNS, EXTRA_COLUMN), rows))
    history = tmp_path / "historical.tsv"
    history.write_bytes(tsv_bytes(
        ("corpus", "endpoint_id", "endpoint_title"),
        [{
            "corpus": historical_candidate["corpus"],
            "endpoint_id": historical_candidate["anchor_id"],
            "endpoint_title": historical_candidate["anchor_title"],
        }],
    ))
    builder = tmp_path / "candidate_builder.json"
    builder.write_text(json.dumps({
        "schema_version": 1,
        "algorithm": "test-candidate-builder-v1",
        "implementation_sha256": "d" * 64,
        "candidate_pool": content_record(pool.read_bytes()),
        "graph": {
            "artifact_sha256": "e" * 64,
            "cumulative_walk_weights": list(FROZEN_WALK_WEIGHTS),
        },
        "nomic": {
            "model_id": FROZEN_NOMIC_MODEL,
            "revision": FROZEN_NOMIC_REVISION,
            "task_prefix": FROZEN_NOMIC_PREFIX,
            "embedding_manifest_sha256": "f" * 64,
        },
        "agreement_thresholds": {"graph_delta_min": 0.0, "nomic_delta_min": 0.0},
    }), encoding="utf-8")
    request = tmp_path / "request_contract.json"
    request.write_text(json.dumps({
        "schema_version": 1,
        "judges": {
            "gpt-5.5-low": {
                "model_id": "gpt-5.5",
                "model_revision": "test-revision-55",
                "prompt_id": "frozen-relation-prompt",
                "prompt_sha256": "a" * 64,
                "reasoning_effort": "low",
                "settings": {"temperature": 0},
                "call_seed": None,
                "stateless": True,
            },
            "gpt-5.6-luna": {
                "model_id": "gpt-5.6-luna",
                "model_revision": "test-revision-luna",
                "prompt_id": "frozen-relation-prompt",
                "prompt_sha256": "a" * 64,
                "reasoning_effort": "low",
                "settings": {"temperature": 0},
                "call_seed": None,
                "stateless": True,
            },
        },
    }), encoding="utf-8")
    return pool, history, builder, request


def tree_payloads(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def data_rows(path):
    return [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]


def test_cli_materializes_reproducible_content_addressed_campaign(tmp_path, capsys):
    pool, history, builder, request = write_inputs(tmp_path)
    first = tmp_path / "campaign-a"
    second = tmp_path / "campaign-b"
    common = [
        "--candidate-pool", str(pool),
        "--candidate-builder-manifest", str(builder),
        "--request-contract", str(request),
        "--historical-endpoints", str(history),
    ]
    assert main([*common, "--out-dir", str(first)]) == 0
    capsys.readouterr()
    assert main([*common, "--out-dir", str(second)]) == 0
    capsys.readouterr()

    assert tree_payloads(first) == tree_payloads(second)
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection"]["component_count"] == 640
    assert manifest["selection"]["row_count"] == 1920
    assert manifest["selection"]["exclusions"] == {"historical_endpoint_id": 1}
    assert manifest["schedule"]["wave_count"] == 6
    assert manifest["schedule"]["records"] == 11520
    assert manifest["schedule"]["same_component_per_request_max"] == 1
    assert manifest["schedule"]["prompt_block_is_inference_cluster"] is True
    assert manifest["classification"] == (
        "protocol-shape-compatible-no-spend-inputs-unverified"
    )
    assert manifest["candidate_builder_verified_by_repository"] is False
    assert manifest["request_contract_approved_for_live_use"] is False
    assert manifest["call_authorized"] is False
    assert str(tmp_path) not in json.dumps(manifest)
    assert "elapsed" not in json.dumps(manifest).lower()
    assert EXTRA_COLUMN in (first / "selected_components.tsv").read_text(encoding="utf-8").splitlines()[0]

    with open(first / "scoring_schedule.tsv", encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    by_request = {}
    for record in records:
        key = (record["wave_id"], record["request_id"])
        by_request.setdefault(key, []).append(record)
    assert by_request
    for request in by_request.values():
        assert 1 <= len(request) <= 10
        assert len({record["component_id"] for record in request}) == len(request)
        assert len({(
            record["corpus"], record["outer_fold"], record["global_inner_fold"]
        ) for record in request}) == 1
        assert len({record["prompt_block_id"] for record in request}) == 1
        assert all(record["prompt_block_id"] == record["inference_cluster_id"] for record in request)

    block_members = {}
    for record in records:
        key = (record["prompt_block_id"], record["role"])
        block_members.setdefault(key, set()).add(record["component_id"])
    for block_id in {record["prompt_block_id"] for record in records}:
        memberships = [block_members[(block_id, role)] for role in ("anchor", "adjacent", "distant")]
        assert memberships[0] == memberships[1] == memberships[2]

    score_inputs = sorted((first / "score_inputs").glob("*.tsv"))
    assert len(score_inputs) == 6
    assert all(len(data_rows(path)) == 1920 for path in score_inputs)
    assert all(path.read_text(encoding="utf-8").startswith("# row_id\t") for path in score_inputs)
    assert (first / "nested_component_folds.tsv").exists()
    assert (first / "response_ingestion_schema.json").exists()
    request_inputs = sorted((first / "request_inputs").rglob("*.tsv"))
    assert len(request_inputs) == len(by_request)
    assert all(1 <= len(data_rows(path)) <= 10 for path in request_inputs)
    output_names = {record["name"] for record in manifest["outputs"]}
    assert output_names == {
        path.relative_to(first).as_posix()
        for path in first.rglob("*") if path.is_file() and path.name != "manifest.json"
    }


def test_cli_marks_direct_per_cell_override_exploratory(tmp_path, capsys):
    pool, history, builder, request = write_inputs(tmp_path)
    out = tmp_path / "invalid-campaign"
    assert main([
        "--candidate-pool", str(pool),
        "--candidate-builder-manifest", str(builder),
        "--request-contract", str(request),
        "--historical-endpoints", str(history),
        "--out-dir", str(out),
        "--per-cell", "9",
    ]) == 0
    capsys.readouterr()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["classification"] == "exploratory-smoke-only"
    assert manifest["call_authorized"] is False
    assert max(manifest["folds"]["component_counts"].values()) - min(
        manifest["folds"]["component_counts"].values()
    ) <= 1


def test_cli_refuses_to_overwrite_an_existing_directory(tmp_path):
    pool, history, builder, request = write_inputs(tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        main([
            "--candidate-pool", str(pool),
            "--candidate-builder-manifest", str(builder),
            "--request-contract", str(request),
            "--historical-endpoints", str(history),
            "--out-dir", str(out),
        ])


def test_cli_rejects_builder_hash_drift_repeats_and_safe_name_collisions(tmp_path):
    pool, history, builder, request = write_inputs(tmp_path)
    common = [
        "--candidate-pool", str(pool),
        "--candidate-builder-manifest", str(builder),
        "--request-contract", str(request),
        "--historical-endpoints", str(history),
    ]
    with pytest.raises(SystemExit, match="at least three"):
        main([*common, "--out-dir", str(tmp_path / "too-few"), "--repeats", "2"])
    with pytest.raises(SystemExit, match="collide"):
        main([
            *common,
            "--out-dir", str(tmp_path / "collision"),
            "--judges", "judge/a", "judge_a",
        ])

    pool.write_bytes(pool.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="does not match"):
        main([*common, "--out-dir", str(tmp_path / "drift")])


def test_materialization_cleans_temporary_directory_after_write_failure(
    tmp_path, monkeypatch
):
    pool, history, builder, request = write_inputs(tmp_path)
    out = tmp_path / "atomic"
    original = sampler._artifact
    count = {"value": 0}

    def fail_on_second(root, relative, payload):
        count["value"] += 1
        if count["value"] == 2:
            raise RuntimeError("injected artifact failure")
        return original(root, relative, payload)

    monkeypatch.setattr(sampler, "_artifact", fail_on_second)
    with pytest.raises(RuntimeError, match="injected"):
        sampler.main([
            "--candidate-pool", str(pool),
            "--candidate-builder-manifest", str(builder),
            "--request-contract", str(request),
            "--historical-endpoints", str(history),
            "--out-dir", str(out),
        ])
    assert not out.exists()
    assert not list(tmp_path.glob(".atomic.*"))


def test_nonfrozen_selector_seed_is_exploratory_only():
    args = sampler.build_arg_parser().parse_args([
        "--candidate-pool", "pool.tsv",
        "--candidate-builder-manifest", "builder.json",
        "--request-contract", "request.json",
        "--historical-endpoints", "history.tsv",
        "--out-dir", "campaign",
    ])
    specs = {
        "gpt-5.5-low": {
            "model_id": "gpt-5.5", "reasoning_effort": "low", "stateless": True,
        },
        "gpt-5.6-luna": {
            "model_id": "gpt-5.6-luna", "reasoning_effort": "low", "stateless": True,
        },
    }
    assert sampler._configuration_classification(args, specs)[0] == (
        "protocol-shape-compatible-no-spend-inputs-unverified"
    )
    args.seed = 1
    classification, deviations = sampler._configuration_classification(args, specs)
    assert classification == "exploratory-smoke-only"
    assert any("selector seed" in deviation for deviation in deviations)
