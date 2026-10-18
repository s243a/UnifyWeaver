#!/usr/bin/env python3
"""Focused tests for the synthetic-only local privacy benchmark."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import benchmark_sm_fs_privacy_local as benchmark


PROVENANCE = {
    "model": {
        "tag": "test:4b",
        "digest": "0123456789ab",
        "ollama_version": "ollama version is 0.test",
        "ollama_cli": "/test/ollama",
    },
    "repo_commit": "a" * 40,
    "endpoint": "http://127.0.0.1:11434",
    "sources": {
        "classifier_sha256": "b" * 64,
        "evaluator_sha256": "c" * 64,
        "llm_bridge_sha256": "d" * 64,
    },
}


def _response(*, wrong_first_label: bool = False) -> str:
    rows = []
    for index, case in enumerate(benchmark.synthetic_cases()):
        interpretation = case["expected"]
        if wrong_first_label and index == 0:
            interpretation = "uncertain"
        rows.append(
            {
                "qid": case["task"]["qid"],
                "interpretation": interpretation,
                "reason": "synthetic test response",
            }
        )
    return json.dumps(rows)


def _install_fake_model(monkeypatch, response: str) -> None:
    monkeypatch.setattr(
        benchmark,
        "_provenance_snapshot",
        lambda _model: copy.deepcopy(PROVENANCE),
    )

    def fake_call(
        _prompt,
        _model,
        _timeout,
        *,
        response_schema,
        max_output_tokens,
        think,
        metadata_out,
    ):
        assert response_schema["minItems"] == 12
        assert response_schema["maxItems"] == 12
        assert max_output_tokens == 2048
        assert think is False
        metadata_out.update(
            {"done_reason": "stop", "eval_count": 71, "total_duration": 73}
        )
        return response

    monkeypatch.setattr(benchmark, "call_ollama_json", fake_call)


def test_benchmark_main_seals_strict_pass_with_bound_provenance(
    tmp_path, monkeypatch
):
    _install_fake_model(monkeypatch, _response())
    output = tmp_path / "private" / "benchmark.json"

    assert (
        benchmark.main(
            [
                "--model",
                "test:4b",
                "--repetitions",
                "2",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == "unifyweaver.sm-fs-privacy-synthetic-benchmark.v2"
    assert report["strict_pass"] is True
    assert report["decision"] == "benchmark_gate_passed_lock_not_implemented"
    assert report["model"] == PROVENANCE["model"]
    assert report["repo_commit"] == PROVENANCE["repo_commit"]
    assert {
        key: report[key] for key in PROVENANCE["sources"]
    } == PROVENANCE["sources"]
    assert report["review_eligible_total_per_run"] == 7
    assert all(run["review_eligible_correct"] == 7 for run in report["runs"])
    assert all(
        run["ollama_eval"]
        == {"done_reason": "stop", "eval_count": 71, "total_duration": 73}
        for run in report["runs"]
    )


def test_benchmark_main_records_fail_without_authorizing_pilot(
    tmp_path, monkeypatch
):
    _install_fake_model(monkeypatch, _response(wrong_first_label=True))
    output = tmp_path / "benchmark.json"

    assert (
        benchmark.main(
            ["--model", "test:4b", "--repetitions", "1", "--out", str(output)]
        )
        == 2
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["strict_pass"] is False
    assert report["decision"] == "advisory_pilot_not_allowed"


def test_benchmark_main_refuses_to_replace_an_artifact(tmp_path, monkeypatch):
    _install_fake_model(monkeypatch, _response())
    output = tmp_path / "benchmark.json"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="File exists"):
        benchmark.main(
            ["--model", "test:4b", "--repetitions", "1", "--out", str(output)]
        )
    assert output.read_text(encoding="utf-8") == "existing\n"


def test_benchmark_main_rejects_midrun_provenance_change(
    tmp_path, monkeypatch
):
    before = copy.deepcopy(PROVENANCE)
    after = copy.deepcopy(PROVENANCE)
    after["model"]["digest"] = "fedcba987654"
    snapshots = iter((before, after))
    monkeypatch.setattr(
        benchmark, "_provenance_snapshot", lambda _model: next(snapshots)
    )

    def fake_call(*_args, metadata_out, **_kwargs):
        metadata_out["done_reason"] = "stop"
        return _response()

    monkeypatch.setattr(benchmark, "call_ollama_json", fake_call)
    output = tmp_path / "benchmark.json"

    with pytest.raises(RuntimeError, match="provenance changed"):
        benchmark.main(
            ["--model", "test:4b", "--repetitions", "1", "--out", str(output)]
        )
    assert not output.exists()


def test_provenance_snapshot_requires_resolved_model_and_commit(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "_model_identity",
        lambda _model: {
            "tag": "test:4b",
            "digest": "",
            "ollama_version": "",
            "ollama_cli": "/test/ollama",
        },
    )
    with pytest.raises(RuntimeError, match="model identity"):
        benchmark._provenance_snapshot("test:4b")

    monkeypatch.setattr(
        benchmark,
        "_model_identity",
        lambda _model: copy.deepcopy(PROVENANCE["model"]),
    )
    monkeypatch.setattr(benchmark, "_command_text", lambda _command: "")
    with pytest.raises(RuntimeError, match="repository commit"):
        benchmark._provenance_snapshot("test:4b")
