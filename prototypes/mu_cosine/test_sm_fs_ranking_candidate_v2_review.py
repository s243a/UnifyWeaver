"""Enforcement and adversarial tests for the rejected SM-FS candidate v2."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

import sm_fs_ranking_lock_verify as supplied
import sm_fs_ranking_pipeline as pipeline
from mu_attention import MuAttention
from routed_policy import canonical_json_bytes
from sm_fs_protocols import ProtocolError
from sm_fs_ranking_candidate_review import EXPECTED_BOOTSTRAP
from sm_fs_ranking_candidate_v2_review import (
    EXPECTED_BLOCKERS,
    EXPECTED_REPLACEMENT_SEQUENCE,
    REVIEW_PATH,
    _review_id,
    load_and_verify_candidate_v2_review,
)


def _review() -> dict:
    return json.loads(REVIEW_PATH.read_text(encoding="utf-8"))


def _write_review_fixture(tmp_path: Path, document: dict) -> Path:
    source = REVIEW_PATH.with_name(document["review_document"])
    copied = tmp_path / document["review_document"]
    copied.write_bytes(source.read_bytes())
    document["review_document_sha256"] = hashlib.sha256(copied.read_bytes()).hexdigest()
    document["review_id"] = _review_id(document)
    path = tmp_path / REVIEW_PATH.name
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def test_committed_v2_review_blocks_every_authorization():
    review = load_and_verify_candidate_v2_review()
    assert set(review["blocking_findings"]) == EXPECTED_BLOCKERS
    assert review["replacement_sequence"] == EXPECTED_REPLACEMENT_SEQUENCE
    assert not any(review["authorization"].values())
    assert review["candidate"]["training_plan_sha256"] is None
    assert review["grounding"]["countersigned_checkpoint_loads_in_pipeline"] is False


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda d: d["authorization"].update(step_4_prereg_amendment_authorized=True),
            "step_4_prereg_amendment_authorized",
        ),
        (
            lambda d: d["authorization"].update(model_fitting_authorized=True),
            "model_fitting_authorized",
        ),
        (
            lambda d: d.update(blocking_findings=d["blocking_findings"][:-1]),
            "candidate v2 blockers",
        ),
        (
            lambda d: d["grounding"].update(countersigned_checkpoint_loads_in_pipeline=True),
            "candidate v2 grounding",
        ),
        (
            lambda d: d["grounding"].update(frozen_bootstrap_implemented=True),
            "candidate v2 grounding",
        ),
        (
            lambda d: d["candidate"].update(training_plan_sha256="0" * 64),
            "acquired a plan",
        ),
        (
            lambda d: d.update(replacement_sequence=d["replacement_sequence"][:-1]),
            "replacement sequence",
        ),
        (
            lambda d: d["frozen_later_amendment_requirements"]["primary"].update(
                minimum_point_gain=0.0
            ),
            "primary decision",
        ),
        (
            lambda d: d["frozen_later_amendment_requirements"]["bootstrap"].update(
                sampler_id="changed"
            ),
            "bootstrap contract",
        ),
        (
            lambda d: d["retention_cascade"].update(execution_authorized=True),
            "retention cascade",
        ),
    ],
)
def test_resealed_v2_review_mutations_fail(tmp_path, mutation, message):
    document = _review()
    mutation(document)
    path = _write_review_fixture(tmp_path, document)
    with pytest.raises(ProtocolError, match=message):
        load_and_verify_candidate_v2_review(path)


def test_review_document_tamper_fails(tmp_path):
    path = _write_review_fixture(tmp_path, _review())
    document_path = path.with_name(_review()["review_document"])
    document_path.write_bytes(document_path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ProtocolError, match="document hash"):
        load_and_verify_candidate_v2_review(path)


def test_hard_bound_review_id_rejects_unenumerated_reseal(tmp_path):
    document = _review()
    document["unreviewed_note"] = True
    path = _write_review_fixture(tmp_path, document)
    with pytest.raises(ProtocolError, match="frozen candidate-v2-review ID"):
        load_and_verify_candidate_v2_review(path)


def test_pipeline_constructor_is_incompatible_with_countersigned_cfg():
    params = inspect.signature(MuAttention.__init__).parameters
    assert "heads" not in params and "layers" not in params
    assert "n_heads" in params and "n_layers" in params
    source = inspect.getsource(pipeline.cmd_fit)
    assert 'MuAttention(**blob["cfg"])' in source
    legacy_cfg = {
        "d_model": 384,
        "heads": 4,
        "layers": 3,
        "judge_name": True,
        "ridge": 0.1,
        "op_name": True,
        "corpus_name": True,
    }
    with pytest.raises(TypeError, match="unexpected keyword argument 'heads'"):
        MuAttention(**legacy_cfg)


def test_supplied_verifier_accepts_tampered_claimed_bindings(monkeypatch):
    baseline = {
        "git_commit": "a" * 40,
        "ranking_bundle_manifest_sha256": "b" * 64,
        "initialized_checkpoints": {"3997001": "c" * 64},
        "title_table_sha256": "d" * 64,
        "e5_revision": "e" * 40,
        "code_sha256": {"x": "f" * 64},
        "adam": {"lr": 0.1},
        "steps": 1,
        "batch_size": 2,
        "query_draws": 1,
        "anchor_weight": 1.0,
        "grad_clip": 1.0,
        "trainable_contract": {"tensor_count": 18, "param_count": 1195782},
        "tie_rule": "ascending-frozen-catalog-column",
        "bootstrap": {"seed": 3997999},
        "schemas": ["x"],
        "environment": {"cuda_available": True},
        "tokenizer_structure": "expected",
        "frozen_reference": "expected",
        "augmentation": {"anchor_rows_augmented": False},
        "early_stopping": False,
        "cleanliness_scope": "expected",
    }
    monkeypatch.setattr(supplied, "_bindings", lambda: dict(baseline))
    lock = dict(baseline)
    lock.update(schema=supplied.CAND_SCHEMA, fitting_authorized=False)
    lock["environment"] = {"cuda_available": False}
    lock["tokenizer_structure"] = "forged"
    lock["frozen_reference"] = "forged"
    lock["augmentation"] = {"anchor_rows_augmented": True}
    lock["early_stopping"] = True
    lock["cleanliness_scope"] = "forged"
    assert supplied.verify_candidate_lock(lock) is True


def test_no_final_lock_or_independent_receipt_emitter_exists():
    assert not hasattr(supplied, "emit_final")
    assert not hasattr(supplied, "emit_verification_receipt")
    source = inspect.getsource(pipeline.fitting_allowed)
    assert 'receipt.get("verifier") not in (None, "", "self")' in source


def test_current_decision_uses_wrong_sampler_and_percentile():
    source = inspect.getsource(pipeline.cmd_decide)
    assert "np.random.default_rng" in source
    assert "np.percentile" in source
    assert EXPECTED_BOOTSTRAP["sampler_id"] not in source


def test_nan_destination_score_is_silently_ranked_first_by_current_expression():
    scores = np.array([0.8, np.nan, 0.7])
    destination_index = 1
    rank = 1 + int(np.sum(scores > scores[destination_index])) + int(
        np.sum(
            (scores == scores[destination_index])
            & (np.arange(len(scores)) < destination_index)
        )
    )
    assert rank == 1
    assert not np.isfinite(scores).all()


def test_install_failure_after_link_leaves_permanent_target(tmp_path, monkeypatch):
    target = tmp_path / "receipt.json"
    real_fsync = pipeline.os.fsync
    calls = 0

    def fail_directory_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(pipeline.os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="injected directory fsync failure"):
        pipeline.install_private(str(target), b"payload\n")
    assert target.exists()
    assert not list(tmp_path.glob(".stage-*"))


def test_review_id_uses_canonical_json():
    document = _review()
    expected = document.pop("review_id")
    assert hashlib.sha256(canonical_json_bytes(document)).hexdigest() == expected
