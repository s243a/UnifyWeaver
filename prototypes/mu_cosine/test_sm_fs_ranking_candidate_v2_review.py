"""Enforcement and adversarial tests for the rejected SM-FS candidate v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from routed_policy import canonical_json_bytes
from sm_fs_protocols import ProtocolError
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


def test_v2_review_permanently_records_checkpoint_and_optimizer_failure():
    review = load_and_verify_candidate_v2_review()
    assert review["grounding"]["countersigned_checkpoint_loads_in_pipeline"] is False
    assert review["grounding"]["optimizer_reachable"] is False
    assert (
        "countersigned-checkpoint-config-incompatible-with-fit-and-evaluate"
        in review["blocking_findings"]
    )


def test_v2_review_permanently_records_incomplete_binding_verification():
    review = load_and_verify_candidate_v2_review()
    assert review["candidate"]["complete_state_reproduction_enforced"] is False
    assert (
        "claimed-environment-tokenizer-reference-augmentation-and-trainable-bindings-not-enforced"
        in review["blocking_findings"]
    )


def test_v2_review_permanently_records_missing_finalization_emitters():
    review = load_and_verify_candidate_v2_review()
    assert review["grounding"]["final_lock_emitter_present"] is False
    assert review["grounding"]["verification_receipt_emitter_present"] is False
    assert (
        "final-lock-and-independent-receipt-not-candidate-bound"
        in review["blocking_findings"]
    )


def test_v2_review_permanently_records_wrong_bootstrap_implementation():
    review = load_and_verify_candidate_v2_review()
    assert review["grounding"]["frozen_bootstrap_implemented"] is False
    assert (
        "frozen-bootstrap-and-nonfinite-ranking-fail-closed-contract-not-implemented"
        in review["blocking_findings"]
    )


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


def test_v2_review_permanently_records_transaction_failure():
    review = load_and_verify_candidate_v2_review()
    assert (
        "transaction-rollback-directory-durability-and-private-mode-contract-incomplete"
        in review["blocking_findings"]
    )


def test_review_id_uses_canonical_json():
    document = _review()
    expected = document.pop("review_id")
    assert hashlib.sha256(canonical_json_bytes(document)).hexdigest() == expected
