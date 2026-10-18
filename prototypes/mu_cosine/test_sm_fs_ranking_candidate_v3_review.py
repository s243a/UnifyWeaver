"""Enforcement tests for the fail-closed SM-FS ranking candidate-v3 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from routed_policy import canonical_json_bytes
from sm_fs_protocols import ProtocolError
from sm_fs_ranking_candidate_v3_review import (
    EXPECTED_BLOCKERS,
    EXPECTED_REVIEW_ID,
    EXPECTED_REPLACEMENT_SEQUENCE,
    REVIEW_PATH,
    _review_id,
    load_and_verify_candidate_v3_review,
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


def test_committed_v3_review_authenticates_and_blocks_every_authorization():
    review = load_and_verify_candidate_v3_review()
    assert review["review_id"] == EXPECTED_REVIEW_ID
    assert set(review["blocking_findings"]) == EXPECTED_BLOCKERS
    assert review["replacement_sequence"] == EXPECTED_REPLACEMENT_SEQUENCE
    assert not any(review["authorization"].values())
    assert review["candidate"]["fitting_authorized"] is False
    assert review["candidate"]["complete_authorization_chain_enforced"] is False


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
            "candidate v3 blockers",
        ),
        (
            lambda d: d["grounding"].update(fabricated_receipts_can_pass_primary_gate=False),
            "candidate v3 grounding",
        ),
        (
            lambda d: d["candidate"].update(sha256="0" * 64),
            "candidate v3 sha256",
        ),
        (
            lambda d: d["candidate"].update(training_plan_sha256="0" * 64),
            "candidate v3 training_plan_sha256",
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
def test_resealed_v3_review_mutations_fail(tmp_path, mutation, message):
    document = _review()
    mutation(document)
    path = _write_review_fixture(tmp_path, document)
    with pytest.raises(ProtocolError, match=message):
        load_and_verify_candidate_v3_review(path)


def test_review_document_tamper_fails(tmp_path):
    path = _write_review_fixture(tmp_path, _review())
    document_path = path.with_name(_review()["review_document"])
    document_path.write_bytes(document_path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ProtocolError, match="document hash"):
        load_and_verify_candidate_v3_review(path)


def test_hard_bound_review_id_rejects_unenumerated_reseal(tmp_path):
    document = _review()
    document["unreviewed_note"] = True
    path = _write_review_fixture(tmp_path, document)
    with pytest.raises(ProtocolError, match="frozen candidate-v3-review ID"):
        load_and_verify_candidate_v3_review(path)


def test_timing_boundary_is_frozen(tmp_path):
    document = _review()
    document["reviewed_before"].remove("held-fold-score")
    path = _write_review_fixture(tmp_path, document)
    with pytest.raises(ProtocolError, match="timing boundary"):
        load_and_verify_candidate_v3_review(path)


def test_parent_review_chain_is_frozen(tmp_path):
    document = _review()
    document["parent_contracts"]["prior_candidate_review_id"] = "0" * 64
    path = _write_review_fixture(tmp_path, document)
    with pytest.raises(ProtocolError, match="parent contracts"):
        load_and_verify_candidate_v3_review(path)


def test_review_id_uses_canonical_json():
    document = _review()
    expected = document.pop("review_id")
    assert hashlib.sha256(canonical_json_bytes(document)).hexdigest() == expected
