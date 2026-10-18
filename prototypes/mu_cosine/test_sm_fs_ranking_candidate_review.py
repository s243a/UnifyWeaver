"""Enforcement tests for the independent SM-FS candidate-lock review."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from routed_policy import canonical_json_bytes
from sm_fs_protocols import ProtocolError
from sm_fs_ranking_candidate_review import (
    CANDIDATE_SCHEMA,
    EXPECTED_BLOCKERS,
    EXPECTED_CANDIDATE,
    EXPECTED_CODE,
    EXPECTED_ENVIRONMENT,
    EXPECTED_INITIALIZED,
    EXPECTED_REPLACEMENT_SEQUENCE,
    REVIEW_PATH,
    _review_id,
    load_and_verify_candidate_review,
    verify_reviewed_candidate,
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


def _candidate() -> dict:
    return {
        "schema": CANDIDATE_SCHEMA,
        "status": "CANDIDATE — fitting stays blocked",
        "git_commit": EXPECTED_CANDIDATE["git_commit"],
        "warm_start": {
            "sha256": EXPECTED_CANDIDATE["warm_start_sha256"],
        },
        "initialized_checkpoints": {
            seed: {"sha256": sha, "authoritative": True}
            for seed, sha in EXPECTED_INITIALIZED.items()
        },
        "source_targets_sha256": EXPECTED_CANDIDATE["source_targets_sha256"],
        "ranking_bundle_manifest_sha256": (
            EXPECTED_CANDIDATE["ranking_bundle_manifest_sha256"]
        ),
        "training_plan_sha256": EXPECTED_CANDIDATE["training_plan_sha256"],
        "code_sha256": EXPECTED_CODE,
        "observed_environment": EXPECTED_ENVIRONMENT,
        "fitting_authorized": False,
    }


def _write_candidate(tmp_path: Path, candidate: dict, review: dict) -> Path:
    path = tmp_path / "candidate_lock.json"
    path.write_bytes(canonical_json_bytes(candidate))
    os.chmod(path, 0o600)
    review["candidate"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def test_committed_review_is_authentic_but_blocks_step_4_and_fitting():
    review = load_and_verify_candidate_review()
    assert review["candidate"]["byte_reproduced"] is True
    assert set(review["blocking_findings"]) == EXPECTED_BLOCKERS
    assert review["replacement_sequence"] == EXPECTED_REPLACEMENT_SEQUENCE
    assert not any(review["authorization"].values())


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda d: d["candidate"].update(fitting_authorized=True),
            "candidate authorized fitting",
        ),
        (
            lambda d: d.update(blocking_findings=d["blocking_findings"][:-1]),
            "candidate blockers",
        ),
        (
            lambda d: d["authorization"].update(step_4_prereg_amendment_authorized=True),
            "step_4_prereg_amendment_authorized",
        ),
        (
            lambda d: d["authorization"].update(model_fitting_authorized=True),
            "model_fitting_authorized",
        ),
        (
            lambda d: d["grounding"].update(optimizer_reachable=True),
            "optimizer became reachable",
        ),
        (
            lambda d: d["frozen_later_amendment_requirements"].update(
                track_t_independent_regrowth=False
            ),
            "Track T independence",
        ),
        (
            lambda d: d.update(replacement_sequence=d["replacement_sequence"][:-1]),
            "replacement sequence",
        ),
        (
            lambda d: d["parent_contracts"].update(ranking_protocol_sha256="0" * 64),
            "parent contracts",
        ),
        (
            lambda d: d["frozen_later_amendment_requirements"]["primary"].update(
                minimum_point_gain=0.0
            ),
            "primary decision",
        ),
        (
            lambda d: d["frozen_later_amendment_requirements"]["bootstrap"].update(
                draws_per_replicate=81
            ),
            "bootstrap contract",
        ),
        (
            lambda d: d["retention_cascade"].update(execution_authorized=True),
            "retention cascade",
        ),
    ],
)
def test_resealed_authority_mutations_fail(tmp_path, mutation, message):
    document = _review()
    mutation(document)
    path = _write_review_fixture(tmp_path, document)
    with pytest.raises(ProtocolError, match=message):
        load_and_verify_candidate_review(path)


def test_review_document_tamper_fails(tmp_path):
    path = _write_review_fixture(tmp_path, _review())
    document_path = path.with_name(_review()["review_document"])
    document_path.write_bytes(document_path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ProtocolError, match="document hash"):
        load_and_verify_candidate_review(path)


def test_hard_bound_review_id_rejects_unenumerated_reseal(tmp_path):
    document = _review()
    document["note"] = "not part of the frozen review"
    path = _write_review_fixture(tmp_path, document)
    with pytest.raises(ProtocolError, match="frozen candidate-review ID"):
        load_and_verify_candidate_review(path)


def test_synthetic_candidate_verifies_as_rejected_not_authorized(tmp_path):
    review = _review()
    path = _write_candidate(tmp_path, _candidate(), review)
    receipt = verify_reviewed_candidate(path, review)
    assert receipt["candidate_authentic"] is True
    assert receipt["step_4_prereg_amendment_authorized"] is False
    assert receipt["model_fitting_authorized"] is False


def test_candidate_tamper_fails(tmp_path):
    review = _review()
    path = _write_candidate(tmp_path, _candidate(), review)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ProtocolError, match="hash differs"):
        verify_reviewed_candidate(path, review)


def test_candidate_hard_link_fails(tmp_path):
    review = _review()
    path = _write_candidate(tmp_path, _candidate(), review)
    os.link(path, tmp_path / "candidate-alias.json")
    with pytest.raises(ProtocolError, match="exactly one hard link"):
        verify_reviewed_candidate(path, review)


def test_candidate_mode_fails(tmp_path):
    review = _review()
    path = _write_candidate(tmp_path, _candidate(), review)
    os.chmod(path, 0o644)
    with pytest.raises(ProtocolError, match="mode must be 0600"):
        verify_reviewed_candidate(path, review)


def test_candidate_cannot_self_upgrade_authority(tmp_path):
    review = _review()
    candidate = _candidate()
    candidate["fitting_authorized"] = True
    path = _write_candidate(tmp_path, candidate, review)
    with pytest.raises(ProtocolError, match="authorized fitting"):
        verify_reviewed_candidate(path, review)


def test_review_id_uses_canonical_json():
    document = _review()
    expected = document.pop("review_id")
    assert hashlib.sha256(canonical_json_bytes(document)).hexdigest() == expected
