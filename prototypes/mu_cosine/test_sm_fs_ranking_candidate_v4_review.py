"""Enforcement tests for the fail-closed SM-FS ranking candidate-v4 review."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from routed_policy import canonical_json_bytes
from sm_fs_protocols import ProtocolError
import sm_fs_ranking_chain as chain
from sm_fs_ranking_candidate_v4_review import (
    EXPECTED_AUTHORIZATION,
    EXPECTED_BLOCKERS,
    EXPECTED_REPLACEMENT_SEQUENCE,
    EXPECTED_REVIEW_ID,
    REVIEW_PATH,
    RUN_DIR,
    _review_id,
    load_and_verify_candidate_v4_review,
    verify_local_candidate_v4,
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


def test_committed_v4_review_authenticates_and_blocks_every_authorization():
    review = load_and_verify_candidate_v4_review()
    assert review["review_id"] == EXPECTED_REVIEW_ID
    assert review["blocking_findings"] == EXPECTED_BLOCKERS
    assert review["authorization"] == EXPECTED_AUTHORIZATION
    assert not any(review["authorization"].values())
    assert review["replacement_sequence"] == EXPECTED_REPLACEMENT_SEQUENCE
    assert review["candidate"]["fitting_authorized"] is False
    assert review["candidate"]["complete_authorization_chain_enforced"] is False
    assert review["candidate"]["model_derived_score_evidence_enforced"] is False


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda d: d["authorization"].update(candidate_accepted=True),
            "authorization boundary",
        ),
        (
            lambda d: d["authorization"].update(model_fitting_authorized=True),
            "authorization boundary",
        ),
        (
            lambda d: d["authorization"].update(held_fold_scoring_authorized=True),
            "authorization boundary",
        ),
        (
            lambda d: d["candidate"].update(sha256="0" * 64),
            "candidate v4 identity",
        ),
        (
            lambda d: d["candidate"].update(git_commit="0" * 40),
            "candidate v4 identity",
        ),
        (
            lambda d: d["candidate"].update(training_plan_sha256="0" * 64),
            "candidate v4 identity",
        ),
        (
            lambda d: d["candidate"]["code_sha256"].update(
                {"sm_fs_ranking_chain.py": "0" * 64}
            ),
            "candidate v4 identity",
        ),
        (
            lambda d: d["parent_contracts"].update(prior_candidate_review_id="0" * 64),
            "parent contracts",
        ),
        (
            lambda d: d.update(blocking_findings=d["blocking_findings"][:-1]),
            "candidate v4 blockers",
        ),
        (
            lambda d: d["grounding"].update(
                fabricated_scores_can_pass_primary_gate=False
            ),
            "enforcement grounding",
        ),
        (
            lambda d: d["adversarial_reproductions"]["fabricated_scores"].update(
                passed_exploratory_gate=False
            ),
            "fabricated-score reproduction",
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
                resamples=100
            ),
            "bootstrap contract",
        ),
        (
            lambda d: d["retention_cascade"].update(reserve_rows_authorized=True),
            "retention cascade",
        ),
    ],
)
def test_resealed_v4_review_mutations_fail(tmp_path, mutation, message):
    document = _review()
    mutation(document)
    path = _write_review_fixture(tmp_path, document)
    with pytest.raises(ProtocolError, match=message):
        load_and_verify_candidate_v4_review(path)


def test_review_document_tamper_fails(tmp_path):
    path = _write_review_fixture(tmp_path, _review())
    document_path = path.with_name(_review()["review_document"])
    document_path.write_bytes(document_path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ProtocolError, match="document hash"):
        load_and_verify_candidate_v4_review(path)


def test_hard_bound_review_id_rejects_unenumerated_reseal(tmp_path):
    document = _review()
    document["unreviewed_note"] = True
    path = _write_review_fixture(tmp_path, document)
    with pytest.raises(ProtocolError, match="top-level fields"):
        load_and_verify_candidate_v4_review(path)


def test_duplicate_key_and_nonfinite_json_fail_strictly(tmp_path):
    path = _write_review_fixture(tmp_path, _review())
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw.replace(
        '"schema": "unifyweaver.sm-fs-ranking-candidate-review.v4",',
        '"schema": "unifyweaver.sm-fs-ranking-candidate-review.v4",'
        '"schema": "unifyweaver.sm-fs-ranking-candidate-review.v4",',
        1,
    ), encoding="utf-8")
    with pytest.raises(ProtocolError):
        load_and_verify_candidate_v4_review(path)

    path = _write_review_fixture(tmp_path, _review())
    raw = path.read_text(encoding="utf-8").replace(
        '"delta_mrr": 0.9878048780487787', '"delta_mrr": NaN'
    )
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ProtocolError):
        load_and_verify_candidate_v4_review(path)


def test_wrong_review_filename_fails(tmp_path):
    path = _write_review_fixture(tmp_path, _review())
    wrong = tmp_path / "renamed.json"
    path.rename(wrong)
    with pytest.raises(ProtocolError, match="filename"):
        load_and_verify_candidate_v4_review(wrong)


def test_tracked_rejection_cannot_satisfy_candidate_acceptance(tmp_path):
    repo = tmp_path / "repo"
    rel = "prototypes/mu_cosine/SM_FS_LINEAGE_RANKING_CANDIDATE_V4_REVIEW.json"
    target = repo / rel
    target.parent.mkdir(parents=True)
    target.write_bytes(canonical_json_bytes(_review()))
    for args in (
        ["init", "-q"],
        ["config", "user.email", "review@test"],
        ["config", "user.name", "review"],
        ["add", rel],
        ["commit", "-q", "-m", "tracked rejection"],
    ):
        subprocess.run(["git"] + args, cwd=repo, check=True, capture_output=True)
    with pytest.raises(chain.ChainError, match="candidate_accepted"):
        chain.verify_accepted_review(
            rel,
            _review()["candidate"]["sha256"],
            str(repo),
        )


def test_local_candidate_v4_authenticates_and_namespace_remains_pre_fit():
    candidate_path = RUN_DIR / "candidate_lock_v4.json"
    if not candidate_path.exists():
        pytest.skip("private local candidate v4 is not present")
    candidate = verify_local_candidate_v4(candidate_path)
    assert candidate["fitting_authorized"] is False


def test_review_id_uses_canonical_json():
    document = _review()
    expected = document.pop("review_id")
    assert hashlib.sha256(canonical_json_bytes(document)).hexdigest() == expected


def test_private_reader_rejects_public_mode_and_hardlinks(tmp_path):
    from sm_fs_ranking_candidate_v4_review import _read_regular

    path = tmp_path / "candidate.json"
    path.write_bytes(b"{}\n")
    os.chmod(path, 0o644)
    with pytest.raises(ProtocolError, match="0600"):
        _read_regular(path, "fixture", private=True)
    os.chmod(path, 0o600)
    link = tmp_path / "candidate-link.json"
    os.link(path, link)
    with pytest.raises(ProtocolError, match="hard link"):
        _read_regular(path, "fixture", private=True)
