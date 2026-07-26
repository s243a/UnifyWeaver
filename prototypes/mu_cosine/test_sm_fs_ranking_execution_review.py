"""Enforcement tests for the #4010 ranking execution-lock rigor review."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from routed_policy import canonical_json_bytes
from sm_fs_protocols import ProtocolError
from sm_fs_ranking_execution_review import (
    EXPECTED_BLOCKERS,
    EXPECTED_FINAL_SEQUENCE,
    EXPECTED_INITIALIZED,
    EXPECTED_NEXT_ARTIFACTS,
    REVIEW_PATH,
    _review_id,
    load_and_verify_execution_review,
    verify_reviewed_local_artifacts,
    verify_reviewed_execution_proposal,
)


def _document() -> dict:
    return json.loads(REVIEW_PATH.read_text(encoding="utf-8"))


def _write_review_fixture(tmp_path: Path, document: dict) -> Path:
    source_review = REVIEW_PATH.with_name(document["review_document"])
    review_document = tmp_path / document["review_document"]
    review_document.write_bytes(source_review.read_bytes())
    document["review_document_sha256"] = hashlib.sha256(review_document.read_bytes()).hexdigest()
    document["review_id"] = _review_id(document)
    path = tmp_path / REVIEW_PATH.name
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def test_committed_review_countersigns_warm_start_but_keeps_fitting_blocked():
    review = load_and_verify_execution_review()
    assert review["warm_start_countersign"]["accepted"] is True
    assert review["warm_start_countersign"]["checkpoint"] == "model_prod_namecond_full.pt"
    assert review["model_fitting_authorized"] is False
    assert set(review["fitting_blockers"]) == EXPECTED_BLOCKERS
    assert set(review["required_next_artifacts"]) == EXPECTED_NEXT_ARTIFACTS
    assert review["final_authorization_sequence"] == EXPECTED_FINAL_SEQUENCE


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda d: d["ranking_bundle"].update(reserve_overlap=1),
            "reserve overlap",
        ),
        (
            lambda d: d["warm_start_countersign"].update(accepted=False),
            "not countersigned",
        ),
        (
            lambda d: d["warm_start_countersign"].update(
                checkpoint="model_pt_filing_lin.pt"
            ),
            "warm-start checkpoint",
        ),
        (
            lambda d: d["warm_start_countersign"]["initialized_checkpoints"][
                "3997001"
            ].update(artifact_sha256="0" * 64),
            "initialized checkpoint bindings",
        ),
        (
            lambda d: d["warm_start_countersign"].update(
                pearltrees_trained_decision_arm_authorized=True
            ),
            "Pearltrees-trained decision arm",
        ),
        (
            lambda d: d["warm_start_countersign"].update(
                track_t_must_regrow_independently=False
            ),
            "Track T independence",
        ),
        (
            lambda d: d["execution_proposal"].update(
                recorded_commit_contains_emitter=True
            ),
            "stale proposal commit",
        ),
        (
            lambda d: d["execution_proposal"].update(actual_ranking_trainer_bound=True),
            "unreviewed ranking trainer",
        ),
        (
            lambda d: d["title_table"].update(mode_gate_passed=True),
            "privacy gate",
        ),
        (
            lambda d: d.update(
                fitting_blockers=d["fitting_blockers"][:-1]
            ),
            "fitting blockers",
        ),
        (
            lambda d: d.update(model_fitting_authorized=True),
            "model fitting became authorized",
        ),
        (
            lambda d: d.update(
                final_authorization_sequence=d["final_authorization_sequence"][:-1]
            ),
            "final authorization sequence",
        ),
        (
            lambda d: d.update(reserve_access_authorized=True),
            "reserve access became authorized",
        ),
    ],
)
def test_resealed_review_mutations_fail(tmp_path, mutation, message):
    document = _document()
    mutation(document)
    path = _write_review_fixture(tmp_path, document)
    with pytest.raises(ProtocolError, match=message):
        load_and_verify_execution_review(path)


def test_review_document_tamper_fails(tmp_path):
    path = _write_review_fixture(tmp_path, _document())
    document_path = path.with_name(_document()["review_document"])
    document_path.write_bytes(document_path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ProtocolError, match="document hash"):
        load_and_verify_execution_review(path)


def test_hard_bound_review_id_rejects_resealed_unenumerated_change(tmp_path):
    document = _document()
    document["reviewed_training_fields"]["optimizer"]["lr"] = 0.0004
    path = _write_review_fixture(tmp_path, document)
    with pytest.raises(ProtocolError, match="reviewed optimizer proposal"):
        load_and_verify_execution_review(path)


def _proposal() -> dict:
    return {
        "schema": "unifyweaver.sm-fs-ranking-execution-lock.v1",
        "status": "PROPOSED — no fitting",
        "prereg_id": "0235521270c565a413d69feaa01a6e101fb533261942451a43ddf963e409aca2",
        "ranking_bundle_manifest_sha256": (
            "e01e1e48b5464bd315cff3c982e035f390cab8ba2b4c3ee60322dac65bf35894"
        ),
        "warm_start": {
            "sha256": "9bb60915e9bcf8d4a89293c538284fd5ac631d9729f355917e128db7353bc8ef"
        },
        "initialized_checkpoints": {
            seed: {"sha256": record["artifact_sha256"]}
            for seed, record in EXPECTED_INITIALIZED.items()
        },
        "tokenizer_text_table": {
            "texts_sha256": (
                "60c8b3bfb7a3739b1e057a407603ec1d9138d5e94f17c5621f28413618763391"
            ),
            "e5_cache_sha256": (
                "bb9342a06cc9c62eedd664bb88c76833829f35e14e464beac0661feba81ed23f"
            ),
        },
        "trainable_parameters": {"tensor_count": 18, "param_count": 1195782},
    }


def _write_proposal(tmp_path: Path, proposal: dict) -> tuple[Path, str]:
    path = tmp_path / "execution_lock.json"
    path.write_bytes(canonical_json_bytes(proposal))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_reviewed_proposal_verifier_checks_bound_fields(tmp_path):
    proposal = _proposal()
    path, sha = _write_proposal(tmp_path, proposal)
    review = _document()
    review["execution_proposal"]["artifact_sha256"] = sha
    observed = verify_reviewed_execution_proposal(path, review)
    assert observed["status"].startswith("PROPOSED")


def test_reviewed_proposal_verifier_rejects_seed_drift(tmp_path):
    proposal = _proposal()
    proposal["initialized_checkpoints"]["3997001"]["sha256"] = "0" * 64
    path, sha = _write_proposal(tmp_path, proposal)
    review = _document()
    review["execution_proposal"]["artifact_sha256"] = sha
    with pytest.raises(ProtocolError, match="3997001"):
        verify_reviewed_execution_proposal(path, review)


def test_review_id_uses_canonical_json():
    document = _document()
    expected = document.pop("review_id")
    assert hashlib.sha256(canonical_json_bytes(document)).hexdigest() == expected


def _write_local_artifact_fixture(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "ranking"
    root.mkdir(mode=0o700)
    os.chmod(root, 0o700)
    review = _document()

    pairs = b'{"pair":1}\n'
    folds = b"map_id\tfold\nm1\t0\n"
    cache = b"private-title-cache"
    (root / "pairs.jsonl").write_bytes(pairs)
    (root / "fold_assignment.tsv").write_bytes(folds)
    (root / "ranking_text_e5.pt").write_bytes(cache)
    review["ranking_bundle"]["pairs_sha256"] = hashlib.sha256(pairs).hexdigest()
    review["ranking_bundle"]["fold_file_sha256"] = hashlib.sha256(folds).hexdigest()
    review["title_table"]["e5_cache_sha256"] = hashlib.sha256(cache).hexdigest()

    for seed, record in review["warm_start_countersign"]["initialized_checkpoints"].items():
        payload = f"checkpoint-{seed}".encode()
        filename = f"init_seed{seed}.pt"
        (root / filename).write_bytes(payload)
        record["artifact_sha256"] = hashlib.sha256(payload).hexdigest()

    manifest = {
        "schema": review["ranking_bundle"]["schema"],
        "outputs": {
            "fold_assignment.tsv": review["ranking_bundle"]["fold_file_sha256"],
            "pairs.jsonl": review["ranking_bundle"]["pairs_sha256"],
        },
        "counts": review["ranking_bundle"]["counts"],
        "model_fitting_authorized": False,
    }
    manifest_payload = canonical_json_bytes(manifest)
    (root / "manifest.json").write_bytes(manifest_payload)
    review["ranking_bundle"]["manifest_sha256"] = hashlib.sha256(manifest_payload).hexdigest()

    proposal = {
        "schema": review["execution_proposal"]["schema"],
        "status": "PROPOSED — no fitting",
        "prereg_id": review["parent"]["ranking_prereg_id"],
        "ranking_bundle_manifest_sha256": review["ranking_bundle"]["manifest_sha256"],
        "warm_start": {"sha256": review["warm_start_countersign"]["checkpoint_sha256"]},
        "initialized_checkpoints": {
            seed: {"sha256": record["artifact_sha256"]}
            for seed, record in review["warm_start_countersign"][
                "initialized_checkpoints"
            ].items()
        },
        "tokenizer_text_table": {
            "texts_sha256": review["title_table"]["texts_sha256"],
            "e5_cache_sha256": review["title_table"]["e5_cache_sha256"],
        },
        "trainable_parameters": {
            "tensor_count": review["reviewed_training_fields"]["trainable_tensor_count"],
            "param_count": review["reviewed_training_fields"]["trainable_parameter_count"],
        },
    }
    proposal_payload = canonical_json_bytes(proposal)
    (root / "execution_lock.json").write_bytes(proposal_payload)
    review["execution_proposal"]["artifact_sha256"] = hashlib.sha256(proposal_payload).hexdigest()

    for path in root.iterdir():
        os.chmod(path, 0o600)
    os.chmod(root / "ranking_text_e5.pt", 0o644)
    return root, review


def test_local_artifact_verifier_grounds_review_but_never_authorizes(tmp_path):
    root, review = _write_local_artifact_fixture(tmp_path)
    receipt = verify_reviewed_local_artifacts(root, review)
    assert receipt["privacy_gate_passed"] is False
    assert receipt["model_fitting_authorized"] is False
    assert receipt["artifact_modes"]["ranking_text_e5.pt"] == "0644"


def test_local_artifact_verifier_rejects_hard_link(tmp_path):
    root, review = _write_local_artifact_fixture(tmp_path)
    os.link(root / "pairs.jsonl", root / "pairs-alias.jsonl")
    with pytest.raises(ProtocolError, match="exactly one hard link"):
        verify_reviewed_local_artifacts(root, review)


def test_local_artifact_verifier_rejects_mode_drift(tmp_path):
    root, review = _write_local_artifact_fixture(tmp_path)
    os.chmod(root / "ranking_text_e5.pt", 0o600)
    with pytest.raises(ProtocolError, match="mode differs"):
        verify_reviewed_local_artifacts(root, review)
