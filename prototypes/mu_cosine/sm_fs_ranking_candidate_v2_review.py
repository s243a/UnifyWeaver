#!/usr/bin/env python3
"""Verify the independent fail-closed review of SM-FS ranking candidate v2.

This verifier authenticates the exact static candidate and the review decision.  It deliberately
cannot authorize a preregistration amendment, fitting, held scoring, or reserve access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

from routed_policy import canonical_json_bytes, strict_json_loads
from sm_fs_protocols import ProtocolError, ROOT, _full_sha, _require
from sm_fs_ranking_candidate_review import (
    EXPECTED_BOOTSTRAP,
    EXPECTED_PRIMARY,
    EXPECTED_RETENTION_CASCADE,
    EXPECTED_REVIEW_ID as PRIOR_REVIEW_ID,
    load_and_verify_candidate_review,
)


REVIEW_PATH = ROOT / "SM_FS_LINEAGE_RANKING_CANDIDATE_V2_REVIEW.json"
REVIEW_DOCUMENT_NAME = "REVIEW_sm_fs_ranking_candidate_v2_lock.md"
REVIEW_SCHEMA = "unifyweaver.sm-fs-ranking-candidate-review.v2"
CANDIDATE_SCHEMA = "unifyweaver.sm-fs-ranking-execution-lock.candidate.v2"
EXPECTED_REVIEW_ID = "916cad2f7b3a9d99fa4b8549cd4423b4849c3f205075cd0734e3672a9044aa2d"

RANK_DIR = Path("~/mu_data/sm_fs_ranking_v1").expanduser()
RUN_DIR = Path("~/mu_data/sm_fs_ranking_run_v2").expanduser()

EXPECTED_PARENT_CONTRACTS = {
    "prior_candidate_review_id": PRIOR_REVIEW_ID,
    "ranking_prereg_id": (
        "0235521270c565a413d69feaa01a6e101fb533261942451a43ddf963e409aca2"
    ),
    "ranking_protocol_sha256": (
        "25bac71a67b317700f6a5123af238f01e9b57225bf4deaf71a6470486bd194f3"
    ),
    "retention_prereg_id": (
        "f3e1d123e6f81191489689d7f7e0fd121e25ff891ada330fe668b8b3238497c1"
    ),
    "retention_protocol_sha256": (
        "5cbe305ed3eb0c2ecbb78b6f032e326c5c0c72bf415d218c538bbe3abdacd3d8"
    ),
}

EXPECTED_CANDIDATE = {
    "sha256": "a490cff99178ca0187c1e11a880c6618e00594bc4b9e7e8757171440159440e3",
    "git_commit": "64ee948ed94d502038e5c46421eab5844c9dd899",
    "ranking_bundle_manifest_sha256": (
        "e01e1e48b5464bd315cff3c982e035f390cab8ba2b4c3ee60322dac65bf35894"
    ),
    "title_table_sha256": (
        "78182f1559123bbbb8ac05a47bb2a32f9651225bee2c574688718a32acf8ba20"
    ),
}

EXPECTED_INITIALIZED = {
    "3997001": "a3bf4c0588cc3e4cf1ad335b66440c113e7ee11fd116bd7307a3cee57447098e",
    "3997002": "fb353e693951819683793641464155e144caad73c553039fc64f1c6e253ad796",
    "3997003": "f42bdea0071a64dca0dad5312178127906b96215ba6890f6a37ad19d87ffdd5a",
}

EXPECTED_CODE = {
    "fine_tune_channel_heads.py": (
        "4e156eebb6675d950a6b5b41bdeaa87e2d2df2780b6fd5706f5c0eb54d15395c"
    ),
    "judge_cards.py": "71c8733a8dcdf782dc4620a65c99846e2575e2400032336fca599aaa1537844d",
    "mu_attention.py": "778664fa6e2b95100bff526bab5751b3907887a2e77ea412897dcffffb788443",
    "sm_fs_protocols.py": (
        "a007663a9339ab371ec7093dfbaacafbd75c779c2149d6e63a26730814a903e4"
    ),
    "sm_fs_ranking_construct.py": (
        "40ed4212f218817166df305b78280966a6af9561d5e3a12440d40af9a2ff67bc"
    ),
    "sm_fs_ranking_lock_verify.py": (
        "1170f39707f4d6343b68628bb5f7934199f2e942679009387a091e89853031b1"
    ),
    "sm_fs_ranking_pipeline.py": (
        "8f81059cdbbb9300d97e8a0bb64bcc79a79a454391917a415e7634eb3850b36a"
    ),
}

EXPECTED_BLOCKERS = {
    "countersigned-checkpoint-config-incompatible-with-fit-and-evaluate",
    "preregistration-id-cycle-remains-and-fresh-v2-plan-absent",
    "final-lock-and-independent-receipt-not-candidate-bound",
    "projection-fit-evaluation-and-decision-receipts-self-authenticate",
    "claimed-environment-tokenizer-reference-augmentation-and-trainable-bindings-not-enforced",
    "frozen-bootstrap-and-nonfinite-ranking-fail-closed-contract-not-implemented",
    "transaction-rollback-directory-durability-and-private-mode-contract-incomplete",
    "decision-bearing-fit-evaluate-finalize-and-bootstrap-tests-absent",
}

EXPECTED_REPLACEMENT_SEQUENCE = [
    "extract-id-free-sampler-and-remove-hard-coded-prereg-ids-from-candidate-bound-code",
    "fix-checkpoint-construction-and-smoke-all-three-initializations-through-optimizer",
    "freeze-and-independently-verify-all-five-train-only-projections-before-candidate",
    "emit-fresh-plan-binding-30-jobs-projections-code-exact-trainables-and-inference",
    "implement-candidate-derived-final-lock-and-independent-review-receipt-verifier",
    "chain-and-verify-fit-evaluation-and-decision-receipts-before-held-access",
    "implement-frozen-sha256-bootstrap-nearest-rank-endpoints-and-finite-score-gates",
    "complete-private-transaction-rollback-directory-fsync-mode-and-byte-handoff",
    "add-real-load-one-step-evaluate-decide-known-answer-and-adversarial-tests",
    "generate-fresh-plan-and-candidate-in-new-no-replace-namespace",
    "independently-review-fresh-candidate",
    "only-then-amend-ranking-prereg-and-cascade-retention-still-blocked",
]

EXPECTED_GROUNDING = {
    "queries": 361,
    "candidates": 359,
    "pairs": 129599,
    "lineage_blocks": 82,
    "fold_query_counts": [73, 72, 72, 72, 72],
    "fold_assignment_sha256": (
        "b5439b1acdfb5020ecb7fb4fad437fd183242408d09056e1ed7ec90d33335f37"
    ),
    "fold0_projection_rows": 103392,
    "fold0_held_queries_excluded": 73,
    "fold0_projection_held_query_overlap": 0,
    "fold0_projection_meta_sha256": (
        "18a418d0abb1ab23c8199e1d6c89227acebdc798e3467e5cb7e31adebeca8695"
    ),
    "fold0_projection_sha256": (
        "3bd7feb3742895d72866d5ba5dc90427db34ad973ec95fa43fe5f878f664b539"
    ),
    "countersigned_checkpoint_loads_in_pipeline": False,
    "optimizer_reachable": False,
    "fresh_training_plan_present": False,
    "final_lock_emitter_present": False,
    "verification_receipt_emitter_present": False,
    "frozen_bootstrap_implemented": False,
    "nonfinite_scores_fail_closed": False,
}

FULL_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _review_id(document: Mapping[str, Any]) -> str:
    core = dict(document)
    core.pop("review_id", None)
    return hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def _read_regular(path: Path, description: str, *, private: bool = False) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ProtocolError(f"cannot open {description}: {exc}") from exc
    try:
        metadata = os.fstat(fd)
        _require(stat.S_ISREG(metadata.st_mode), f"{description} must be a regular file")
        _require(metadata.st_nlink == 1, f"{description} must have exactly one hard link")
        if private:
            _require(
                stat.S_IMODE(metadata.st_mode) == 0o600,
                f"{description} mode must be 0600",
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _load_json(path: Path, description: str, *, private: bool = False) -> dict[str, Any]:
    payload = _read_regular(path, description, private=private)
    try:
        document = strict_json_loads(payload, source=str(path))
    except ValueError as exc:
        raise ProtocolError(f"cannot parse {description}: {exc}") from exc
    _require(isinstance(document, dict), f"{description} must be a JSON object")
    return document


def _sha(path: Path, description: str, *, private: bool = False) -> str:
    return hashlib.sha256(_read_regular(path, description, private=private)).hexdigest()


def load_and_verify_candidate_v2_review(
    path: str | Path = REVIEW_PATH,
) -> dict[str, Any]:
    path = Path(path)
    _require(path.name == REVIEW_PATH.name, "unexpected candidate-v2-review filename")
    document = _load_json(path, "candidate v2 review")
    _require(document.get("schema") == REVIEW_SCHEMA, "unsupported candidate-v2-review schema")
    _require(
        document.get("review_document") == REVIEW_DOCUMENT_NAME,
        "candidate-v2-review document path drifted",
    )
    review_document = path.with_name(REVIEW_DOCUMENT_NAME)
    observed_document_sha = _sha(review_document, "candidate v2 review document")
    _require(
        document.get("review_document_sha256") == observed_document_sha,
        "candidate-v2-review document hash differs",
    )
    _require(document.get("review_id") == _review_id(document), "candidate-v2-review ID mismatch")
    _require(
        document.get("review_status")
        == "authentic-static-artifact-request-changes-before-step-4",
        "candidate-v2-review status drifted",
    )
    _require(
        document.get("reviewed_before")
        == [
            "ranking-optimizer-step",
            "held-fold-score",
            "reserve-access",
            "step-4-prereg-amendment",
        ],
        "candidate-v2-review timing boundary drifted",
    )

    prior = load_and_verify_candidate_review()
    _require(prior.get("review_id") == PRIOR_REVIEW_ID, "prior candidate review drifted")
    _require(
        document.get("parent_contracts") == EXPECTED_PARENT_CONTRACTS,
        "candidate-v2-review parent contracts drifted",
    )

    candidate = document.get("candidate", {})
    _require(candidate.get("schema") == CANDIDATE_SCHEMA, "candidate v2 schema drifted")
    for field, expected in EXPECTED_CANDIDATE.items():
        if field == "git_commit":
            _require(
                isinstance(candidate.get(field), str)
                and FULL_GIT_SHA_RE.fullmatch(candidate[field]) is not None,
                "candidate v2 git commit must be a full lowercase SHA",
            )
        else:
            _full_sha(candidate.get(field), f"candidate v2 {field}")
        _require(candidate.get(field) == expected, f"candidate v2 {field} drifted")
    _require(candidate.get("training_plan_sha256") is None, "candidate v2 acquired a plan")
    _require(
        candidate.get("initialized_checkpoints") == EXPECTED_INITIALIZED,
        "candidate v2 initialized checkpoints drifted",
    )
    _require(candidate.get("code_sha256") == EXPECTED_CODE, "candidate v2 code drifted")
    _require(candidate.get("fitting_authorized") is False, "candidate v2 authorized fitting")
    _require(candidate.get("static_hashes_match") is True, "candidate v2 grounding disappeared")
    _require(
        candidate.get("supplied_verifier_accepts") is True
        and candidate.get("complete_state_reproduction_enforced") is False,
        "candidate v2 verifier characterization drifted",
    )

    _require(document.get("grounding") == EXPECTED_GROUNDING, "candidate v2 grounding drifted")
    _require(
        set(document.get("blocking_findings", [])) == EXPECTED_BLOCKERS,
        "candidate v2 blockers drifted",
    )
    authorization = document.get("authorization", {})
    expected_false = {
        "step_4_prereg_amendment_authorized",
        "retention_cascade_authorized",
        "model_fitting_authorized",
        "held_fold_scoring_authorized",
        "reserve_access_authorized",
        "checkpoint_release_authorized",
    }
    _require(set(authorization) == expected_false, "candidate v2 authorization fields drifted")
    for field in expected_false:
        _require(authorization.get(field) is False, f"{field} became authorized")
    _require(
        document.get("replacement_sequence") == EXPECTED_REPLACEMENT_SEQUENCE,
        "candidate v2 replacement sequence drifted",
    )

    later = document.get("frozen_later_amendment_requirements", {})
    _require(later.get("ranking_estimand_unchanged") is True, "ranking estimand became mutable")
    _require(
        (
            later.get("queries"),
            later.get("candidates"),
            later.get("folds"),
            later.get("lineage_blocks"),
        )
        == (361, 359, 5, 82),
        "later ranking population drifted",
    )
    _require(later.get("primary") == EXPECTED_PRIMARY, "later primary decision drifted")
    _require(later.get("bootstrap") == EXPECTED_BOOTSTRAP, "later bootstrap contract drifted")
    _require(later.get("reserve_rows_untouched") == 1481, "reserve population drifted")
    _require(
        later.get("track_t_independent_regrowth") is True
        and later.get("track_t_seeds") == [3998101, 3998102, 3998103],
        "Track T independence drifted",
    )
    _require(
        document.get("retention_cascade") == EXPECTED_RETENTION_CASCADE,
        "later retention cascade drifted",
    )
    _require(
        document.get("review_id") == EXPECTED_REVIEW_ID,
        "frozen candidate-v2-review ID drifted",
    )
    return document


def verify_reviewed_candidate_v2(
    candidate_path: str | Path,
    review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the exact rejected candidate and its currently present static inputs."""

    review = dict(review or load_and_verify_candidate_v2_review())
    candidate_path = Path(candidate_path)
    payload = _read_regular(candidate_path, "reviewed candidate v2", private=True)
    _require(
        hashlib.sha256(payload).hexdigest() == review["candidate"]["sha256"],
        "reviewed candidate v2 hash differs",
    )
    try:
        candidate = strict_json_loads(payload, source=str(candidate_path))
    except ValueError as exc:
        raise ProtocolError(f"reviewed candidate v2 JSON is invalid: {exc}") from exc
    _require(isinstance(candidate, dict), "reviewed candidate v2 must be an object")
    _require(candidate.get("schema") == CANDIDATE_SCHEMA, "reviewed candidate v2 schema drifted")
    _require(
        candidate.get("git_commit") == review["candidate"]["git_commit"],
        "reviewed candidate v2 commit drifted",
    )
    _require(
        candidate.get("ranking_bundle_manifest_sha256")
        == review["candidate"]["ranking_bundle_manifest_sha256"],
        "reviewed candidate v2 bundle drifted",
    )
    _require(
        candidate.get("title_table_sha256") == review["candidate"]["title_table_sha256"],
        "reviewed candidate v2 title table drifted",
    )
    _require(
        candidate.get("initialized_checkpoints") == EXPECTED_INITIALIZED,
        "reviewed candidate v2 initialization bindings drifted",
    )
    _require(candidate.get("code_sha256") == EXPECTED_CODE, "reviewed candidate v2 code drifted")
    _require(candidate.get("fitting_authorized") is False, "reviewed candidate v2 authorized fitting")

    for filename, expected in EXPECTED_CODE.items():
        _require(
            _sha(ROOT / filename, f"candidate-bound code {filename}") == expected,
            f"candidate-bound code {filename} differs",
        )
    _require(
        _sha(RANK_DIR / "manifest.json", "ranking manifest", private=True)
        == EXPECTED_CANDIDATE["ranking_bundle_manifest_sha256"],
        "ranking manifest differs",
    )
    manifest = _load_json(RANK_DIR / "manifest.json", "ranking manifest", private=True)
    _require(
        manifest.get("counts", {}).get("queries") == 361
        and manifest.get("counts", {}).get("candidates") == 359
        and manifest.get("counts", {}).get("pairs") == 129599,
        "ranking manifest population differs",
    )
    _require(
        manifest.get("fold", {}).get("blocks") == 82
        and manifest.get("fold", {}).get("fold_map_counts") == [73, 72, 72, 72, 72]
        and manifest.get("fold", {}).get("assignment_sha256")
        == EXPECTED_GROUNDING["fold_assignment_sha256"],
        "ranking manifest fold contract differs",
    )
    output_payloads: dict[str, bytes] = {}
    for filename, expected in manifest.get("outputs", {}).items():
        payload = _read_regular(
            RANK_DIR / filename,
            f"ranking output {filename}",
            private=True,
        )
        _require(
            hashlib.sha256(payload).hexdigest() == expected,
            f"ranking output {filename} differs",
        )
        output_payloads[filename] = payload
    _require(
        _sha(RUN_DIR / "titles.json", "title table", private=True)
        == EXPECTED_CANDIDATE["title_table_sha256"],
        "title table differs",
    )
    for seed, expected in EXPECTED_INITIALIZED.items():
        _require(
            _sha(RANK_DIR / f"init_seed{seed}.pt", f"initialized checkpoint {seed}", private=True)
            == expected,
            f"initialized checkpoint {seed} differs",
        )
    projection_meta_path = RUN_DIR / "fold0" / "train_projection.meta.json"
    _require(
        _sha(projection_meta_path, "fold-0 projection metadata", private=True)
        == EXPECTED_GROUNDING["fold0_projection_meta_sha256"],
        "fold-0 projection metadata hash differs",
    )
    projection_meta = _load_json(
        projection_meta_path,
        "fold-0 projection metadata",
        private=True,
    )
    _require(
        projection_meta.get("schema") == "unifyweaver.sm-fs-ranking-train-projection.v1"
        and projection_meta.get("fold") == 0
        and projection_meta.get("rows") == EXPECTED_GROUNDING["fold0_projection_rows"]
        and projection_meta.get("held_queries_excluded")
        == EXPECTED_GROUNDING["fold0_held_queries_excluded"]
        and projection_meta.get("source_manifest_sha256")
        == EXPECTED_CANDIDATE["ranking_bundle_manifest_sha256"],
        "fold-0 projection metadata differs",
    )
    projection_payload = _read_regular(
        RUN_DIR / "fold0" / "train_projection.jsonl",
        "fold-0 projection",
        private=True,
    )
    _require(
        hashlib.sha256(projection_payload).hexdigest()
        == projection_meta.get("projection_sha256")
        == EXPECTED_GROUNDING["fold0_projection_sha256"],
        "fold-0 projection hash differs",
    )
    pairs = [
        json.loads(line)
        for line in output_payloads["pairs.jsonl"].decode("utf-8").splitlines()
    ]
    held_queries = {row["query"] for row in pairs if row["fold"] == 0}
    projection_rows = [
        json.loads(line) for line in projection_payload.decode("utf-8").splitlines()
    ]
    _require(
        len(projection_rows) == EXPECTED_GROUNDING["fold0_projection_rows"],
        "fold-0 projection row count differs",
    )
    _require(
        not ({row["query"] for row in projection_rows} & held_queries),
        "fold-0 projection contains held queries",
    )
    _require(not (RUN_DIR / "training_plan.json").exists(), "rejected v2 namespace was modified")
    return {
        "candidate_sha256": EXPECTED_CANDIDATE["sha256"],
        "candidate_authentic": True,
        "step_4_prereg_amendment_authorized": False,
        "model_fitting_authorized": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate")
    args = parser.parse_args(argv)
    review = load_and_verify_candidate_v2_review()
    print(f"ranking-candidate-v2-review\t{review['review_id']}")
    print("review-record\tauthenticated")
    print("step-4-amendment\tblocked")
    print("model-fitting\tblocked")
    if args.candidate:
        verify_reviewed_candidate_v2(args.candidate, review)
        print("local-candidate\tstatic-artifact-authentic-verified-rejected")


if __name__ == "__main__":
    main()
