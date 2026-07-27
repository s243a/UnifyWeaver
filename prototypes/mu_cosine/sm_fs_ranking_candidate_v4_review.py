#!/usr/bin/env python3
"""Authenticate the independent fail-closed review of SM-FS ranking candidate v4.

The optional local-candidate check independently binds the immutable candidate, plan, projections,
code, and still-blocked preregistrations. This module can only report rejection; it cannot
authorize an amendment, fitting, held scoring, reserve access, or checkpoint release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from routed_policy import canonical_json_bytes, strict_json_loads
from sm_fs_protocols import ProtocolError, ROOT, _require
from sm_fs_ranking_candidate_v3_review import (
    EXPECTED_REVIEW_ID as PRIOR_REVIEW_ID,
    load_and_verify_candidate_v3_review,
)


REVIEW_PATH = ROOT / "SM_FS_LINEAGE_RANKING_CANDIDATE_V4_REVIEW.json"
REVIEW_DOCUMENT_NAME = "REVIEW_sm_fs_ranking_candidate_v4_lock.md"
REVIEW_SCHEMA = "unifyweaver.sm-fs-ranking-candidate-review.v4"
CANDIDATE_SCHEMA = "unifyweaver.sm-fs-ranking-execution-lock.candidate.v4"
PLAN_SCHEMA = "unifyweaver.sm-fs-ranking-training-plan.v4"
EXPECTED_REVIEW_ID = "3c08f4165054360c92a43c19a21d5cbf613a49ef4c30f9b0913b9cb9e712e090"

RANK_DIR = Path("~/mu_data/sm_fs_ranking_v1").expanduser()
RUN_DIR = Path("~/mu_data/sm_fs_ranking_run_v4").expanduser()

EXPECTED_PARENT_CONTRACTS = {
    "prior_candidate_review_id": PRIOR_REVIEW_ID,
    "ranking_prereg_id": (
        "0235521270c565a413d69feaa01a6e101fb533261942451a43ddf963e409aca2"
    ),
    "ranking_prereg_sha256": (
        "2cefb96f96a1bfb76ed0dbe54dfd8b84e9da79f4d96fe609ade1178a5df68729"
    ),
    "ranking_protocol_sha256": (
        "25bac71a67b317700f6a5123af238f01e9b57225bf4deaf71a6470486bd194f3"
    ),
    "retention_prereg_id": (
        "f3e1d123e6f81191489689d7f7e0fd121e25ff891ada330fe668b8b3238497c1"
    ),
    "retention_prereg_sha256": (
        "b40feb04d94dc9e1da05bcb8e2c74204357492649fb6fbd45c7dba0de3f2fc7a"
    ),
    "retention_protocol_sha256": (
        "5cbe305ed3eb0c2ecbb78b6f032e326c5c0c72bf415d218c538bbe3abdacd3d8"
    ),
}

EXPECTED_CODE = {
    "fine_tune_channel_heads.py": (
        "4e156eebb6675d950a6b5b41bdeaa87e2d2df2780b6fd5706f5c0eb54d15395c"
    ),
    "judge_cards.py": "71c8733a8dcdf782dc4620a65c99846e2575e2400032336fca599aaa1537844d",
    "mu_attention.py": "778664fa6e2b95100bff526bab5751b3907887a2e77ea412897dcffffb788443",
    "sm_fs_bootstrap.py": (
        "76c6be512542210b72a971d0b15dc1f236dd174044f1fba0fe93ee145b203f00"
    ),
    "sm_fs_ranking_chain.py": (
        "39079cecc8a76e4aaaf236d569205eae05f5c6c51c57c20c67b672778c1bea09"
    ),
    "sm_fs_ranking_construct.py": (
        "40ed4212f218817166df305b78280966a6af9561d5e3a12440d40af9a2ff67bc"
    ),
    "sm_fs_ranking_lock_verify.py": (
        "6262c60b861a25f9a9a240c2b4abaf2b63e9809b2f614f7f5c240d7af27fcf79"
    ),
    "sm_fs_ranking_pipeline.py": (
        "3ed0029ba36b262a283493918c741326b1ac0e99729b7944e10ca23823e7721e"
    ),
    "sm_fs_sampler.py": "790225b25f9ccc20b2b3d8e988441e5158f8388b12460dc63f74b72b18339267",
}

EXPECTED_INITIALIZED = {
    "3997001": "a3bf4c0588cc3e4cf1ad335b66440c113e7ee11fd116bd7307a3cee57447098e",
    "3997002": "fb353e693951819683793641464155e144caad73c553039fc64f1c6e253ad796",
    "3997003": "f42bdea0071a64dca0dad5312178127906b96215ba6890f6a37ad19d87ffdd5a",
}

EXPECTED_CANDIDATE = {
    "path": "~/mu_data/sm_fs_ranking_run_v4/candidate_lock_v4.json",
    "schema": CANDIDATE_SCHEMA,
    "sha256": "b5a2a7e7034a937d8241e437d48536592c579bca13dcd7af59b0d9bc4e21f4fa",
    "git_commit": "fe87d08c8c96b5b0b886dc3154dc18d14efd8250",
    "ranking_bundle_manifest_sha256": (
        "e01e1e48b5464bd315cff3c982e035f390cab8ba2b4c3ee60322dac65bf35894"
    ),
    "title_table_sha256": (
        "78182f1559123bbbb8ac05a47bb2a32f9651225bee2c574688718a32acf8ba20"
    ),
    "training_plan_sha256": (
        "1cdb659bd3baa291af1847de764f46148318df98f95ea4b286270868f8a486d1"
    ),
    "initialized_checkpoints": EXPECTED_INITIALIZED,
    "code_sha256": EXPECTED_CODE,
    "fitting_authorized": False,
    "static_hashes_match": True,
    "supplied_candidate_verifier_accepts": True,
    "complete_authorization_chain_enforced": False,
    "model_derived_score_evidence_enforced": False,
}

EXPECTED_PROJECTIONS = {
    "0": {
        "rows": 103392,
        "held_queries": 73,
        "held_query_overlap": 0,
        "sha256": "3bd7feb3742895d72866d5ba5dc90427db34ad973ec95fa43fe5f878f664b539",
    },
    "1": {
        "rows": 103751,
        "held_queries": 72,
        "held_query_overlap": 0,
        "sha256": "1699ff88e2d6e3bfe3f7622e01bd5c6f1aed841c9dc603189d3a9752791d6f20",
    },
    "2": {
        "rows": 103751,
        "held_queries": 72,
        "held_query_overlap": 0,
        "sha256": "9c605c9abaea5142c78d551d355abeedfa23026a19e5246610f7c32c1e3084cc",
    },
    "3": {
        "rows": 103751,
        "held_queries": 72,
        "held_query_overlap": 0,
        "sha256": "eb9ea3a4633d2289a8279af856578f7eb9e00a011889d038bf8388a9cfd0e0b5",
    },
    "4": {
        "rows": 103751,
        "held_queries": 72,
        "held_query_overlap": 0,
        "sha256": "420fbb7772d9f173001fc3c371ea06ce60ea2af7f98ef70893f37ab93e102891",
    },
}

EXPECTED_BLOCKERS = [
    "caller-authored-score-vectors-can-pass-primary-decision",
    "preregistration-amendment-and-retention-cascade-unrestricted",
    "review-and-verification-authority-caller-forgeable",
    "registered-secondary-control-and-reporting-contract-incomplete",
    "verified-byte-handoff-and-decision-bearing-integration-tests-incomplete",
]

EXPECTED_AUTHORIZATION = {
    "candidate_accepted": False,
    "step_4_prereg_amendment_authorized": False,
    "retention_cascade_authorized": False,
    "model_fitting_authorized": False,
    "held_fold_scoring_authorized": False,
    "reserve_access_authorized": False,
    "checkpoint_release_authorized": False,
}

EXPECTED_REPLACEMENT_SEQUENCE = [
    "use-one-exact-schema-and-fixed-path-with-schema-specific-review-verification",
    "anchor-review-acceptance-to-user-merged-or-cryptographically-countersigned-authority",
    "bind-candidate-commit-sha-plan-report-protocols-and-explicit-amendment-contract",
    "recover-starting-preregistration-blobs-from-candidate-commit-and-verify-their-hashes",
    "structurally-diff-ranking-amendment-and-enforce-only-enumerated-json-pointer-values",
    "rederive-retention-id-link-and-enforce-exact-still-blocked-narrow-cascade",
    "emit-final-lock-and-verification-receipt-through-reviewed-private-no-replace-transactions",
    "authenticate-fitted-checkpoint-provenance-before-held-data-access",
    "reproduce-scores-from-authenticated-checkpoints-and-frozen-evaluator-inputs-before-decision",
    "carry-candidate-bound-source-bytes-or-hashes-through-every-held-data-loader",
    "complete-controls-binary-zero-sensitivity-secondary-diagnostics-and-whole-population-report",
    "add-end-to-end-chain-fit-evaluate-decide-and-adversarial-evidence-tests",
    "generate-fresh-plan-and-candidate-v5-in-a-new-no-replace-namespace",
    "independently-review-fresh-candidate-before-any-amendment-or-fit",
]

EXPECTED_NAMESPACE = {
    "candidate_lock_v4.json",
    "titles.json",
    "training_plan.json",
    *(f"fold{fold}/train_projection.jsonl" for fold in range(5)),
}


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
            _require(stat.S_IMODE(metadata.st_mode) == 0o600, f"{description} mode must be 0600")
            _require(metadata.st_uid == os.geteuid(), f"{description} owner differs")
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
        value = strict_json_loads(payload, source=str(path))
    except ValueError as exc:
        raise ProtocolError(f"cannot parse {description}: {exc}") from exc
    _require(isinstance(value, dict), f"{description} must be a JSON object")
    return value


def _sha(path: Path, description: str, *, private: bool = False) -> str:
    return hashlib.sha256(_read_regular(path, description, private=private)).hexdigest()


def load_and_verify_candidate_v4_review(
    path: str | Path = REVIEW_PATH,
) -> dict[str, Any]:
    """Verify the committed rejection authority and its preserved scientific contract."""
    path = Path(path)
    _require(path.name == REVIEW_PATH.name, "unexpected candidate-v4-review filename")
    document = _load_json(path, "candidate v4 review")
    _require(set(document) == {
        "schema", "review_document", "review_document_sha256", "review_status",
        "reviewed_before", "parent_contracts", "candidate", "grounding",
        "adversarial_reproductions", "blocking_findings", "authorization",
        "replacement_sequence", "frozen_later_amendment_requirements",
        "retention_cascade", "review_id",
    }, "candidate v4 review top-level fields drifted")
    _require(document.get("schema") == REVIEW_SCHEMA, "unsupported candidate-v4-review schema")
    _require(
        document.get("review_document") == REVIEW_DOCUMENT_NAME,
        "candidate-v4-review document path drifted",
    )
    document_path = path.with_name(REVIEW_DOCUMENT_NAME)
    _require(
        document.get("review_document_sha256")
        == _sha(document_path, "candidate v4 review document"),
        "candidate-v4-review document hash differs",
    )
    _require(
        document.get("review_status")
        == "authentic-executable-static-artifact-request-changes-before-step-4",
        "candidate-v4-review status drifted",
    )
    _require(
        document.get("reviewed_before")
        == [
            "decision-bearing-ranking-optimizer-step",
            "held-fold-score",
            "reserve-access",
            "step-4-prereg-amendment",
        ],
        "candidate-v4-review timing boundary drifted",
    )
    prior = load_and_verify_candidate_v3_review()
    _require(prior["review_id"] == PRIOR_REVIEW_ID, "candidate v3 parent review drifted")
    _require(
        document.get("parent_contracts") == EXPECTED_PARENT_CONTRACTS,
        "candidate-v4-review parent contracts drifted",
    )
    _require(document.get("candidate") == EXPECTED_CANDIDATE, "candidate v4 identity drifted")

    grounding = document.get("grounding")
    _require(isinstance(grounding, dict), "candidate v4 grounding missing")
    _require(
        {key: grounding.get(key) for key in (
            "queries", "candidates", "pairs", "lineage_blocks", "fold_query_counts",
            "fold_assignment_sha256", "jobs", "fold_projections",
        )}
        == {
            "queries": 361,
            "candidates": 359,
            "pairs": 129599,
            "lineage_blocks": 82,
            "fold_query_counts": [73, 72, 72, 72, 72],
            "fold_assignment_sha256": (
                "b5439b1acdfb5020ecb7fb4fad437fd183242408d09056e1ed7ec90d33335f37"
            ),
            "jobs": 30,
            "fold_projections": EXPECTED_PROJECTIONS,
        },
        "candidate v4 population grounding drifted",
    )
    expected_grounding_flags = {
        "checkpoint_loader_passes_all_seeds": True,
        "exact_trainable_tensor_count": 18,
        "exact_trainable_parameter_count": 1195782,
        "synthetic_cpu_optimizer_step_passes": True,
        "synthetic_gpu_optimizer_step_passes": True,
        "all_training_sampler_known_answers_match": True,
        "frozen_bootstrap_implemented_exactly": True,
        "nonfinite_catalog_scores_fail_closed": True,
        "decision_recomputes_rank_from_scores": True,
        "decision_authenticates_scores_from_checkpoint": False,
        "fabricated_scores_can_pass_primary_gate": True,
        "arbitrary_preregistration_amendments_can_pass": True,
        "exact_review_schema_and_authority_enforced": False,
        "final_lock_emitter_present": False,
        "verification_receipt_emitter_present": False,
        "complete_secondary_and_control_report_enforced": False,
        "decision_bearing_end_to_end_tests_present": False,
    }
    _require(
        {key: grounding.get(key) for key in expected_grounding_flags}
        == expected_grounding_flags,
        "candidate v4 enforcement grounding drifted",
    )
    reproduction = document.get("adversarial_reproductions", {})
    _require(
        reproduction.get("fabricated_scores")
        == {
            "checkpoint_files": 0,
            "jobs": 30,
            "lineage_blocks": 82,
            "query_count": 82,
            "bootstrap_resamples": 9999,
            "sampler_attempts": 819918,
            "delta_mrr": 0.9878048780487787,
            "ci95": [0.9878048780487787, 0.9878048780487787],
            "bootstrap_mean": 0.9878048780485253,
            "positive_only_seed_mrr": 0.01219512195121952,
            "graded_negative_seed_mrr": 1.0,
            "passed_exploratory_gate": True,
            "authorizes": "new-reserve-preregistration-only",
        },
        "fabricated-score reproduction drifted",
    )
    _require(
        reproduction.get("preregistration_rewrite_accepted") is True
        and reproduction.get("caller_authored_review_accepted") is True
        and reproduction.get("caller_authored_verification_receipt_accepted") is True,
        "authorization-chain reproduction drifted",
    )
    _require(document.get("blocking_findings") == EXPECTED_BLOCKERS,
             "candidate v4 blockers drifted")
    _require(document.get("authorization") == EXPECTED_AUTHORIZATION,
             "candidate v4 authorization boundary drifted")
    _require(document.get("replacement_sequence") == EXPECTED_REPLACEMENT_SEQUENCE,
             "candidate v4 replacement sequence drifted")

    frozen = document.get("frozen_later_amendment_requirements", {})
    primary = frozen.get("primary", {})
    bootstrap = frozen.get("bootstrap", {})
    _require(
        frozen.get("ranking_estimand_unchanged") is True
        and (frozen.get("queries"), frozen.get("candidates"), frozen.get("folds"),
             frozen.get("lineage_blocks")) == (361, 359, 5, 82)
        and frozen.get("reserve_rows_untouched") == 1481
        and frozen.get("ranking_seeds") == [3997001, 3997002, 3997003]
        and frozen.get("track_t_seeds") == [3998101, 3998102, 3998103],
        "frozen candidate-v4 population/seed contract drifted",
    )
    _require(
        primary
        == {
            "metric": "exact-destination-mrr",
            "contrast": "graded-negative-minus-positive-only",
            "minimum_point_gain": 0.01,
            "interval_lower_strictly_greater_than": 0.0,
            "tie_rule": "ascending-frozen-catalog-column",
            "secondary_rescue_authorized": False,
            "passing_authorizes": "new-reserve-preregistration-only",
        },
        "frozen candidate-v4 primary decision drifted",
    )
    _require(
        bootstrap.get("resamples") == 9999
        and bootstrap.get("seed") == 3997999
        and bootstrap.get("draws_per_replicate") == 82
        and bootstrap.get("lower_order_zero_based") == 249
        and bootstrap.get("upper_order_zero_based") == 9749
        and bootstrap.get("same_multiplicities_both_arms") is True
        and bootstrap.get("query_weighted_mean_preserved") is True,
        "frozen candidate-v4 bootstrap contract drifted",
    )
    cascade = document.get("retention_cascade", {})
    _require(
        cascade.get("only_allowed_source_changes")
        == [
            "ranking_protocol_sha256",
            "ranking_prereg_id",
            "negative_bundle_status",
            "negative_bundle_sha256",
        ]
        and cascade.get("execution_authorized") is False
        and cascade.get("reserve_rows_authorized") is False
        and cascade.get("quarantined_rows_authorized") is False
        and cascade.get("privacy_and_checkpoint_release_gates_unchanged") is True,
        "candidate v4 retention cascade drifted",
    )
    _require(document.get("review_id") == _review_id(document),
             "candidate-v4-review ID does not rederive")
    _require(document["review_id"] == EXPECTED_REVIEW_ID,
             "candidate-v4-review ID differs from frozen independent review")
    return document


def verify_local_candidate_v4(
    candidate_path: str | Path = RUN_DIR / "candidate_lock_v4.json",
) -> dict[str, Any]:
    """Optionally verify the private pre-fit namespace against the committed rejection."""
    review = load_and_verify_candidate_v4_review()
    candidate_path = Path(candidate_path)
    _require(candidate_path.name == "candidate_lock_v4.json", "unexpected v4 candidate filename")
    candidate_bytes = _read_regular(candidate_path, "candidate v4 lock", private=True)
    _require(
        hashlib.sha256(candidate_bytes).hexdigest() == review["candidate"]["sha256"],
        "local candidate v4 SHA-256 differs from reviewed bytes",
    )
    candidate = strict_json_loads(candidate_bytes, source=str(candidate_path))
    _require(isinstance(candidate, dict), "candidate v4 lock must be an object")
    _require(candidate_bytes == canonical_json_bytes(candidate), "candidate v4 JSON is noncanonical")
    for key in (
        "schema", "git_commit", "ranking_bundle_manifest_sha256", "title_table_sha256",
        "training_plan_sha256", "initialized_checkpoints", "code_sha256",
        "fitting_authorized",
    ):
        _require(
            candidate.get(key) == review["candidate"].get(key),
            f"local candidate v4 {key} differs from review",
        )
    _require(candidate.get("fitting_authorized") is False,
             "local candidate v4 unexpectedly authorizes fitting")

    plan_path = candidate_path.with_name("training_plan.json")
    plan_bytes = _read_regular(plan_path, "candidate v4 training plan", private=True)
    _require(hashlib.sha256(plan_bytes).hexdigest() == EXPECTED_CANDIDATE["training_plan_sha256"],
             "candidate v4 plan hash differs")
    plan = strict_json_loads(plan_bytes, source=str(plan_path))
    _require(plan_bytes == canonical_json_bytes(plan), "candidate v4 plan is noncanonical")
    _require(plan.get("schema") == PLAN_SCHEMA and plan.get("fitting_authorized") is False,
             "candidate v4 plan schema/authorization drifted")
    expected_jobs = {
        (fold, arm, seed)
        for fold in range(5)
        for arm in ("positive_only", "graded_negative")
        for seed in (3997001, 3997002, 3997003)
    }
    jobs = {(job.get("fold"), job.get("arm"), job.get("seed")) for job in plan.get("jobs", [])}
    _require(plan.get("job_count") == 30 and len(plan.get("jobs", [])) == 30
             and jobs == expected_jobs, "candidate v4 job matrix drifted")
    _require(plan.get("control_arms") == ["control_warm_start", "control_e5_cosine"],
             "candidate v4 registered controls drifted")
    for fold, expected in EXPECTED_PROJECTIONS.items():
        projection = candidate_path.parent / f"fold{fold}" / "train_projection.jsonl"
        payload = _read_regular(projection, f"fold {fold} projection", private=True)
        _require(hashlib.sha256(payload).hexdigest() == expected["sha256"],
                 f"fold {fold} projection hash differs")
        _require(len(payload.splitlines()) == expected["rows"],
                 f"fold {fold} projection row count differs")

    observed = {
        path.relative_to(candidate_path.parent).as_posix()
        for path in candidate_path.parent.rglob("*")
        if path.is_file()
    }
    _require(observed == EXPECTED_NAMESPACE,
             f"candidate v4 namespace contains unauthorized/missing outputs: {sorted(observed)}")

    root = ROOT
    for name, expected in EXPECTED_CODE.items():
        _require(_sha(root / name, f"candidate v4 code {name}") == expected,
                 f"candidate v4 code {name} differs")
    _require(_sha(candidate_path.with_name("titles.json"), "candidate v4 titles", private=True)
             == EXPECTED_CANDIDATE["title_table_sha256"], "candidate v4 title table differs")

    for filename, hash_key, auth_key in (
        ("SM_FS_LINEAGE_RANKING_PREREG.json", "ranking_prereg_sha256",
         "model_fitting_authorized"),
        ("SM_FS_RETENTION_TRANSFER_PREREG.json", "retention_prereg_sha256",
         "execution_authorized"),
    ):
        prereg_path = ROOT / filename
        prereg_bytes = _read_regular(prereg_path, filename)
        _require(
            hashlib.sha256(prereg_bytes).hexdigest()
            == EXPECTED_PARENT_CONTRACTS[hash_key],
            f"live {filename} bytes changed before authorization",
        )
        prereg = strict_json_loads(prereg_bytes, source=str(prereg_path))
        _require(prereg.get(auth_key) is False, f"live {filename} is no longer blocked")
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", default=str(REVIEW_PATH))
    parser.add_argument("--verify-local-candidate", action="store_true")
    parser.add_argument("--candidate", default=str(RUN_DIR / "candidate_lock_v4.json"))
    args = parser.parse_args(argv)
    document = load_and_verify_candidate_v4_review(args.review)
    if args.verify_local_candidate:
        verify_local_candidate_v4(args.candidate)
    print(
        json.dumps(
            {
                "review_id": document["review_id"],
                "candidate_sha256": document["candidate"]["sha256"],
                "candidate_accepted": False,
                "model_fitting_authorized": False,
                "held_fold_scoring_authorized": False,
                "reserve_access_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
