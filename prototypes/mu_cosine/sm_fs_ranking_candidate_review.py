#!/usr/bin/env python3
"""Verify the independent fail-safe review of the SM-FS ranking candidate lock.

The reviewed candidate is authentic but incomplete.  This module intentionally
cannot authorize a preregistration amendment or fitting.  A replacement
candidate needs a new review ID and a new review artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import stat
from pathlib import Path
from typing import Any, Mapping

from routed_policy import canonical_json_bytes, strict_json_loads
from sm_fs_protocols import (
    ProtocolError,
    ROOT,
    _full_sha,
    _require,
    load_and_verify_ranking,
    load_and_verify_transfer,
)


REVIEW_PATH = ROOT / "SM_FS_LINEAGE_RANKING_CANDIDATE_REVIEW.json"
REVIEW_DOCUMENT_NAME = "REVIEW_sm_fs_ranking_candidate_lock.md"
REVIEW_SCHEMA = "unifyweaver.sm-fs-ranking-candidate-review.v1"
CANDIDATE_SCHEMA = "unifyweaver.sm-fs-ranking-execution-lock.candidate.v1"
EXPECTED_REVIEW_ID = "68e036c59c210002c50eafcbf6333cdb43393383fe916cd1449942f9fcc584f1"

EXPECTED_CANDIDATE = {
    "sha256": "6c4801cf387f0f8af075a22fbca5a411b8b0a659af899e99ca219f670d09f881",
    "git_commit": "094de6345e3b7b88b36657fc8fc9c0d8ed340dcf",
    "training_plan_sha256": "01f34ab74478a9642a54e328fb4e2dec7f290c33fd95c695b7eb81ea03ea26df",
    "ranking_bundle_manifest_sha256": (
        "e01e1e48b5464bd315cff3c982e035f390cab8ba2b4c3ee60322dac65bf35894"
    ),
    "source_targets_sha256": (
        "c3b298d5335ee901111f4985bbf5f7c5feb017c503a8c81db60dba1b947ac051"
    ),
    "warm_start_sha256": (
        "9bb60915e9bcf8d4a89293c538284fd5ac631d9729f355917e128db7353bc8ef"
    ),
}

EXPECTED_INITIALIZED = {
    "3997001": "a3bf4c0588cc3e4cf1ad335b66440c113e7ee11fd116bd7307a3cee57447098e",
    "3997002": "fb353e693951819683793641464155e144caad73c553039fc64f1c6e253ad796",
    "3997003": "f42bdea0071a64dca0dad5312178127906b96215ba6890f6a37ad19d87ffdd5a",
}

EXPECTED_CODE = {
    "emit_ranking_execution_lock.py": (
        "7188304a01aea76fcc1d9051f39c9417bf764fdde3bcfb9c41d77e8e167aaafb"
    ),
    "fine_tune_channel_heads.py": (
        "4e156eebb6675d950a6b5b41bdeaa87e2d2df2780b6fd5706f5c0eb54d15395c"
    ),
    "fine_tune_pearltrees_filing.py": (
        "257b69aa5d6ce1939717511b9870605671b9d197e0e99777ba5767098a1efed6"
    ),
    "judge_cards.py": "71c8733a8dcdf782dc4620a65c99846e2575e2400032336fca599aaa1537844d",
    "mu_attention.py": "778664fa6e2b95100bff526bab5751b3907887a2e77ea412897dcffffb788443",
    "sm_fs_protocols.py": (
        "a007663a9339ab371ec7093dfbaacafbd75c779c2149d6e63a26730814a903e4"
    ),
    "sm_fs_ranking_construct.py": (
        "40ed4212f218817166df305b78280966a6af9561d5e3a12440d40af9a2ff67bc"
    ),
    "sm_fs_ranking_execution_review.py": (
        "f4912929cca2af90719f81eeef84a40d260c7d546b0e01dd8fe79125a56d522b"
    ),
    "sm_fs_ranking_train.py": (
        "c30cfe1bf4cc6bfc39474b33efe8ed483658d90bab9b9dfa2bb521fe1e65649e"
    ),
}

EXPECTED_ENVIRONMENT = {
    "cublas_workspace": ":4096:8",
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "deterministic_algorithms": True,
    "matmul_precision": "highest",
    "tf32_cudnn": False,
    "tf32_matmul": False,
    "threads": 4,
    "torch": "2.13.0+cu130",
}

EXPECTED_PARENT_CONTRACTS = {
    "ranking_prereg_id": "0235521270c565a413d69feaa01a6e101fb533261942451a43ddf963e409aca2",
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

EXPECTED_BLOCKERS = {
    "complete-fit-evaluate-decide-transaction-absent",
    "final-lock-gate-trusts-self-asserted-independent-verification",
    "held-outcomes-not-capability-separated-from-fitter",
    "required-training-tokenizer-optimizer-rng-evaluation-bindings-absent",
    "candidate-bound-execution-and-final-lock-verifier-absent",
    "crash-atomic-transaction-and-descriptor-bound-inputs-absent",
    "fail-closed-perimeter-and-test-coverage-incomplete",
}

EXPECTED_REPLACEMENT_SEQUENCE = [
    "land-cycle-free-prereg-id-and-id-free-sampler-refactor",
    "land-complete-still-gated-fit-evaluate-decide-transaction",
    "land-cycle-free-independent-final-lock-receipt-verifier",
    "land-train-only-capability-boundary-and-post-seal-held-evaluator",
    "bind-and-enforce-complete-training-tokenizer-rng-environment-and-output-contract",
    "install-inputs-and-outputs-through-private-crash-atomic-no-replace-transactions",
    "generate-fresh-plan-and-candidate-in-new-namespace-from-clean-landed-commit",
    "independently-review-fresh-candidate",
    "only-then-amend-ranking-prereg-and-cascade-retention-still-blocked",
]

EXPECTED_PRIMARY = {
    "metric": "exact-destination-mrr",
    "contrast": "graded-negative-minus-positive-only",
    "minimum_point_gain": 0.01,
    "interval_lower_strictly_greater_than": 0.0,
    "tie_rule": "ascending-frozen-catalog-column",
    "secondary_rescue_authorized": False,
    "passing_authorizes": "new-reserve-preregistration-only",
}

EXPECTED_BOOTSTRAP = {
    "seed_average_within_query_before_contrast": True,
    "unit": "frozen-adaptive-lineage-block",
    "observed_eligible_blocks": 82,
    "draws_per_replicate": 82,
    "resamples": 9999,
    "seed": 3997999,
    "same_multiplicities_both_arms": True,
    "query_weighted_mean_preserved": True,
    "replicate_formula": "sum_b(m_b*sum_q_in_b(d_q))/sum_b(m_b*n_b)",
    "sampler_schema": "unifyweaver.sm-fs-ranking-bootstrap-key.v1",
    "sampler_id": "sm-fs-ranking-bootstrap-v1",
    "sampler_serialization": "canonical-json-sorted-keys-compact-utf8-terminal-lf",
    "sampler_key_fields": [
        "draw",
        "replicate",
        "retry",
        "sampler_id",
        "schema",
        "seed",
    ],
    "sampler_index": "sha256-unsigned-big-endian-rejection-then-modulo-82",
    "lower_order_zero_based": 249,
    "upper_order_zero_based": 9749,
    "order_rule": "nearest-rank-ceil-p-times-resamples-one-based",
    "minimum_observed_blocks": 20,
    "minimum_unique_blocks_per_replicate": None,
}

EXPECTED_RETENTION_CASCADE = {
    "only_allowed_source_changes": [
        "ranking_protocol_sha256",
        "ranking_prereg_id",
        "negative_bundle_status",
        "negative_bundle_sha256",
    ],
    "execution_authorized": False,
    "null_ledger_status": "blocked-until-frozen-and-verified",
    "null_ledger_sha256": None,
    "behavior_panel_sha256": None,
    "fresh_target_cohort_stays_blocked": True,
    "reserve_rows_authorized": False,
    "quarantined_rows_authorized": False,
    "track_r_contract_unchanged": True,
    "track_t_contract_unchanged": True,
    "inference_and_bonferroni_unchanged": True,
    "privacy_and_checkpoint_release_gates_unchanged": True,
}

FULL_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _review_id(document: Mapping[str, Any]) -> str:
    core = dict(document)
    core.pop("review_id", None)
    return hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"{description} is missing")
    _require(not path.is_symlink(), f"{description} may not be a symlink")
    try:
        document = strict_json_loads(path.read_bytes(), source=str(path))
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"cannot load {description}: {exc}") from exc
    _require(isinstance(document, dict), f"{description} must be a JSON object")
    return document


def load_and_verify_candidate_review(
    path: str | Path = REVIEW_PATH,
) -> dict[str, Any]:
    """Verify the committed review and its fail-closed decision."""

    path = Path(path)
    _require(path.name == REVIEW_PATH.name, "unexpected candidate-review filename")
    document = _load_json(path, "candidate review")
    _require(document.get("schema") == REVIEW_SCHEMA, "unsupported candidate-review schema")
    _require(
        document.get("review_document") == REVIEW_DOCUMENT_NAME,
        "candidate-review document path drifted",
    )
    review_document = path.with_name(REVIEW_DOCUMENT_NAME)
    _require(review_document.is_file(), "candidate-review document is missing")
    _require(not review_document.is_symlink(), "candidate-review document may not be a symlink")
    observed_document_sha = hashlib.sha256(review_document.read_bytes()).hexdigest()
    _require(
        document.get("review_document_sha256") == observed_document_sha,
        "candidate-review document hash differs",
    )
    _require(document.get("review_id") == _review_id(document), "candidate-review ID mismatch")
    _require(
        document.get("review_status")
        == "authentic-preflight-request-changes-before-step-4",
        "candidate-review status drifted",
    )
    _require(
        document.get("reviewed_before")
        == [
            "ranking-optimizer-step",
            "held-fold-score",
            "reserve-access",
            "step-4-prereg-amendment",
        ],
        "candidate-review timing boundary drifted",
    )
    _require(
        document.get("parent_contracts") == EXPECTED_PARENT_CONTRACTS,
        "candidate-review parent contracts drifted",
    )
    ranking_protocol = ROOT / "PROTOCOL_sm_fs_lineage_ranking.md"
    retention_protocol = ROOT / "PROTOCOL_sm_fs_retention_transfer.md"
    _require(
        hashlib.sha256(ranking_protocol.read_bytes()).hexdigest()
        == EXPECTED_PARENT_CONTRACTS["ranking_protocol_sha256"],
        "live ranking protocol differs from reviewed parent",
    )
    _require(
        hashlib.sha256(retention_protocol.read_bytes()).hexdigest()
        == EXPECTED_PARENT_CONTRACTS["retention_protocol_sha256"],
        "live retention protocol differs from reviewed parent",
    )
    ranking_prereg = load_and_verify_ranking()
    retention_prereg = load_and_verify_transfer()
    _require(
        ranking_prereg.get("prereg_id")
        == EXPECTED_PARENT_CONTRACTS["ranking_prereg_id"],
        "live ranking preregistration differs from reviewed parent",
    )
    _require(
        retention_prereg.get("prereg_id")
        == EXPECTED_PARENT_CONTRACTS["retention_prereg_id"],
        "live retention preregistration differs from reviewed parent",
    )

    candidate = document.get("candidate", {})
    _require(candidate.get("schema") == CANDIDATE_SCHEMA, "candidate schema drifted")
    for field, expected in EXPECTED_CANDIDATE.items():
        if field == "git_commit":
            _require(
                isinstance(candidate.get(field), str)
                and FULL_GIT_SHA_RE.fullmatch(candidate[field]) is not None,
                "git_commit must be a full lowercase Git SHA",
            )
        else:
            _full_sha(candidate.get(field), field)
        _require(candidate.get(field) == expected, f"candidate {field} drifted")
    _require(
        candidate.get("initialized_checkpoints") == EXPECTED_INITIALIZED,
        "candidate initialized checkpoint bindings drifted",
    )
    _require(candidate.get("code_sha256") == EXPECTED_CODE, "candidate code bindings drifted")
    _require(
        candidate.get("observed_environment") == EXPECTED_ENVIRONMENT,
        "candidate environment receipt drifted",
    )
    _require(candidate.get("fitting_authorized") is False, "candidate authorized fitting")
    _require(candidate.get("byte_reproduced") is True, "candidate byte reproduction disappeared")
    _require(
        candidate.get("recorded_hashes_match") is True,
        "candidate grounding no longer passes",
    )
    _require(
        candidate.get("reviewed_artifact_modes_match") is True,
        "candidate mode grounding no longer passes",
    )

    grounding = document.get("grounding", {})
    _require(grounding.get("job_count") == 30, "candidate job count drifted")
    _require(grounding.get("folds") == 5, "candidate fold count drifted")
    _require(
        grounding.get("arms") == ["positive_only", "graded_negative"],
        "candidate arms drifted",
    )
    _require(
        grounding.get("seeds") == [3997001, 3997002, 3997003],
        "candidate seeds drifted",
    )
    _require(grounding.get("optimizer_reachable") is False, "optimizer became reachable")
    _require(grounding.get("current_fit_exit") == 3, "current fit exit drifted")
    _require(
        grounding.get("reviewed_files_regular_single_link_non_symlink") is True,
        "reviewed file-shape grounding drifted",
    )
    _require(
        grounding.get("run_directory_mode") == "0700"
        and grounding.get("ranking_directory_mode") == "0700"
        and grounding.get("reviewed_file_mode") == "0600",
        "reviewed privacy modes drifted",
    )

    _require(
        set(document.get("blocking_findings", [])) == EXPECTED_BLOCKERS,
        "candidate blockers drifted",
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
    _require(set(authorization) == expected_false, "candidate authorization fields drifted")
    for field in expected_false:
        _require(authorization.get(field) is False, f"{field} became authorized")
    _require(
        document.get("replacement_sequence") == EXPECTED_REPLACEMENT_SEQUENCE,
        "replacement sequence drifted",
    )

    later = document.get("frozen_later_amendment_requirements", {})
    _require(later.get("ranking_estimand_unchanged") is True, "ranking estimand became mutable")
    _require(
        (later.get("queries"), later.get("candidates"), later.get("folds"), later.get("lineage_blocks"))
        == (361, 359, 5, 82),
        "later ranking population drifted",
    )
    _require(
        later.get("fold_assignment_sha256")
        == "b5439b1acdfb5020ecb7fb4fad437fd183242408d09056e1ed7ec90d33335f37",
        "later fold assignment drifted",
    )
    _require(
        later.get("evidence_scope")
        == "map-near-lineage-blocked-catalog-transductive",
        "later evidence scope drifted",
    )
    _require(later.get("primary") == EXPECTED_PRIMARY, "later primary decision drifted")
    _require(later.get("reserve_rows_untouched") == 1481, "reserve population drifted")
    _require(
        later.get("ranking_seeds") == [3997001, 3997002, 3997003],
        "later ranking seeds drifted",
    )
    _require(
        later.get("track_t_independent_regrowth") is True
        and later.get("track_t_seeds") == [3998101, 3998102, 3998103],
        "Track T independence drifted",
    )
    _require(
        later.get("bootstrap") == EXPECTED_BOOTSTRAP,
        "later bootstrap contract drifted",
    )
    _require(
        document.get("retention_cascade") == EXPECTED_RETENTION_CASCADE,
        "later retention cascade drifted",
    )
    _require(
        document.get("review_id") == EXPECTED_REVIEW_ID,
        "frozen candidate-review ID drifted",
    )
    return document


def verify_reviewed_candidate(
    candidate_path: str | Path,
    review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the exact rejected candidate without upgrading its authority."""

    review = dict(review or load_and_verify_candidate_review())
    path = Path(candidate_path)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProtocolError(f"cannot stat reviewed candidate: {exc}") from exc
    _require(stat.S_ISREG(metadata.st_mode), "reviewed candidate must be a regular file")
    _require(not path.is_symlink(), "reviewed candidate may not be a symlink")
    _require(metadata.st_nlink == 1, "reviewed candidate must have exactly one hard link")
    _require(
        f"{stat.S_IMODE(metadata.st_mode):04o}" == "0600",
        "reviewed candidate mode must be 0600",
    )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProtocolError(f"cannot read reviewed candidate: {exc}") from exc
    _require(
        hashlib.sha256(payload).hexdigest() == review["candidate"]["sha256"],
        "reviewed candidate hash differs",
    )
    try:
        candidate = strict_json_loads(payload, source=str(path))
    except ValueError as exc:
        raise ProtocolError(f"reviewed candidate JSON is invalid: {exc}") from exc
    _require(isinstance(candidate, dict), "reviewed candidate must be a JSON object")
    _require(candidate.get("schema") == CANDIDATE_SCHEMA, "reviewed candidate schema drifted")
    _require(
        candidate.get("git_commit") == review["candidate"]["git_commit"],
        "reviewed candidate commit drifted",
    )
    _require(
        candidate.get("training_plan_sha256")
        == review["candidate"]["training_plan_sha256"],
        "reviewed candidate plan drifted",
    )
    _require(
        candidate.get("ranking_bundle_manifest_sha256")
        == review["candidate"]["ranking_bundle_manifest_sha256"],
        "reviewed candidate bundle drifted",
    )
    _require(
        candidate.get("source_targets_sha256")
        == review["candidate"]["source_targets_sha256"],
        "reviewed candidate source targets drifted",
    )
    _require(
        candidate.get("warm_start", {}).get("sha256")
        == review["candidate"]["warm_start_sha256"],
        "reviewed candidate warm start drifted",
    )
    initialized = candidate.get("initialized_checkpoints", {})
    _require(set(initialized) == set(EXPECTED_INITIALIZED), "reviewed candidate seeds drifted")
    for seed, expected in EXPECTED_INITIALIZED.items():
        _require(
            initialized.get(seed, {}).get("sha256") == expected
            and initialized.get(seed, {}).get("authoritative") is True,
            f"reviewed candidate initialized checkpoint {seed} drifted",
        )
    _require(candidate.get("code_sha256") == EXPECTED_CODE, "reviewed candidate code drifted")
    _require(
        candidate.get("observed_environment") == EXPECTED_ENVIRONMENT,
        "reviewed candidate environment drifted",
    )
    _require(candidate.get("fitting_authorized") is False, "reviewed candidate authorized fitting")
    _require(
        str(candidate.get("status", "")).startswith("CANDIDATE"),
        "reviewed candidate status drifted",
    )
    return {
        "candidate_sha256": review["candidate"]["sha256"],
        "candidate_authentic": True,
        "step_4_prereg_amendment_authorized": False,
        "model_fitting_authorized": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate")
    args = parser.parse_args(argv)
    review = load_and_verify_candidate_review()
    print(f"ranking-candidate-review\t{review['review_id']}")
    print("candidate\tauthentic")
    print("step-4-amendment\tblocked")
    print("model-fitting\tblocked")
    if args.candidate:
        verify_reviewed_candidate(args.candidate, review)
        print("local-candidate\tverified-rejected")


if __name__ == "__main__":
    main()
