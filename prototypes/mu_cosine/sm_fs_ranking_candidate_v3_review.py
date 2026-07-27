#!/usr/bin/env python3
"""Authenticate the independent fail-closed review of SM-FS ranking candidate v3.

The optional local-candidate check rederives the exact static candidate, plan, and all five
train-only projections.  This verifier can only report rejection; it cannot authorize amendment,
fitting, held scoring, reserve access, or checkpoint release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from routed_policy import canonical_json_bytes, strict_json_loads
from sm_fs_protocols import ProtocolError, ROOT, _require
from sm_fs_ranking_candidate_review import (
    EXPECTED_BOOTSTRAP,
    EXPECTED_PRIMARY,
    EXPECTED_RETENTION_CASCADE,
)
from sm_fs_ranking_candidate_v2_review import (
    EXPECTED_REVIEW_ID as PRIOR_REVIEW_ID,
    load_and_verify_candidate_v2_review,
)


REVIEW_PATH = ROOT / "SM_FS_LINEAGE_RANKING_CANDIDATE_V3_REVIEW.json"
REVIEW_DOCUMENT_NAME = "REVIEW_sm_fs_ranking_candidate_v3_lock.md"
REVIEW_SCHEMA = "unifyweaver.sm-fs-ranking-candidate-review.v3"
CANDIDATE_SCHEMA = "unifyweaver.sm-fs-ranking-execution-lock.candidate.v3"
PLAN_SCHEMA = "unifyweaver.sm-fs-ranking-training-plan.v3"
EXPECTED_REVIEW_ID = "421fbab584df2c02eb34d11d4d4a8ee3a536ef06ab38f38ba8eb585b68da71f6"

RANK_DIR = Path("~/mu_data/sm_fs_ranking_v1").expanduser()
RUN_DIR = Path("~/mu_data/sm_fs_ranking_run_v3").expanduser()

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
    "sha256": "33628b5da5f3bed7e6f7afb0f21a633decac27f348bc256bd52079dd4f47efcf",
    "git_commit": "db5dea709dff77f39a1b772c3bb9ea690b41e077",
    "ranking_bundle_manifest_sha256": (
        "e01e1e48b5464bd315cff3c982e035f390cab8ba2b4c3ee60322dac65bf35894"
    ),
    "pairs_sha256": "a0f3e30ce091567516db3dde2cdf6025ba44f526a7588fbd6d01125d448c9b26",
    "fold_file_sha256": (
        "a03e2ef7d584f58adc1b4f6b1abb6d042c7d0cfbff1d7ca18b5d132f81703c8e"
    ),
    "title_table_sha256": (
        "78182f1559123bbbb8ac05a47bb2a32f9651225bee2c574688718a32acf8ba20"
    ),
    "training_plan_sha256": (
        "ff81a8fda9f2cea9c21c21e0d27341661d42046368925fd04e4c3c11f6e2856d"
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
    "sm_fs_bootstrap.py": (
        "76c6be512542210b72a971d0b15dc1f236dd174044f1fba0fe93ee145b203f00"
    ),
    "sm_fs_ranking_construct.py": (
        "40ed4212f218817166df305b78280966a6af9561d5e3a12440d40af9a2ff67bc"
    ),
    "sm_fs_ranking_lock_verify.py": (
        "1b12faf543c847a952cc5a8d6ab82c3541606a002c24b40f417b743b8c8a9f99"
    ),
    "sm_fs_ranking_pipeline.py": (
        "b90bdadfa37bde3f710d9cd3fcda16ec7b92064a841c1e7289c8d2cf9ccc3b18"
    ),
    "sm_fs_sampler.py": "790225b25f9ccc20b2b3d8e988441e5158f8388b12460dc63f74b72b18339267",
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

EXPECTED_BLOCKERS = {
    "accepted-review-final-lock-and-post-review-state-chain-caller-forgeable",
    "fit-evaluation-and-primary-decision-receipts-self-authenticate",
    "registered-diagnostics-runtime-and-seed-reporting-contract-incomplete",
    "transaction-rollback-durability-and-verified-byte-handoff-incomplete",
    "finalization-evaluation-decision-and-adversarial-integration-tests-absent",
}

EXPECTED_REPLACEMENT_SEQUENCE = [
    "emit-final-lock-and-verification-receipt-through-explicit-private-no-replace-transactions",
    "authenticate-canonical-review-schema-id-and-git-tracked-bytes-at-reviewed-commit",
    "compare-every-final-binding-to-reviewed-candidate-with-machine-checked-amendment-whitelist",
    "bind-and-verify-ranking-and-retention-parent-documents-amendment-and-blocked-cascade",
    "bind-fit-receipts-exactly-to-candidate-final-review-plan-job-runtime-and-checkpoint",
    "require-authenticated-fit-chain-before-held-outcome-access",
    "bind-evaluation-receipts-and-recompute-destination-rank-and-rr-from-finite-scores",
    "validate-complete-30-job-query-fold-catalog-runtime-and-diagnostic-population-at-decision",
    "implement-weighted-diagnostics-secondary-contract-and-three-whole-population-seed-results",
    "complete-rollback-fsync-errors-and-descriptor-relative-or-verified-byte-handoff",
    "add-end-to-end-finalize-evaluate-decide-forgery-drift-and-rollback-adversarial-tests",
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
    "jobs": 30,
    "fold_projections": EXPECTED_PROJECTIONS,
    "checkpoint_loader_passes_all_seeds": True,
    "exact_trainable_tensor_count": 18,
    "exact_trainable_parameter_count": 1195782,
    "synthetic_cpu_optimizer_step_passes": True,
    "synthetic_gpu_optimizer_step_passes": True,
    "all_training_sampler_known_answers_match": True,
    "frozen_bootstrap_implemented_exactly": True,
    "nonfinite_catalog_scores_fail_closed": True,
    "final_lock_emitter_present": False,
    "verification_receipt_emitter_present": False,
    "review_artifact_git_identity_enforced": False,
    "final_state_equals_reviewed_candidate_enforced": False,
    "fit_receipt_chain_enforced": False,
    "evaluation_receipt_chain_enforced": False,
    "decision_recomputes_rank_from_scores": False,
    "fabricated_receipts_can_pass_primary_gate": True,
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


def load_and_verify_candidate_v3_review(
    path: str | Path = REVIEW_PATH,
) -> dict[str, Any]:
    path = Path(path)
    _require(path.name == REVIEW_PATH.name, "unexpected candidate-v3-review filename")
    document = _load_json(path, "candidate v3 review")
    _require(document.get("schema") == REVIEW_SCHEMA, "unsupported candidate-v3-review schema")
    _require(
        document.get("review_document") == REVIEW_DOCUMENT_NAME,
        "candidate-v3-review document path drifted",
    )
    observed_document_sha = _sha(path.with_name(REVIEW_DOCUMENT_NAME), "candidate v3 review document")
    _require(
        document.get("review_document_sha256") == observed_document_sha,
        "candidate-v3-review document hash differs",
    )
    _require(document.get("review_id") == _review_id(document), "candidate-v3-review ID mismatch")
    _require(
        document.get("review_status")
        == "authentic-executable-static-artifact-request-changes-before-step-4",
        "candidate-v3-review status drifted",
    )
    _require(
        document.get("reviewed_before")
        == [
            "decision-bearing-ranking-optimizer-step",
            "held-fold-score",
            "reserve-access",
            "step-4-prereg-amendment",
        ],
        "candidate-v3-review timing boundary drifted",
    )

    prior = load_and_verify_candidate_v2_review()
    _require(prior.get("review_id") == PRIOR_REVIEW_ID, "candidate-v2 parent review drifted")
    _require(
        document.get("parent_contracts") == EXPECTED_PARENT_CONTRACTS,
        "candidate-v3-review parent contracts drifted",
    )

    candidate = document.get("candidate", {})
    _require(candidate.get("schema") == CANDIDATE_SCHEMA, "candidate v3 schema drifted")
    for field, expected in EXPECTED_CANDIDATE.items():
        _require(candidate.get(field) == expected, f"candidate v3 {field} drifted")
    _require(
        candidate.get("initialized_checkpoints") == EXPECTED_INITIALIZED,
        "candidate v3 initialized checkpoints drifted",
    )
    _require(candidate.get("code_sha256") == EXPECTED_CODE, "candidate v3 code drifted")
    _require(candidate.get("fitting_authorized") is False, "candidate v3 authorized fitting")
    _require(candidate.get("static_hashes_match") is True, "candidate v3 grounding disappeared")
    _require(
        candidate.get("supplied_candidate_verifier_accepts") is True
        and candidate.get("complete_authorization_chain_enforced") is False,
        "candidate v3 verifier characterization drifted",
    )

    _require(document.get("grounding") == EXPECTED_GROUNDING, "candidate v3 grounding drifted")
    _require(
        set(document.get("blocking_findings", [])) == EXPECTED_BLOCKERS,
        "candidate v3 blockers drifted",
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
    _require(set(authorization) == expected_false, "candidate v3 authorization fields drifted")
    for field in expected_false:
        _require(authorization.get(field) is False, f"{field} became authorized")
    _require(
        document.get("replacement_sequence") == EXPECTED_REPLACEMENT_SEQUENCE,
        "candidate v3 replacement sequence drifted",
    )

    later = document.get("frozen_later_amendment_requirements", {})
    _require(later.get("ranking_estimand_unchanged") is True, "ranking estimand became mutable")
    _require(
        (later.get("queries"), later.get("candidates"), later.get("folds"), later.get("lineage_blocks"))
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
        "frozen candidate-v3-review ID drifted",
    )
    return document


def verify_reviewed_candidate_v3(
    candidate_path: str | Path,
    review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the exact rejected candidate and independently rederive its static plan."""

    review = dict(review or load_and_verify_candidate_v3_review())
    candidate_path = Path(candidate_path)
    payload = _read_regular(candidate_path, "reviewed candidate v3", private=True)
    _require(
        hashlib.sha256(payload).hexdigest() == EXPECTED_CANDIDATE["sha256"],
        "reviewed candidate v3 hash differs",
    )
    try:
        candidate = strict_json_loads(payload, source=str(candidate_path))
    except ValueError as exc:
        raise ProtocolError(f"reviewed candidate v3 JSON is invalid: {exc}") from exc
    _require(isinstance(candidate, dict), "reviewed candidate v3 must be an object")
    _require(candidate.get("schema") == CANDIDATE_SCHEMA, "reviewed candidate v3 schema drifted")
    _require(candidate.get("fitting_authorized") is False, "reviewed candidate v3 authorized fitting")
    for field in (
        "git_commit",
        "ranking_bundle_manifest_sha256",
        "title_table_sha256",
        "training_plan_sha256",
    ):
        _require(candidate.get(field) == EXPECTED_CANDIDATE[field], f"candidate v3 {field} differs")
    _require(candidate.get("initialized_checkpoints") == EXPECTED_INITIALIZED, "init bindings differ")
    _require(candidate.get("code_sha256") == EXPECTED_CODE, "candidate code bindings differ")
    inventory = candidate.get("trainable_names_shapes")
    _require(isinstance(inventory, list) and len(inventory) == 18, "trainable inventory differs")
    parameter_count = 0
    for item in inventory:
        _require(
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], list)
            and all(type(n) is int and n > 0 for n in item[1]),
            "malformed trainable inventory",
        )
        count = 1
        for size in item[1]:
            count *= size
        parameter_count += count
    _require(parameter_count == 1195782, "trainable parameter count differs")

    for filename, expected in EXPECTED_CODE.items():
        _require(_sha(ROOT / filename, f"candidate-bound code {filename}") == expected,
                 f"candidate-bound code {filename} differs")

    manifest_payload = _read_regular(RANK_DIR / "manifest.json", "ranking manifest", private=True)
    _require(hashlib.sha256(manifest_payload).hexdigest()
             == EXPECTED_CANDIDATE["ranking_bundle_manifest_sha256"],
             "ranking manifest differs")
    manifest = strict_json_loads(manifest_payload, source="ranking manifest")
    _require(
        manifest.get("counts", {}).get("queries") == 361
        and manifest.get("counts", {}).get("candidates") == 359
        and manifest.get("counts", {}).get("pairs") == 129599,
        "ranking population differs",
    )
    _require(
        manifest.get("fold", {}).get("blocks") == 82
        and manifest.get("fold", {}).get("fold_map_counts") == [73, 72, 72, 72, 72]
        and manifest.get("fold", {}).get("assignment_sha256")
        == EXPECTED_GROUNDING["fold_assignment_sha256"],
        "ranking fold contract differs",
    )
    pairs_payload = _read_regular(RANK_DIR / "pairs.jsonl", "ranking pairs", private=True)
    folds_payload = _read_regular(RANK_DIR / "fold_assignment.tsv", "fold assignment", private=True)
    _require(hashlib.sha256(pairs_payload).hexdigest() == EXPECTED_CANDIDATE["pairs_sha256"],
             "ranking pairs differ")
    _require(hashlib.sha256(folds_payload).hexdigest() == EXPECTED_CANDIDATE["fold_file_sha256"],
             "fold assignment differs")
    _require(
        manifest.get("outputs", {}).get("pairs.jsonl") == EXPECTED_CANDIDATE["pairs_sha256"]
        and manifest.get("outputs", {}).get("fold_assignment.tsv")
        == EXPECTED_CANDIDATE["fold_file_sha256"],
        "manifest output bindings differ",
    )
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

    plan_payload = _read_regular(RUN_DIR / "training_plan.json", "training plan", private=True)
    _require(hashlib.sha256(plan_payload).hexdigest()
             == EXPECTED_CANDIDATE["training_plan_sha256"], "training plan hash differs")
    plan = strict_json_loads(plan_payload, source="training plan")
    _require(plan.get("schema") == PLAN_SCHEMA and plan.get("fitting_authorized") is False,
             "training plan authorization/schema differs")
    expected_jobs = {
        (fold, arm, seed)
        for fold in range(5)
        for seed in (3997001, 3997002, 3997003)
        for arm in ("positive_only", "graded_negative")
    }
    observed_jobs = {
        (job.get("fold"), job.get("arm"), job.get("seed")) for job in plan.get("jobs", [])
    }
    _require(
        observed_jobs == expected_jobs
        and len(plan.get("jobs", [])) == len(observed_jobs) == plan.get("job_count") == 30,
        "training plan job matrix differs",
    )

    pairs = [strict_json_loads(line, source="ranking pair") for line in pairs_payload.splitlines()]
    _require(len(pairs) == 129599, "ranking pair count differs")
    for fold in range(5):
        held = {row["query"] for row in pairs if row["fold"] == fold}
        train_rows = [row for row in pairs if row["query"] not in held]
        projection_payload = b"".join(canonical_json_bytes(row) for row in train_rows)
        expected = EXPECTED_PROJECTIONS[str(fold)]
        _require(len(held) == expected["held_queries"], f"fold {fold} held population differs")
        _require(len(train_rows) == expected["rows"], f"fold {fold} train population differs")
        _require(hashlib.sha256(projection_payload).hexdigest() == expected["sha256"],
                 f"fold {fold} derived projection differs")
        disk = _read_regular(
            RUN_DIR / f"fold{fold}" / "train_projection.jsonl",
            f"fold {fold} projection",
            private=True,
        )
        _require(disk == projection_payload, f"fold {fold} projection bytes differ")
        _require(
            plan.get("projections", {}).get(str(fold), {}).get("projection_sha256")
            == expected["sha256"],
            f"fold {fold} plan projection differs",
        )
        _require(
            not ({row["query"] for row in train_rows} & held),
            f"fold {fold} projection contains held queries",
        )

    ranking_prereg = _load_json(ROOT / "SM_FS_LINEAGE_RANKING_PREREG.json", "ranking prereg")
    retention_prereg = _load_json(
        ROOT / "SM_FS_RETENTION_TRANSFER_PREREG.json", "retention prereg"
    )
    _require(ranking_prereg.get("model_fitting_authorized") is False,
             "live ranking prereg authorized fitting")
    _require(retention_prereg.get("execution_authorized") is False,
             "live retention prereg authorized execution")

    forbidden = list(RUN_DIR.glob("fold*/fit_*")) + list(RUN_DIR.glob("fold*/eval_*"))
    forbidden += [
        RUN_DIR / "decision.json",
        RUN_DIR / "final_lock_v2.json",
        RUN_DIR / "verification_receipt_v2.json",
    ]
    _require(not any(path.exists() for path in forbidden), "rejected v3 namespace has run outputs")
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
    review = load_and_verify_candidate_v3_review()
    print(f"ranking-candidate-v3-review\t{review['review_id']}")
    print("review-record\tauthenticated")
    print("step-4-amendment\tblocked")
    print("model-fitting\tblocked")
    if args.candidate:
        verify_reviewed_candidate_v3(args.candidate, review)
        print("local-candidate\tstatic-artifact-and-plan-authentic-verified-rejected")


if __name__ == "__main__":
    main()
