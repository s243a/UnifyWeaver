#!/usr/bin/env python3
"""Verify the rigor review of the proposed SM-FS ranking execution lock.

This module deliberately does not authorize fitting.  It content-binds the
accepted pre-Pearltrees warm start and the exact proposal reviewed after PR
#4010, while enforcing the unresolved execution blockers.  The eventual final
training plan must produce a new preregistration amendment and a new ID.
"""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path
from typing import Any, Mapping

from routed_policy import canonical_json_bytes, strict_json_loads
from sm_fs_protocols import (
    EXPECTED_RANKING_PREREG_ID,
    ProtocolError,
    RANKING_PROTOCOL_NAME,
    ROOT,
    _full_sha,
    _require,
    load_and_verify_ranking,
)


REVIEW_PATH = ROOT / "SM_FS_LINEAGE_RANKING_EXECUTION_REVIEW.json"
REVIEW_DOCUMENT_NAME = "REVIEW_sm_fs_ranking_execution_lock.md"
REVIEW_SCHEMA = "unifyweaver.sm-fs-ranking-execution-review.v1"
EXECUTION_LOCK_SCHEMA = "unifyweaver.sm-fs-ranking-execution-lock.v1"
EXPECTED_REVIEW_ID = "cb377b6c6aabcfe572f7cf320f8ec32ba5211b85b1747c3334b69ca6144f4894"

EXPECTED_INITIALIZED = {
    "3997001": {
        "artifact_sha256": "a3bf4c0588cc3e4cf1ad335b66440c113e7ee11fd116bd7307a3cee57447098e",
        "canonical_state_sha256": "0d842ed6eeb8e69036605e50697aa2cb3a5f8955255deb7becfbfa1c8b3154ad",
    },
    "3997002": {
        "artifact_sha256": "fb353e693951819683793641464155e144caad73c553039fc64f1c6e253ad796",
        "canonical_state_sha256": "f3eae318f41f17332f5bdc589958dd09d80d411afab58cca0fe63bbb7c90cdd3",
    },
    "3997003": {
        "artifact_sha256": "f42bdea0071a64dca0dad5312178127906b96215ba6890f6a37ad19d87ffdd5a",
        "canonical_state_sha256": "e9cedab4e2df1405c929e9c42af5adc439c37117ba62ef5e2b43e29b40cac8ce",
    },
}

EXPECTED_BLOCKERS = {
    "no-content-bound-ranking-trainer-evaluator-or-complete-training-plan",
    "no-fold-arm-seed-job-ledger-or-held-fold-capability-boundary",
    "trainable-allowlist-optimizer-reference-and-augmentation-not-enforced",
    "tokenizer-structure-and-numpy-torch-rng-order-not-bound",
    "determinism-settings-recorded-as-requirements-not-observed-assertions",
    "execution-proposal-commit-predates-emitter",
    "growth-provenance-omits-transitive-code-and-card-cache-inputs",
    "title-cache-mode-0644-violates-0600",
    "artifacts-written-in-place-with-truncation-not-atomic-no-replace",
    "source-projection-reopened-without-full-descriptor-bound-verification",
}

EXPECTED_NEXT_ARTIFACTS = {
    "landed-ranking-trainer-coordinator-evaluator-and-verifier",
    "complete-content-bound-training-plan-and-job-ledger",
    "private-atomic-no-replace-run-input-transaction",
    "enforced-and-observed-deterministic-environment-receipt",
    "new-execution-lock-from-clean-landed-commit",
}

EXPECTED_FINAL_SEQUENCE = [
    "land-trainer-plan-and-candidate-lock-verifier",
    "generate-candidate-proposal-from-clean-landed-commit",
    "independently-review-candidate-proposal",
    "land-authorized-prereg-protocol-new-id-and-cycle-free-verifier",
    "regenerate-final-lock-from-exact-authorized-commit-and-new-prereg-id",
    "independently-verify-final-lock-against-landed-state",
    "permit-first-optimizer-step",
]


def _review_id(document: Mapping[str, Any]) -> str:
    core = dict(document)
    core.pop("review_id", None)
    return hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def _load_json_file(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"{description} is missing")
    _require(not path.is_symlink(), f"{description} may not be a symlink")
    try:
        document = strict_json_loads(path.read_bytes(), source=str(path))
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"cannot load {description}: {exc}") from exc
    _require(isinstance(document, dict), f"{description} must be a JSON object")
    return document


def load_and_verify_execution_review(
    path: str | Path = REVIEW_PATH,
) -> dict[str, Any]:
    path = Path(path)
    _require(path.name == REVIEW_PATH.name, "unexpected execution-review filename")
    document = _load_json_file(path, "execution review")
    _require(document.get("schema") == REVIEW_SCHEMA, "unsupported execution-review schema")
    _require(
        document.get("review_document") == REVIEW_DOCUMENT_NAME,
        "execution-review document path drifted",
    )
    review_document = path.with_name(REVIEW_DOCUMENT_NAME)
    _require(review_document.is_file(), "execution-review document is missing")
    _require(not review_document.is_symlink(), "execution-review document may not be a symlink")
    observed_document_sha = hashlib.sha256(review_document.read_bytes()).hexdigest()
    _require(
        document.get("review_document_sha256") == observed_document_sha,
        "execution-review document hash differs",
    )
    _require(
        document.get("review_id") == _review_id(document),
        "execution-review ID mismatch",
    )
    _require(
        document.get("review_status")
        == "warm-start-countersigned-execution-not-authorized",
        "execution-review status drifted",
    )
    _require(
        document.get("reviewed_before")
        == ["ranking-optimizer-step", "held-fold-score", "reserve-access"],
        "review timing boundary drifted",
    )

    ranking = load_and_verify_ranking()
    parent = document.get("parent", {})
    _require(
        parent.get("ranking_prereg_id")
        == ranking.get("prereg_id")
        == EXPECTED_RANKING_PREREG_ID,
        "parent ranking preregistration drifted",
    )
    observed_protocol_sha = hashlib.sha256((ROOT / RANKING_PROTOCOL_NAME).read_bytes()).hexdigest()
    _require(
        parent.get("ranking_protocol_sha256") == observed_protocol_sha,
        "parent ranking protocol hash drifted",
    )

    bundle = document.get("ranking_bundle", {})
    _require(
        bundle.get("schema") == "unifyweaver.sm-fs-lineage-ranking-bundle.v1",
        "ranking bundle schema drifted",
    )
    expected_bundle_hashes = {
        "manifest_sha256": "e01e1e48b5464bd315cff3c982e035f390cab8ba2b4c3ee60322dac65bf35894",
        "pairs_sha256": "a0f3e30ce091567516db3dde2cdf6025ba44f526a7588fbd6d01125d448c9b26",
        "fold_file_sha256": "a03e2ef7d584f58adc1b4f6b1abb6d042c7d0cfbff1d7ca18b5d132f81703c8e",
        "fold_assignment_sha256": "b5439b1acdfb5020ecb7fb4fad437fd183242408d09056e1ed7ec90d33335f37",
        "constructor_sha256": "40ed4212f218817166df305b78280966a6af9561d5e3a12440d40af9a2ff67bc",
    }
    for field, expected in expected_bundle_hashes.items():
        _full_sha(bundle.get(field), field)
        _require(bundle.get(field) == expected, f"{field} drifted")
    _require(
        bundle.get("constructor_commit")
        == "b8b969ac1a5e66c7ad864aea4e4be35a0f45113b",
        "constructor commit drifted",
    )
    _require(
        bundle.get("counts")
        == {
            "queries": 361,
            "candidates": 359,
            "pairs": 129599,
            "positives": 2792,
            "nonancestors": 126807,
            "hard": 1819,
            "medium": 7814,
            "easy": 117174,
        },
        "reviewed ranking counts drifted",
    )
    _require(bundle.get("reserve_overlap") == 0, "reserve overlap became nonzero")
    _require(bundle.get("cross_lineage_overlap") == 0, "cross-lineage overlap became nonzero")

    warm = document.get("warm_start_countersign", {})
    _require(warm.get("accepted") is True, "pre-Pearltrees warm start is not countersigned")
    _require(
        warm.get("checkpoint") == "model_prod_namecond_full.pt",
        "warm-start checkpoint drifted",
    )
    _require(
        warm.get("checkpoint_sha256")
        == "9bb60915e9bcf8d4a89293c538284fd5ac631d9729f355917e128db7353bc8ef",
        "warm-start bytes drifted",
    )
    _require(
        warm.get("role") == "common-paired-ranking-initialization-only",
        "warm-start role drifted",
    )
    _require(
        warm.get("growth_description")
        == "expand-current-judges-and-add-lineage-plus-lineage-rank-operators-refresh-name-cards-and-create-fresh-readouts",
        "growth description drifted",
    )
    _require(warm.get("seeds") == [3997001, 3997002, 3997003], "ranking seeds drifted")
    _require(
        warm.get("initialized_checkpoints") == EXPECTED_INITIALIZED,
        "initialized checkpoint bindings drifted",
    )
    _require(
        warm.get("canonical_state_hash_algorithm")
        == "sha256-sorted-state-keys-key-nul-dtype-nul-shape-nul-contiguous-cpu-numpy-c-order-bytes",
        "canonical state hash algorithm drifted",
    )
    _require(
        warm.get("canonical_config_sha256")
        == "01a1236ab7debd57b42818210e7e4fa2d711244ea67f408cd02472417374e98d",
        "initialized checkpoint configuration drifted",
    )
    _require(
        warm.get("canonical_config_hash_algorithm")
        == "sha256-sorted-key-compact-json-utf8-terminal-lf",
        "canonical configuration hash algorithm drifted",
    )
    _require(
        warm.get("reuse_rule")
        == "reload-one-seed-initialized-checkpoint-byte-identically-before-every-fold-and-both-arms",
        "paired initialization reuse rule drifted",
    )
    _require(
        warm.get("pearltrees_trained_decision_arm_authorized") is False,
        "Pearltrees-trained decision arm became authorized",
    )
    _require(warm.get("transfer_evidence") is False, "ranking warm start became transfer evidence")
    _require(
        warm.get("track_t_must_regrow_independently") is True,
        "Track T independence drifted",
    )
    _require(warm.get("track_t_seeds") == [3998101, 3998102, 3998103], "Track T seeds drifted")

    proposal = document.get("execution_proposal", {})
    _require(proposal.get("schema") == EXECUTION_LOCK_SCHEMA, "execution-lock schema drifted")
    _require(
        proposal.get("artifact_sha256")
        == "0752cc16256a5699924e150d665dff3a556d0f92c3739e652b9892de97d4e47e",
        "reviewed execution-lock bytes drifted",
    )
    _require(proposal.get("status") == "PROPOSED", "execution-lock status drifted")
    _require(
        proposal.get("recorded_git_commit")
        == "e18076a9bb07dd87f6b5854d7de0f9ce3e9d32a0",
        "recorded proposal commit drifted",
    )
    _require(
        proposal.get("recorded_commit_contains_emitter") is False,
        "stale proposal commit became accepted",
    )
    _require(
        proposal.get("emitter_sha256")
        == "97b49380dec4c041c25cd05a3efdf3f4a910f7975dca7899c3701e01de7899f2",
        "emitter bytes drifted",
    )
    _require(
        proposal.get("emitter_commit")
        == "f4a172a6ee2d76f05adeacdfb43db2a2caca7a4a",
        "emitter commit drifted",
    )
    _require(
        proposal.get("emitter_merge_commit")
        == "01da72ba31868d3383bdb172f2b8ce2279181928",
        "emitter merge commit drifted",
    )
    _require(
        proposal.get("training_plan_hash_is_emitter_only") is True,
        "proposal training-plan limitation disappeared",
    )
    _require(
        proposal.get("actual_ranking_trainer_bound") is False,
        "unreviewed ranking trainer became bound",
    )
    _require(
        proposal.get("actual_ranking_evaluator_bound") is False,
        "unreviewed ranking evaluator became bound",
    )

    title = document.get("title_table", {})
    _require(title.get("identities") == 720, "title-table identity count drifted")
    _require(title.get("query_identities") == 361, "title-table query count drifted")
    _require(title.get("candidate_identities") == 359, "title-table candidate count drifted")
    _require(title.get("identity_overlap") == 0, "title-table identity overlap drifted")
    expected_title_hashes = {
        "names_sha256": "9c81701c10e00215f1ff6e734bd1910dfe89ee18fa903959b4dce8c6226cde42",
        "texts_sha256": "60c8b3bfb7a3739b1e057a407603ec1d9138d5e94f17c5621f28413618763391",
        "e5_cache_sha256": "bb9342a06cc9c62eedd664bb88c76833829f35e14e464beac0661feba81ed23f",
    }
    for field, expected in expected_title_hashes.items():
        _full_sha(title.get(field), field)
        _require(title.get(field) == expected, f"{field} drifted")
    _require(
        title.get("e5_revision") == "ffb93f3bd4047442299a41ebb6fa998a38507c52",
        "title-table E5 revision drifted",
    )
    _require(
        title.get("embedding_rule") == "certified-map-title-and-candidate-leaf-title-only",
        "title embedding rule drifted",
    )
    _require(title.get("path_components_used_as_text") is False, "path text became authorized")
    _require(title.get("observed_mode") == "0644", "observed title-cache mode drifted")
    _require(title.get("required_mode") == "0600", "required title-cache mode drifted")
    _require(title.get("mode_gate_passed") is False, "0644 title cache passed the privacy gate")

    training = document.get("reviewed_training_fields", {})
    _require(
        training.get("optimizer")
        == {
            "class": "torch.optim.Adam",
            "lr": 0.0005,
            "betas": [0.9, 0.999],
            "eps": 1e-08,
            "weight_decay": 0,
        },
        "reviewed optimizer proposal drifted",
    )
    _require(training.get("trainable_tensor_count") == 18, "trainable tensor count drifted")
    _require(
        training.get("trainable_parameter_count") == 1195782,
        "trainable parameter count drifted",
    )
    _require(
        training.get("frozen_reference_rule")
        == "deepcopy-exact-initialized-model-before-optimizer-eval-requires-grad-false",
        "frozen reference rule drifted",
    )
    _require(training.get("anchor_rows_augmented") is False, "anchor augmentation became enabled")
    _require(
        training.get("sampler_module_sha256")
        == "a007663a9339ab371ec7093dfbaacafbd75c779c2149d6e63a26730814a903e4",
        "sampler module binding drifted",
    )

    _require(set(document.get("fitting_blockers", [])) == EXPECTED_BLOCKERS, "fitting blockers drifted")
    _require(
        set(document.get("required_next_artifacts", [])) == EXPECTED_NEXT_ARTIFACTS,
        "required next artifacts drifted",
    )
    _require(
        document.get("final_authorization_sequence") == EXPECTED_FINAL_SEQUENCE,
        "final authorization sequence drifted",
    )
    _require(document.get("model_fitting_authorized") is False, "model fitting became authorized")
    _require(document.get("reserve_access_authorized") is False, "reserve access became authorized")
    _require(
        document.get("checkpoint_release_authorized") is False,
        "checkpoint release became authorized",
    )
    _require(
        document.get("review_id") == EXPECTED_REVIEW_ID,
        "frozen execution-review ID drifted",
    )
    return document


def verify_reviewed_execution_proposal(
    proposal_path: str | Path,
    review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the exact reviewed local proposal without treating it as executable."""

    review = dict(review or load_and_verify_execution_review())
    path = Path(proposal_path)
    proposal = _load_json_file(path, "execution-lock proposal")
    observed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    _require(
        observed_sha == review["execution_proposal"]["artifact_sha256"],
        "execution-lock proposal hash differs from review",
    )
    _require(proposal.get("schema") == EXECUTION_LOCK_SCHEMA, "execution-lock schema drifted")
    _require(proposal.get("prereg_id") == EXPECTED_RANKING_PREREG_ID, "proposal parent prereg drifted")
    _require(
        proposal.get("ranking_bundle_manifest_sha256")
        == review["ranking_bundle"]["manifest_sha256"],
        "proposal ranking bundle drifted",
    )
    _require(
        proposal.get("warm_start", {}).get("sha256")
        == review["warm_start_countersign"]["checkpoint_sha256"],
        "proposal warm start drifted",
    )
    proposal_initial = proposal.get("initialized_checkpoints", {})
    _require(set(proposal_initial) == set(EXPECTED_INITIALIZED), "proposal seed set drifted")
    for seed, expected in EXPECTED_INITIALIZED.items():
        _require(
            proposal_initial.get(seed, {}).get("sha256") == expected["artifact_sha256"],
            f"proposal initialized checkpoint {seed} drifted",
        )
    _require(
        proposal.get("tokenizer_text_table", {}).get("texts_sha256")
        == review["title_table"]["texts_sha256"],
        "proposal title table drifted",
    )
    _require(
        proposal.get("tokenizer_text_table", {}).get("e5_cache_sha256")
        == review["title_table"]["e5_cache_sha256"],
        "proposal E5 cache drifted",
    )
    _require(
        proposal.get("trainable_parameters", {}).get("tensor_count") == 18,
        "proposal trainable tensor count drifted",
    )
    _require(
        proposal.get("trainable_parameters", {}).get("param_count") == 1195782,
        "proposal trainable parameter count drifted",
    )
    _require(
        proposal.get("status", "").startswith("PROPOSED"),
        "reviewed proposal no longer identifies itself as proposed",
    )
    return proposal


def _verified_regular_file(path: Path, description: str) -> tuple[bytes, str]:
    """Read one reviewed input after rejecting aliases and mutable link shapes."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProtocolError(f"cannot stat {description}: {exc}") from exc
    _require(stat.S_ISREG(metadata.st_mode), f"{description} must be a regular file")
    _require(not path.is_symlink(), f"{description} may not be a symlink")
    _require(metadata.st_nlink == 1, f"{description} must have exactly one hard link")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProtocolError(f"cannot read {description}: {exc}") from exc
    return payload, f"{stat.S_IMODE(metadata.st_mode):04o}"


def verify_reviewed_local_artifacts(
    bundle_dir: str | Path,
    review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Ground the reviewed proposal in its exact local files.

    This verifies the state that was reviewed, including the known ``0644``
    title-cache failure.  It therefore returns a privacy-gate result but can
    never authorize fitting.  A corrected input tree needs a new lock and a
    new review; silently changing this tree makes this review fail.
    """

    review = dict(review or load_and_verify_execution_review())
    root = Path(bundle_dir)
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise ProtocolError(f"cannot stat reviewed ranking bundle: {exc}") from exc
    _require(stat.S_ISDIR(root_metadata.st_mode), "reviewed ranking bundle must be a directory")
    _require(not root.is_symlink(), "reviewed ranking bundle may not be a symlink")
    observed_root_mode = f"{stat.S_IMODE(root_metadata.st_mode):04o}"
    _require(observed_root_mode == "0700", "reviewed ranking bundle mode must be 0700")

    initialized = review["warm_start_countersign"]["initialized_checkpoints"]
    expected_hashes = {
        "manifest.json": review["ranking_bundle"]["manifest_sha256"],
        "pairs.jsonl": review["ranking_bundle"]["pairs_sha256"],
        "fold_assignment.tsv": review["ranking_bundle"]["fold_file_sha256"],
        "execution_lock.json": review["execution_proposal"]["artifact_sha256"],
        "ranking_text_e5.pt": review["title_table"]["e5_cache_sha256"],
        **{
            f"init_seed{seed}.pt": record["artifact_sha256"]
            for seed, record in initialized.items()
        },
    }
    payloads: dict[str, bytes] = {}
    observed_modes: dict[str, str] = {}
    for filename, expected_sha in expected_hashes.items():
        payload, mode = _verified_regular_file(root / filename, filename)
        observed_sha = hashlib.sha256(payload).hexdigest()
        _require(observed_sha == expected_sha, f"{filename} hash differs from review")
        payloads[filename] = payload
        observed_modes[filename] = mode

    for filename, mode in observed_modes.items():
        if filename != "ranking_text_e5.pt":
            _require(mode == "0600", f"{filename} mode must be 0600")
    _require(
        observed_modes["ranking_text_e5.pt"] == review["title_table"]["observed_mode"],
        "ranking_text_e5.pt mode differs from reviewed proposal",
    )
    privacy_gate_passed = (
        observed_modes["ranking_text_e5.pt"] == review["title_table"]["required_mode"]
    )
    _require(
        privacy_gate_passed is review["title_table"]["mode_gate_passed"],
        "title-cache privacy-gate result differs from review",
    )

    try:
        manifest = strict_json_loads(payloads["manifest.json"], source=str(root / "manifest.json"))
        proposal = strict_json_loads(
            payloads["execution_lock.json"],
            source=str(root / "execution_lock.json"),
        )
    except ValueError as exc:
        raise ProtocolError(f"reviewed local JSON is invalid: {exc}") from exc
    _require(isinstance(manifest, dict), "ranking manifest must be a JSON object")
    _require(isinstance(proposal, dict), "execution-lock proposal must be a JSON object")
    _require(
        manifest.get("schema") == review["ranking_bundle"]["schema"],
        "local ranking manifest schema drifted",
    )
    _require(
        manifest.get("outputs")
        == {
            "fold_assignment.tsv": review["ranking_bundle"]["fold_file_sha256"],
            "pairs.jsonl": review["ranking_bundle"]["pairs_sha256"],
        },
        "local ranking manifest outputs drifted",
    )
    _require(
        manifest.get("counts") == review["ranking_bundle"]["counts"],
        "local ranking manifest counts drifted",
    )
    _require(
        manifest.get("model_fitting_authorized") is False,
        "local ranking manifest authorizes fitting",
    )

    _require(proposal.get("schema") == EXECUTION_LOCK_SCHEMA, "local execution-lock schema drifted")
    _require(
        proposal.get("prereg_id") == review["parent"]["ranking_prereg_id"],
        "local execution-lock parent prereg drifted",
    )
    _require(
        proposal.get("ranking_bundle_manifest_sha256")
        == review["ranking_bundle"]["manifest_sha256"],
        "local execution-lock ranking bundle drifted",
    )
    _require(
        proposal.get("warm_start", {}).get("sha256")
        == review["warm_start_countersign"]["checkpoint_sha256"],
        "local execution-lock warm start drifted",
    )
    local_initialized = proposal.get("initialized_checkpoints", {})
    _require(set(local_initialized) == set(initialized), "local execution-lock seed set drifted")
    for seed, record in initialized.items():
        _require(
            local_initialized.get(seed, {}).get("sha256") == record["artifact_sha256"],
            f"local execution-lock initialized checkpoint {seed} drifted",
        )
    _require(
        proposal.get("tokenizer_text_table", {}).get("texts_sha256")
        == review["title_table"]["texts_sha256"],
        "local execution-lock title table drifted",
    )
    _require(
        proposal.get("tokenizer_text_table", {}).get("e5_cache_sha256")
        == review["title_table"]["e5_cache_sha256"],
        "local execution-lock E5 cache drifted",
    )
    _require(
        proposal.get("trainable_parameters", {}).get("tensor_count")
        == review["reviewed_training_fields"]["trainable_tensor_count"],
        "local execution-lock trainable tensor count drifted",
    )
    _require(
        proposal.get("trainable_parameters", {}).get("param_count")
        == review["reviewed_training_fields"]["trainable_parameter_count"],
        "local execution-lock trainable parameter count drifted",
    )
    _require(
        proposal.get("status", "").startswith("PROPOSED"),
        "local execution lock no longer identifies itself as proposed",
    )

    return {
        "bundle_mode": observed_root_mode,
        "artifact_modes": observed_modes,
        "privacy_gate_passed": privacy_gate_passed,
        "model_fitting_authorized": False,
    }


def main() -> None:
    review = load_and_verify_execution_review()
    print(f"ranking-execution-review\t{review['review_id']}")
    print("warm-start\tcountersigned")
    print("model-fitting\tblocked")


if __name__ == "__main__":
    main()
