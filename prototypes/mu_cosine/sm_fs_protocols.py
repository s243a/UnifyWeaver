#!/usr/bin/env python3
"""Verify the frozen SM-FS ranking and retention/transfer protocols.

These validators protect the statistical choices that are cheap to mutate after
results are visible. They do not construct private data, fit a model, or authorize
the currently blocked transfer run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from routed_policy import canonical_json_bytes, strict_json_loads


ROOT = Path(__file__).resolve().parent
RANKING_PREREG_PATH = ROOT / "SM_FS_LINEAGE_RANKING_PREREG.json"
TRANSFER_PREREG_PATH = ROOT / "SM_FS_RETENTION_TRANSFER_PREREG.json"
RANKING_SCHEMA = "unifyweaver.sm-fs-lineage-ranking-prereg.v1"
TRANSFER_SCHEMA = "unifyweaver.sm-fs-retention-transfer-prereg.v1"
RANKING_PROTOCOL_NAME = "PROTOCOL_sm_fs_lineage_ranking.md"
TRANSFER_PROTOCOL_NAME = "PROTOCOL_sm_fs_retention_transfer.md"
EXPECTED_RANKING_PREREG_ID = "0235521270c565a413d69feaa01a6e101fb533261942451a43ddf963e409aca2"
EXPECTED_TRANSFER_PREREG_ID = "f3e1d123e6f81191489689d7f7e0fd121e25ff891ada330fe668b8b3238497c1"
SAMPLER_KEY_SCHEMA = "unifyweaver.sm-fs-ranking-sampler-key.v1"
SAMPLER_ID = "sm-fs-ranking-sampler-v1"
SAMPLER_ROLES = frozenset(
    {
        "query",
        "common-positive",
        "contrast-positive",
        "negative-bucket",
        "negative-candidate",
    }
)


class ProtocolError(ValueError):
    """A protocol is malformed, internally inconsistent, or has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _prereg_id(document: Mapping[str, Any]) -> str:
    core = dict(document)
    core.pop("prereg_id", None)
    return hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def _exact_int(value: Any, expected: int, description: str) -> None:
    _require(type(value) is int and value == expected, f"{description} drifted")


def _exact_number(value: Any, expected: int | float, description: str) -> None:
    _require(
        type(value) in {int, float} and value == expected,
        f"{description} drifted",
    )


def _load_document(
    path: str | Path,
    schema: str,
    preregistration_name: str,
    protocol_name: str,
) -> dict[str, Any]:
    path = Path(path)
    _require(path.name == preregistration_name, "unexpected preregistration filename")
    _require(not path.is_symlink(), "preregistration may not be a symlink")
    try:
        document = strict_json_loads(path.read_bytes(), source=str(path))
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"cannot load protocol preregistration: {exc}") from exc
    _require(isinstance(document, dict), "preregistration must be a JSON object")
    _require(document.get("schema") == schema, "unsupported preregistration schema")
    _require(document.get("protocol_path") == protocol_name, "protocol path drifted")
    protocol_path = path.parent / protocol_name
    _require(protocol_path.is_file(), "protocol document is missing")
    _require(not protocol_path.is_symlink(), "protocol document may not be a symlink")
    observed = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    _require(
        document.get("protocol_sha256") == observed,
        "protocol document hash differs from preregistration",
    )
    _require(document.get("prereg_id") == _prereg_id(document), "preregistration ID mismatch")
    return document


def _full_sha(value: Any, description: str) -> None:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{description} must be a full lowercase SHA-256",
    )


def sampler_key_bytes(
    *,
    fold: int,
    seed: int,
    step: int,
    draw: int,
    role: str,
    query_id: str = "",
    bucket: str = "",
    retry: int = 0,
) -> bytes:
    """Canonical bytes for one counter-based ranking-sampler draw.

    Fold, step, draw, and retry are zero-based nonnegative integers. JSON
    numbers are their canonical base-10 representation; strings are UTF-8;
    ``canonical_json_bytes`` fixes key order, separators, escaping, and the
    final newline.
    """

    for value, name in (
        (fold, "fold"),
        (seed, "seed"),
        (step, "step"),
        (draw, "draw"),
        (retry, "retry"),
    ):
        _require(
            type(value) is int and value >= 0,
            f"sampler {name} must be a nonnegative integer",
        )
    _require(role in SAMPLER_ROLES, "unknown sampler role")
    _require(isinstance(query_id, str), "sampler query ID must be a string")
    _require(isinstance(bucket, str), "sampler bucket must be a string")
    _require(
        (role != "query" or (query_id == "" and bucket == "")),
        "query draw must not bind a selected query or bucket",
    )
    _require(
        (role == "query" or query_id != ""),
        "post-query draw must bind the selected query",
    )
    _require(
        (role == "negative-candidate" or bucket == ""),
        "only a negative-candidate draw may bind a bucket",
    )
    _require(
        (role != "negative-candidate" or bucket in {"hard", "medium", "easy"}),
        "negative-candidate draw must bind its bucket",
    )
    return canonical_json_bytes(
        {
            "bucket": bucket,
            "draw": draw,
            "fold": fold,
            "query_id": query_id,
            "retry": retry,
            "role": role,
            "sampler_id": SAMPLER_ID,
            "schema": SAMPLER_KEY_SCHEMA,
            "seed": seed,
            "step": step,
        }
    )


def sampler_index(
    n: int,
    *,
    fold: int,
    seed: int,
    step: int,
    draw: int,
    role: str,
    query_id: str = "",
    bucket: str = "",
) -> tuple[int, str, int]:
    """Return an unbiased deterministic index, accepted digest, and retry."""

    _require(type(n) is int and n > 0, "sampler population must be positive")
    modulus = 1 << 256
    limit = (modulus // n) * n
    retry = 0
    while True:
        digest = hashlib.sha256(
            sampler_key_bytes(
                fold=fold,
                seed=seed,
                step=step,
                draw=draw,
                role=role,
                query_id=query_id,
                bucket=bucket,
                retry=retry,
            )
        ).digest()
        value = int.from_bytes(digest, "big", signed=False)
        if value < limit:
            return value % n, digest.hex(), retry
        retry += 1


def load_and_verify_ranking(
    path: str | Path = RANKING_PREREG_PATH,
) -> dict[str, Any]:
    document = _load_document(
        path,
        RANKING_SCHEMA,
        RANKING_PREREG_PATH.name,
        RANKING_PROTOCOL_NAME,
    )
    _require(
        document.get("evidence_status")
        == "prospective-exploratory-catalog-transductive",
        "ranking evidence status drifted",
    )

    prior = document.get("prior_pilot", {})
    _require(prior.get("decision_eligible") is False, "pilot must remain decision-ineligible")
    _require(prior.get("ranking_evidence") is False, "pilot is not ranking evidence")
    _require(prior.get("transfer_evidence") is False, "pilot is not transfer evidence")
    _require(
        prior.get("reported_warm_start_0.37_is_content_bound") is False,
        "unsupported warm-start baseline became authorized",
    )
    _require(
        prior.get("validation_rows_with_train_ancestor") == 356,
        "pilot lineage-overlap audit drifted",
    )

    source = document.get("source_bundle", {})
    _require(
        source.get("schema") == "unifyweaver.sm-fs-training-bundle.v1",
        "wrong source bundle schema",
    )
    _require(source.get("privacy_policy") == "public-only", "public-only source required")
    _require(
        source.get("process_expression") == "lineage(fs,decay=0.85)",
        "source process expression drifted",
    )
    _require(
        source.get("e5_revision") == "ffb93f3bd4047442299a41ebb6fa998a38507c52",
        "source E5 revision drifted",
    )
    expected_source_hashes = {
        "manifest_file_sha256": "466adfcf2e7c5914dad27b548c3f009804fec12e633498e187b0b4df71a98d61",
        "manifest_internal_sha256": "e36164b7fef0569acfaeafb667628062cf37ced7ad04df0f34543c899086efe2",
        "ledger_sha256": "b45ff8ded88f2e5d5ded78664b2d381a017a5722464f956f6b44d660f5060b40",
        "targets_sha256": "c3b298d5335ee901111f4985bbf5f7c5feb017c503a8c81db60dba1b947ac051",
    }
    for field, expected in expected_source_hashes.items():
        _full_sha(source.get(field), field)
        _require(source.get(field) == expected, f"{field} drifted")
    _require(
        source.get("constructor_input") == "verified-target-projection-only",
        "constructor must receive only the verified target projection",
    )
    _require(source.get("ledger_catalog_authorized") is False, "full ledger catalog is forbidden")
    _require(source.get("reserve_rows_authorized") is False, "reserve rows are forbidden")
    _require(
        source.get("cross_lineage_rows_authorized") is False,
        "cross-lineage rows are forbidden",
    )

    construction = document.get("construction", {})
    _require(
        construction.get("schema") == "unifyweaver.sm-fs-lineage-ranking-bundle.v1",
        "ranking bundle schema drifted",
    )
    _require(
        construction.get("catalog") == "sorted-union-of-explore-positive-ancestor-paths",
        "explore-only catalog rule drifted",
    )
    _require(construction.get("candidate_enumeration") == "exhaustive", "negatives must be exhaustive")
    _require(
        construction.get("negative_means_semantically_unsuitable") is False,
        "structural alternatives cannot become semantic labels",
    )
    counts = construction.get("counts", {})
    _require(
        counts
        == {
            "queries": 361,
            "candidates": 359,
            "pairs": 129599,
            "positives": 2792,
            "nonancestors": 126807,
            "hard_nonancestors": 1819,
            "medium_nonancestors": 7814,
            "easy_nonancestors": 117174,
        },
        "ranking construction counts drifted",
    )
    _require(
        counts["positives"] + counts["nonancestors"] == counts["pairs"],
        "positive and nonancestor counts do not close",
    )
    _require(
        counts["hard_nonancestors"]
        + counts["medium_nonancestors"]
        + counts["easy_nonancestors"]
        == counts["nonancestors"],
        "hardness counts do not close",
    )
    graph_target = construction.get("graph_target", {})
    _exact_number(graph_target.get("decay"), 0.85, "graph-target decay")
    _exact_number(graph_target.get("floor"), 0.02, "graph-target floor")
    _require(
        graph_target.get("formula")
        == "max(floor,decay^distance*depth(lca)/depth(destination))",
        "graph-target formula drifted",
    )
    _require(
        graph_target.get("positive_representation")
        == "certified-six-decimal-ascii-parsed-to-binary64",
        "positive target representation drifted",
    )
    _require(
        graph_target.get("negative_representation")
        == "exact-reduced-rational-then-correctly-rounded-binary64",
        "negative target representation drifted",
    )
    _exact_number(
        graph_target.get("different_root_lca_fraction"),
        0.0,
        "different-root LCA fraction",
    )
    _require(
        graph_target.get("rounding_record") == "reduced-rational-plus-float.hex",
        "target rounding record drifted",
    )
    _require(
        construction.get("relations")
        == {
            "descendant": "destination-is-strict-prefix-of-candidate",
            "sibling": "same-immediate-parent-and-equal-depth",
            "near_branch": "shared-root-not-descendant-or-sibling-and-distance-at-most-4",
            "same_root_far": "shared-root-not-descendant-or-sibling-and-distance-at-least-5",
            "cross_root": "longest-common-prefix-length-zero",
        },
        "relation definitions drifted",
    )
    _require(
        construction.get("hardness")
        == {
            "hard": "reachable-distance-1-or-2",
            "medium": "reachable-distance-3-or-4",
            "easy": "reachable-distance-at-least-5-or-cross-root",
        },
        "hardness definitions drifted",
    )
    mass = construction.get("query_mass", {})
    _require(mass.get("positive") == 0.5, "positive query mass drifted")
    _require(mass.get("nonancestor") == 0.5, "nonancestor query mass drifted")
    _require(
        mass.get("negative_bucket_ratio") == {"hard": 3, "medium": 2, "easy": 1},
        "negative bucket weights drifted",
    )
    _require(mass.get("empty_bucket_topup") is False, "empty buckets may not be topped up")

    split = document.get("split", {})
    _require(
        split.get("kind") == "adaptive-depth-3-lineage-blocked-five-fold",
        "split kind drifted",
    )
    _exact_int(split.get("folds"), 5, "fold count")
    _exact_int(split.get("cap"), 28, "lineage-block cap")
    _exact_int(split.get("deepened_prefixes"), 14, "deepened-prefix count")
    _exact_int(split.get("blocks"), 82, "lineage-block count")
    _require(split.get("fold_map_counts") == [73, 72, 72, 72, 72], "fold map counts drifted")
    _require(split.get("fold_block_counts") == [16, 16, 16, 17, 17], "fold block counts drifted")
    _require(
        split.get("assignment")
        == "descending-count-then-sha256-salted-block-then-utf8-greedy-lightest-fold",
        "fold assignment algorithm drifted",
    )
    _require(
        split.get("assignment_sha256")
        == "b5439b1acdfb5020ecb7fb4fad437fd183242408d09056e1ed7ec90d33335f37",
        "fold assignment hash drifted",
    )
    _require(split.get("block_split_authorized") is False, "lineage blocks may not split")
    _require(
        split.get("evidence_scope") == "map-near-lineage-blocked-catalog-transductive",
        "ranking evidence scope drifted",
    )

    training = document.get("training", {})
    _require(training.get("seeds") == [3997001, 3997002, 3997003], "ranking seeds drifted")
    _exact_int(training.get("steps"), 800, "training steps")
    _exact_int(training.get("batch_size"), 48, "batch size")
    _exact_number(training.get("learning_rate"), 0.0005, "learning rate")
    _exact_number(training.get("anchor_weight"), 1.0, "anchor weight")
    _exact_number(training.get("gradient_clip_norm"), 1.0, "gradient clipping")
    _require(training.get("early_stopping") is False, "held early stopping is forbidden")
    _require(
        training.get(
            "matched_initialization_budget_query_schedule_common_positive_batches_and_augmentation"
        )
        is True,
        "paired sampler contract drifted",
    )
    sampler = training.get("sampler", {})
    sampler_without_vectors = dict(sampler)
    vectors = sampler_without_vectors.pop("known_answer_vectors", None)
    _require(
        sampler_without_vectors
        == {
            "kind": "paired-query-two-slot-sha256-rejection-v1",
            "sampler_id": SAMPLER_ID,
            "key_schema": SAMPLER_KEY_SCHEMA,
            "serialization": "canonical-json-utf8-sorted-keys-compact-separators-terminal-lf",
            "indexing": "fold-step-draw-retry-zero-based",
            "query_draws_per_step": 24,
            "rows_per_step": 48,
            "with_replacement": True,
            "query_population": "canonical-utf8-sorted-training-query-list",
            "domain_separated_roles": [
                "query",
                "common-positive",
                "contrast-positive",
                "negative-bucket",
                "negative-candidate",
            ],
            "common_slot": "uniform-positive-both-arms",
            "contrast_slot_positive_only": "independent-uniform-positive",
            "contrast_slot_graded_negative": "renormalized-3-2-1-bucket-then-uniform-candidate",
            "bucket_order": ["hard", "medium", "easy"],
            "batch_loss": "unweighted-mean-squared-error-over-sampled-rows",
            "stored_objective_weight_multiplied_again": False,
            "anchor_rows": "common-positive-slots-only-identical-across-arms",
            "anchor_training_augmentation": False,
        },
        "minibatch sampler drifted",
    )
    _require(isinstance(vectors, list) and len(vectors) == 4, "sampler vectors drifted")
    for vector in vectors:
        _require(isinstance(vector, dict), "sampler vector is malformed")
        index, digest, retry = sampler_index(
            vector.get("n"),
            fold=vector.get("fold"),
            seed=vector.get("seed"),
            step=vector.get("step"),
            draw=vector.get("draw"),
            role=vector.get("role"),
            query_id=vector.get("query_id"),
            bucket=vector.get("bucket"),
        )
        _require(
            (index, digest, retry)
            == (vector.get("index"), vector.get("digest"), vector.get("retry")),
            "sampler known-answer vector drifted",
        )
    _require(
        training.get("decision_arms") == ["positive_only", "graded_negative"],
        "decision arms drifted",
    )
    _require(
        training.get("lineage_rank_softmax_authorized") is False,
        "listwise head is a separate future arm",
    )
    _require(
        training.get("path_components_in_embedding_text") is False,
        "path components may not leak into embedding text",
    )
    _require(
        training.get("embedding_text")
        == "certified-map-title-and-candidate-leaf-title-only",
        "embedding text contract drifted",
    )

    primary = document.get("primary", {})
    _require(primary.get("metric") == "exact-destination-mrr", "primary metric drifted")
    _require(
        primary.get("contrast") == "graded-negative-minus-positive-only",
        "primary contrast drifted",
    )
    _require(
        primary.get("seed_aggregation") == "mean-per-query-before-contrast",
        "ranking seed aggregation drifted",
    )
    _exact_number(primary.get("minimum_point_gain"), 0.01, "practical floor")
    _exact_number(
        primary.get("interval_lower_strictly_greater_than"),
        0.0,
        "interval gate",
    )
    bootstrap = primary.get("bootstrap", {})
    _require(bootstrap.get("unit") == "adaptive-lineage-block", "bootstrap unit drifted")
    _exact_int(bootstrap.get("resamples"), 9999, "bootstrap count")
    _exact_int(bootstrap.get("seed"), 3997999, "bootstrap seed")
    _exact_number(bootstrap.get("confidence"), 0.95, "bootstrap confidence")
    _exact_int(bootstrap.get("minimum_blocks_for_decision"), 20, "block floor")
    _require(
        primary.get("interval_scope")
        == "fixed-fit-conditional-on-frozen-corpus-folds-and-three-seeds",
        "ranking interval scope drifted",
    )
    _require(
        primary.get("passing_authorizes") == "new-reserve-preregistration-only",
        "passing result cannot directly authorize reserve scoring",
    )

    reserve = document.get("reserve", {})
    _require(reserve.get("reported_count") == 1481, "reserve count drifted")
    _require(reserve.get("opened_by_constructor") is False, "constructor may not open reserve")
    _require(reserve.get("opened_by_trainer") is False, "trainer may not open reserve")
    _require(reserve.get("scored_by_protocol") is False, "reserve must remain unscored")

    privacy = document.get("privacy", {})
    _require(privacy.get("local_only") is True, "artifacts must remain local")
    _require(privacy.get("directory_mode") == "0700", "private directory mode required")
    _require(privacy.get("file_mode") == "0600", "private file mode required")
    _require(privacy.get("atomic_no_replace") is True, "atomic no-replace required")
    _require(
        privacy.get("provider_calls_with_source_content") is False,
        "source content may not reach a provider",
    )
    _require(
        privacy.get("checkpoint_release_authorized") is False,
        "protocol cannot authorize checkpoint release",
    )
    _require(
        document.get("constructor_implementation_authorized") is True,
        "ranking constructor should be implementable",
    )
    _require(document.get("model_fitting_authorized") is False, "model fitting must remain blocked")
    _require(
        set(document.get("fitting_blocked_on", []))
        == {
            "content-bound-initialized-checkpoint-and-growth",
            "content-bound-tokenizer-and-text-table",
            "content-bound-training-code-and-plan",
            "optimizer-trainable-parameter-anchor-and-augmentation-lock",
            "numeric-environment-and-determinism-lock",
        },
        "ranking fitting blockers drifted",
    )
    _require(
        document.get("prereg_id") == EXPECTED_RANKING_PREREG_ID,
        "frozen ranking preregistration ID drifted",
    )
    return document


def load_and_verify_transfer(
    path: str | Path = TRANSFER_PREREG_PATH,
) -> dict[str, Any]:
    document = _load_document(
        path,
        TRANSFER_SCHEMA,
        TRANSFER_PREREG_PATH.name,
        TRANSFER_PROTOCOL_NAME,
    )
    _require(
        document.get("evidence_status") == "prospective-blocked-design-lock",
        "transfer evidence status drifted",
    )
    source = document.get("source", {})
    ranking_contract = load_and_verify_ranking()
    _require(
        source.get("ranking_protocol_schema") == RANKING_SCHEMA,
        "wrong source ranking protocol",
    )
    _require(
        source.get("ranking_protocol_sha256")
        == hashlib.sha256((ROOT / RANKING_PROTOCOL_NAME).read_bytes()).hexdigest(),
        "source ranking protocol hash drifted",
    )
    _require(
        source.get("ranking_prereg_id")
        == ranking_contract.get("prereg_id")
        == EXPECTED_RANKING_PREREG_ID,
        "source ranking preregistration drifted",
    )
    _require(
        source.get("negative_bundle_status") == "blocked-until-frozen-and-verified",
        "negative-bundle status drifted",
    )
    _require(
        source.get("null_ledger_status") == "blocked-until-frozen-and-verified",
        "null-ledger status drifted",
    )
    _require(source.get("negative_bundle_sha256") is None, "unbound negative bundle must stay null")
    _require(source.get("null_ledger_sha256") is None, "unbound null ledger must stay null")
    _require(source.get("reserve_rows_authorized") is False, "reserve rows are forbidden")
    _require(source.get("quarantined_rows_authorized") is False, "quarantined rows are forbidden")

    expected_checkpoints = {
        "pre_pt_base": "c1cfc3a3827e42a1993f4286b6a881aee7ff10eb56a76367735b9ec8fdf11f7d",
        "pre_pt_migrated": "9bb60915e9bcf8d4a89293c538284fd5ac631d9729f355917e128db7353bc8ef",
        "pt_retention_base": "55834a204093e1cd525b7189d372cdec6eea4b424518e938912955f5dfcf3c76",
        "sm_fs_positive_pilot": "e541e8b812a00be457a27f3f7484ffcb53f5ba65c54c6c3bbd9ebcba90c9be19",
    }
    checkpoints = document.get("checkpoints", {})
    for name, expected in expected_checkpoints.items():
        record = checkpoints.get(name, {})
        _full_sha(record.get("sha256"), f"{name} checkpoint")
        _require(record.get("sha256") == expected, f"{name} checkpoint drifted")
    _require(
        checkpoints.get("pre_pt_base", {}).get("role") == "migration-provenance-only",
        "raw pre-PT checkpoint role drifted",
    )
    _require(
        checkpoints.get("pre_pt_migrated", {}).get("role") == "operative-track-t-source",
        "operative transfer checkpoint drifted",
    )
    _require(
        checkpoints.get("sm_fs_positive_pilot", {}).get("decision_eligible") is False,
        "pilot must remain decision-ineligible",
    )

    training = document.get("training", {})
    _require(training.get("seeds") == [3998101, 3998102, 3998103], "transfer seeds drifted")
    _require(
        training.get("matched_initial_state_budget_batches_and_augmentation") is True,
        "paired training contract drifted",
    )
    _require(training.get("seed_zero_pilot_eligible") is False, "pilot seed is ineligible")
    _require(
        training.get("pearltrees_outcome_selection_authorized") is False,
        "target outcomes cannot select source training",
    )
    _require(
        training.get("sm_fs_node_types") == ["mindmap_node", "mindmap_node"],
        "SM-FS node-type contract drifted",
    )
    _require(training.get("one_gpu_job_at_a_time") is True, "GPU scheduling lock drifted")

    null = document.get("null_control", {})
    _require(null.get("required") is True, "matched source control is required")
    _require(
        null.get("kind") == "within-source-list-constrained-joint-payload-permutation",
        "null construction drifted",
    )
    _require(
        null.get("preserves_list_level")
        == [
            "items",
            "candidate_text",
            "list_sizes",
            "target_marginals",
            "positive-negative-types",
            "row_weights",
            "folds",
            "batches",
            "optimizer_budget",
        ],
        "null list-level invariants drifted",
    )
    _require(
        null.get("payload")
        == [
            "target-bytes",
            "objective-mass",
            "positive-nonancestor-status",
            "relation",
            "hardness",
        ],
        "null supervision payload drifted",
    )
    _require(null.get("rowwise_type_or_weight_preserved") is False, "null must move row metadata")
    _require(
        null.get("constraints")
        == [
            "every-payload-index-moves",
            "every-true-ancestor-receives-nonancestor-payload",
            "every-positive-payload-moves-to-true-nonancestor",
            "target-one-parent-payload-moves",
        ],
        "null permutation constraints drifted",
    )
    _require(
        null.get("fixed_point_definition") == "payload-index-not-numeric-value",
        "null fixed-point definition drifted",
    )
    _require(
        null.get("unchanged_numeric_target_fraction_recorded") is True,
        "null unchanged-value diagnostic required",
    )
    _require(null.get("pearltrees_inputs_authorized") is False, "null may not use target inputs")

    retention = document.get("retention", {})
    _require(
        retention.get("arms") == ["R0_unchanged_pt", "R1_correct_sm_fs", "Rnull_scrambled_sm_fs"],
        "retention arms drifted",
    )
    _require(retention.get("primary_contrast") == "R1-minus-R0", "retention contrast drifted")
    _require(retention.get("metric") == "exact-destination-mrr", "retention metric drifted")
    _exact_number(
        retention.get("interval_lower_strictly_greater_than"),
        -0.005,
        "retention margin",
    )
    _exact_number(retention.get("maximum_individual_seed_loss"), 0.01, "per-seed retention floor")
    _require(retention.get("correct_candidate_lineage") is True, "retention lineage regime drifted")
    _require(retention.get("behavior_panel_required") is True, "behavior panel is required")
    _require(retention.get("behavior_panel_sha256") is None, "unbound behavior panel must stay null")
    _require(
        retention.get("status") == "blocked-behavior-panel-and-source-ledgers",
        "retention status drifted",
    )

    transfer = document.get("transfer", {})
    _require(
        transfer.get("arms")
        == ["T0_common_expanded_pre_pt", "T1_correct_sm_fs", "Tnull_scrambled_sm_fs"],
        "transfer arms drifted",
    )
    _require(transfer.get("common_expansion_once_per_seed") is True, "common expansion required")
    _require(
        transfer.get("operative_source") == "model_prod_namecond_full.pt",
        "operative transfer source drifted",
    )
    _require(transfer.get("total_contrast") == "T1-minus-T0", "total transfer contrast drifted")
    _exact_number(transfer.get("total_minimum_point_gain"), 0.01, "transfer floor")
    _exact_number(
        transfer.get("total_interval_lower_strictly_greater_than"),
        0.0,
        "total-transfer interval gate",
    )
    _require(
        transfer.get("specificity_contrast") == "T1-minus-Tnull",
        "specificity contrast drifted",
    )
    _exact_number(transfer.get("specificity_minimum_point_gain"), 0.005, "specificity floor")
    _exact_number(
        transfer.get("specificity_interval_lower_strictly_greater_than"),
        0.0,
        "specificity interval gate",
    )
    _require(transfer.get("gate_specificity_after_total") is True, "gatekeeping drifted")
    _require(
        transfer.get("e5_noninferiority_contrast") == "T1-minus-e5",
        "e5 safety contrast drifted",
    )
    _exact_number(transfer.get("e5_noninferiority_margin"), -0.01, "e5 safety margin")
    _require(
        transfer.get("e5_interval_lower_strictly_greater_than_margin") is True,
        "e5 safety interval gate drifted",
    )
    _require(transfer.get("title_only_lineage_primary") is True, "title-only primary required")
    _require(
        transfer.get("correct_candidate_lineage_primary") is False,
        "candidate-lineage transfer primary is forbidden",
    )
    _require(
        transfer.get("status") == "blocked-fresh-target-cohort-and-source-ledgers",
        "transfer status drifted",
    )

    target = document.get("target", {})
    _require(
        target.get("fresh_post_freeze_cohort_required_for_confirmation") is True,
        "fresh cohort required for confirmation",
    )
    _require(target.get("catalog_frozen_before_placement_labels") is True, "catalog timing drifted")
    _require(
        target.get("predictions_sealed_before_destination_join") is True,
        "prediction/label temporal boundary drifted",
    )
    _require(
        target.get("historical_identity_overlap_authorized") is False,
        "historical overlap is forbidden",
    )
    _require(
        target.get("historical_resource_overlap_authorized") is False,
        "historical resource overlap is forbidden",
    )
    _require(
        target.get("canonical_resource_hash_required_for_confirmation") is True,
        "canonical resource identity is required",
    )
    _require(
        target.get("historical_population_status")
        == "exploratory-transductive-locked-audit-only",
        "historical population status drifted",
    )
    _require(
        target.get("e5_revision") == "ffb93f3bd4047442299a41ebb6fa998a38507c52",
        "target E5 revision drifted",
    )
    _exact_int(target.get("candidate_pool_size"), 100, "candidate pool")
    _exact_number(target.get("missing_destination_rr"), 0.0, "candidate miss rule")
    _require(target.get("exact_id_primary") is True, "exact destination identity required")
    _require(target.get("title_equivalence_primary") is False, "title equivalence is secondary")

    inference = document.get("inference", {})
    _require(inference.get("function") == "paired_node_bootstrap_ci", "bootstrap function drifted")
    _require(inference.get("typed_endpoints") == ["resource", "folder"], "typed endpoints drifted")
    _exact_int(inference.get("resamples"), 9999, "bootstrap count")
    _exact_int(inference.get("seed"), 3998999, "bootstrap seed")
    _exact_number(inference.get("confidence"), 0.95, "bootstrap confidence")
    _exact_int(
        inference.get("minimum_endpoint_components_for_decision"),
        20,
        "endpoint-component floor",
    )
    _require(
        inference.get("seed_aggregation") == "mean-per-query-before-contrast",
        "transfer seed aggregation drifted",
    )
    _require(
        inference.get("interval_scope")
        == "conditional-on-frozen-target-cohort-and-fitted-checkpoints",
        "transfer interval scope drifted",
    )
    _require(
        inference.get("root_decisions") == ["retention", "total-transfer"],
        "root decision family drifted",
    )
    _require(inference.get("root_interval") == "central-95-percent", "root interval drifted")
    _exact_number(inference.get("root_lower_tail_alpha_each"), 0.025, "root tail alpha")
    _exact_number(
        inference.get("root_family_bonferroni_upper_bound"),
        0.05,
        "root family error bound",
    )
    _require(
        inference.get("specificity_and_e5_are_serially_gated") is True,
        "serial gate drifted",
    )

    privacy = document.get("privacy", {})
    _require(privacy.get("local_only") is True, "artifacts must remain local")
    _require(privacy.get("directory_mode") == "0700", "private directory mode required")
    _require(privacy.get("file_mode") == "0600", "private file mode required")
    _require(privacy.get("atomic_no_replace") is True, "atomic no-replace required")
    _require(
        privacy.get("provider_calls_with_source_content") is False,
        "source content may not reach a provider",
    )
    _require(
        privacy.get("checkpoint_release_authorized") is False,
        "protocol cannot authorize checkpoint release",
    )
    _require(document.get("execution_authorized") is False, "blocked design must not authorize execution")
    _require(
        set(document.get("blocked_on", []))
        == {
            "negative-ranking-bundle",
            "matched-null-ledger",
            "retention-behavior-panel",
            "fresh-certified-public-pearltrees-cohort",
            "exact-training-plan-and-environment-lock",
        },
        "transfer blocking set drifted",
    )
    _require(
        document.get("prereg_id") == EXPECTED_TRANSFER_PREREG_ID,
        "frozen transfer preregistration ID drifted",
    )
    return document


def main() -> None:
    ranking = load_and_verify_ranking()
    transfer = load_and_verify_transfer()
    print(f"ranking\t{ranking['prereg_id']}")
    print(f"retention-transfer\t{transfer['prereg_id']}\tblocked")


if __name__ == "__main__":
    main()
