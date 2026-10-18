#!/usr/bin/env python3
"""Extract the exact topology source design used by Stage-A power runs.

The reviewed source-dependence artifact is intentionally a compact projection:
it records hashes for exposure matrices and allocation arrays but omits their
values.  Stage A must not pretend those hashes are the arrays.  This module
extracts the exact values from the content-addressed full payload, validates
every omitted-array record against the tracked projection, and writes a
smaller, runner-oriented, path-free bundle.

The compact encoding stores region IDs once per ``(corpus, K)``.  Assignment,
count, and capacity arrays then contain integer indices/counts in that exact
region order.  Their original ID-keyed/list representations are reconstructed
and re-hashed on every load.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY_PATH = (
    HERE / "repro" / "repeated_judge_source_dependence" / "summary.json"
)
SCHEMA_VERSION = 1
ALGORITHM = "repeated-judge-source-power-design-bundle-v1"
PARENT_ALGORITHM = "repeated-judge-topology-source-dependence-bridge-v1"
EXPECTED_CORPORA = ("exploratory", "fresh")
EXPECTED_REGION_COUNTS = (64, 96, 128)
EXPECTED_COMPONENT_COUNTS = (160, 320, 512, 800)
EXPECTED_SOURCE_ETA_GRID = (0.0, 0.025, 0.05, 0.10, 0.20)
_SHA256_HEX_LENGTH = 64
REGION_INDEX_ENCODING = (
    "assignment_region_indices[i] indexes region_ids; counts_by_region and "
    "capacities_by_region use the same region_ids order"
)
AUTHORIZATION = {
    "attempted_input_identity_inventory_unlocked": False,
    "candidate_enumeration_unlocked": False,
    "nomic_embedding_unlocked": False,
    "judge_calls_authorized": False,
    "covariance_deployment_unlocked": False,
    "qr_specialization_unlocked": False,
    "cuda_claim_unlocked": False,
}
NON_CLAIMS = [
    "region-average topology exposure is not empirical residual covariance",
    "the allocation is a prospective diagnostic quota, not exact candidate packability",
    "this exact source bundle does not itself unlock any downstream work",
]


class SourcePowerBundleError(ValueError):
    """Raised when the exact source-design contract cannot be verified."""


def _object_without_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SourcePowerBundleError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_json(value):
    raise SourcePowerBundleError(f"non-finite JSON constant: {value}")


def _loads(data: bytes, label: str) -> dict:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourcePowerBundleError(f"invalid UTF-8 JSON for {label}") from exc
    if not isinstance(value, dict):
        raise SourcePowerBundleError(f"{label} must be a JSON object")
    return value


def canonical_json_bytes(value: Mapping) -> bytes:
    """Return the path-independent canonical bytes used for bundle identity."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SourcePowerBundleError("bundle is not finite canonical JSON") from exc
    return (text + "\n").encode("utf-8")


def content_record(data: bytes) -> dict:
    return {
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def canonical_value_record(value) -> dict:
    """Match ``tracked-summary-v1`` omitted-value hashing exactly."""
    try:
        data = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SourcePowerBundleError("omitted value is not finite canonical JSON") from exc
    return content_record(data)


def source_design_bundle_identity(bundle: Mapping) -> dict:
    """Return the canonical content identity consumed by Stage-A fingerprints."""
    return content_record(canonical_json_bytes(bundle))


def _require_exact_keys(value, expected, label):
    if not isinstance(value, dict):
        raise SourcePowerBundleError(f"{label} must be an object")
    observed = set(value)
    expected = set(expected)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise SourcePowerBundleError(
            f"{label} keys mismatch; missing={missing}, extra={extra}"
        )


def _validated_record(value, label):
    _require_exact_keys(value, ("size_bytes", "sha256"), label)
    size = value["size_bytes"]
    digest = value["sha256"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise SourcePowerBundleError(f"{label}.size_bytes must be nonnegative int")
    if (
        not isinstance(digest, str)
        or len(digest) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise SourcePowerBundleError(f"{label}.sha256 must be lowercase SHA-256")
    return {"size_bytes": size, "sha256": digest}


def _require_record(value, expected, label):
    observed = canonical_value_record(value)
    expected = _validated_record(expected, label)
    if observed != expected:
        raise SourcePowerBundleError(f"{label} canonical content record mismatch")
    return expected


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SourcePowerBundleError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourcePowerBundleError(f"{label} must be a nonnegative integer")
    return value


def _exact_int_grid(value, expected, label):
    if not isinstance(value, list):
        raise SourcePowerBundleError(f"{label} must be a list")
    output = tuple(_positive_int(item, f"{label}[]") for item in value)
    if output != tuple(expected):
        raise SourcePowerBundleError(
            f"{label} must equal {list(expected)}, observed {list(output)}"
        )
    return output


def _exact_float_grid(value, expected, label):
    if not isinstance(value, list):
        raise SourcePowerBundleError(f"{label} must be a list")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise SourcePowerBundleError(f"{label} must contain finite numeric values")
    output = tuple(map(float, value))
    if output != tuple(expected):
        raise SourcePowerBundleError(
            f"{label} must equal {list(expected)}, observed {list(output)}"
        )
    return output


def _validate_parent_contract(full, summary, full_bytes, summary_bytes):
    if "artifact_projection" in full or "full_payload_record" in full:
        raise SourcePowerBundleError(
            "full source payload must be unprojected, not tracked-summary-v1"
        )
    if summary.get("artifact_projection") != "tracked-summary-v1":
        raise SourcePowerBundleError("reviewed summary is not tracked-summary-v1")
    parent_record = _validated_record(
        summary.get("full_payload_record"), "summary.full_payload_record"
    )
    if content_record(full_bytes) != parent_record:
        raise SourcePowerBundleError(
            "full source payload does not match the reviewed parent record"
        )
    if full.get("algorithm") != PARENT_ALGORITHM:
        raise SourcePowerBundleError("unexpected source-dependence algorithm")
    if summary.get("algorithm") != full.get("algorithm"):
        raise SourcePowerBundleError("summary/full algorithm mismatch")
    if full.get("schema_version") != summary.get("schema_version"):
        raise SourcePowerBundleError("summary/full schema mismatch")
    for key in ("configuration", "inputs"):
        if summary.get(key) != full.get(key):
            raise SourcePowerBundleError(f"summary/full {key} identity mismatch")
    configuration = full.get("configuration")
    if not isinstance(configuration, dict):
        raise SourcePowerBundleError("source configuration must be an object")
    corpora = configuration.get("required_corpora")
    if corpora != list(EXPECTED_CORPORA):
        raise SourcePowerBundleError("source corpus grid is not the frozen Stage-A grid")
    _exact_int_grid(
        configuration.get("region_count_grid"),
        EXPECTED_REGION_COUNTS,
        "configuration.region_count_grid",
    )
    _exact_int_grid(
        configuration.get("registered_components_per_corpus"),
        EXPECTED_COMPONENT_COUNTS,
        "configuration.registered_components_per_corpus",
    )
    _exact_float_grid(
        configuration.get("rho_grid"),
        EXPECTED_SOURCE_ETA_GRID,
        "configuration.rho_grid (legacy source eta grid)",
    )
    if set(full.get("results", {})) != set(EXPECTED_CORPORA):
        raise SourcePowerBundleError("full results do not contain exactly both corpora")
    if set(summary.get("results", {})) != set(EXPECTED_CORPORA):
        raise SourcePowerBundleError("summary results do not contain exactly both corpora")
    return {
        "source_dependence_algorithm": full["algorithm"],
        "source_dependence_schema_version": full["schema_version"],
        "full_payload_record": parent_record,
        "reviewed_summary_record": content_record(summary_bytes),
    }


def _validated_region_ids(value, region_count, label):
    if not isinstance(value, list) or len(value) != region_count:
        raise SourcePowerBundleError(f"{label} must contain {region_count} IDs")
    if any(not isinstance(item, str) or not item for item in value):
        raise SourcePowerBundleError(f"{label} must contain nonempty strings")
    if len(set(value)) != len(value):
        raise SourcePowerBundleError(f"{label} must contain unique IDs")
    return tuple(value)


def _validated_matrix(value, region_count, label):
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SourcePowerBundleError(f"{label} must be numeric") from exc
    if matrix.shape != (region_count, region_count):
        raise SourcePowerBundleError(
            f"{label} must have shape {(region_count, region_count)}"
        )
    if not np.isfinite(matrix).all():
        raise SourcePowerBundleError(f"{label} must be finite")
    if not np.array_equal(matrix, matrix.T):
        raise SourcePowerBundleError(f"{label} must be exactly symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0.0):
        raise SourcePowerBundleError(f"{label} must have unit diagonal")
    if np.any(matrix < -1e-12) or np.any(matrix > 1.0 + 1e-12):
        raise SourcePowerBundleError(f"{label} entries must lie in [0,1]")
    minimum = float(np.linalg.eigvalsh(matrix)[0])
    if minimum < -1e-10:
        raise SourcePowerBundleError(
            f"{label} must be positive semidefinite; minimum eigenvalue={minimum}"
        )
    # Return original JSON scalars so the reviewed canonical hash is preserved.
    return copy.deepcopy(value)


def _mapping_counts(value, region_ids, label):
    if not isinstance(value, dict) or set(value) != set(region_ids):
        raise SourcePowerBundleError(f"{label} must align exactly with region IDs")
    return tuple(
        _nonnegative_int(value[region_id], f"{label}[{region_id!r}]")
        for region_id in region_ids
    )


def _assignment_indices(value, region_ids, components, label):
    if not isinstance(value, list) or len(value) != components:
        raise SourcePowerBundleError(
            f"{label} must contain exactly {components} assignments"
        )
    index = {region_id: position for position, region_id in enumerate(region_ids)}
    try:
        return tuple(index[item] for item in value)
    except (KeyError, TypeError) as exc:
        raise SourcePowerBundleError(f"{label} contains an unknown region ID") from exc


def _validate_quadratic_exposure(matrix, counts, observed, label):
    try:
        observed = float(observed)
    except (TypeError, ValueError) as exc:
        raise SourcePowerBundleError(f"{label} must be finite numeric") from exc
    if not math.isfinite(observed):
        raise SourcePowerBundleError(f"{label} must be finite numeric")
    vector = np.asarray(counts, dtype=float)
    expected = float(vector @ np.asarray(matrix, dtype=float) @ vector)
    if not math.isclose(observed, expected, rel_tol=1e-11, abs_tol=1e-8):
        raise SourcePowerBundleError(
            f"{label} does not match counts.T @ exposure @ counts"
        )
    return observed


def _extract_allocation(
    full_row,
    summary_row,
    *,
    matrix,
    region_ids,
    components,
    label,
):
    if full_row.get("components_per_corpus") != components:
        raise SourcePowerBundleError(f"{label} component count mismatch")
    allocation = full_row.get("allocation")
    summary_allocation = summary_row.get("allocation")
    if not isinstance(allocation, dict) or not isinstance(summary_allocation, dict):
        raise SourcePowerBundleError(f"{label}.allocation must be an object")
    assignments = allocation.get("assignment_region_ids")
    counts_by_region = allocation.get("counts_by_region")
    capacities_by_region = full_row.get("capacities_by_region")
    _require_record(
        assignments,
        summary_allocation.get("assignment_region_ids_record"),
        f"{label}.assignment_region_ids_record",
    )
    _require_record(
        counts_by_region,
        summary_allocation.get("counts_by_region_record"),
        f"{label}.counts_by_region_record",
    )
    _require_record(
        capacities_by_region,
        summary_row.get("capacities_by_region_record"),
        f"{label}.capacities_by_region_record",
    )
    assignment_indices = _assignment_indices(
        assignments, region_ids, components, f"{label}.assignment_region_ids"
    )
    counts = _mapping_counts(
        counts_by_region, region_ids, f"{label}.counts_by_region"
    )
    capacities = _mapping_counts(
        capacities_by_region, region_ids, f"{label}.capacities_by_region"
    )
    reconstructed = Counter(assignment_indices)
    expected_counts = tuple(reconstructed.get(index, 0) for index in range(len(region_ids)))
    if counts != expected_counts or sum(counts) != components:
        raise SourcePowerBundleError(f"{label} assignment/count mismatch")
    if any(count > capacity for count, capacity in zip(counts, capacities)):
        raise SourcePowerBundleError(f"{label} exceeds a region capacity")
    used = sum(count > 0 for count in counts)
    if allocation.get("used_region_count") != used:
        raise SourcePowerBundleError(f"{label}.used_region_count mismatch")
    quadratic = _validate_quadratic_exposure(
        matrix,
        counts,
        allocation.get("quadratic_exposure"),
        f"{label}.quadratic_exposure",
    )
    return {
        "components_per_corpus": components,
        "assignment_region_indices": list(assignment_indices),
        "counts_by_region": list(counts),
        "capacities_by_region": list(capacities),
        "used_region_count": used,
        "quadratic_exposure": quadratic,
        "source_records": {
            "assignment_region_ids": copy.deepcopy(
                summary_allocation["assignment_region_ids_record"]
            ),
            "counts_by_region": copy.deepcopy(
                summary_allocation["counts_by_region_record"]
            ),
            "capacities_by_region": copy.deepcopy(
                summary_row["capacities_by_region_record"]
            ),
        },
    }


def _extract_design(full_audit, summary_audit, corpus, region_count):
    label = f"results.{corpus}.{region_count}"
    if not isinstance(full_audit, dict) or not isinstance(summary_audit, dict):
        raise SourcePowerBundleError(f"{label} must be an object")
    for key in ("target_region_count", "actual_region_count"):
        if full_audit.get(key) != region_count or summary_audit.get(key) != region_count:
            raise SourcePowerBundleError(f"{label}.{key} mismatch")
    exposure = full_audit.get("exposure")
    summary_exposure = summary_audit.get("exposure")
    if not isinstance(exposure, dict) or not isinstance(summary_exposure, dict):
        raise SourcePowerBundleError(f"{label}.exposure must be an object")
    if "matrix" not in exposure or "matrix" in summary_exposure:
        raise SourcePowerBundleError(
            f"{label} must pair an exact matrix with an omitted-matrix summary"
        )
    region_ids = _validated_region_ids(
        exposure.get("region_ids"), region_count, f"{label}.exposure.region_ids"
    )
    if summary_exposure.get("region_ids") != list(region_ids):
        raise SourcePowerBundleError(f"{label} summary/full region-ID mismatch")
    if summary_exposure.get("matrix_shape") != [region_count, region_count]:
        raise SourcePowerBundleError(f"{label} projected matrix shape mismatch")
    matrix = _validated_matrix(
        exposure["matrix"], region_count, f"{label}.exposure.matrix"
    )
    matrix_record = _require_record(
        matrix,
        summary_exposure.get("matrix_record"),
        f"{label}.exposure.matrix_record",
    )
    full_sizes = full_audit.get("registered_size_results")
    summary_sizes = summary_audit.get("registered_size_results")
    expected_keys = {str(value) for value in EXPECTED_COMPONENT_COUNTS}
    if not isinstance(full_sizes, dict) or set(full_sizes) != expected_keys:
        raise SourcePowerBundleError(f"{label} full size grid mismatch")
    if not isinstance(summary_sizes, dict) or set(summary_sizes) != expected_keys:
        raise SourcePowerBundleError(f"{label} summary size grid mismatch")
    allocations = {
        str(components): _extract_allocation(
            full_sizes[str(components)],
            summary_sizes[str(components)],
            matrix=matrix,
            region_ids=region_ids,
            components=components,
            label=f"{label}.registered_size_results.{components}",
        )
        for components in EXPECTED_COMPONENT_COUNTS
    }
    partition_record = _validated_record(
        full_audit.get("partition_assignment_record"),
        f"{label}.partition_assignment_record",
    )
    if summary_audit.get("partition_assignment_record") != partition_record:
        raise SourcePowerBundleError(f"{label} partition record mismatch")
    return {
        "region_count": region_count,
        "region_ids": list(region_ids),
        "exposure_matrix": matrix,
        "exposure_matrix_record": matrix_record,
        "partition_assignment_record": partition_record,
        "region_index_encoding": REGION_INDEX_ENCODING,
        "allocations": allocations,
    }


def build_source_design_bundle(full_bytes: bytes, summary_bytes: bytes) -> dict:
    """Validate the reviewed source payload and return its exact Stage-A bundle."""
    full = _loads(full_bytes, "full source-dependence payload")
    summary = _loads(summary_bytes, "reviewed source-dependence summary")
    parent = _validate_parent_contract(full, summary, full_bytes, summary_bytes)
    designs = {}
    for corpus in EXPECTED_CORPORA:
        full_corpus = full["results"][corpus]
        summary_corpus = summary["results"][corpus]
        expected_keys = {str(value) for value in EXPECTED_REGION_COUNTS}
        if not isinstance(full_corpus, dict) or set(full_corpus) != expected_keys:
            raise SourcePowerBundleError(f"{corpus} full region grid mismatch")
        if not isinstance(summary_corpus, dict) or set(summary_corpus) != expected_keys:
            raise SourcePowerBundleError(f"{corpus} summary region grid mismatch")
        designs[corpus] = {
            str(region_count): _extract_design(
                full_corpus[str(region_count)],
                summary_corpus[str(region_count)],
                corpus,
                region_count,
            )
            for region_count in EXPECTED_REGION_COUNTS
        }
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "parent": parent,
        "configuration": {
            "required_corpora": list(EXPECTED_CORPORA),
            "region_count_grid": list(EXPECTED_REGION_COUNTS),
            "components_per_corpus_grid": list(EXPECTED_COMPONENT_COUNTS),
            # The parent artifact predates the Stage-A notation split and
            # calls this source-dependence axis ``rho_grid``.  Rename it at
            # the bundle boundary so it can never be confused with the
            # within-component ``rho_item`` selector field.
            "source_eta_grid": copy.deepcopy(
                full["configuration"]["rho_grid"]
            ),
            "cumulative_walk_weights": copy.deepcopy(
                full["configuration"]["cumulative_walk_weights"]
            ),
            "source_region_cap_fraction": full["configuration"][
                "source_region_cap_fraction"
            ],
            "endpoints_charged_per_component": full["configuration"][
                "endpoints_charged_per_campaign_component"
            ],
            "source_regions_claimed_independent": False,
        },
        "input_identity": {
            "graph_bundle": copy.deepcopy(full["inputs"]["graph_bundle"]),
            "graphs": copy.deepcopy(full["inputs"]["graphs"]),
            "source_inputs_record": canonical_value_record(full["inputs"]),
            "outcomes_consumed": False,
            "historical_inventory_consumed": False,
            "nomic_cache_consumed": False,
            "candidate_pool_consumed": False,
            "judge_responses_consumed": False,
        },
        "designs": designs,
        "authorization": copy.deepcopy(AUTHORIZATION),
        "non_claims": list(NON_CLAIMS),
    }
    # Exercise the same loader invariants before a caller can write the bundle.
    _validate_bundle_against_summary(bundle, summary, summary_bytes)
    return bundle


def _validate_bundle_allocation(allocation, summary_row, matrix, region_ids, label):
    required = {
        "components_per_corpus",
        "assignment_region_indices",
        "counts_by_region",
        "capacities_by_region",
        "used_region_count",
        "quadratic_exposure",
        "source_records",
    }
    _require_exact_keys(allocation, required, label)
    components = _positive_int(allocation["components_per_corpus"], f"{label}.components")
    assignments = allocation["assignment_region_indices"]
    counts = allocation["counts_by_region"]
    capacities = allocation["capacities_by_region"]
    if not isinstance(assignments, list) or len(assignments) != components:
        raise SourcePowerBundleError(f"{label} assignment length mismatch")
    assignments = tuple(
        _nonnegative_int(value, f"{label}.assignment_region_indices[]")
        for value in assignments
    )
    if any(value >= len(region_ids) for value in assignments):
        raise SourcePowerBundleError(f"{label} assignment index out of range")
    if not isinstance(counts, list) or len(counts) != len(region_ids):
        raise SourcePowerBundleError(f"{label} count-vector length mismatch")
    if not isinstance(capacities, list) or len(capacities) != len(region_ids):
        raise SourcePowerBundleError(f"{label} capacity-vector length mismatch")
    counts = tuple(_nonnegative_int(value, f"{label}.counts[]") for value in counts)
    capacities = tuple(
        _nonnegative_int(value, f"{label}.capacities[]") for value in capacities
    )
    reconstructed = Counter(assignments)
    if counts != tuple(reconstructed.get(i, 0) for i in range(len(region_ids))):
        raise SourcePowerBundleError(f"{label} assignment/count mismatch")
    if any(count > capacity for count, capacity in zip(counts, capacities)):
        raise SourcePowerBundleError(f"{label} exceeds capacity")
    if allocation["used_region_count"] != sum(value > 0 for value in counts):
        raise SourcePowerBundleError(f"{label} used-region mismatch")
    _validate_quadratic_exposure(
        matrix, counts, allocation["quadratic_exposure"], f"{label}.quadratic_exposure"
    )
    records = allocation["source_records"]
    _require_exact_keys(
        records,
        ("assignment_region_ids", "counts_by_region", "capacities_by_region"),
        f"{label}.source_records",
    )
    assignment_ids = [region_ids[index] for index in assignments]
    count_mapping = {region_id: counts[i] for i, region_id in enumerate(region_ids)}
    capacity_mapping = {
        region_id: capacities[i] for i, region_id in enumerate(region_ids)
    }
    summary_allocation = summary_row["allocation"]
    expected_records = {
        "assignment_region_ids": summary_allocation["assignment_region_ids_record"],
        "counts_by_region": summary_allocation["counts_by_region_record"],
        "capacities_by_region": summary_row["capacities_by_region_record"],
    }
    values = {
        "assignment_region_ids": assignment_ids,
        "counts_by_region": count_mapping,
        "capacities_by_region": capacity_mapping,
    }
    for key in values:
        if records[key] != expected_records[key]:
            raise SourcePowerBundleError(f"{label}.{key} reviewed record mismatch")
        _require_record(values[key], records[key], f"{label}.{key}")


def _validate_bundle_against_summary(bundle, summary, summary_bytes):
    top_keys = {
        "schema_version",
        "algorithm",
        "parent",
        "configuration",
        "input_identity",
        "designs",
        "authorization",
        "non_claims",
    }
    _require_exact_keys(bundle, top_keys, "bundle")
    if bundle["schema_version"] != SCHEMA_VERSION or bundle["algorithm"] != ALGORITHM:
        raise SourcePowerBundleError("unsupported source-design bundle schema/algorithm")
    parent = bundle["parent"]
    _require_exact_keys(
        parent,
        (
            "source_dependence_algorithm",
            "source_dependence_schema_version",
            "full_payload_record",
            "reviewed_summary_record",
        ),
        "bundle.parent",
    )
    if parent["source_dependence_algorithm"] != PARENT_ALGORITHM:
        raise SourcePowerBundleError("bundle parent algorithm mismatch")
    if parent["source_dependence_schema_version"] != summary.get("schema_version"):
        raise SourcePowerBundleError("bundle parent schema mismatch")
    if parent["full_payload_record"] != summary.get("full_payload_record"):
        raise SourcePowerBundleError("bundle parent payload record mismatch")
    if parent["reviewed_summary_record"] != content_record(summary_bytes):
        raise SourcePowerBundleError("bundle reviewed-summary record mismatch")
    configuration = bundle["configuration"]
    _require_exact_keys(
        configuration,
        (
            "required_corpora",
            "region_count_grid",
            "components_per_corpus_grid",
            "source_eta_grid",
            "cumulative_walk_weights",
            "source_region_cap_fraction",
            "endpoints_charged_per_component",
            "source_regions_claimed_independent",
        ),
        "bundle.configuration",
    )
    if configuration.get("required_corpora") != list(EXPECTED_CORPORA):
        raise SourcePowerBundleError("bundle corpus grid mismatch")
    if configuration.get("region_count_grid") != list(EXPECTED_REGION_COUNTS):
        raise SourcePowerBundleError("bundle region grid mismatch")
    if configuration.get("components_per_corpus_grid") != list(EXPECTED_COMPONENT_COUNTS):
        raise SourcePowerBundleError("bundle component grid mismatch")
    _exact_float_grid(
        configuration.get("source_eta_grid"),
        EXPECTED_SOURCE_ETA_GRID,
        "bundle.configuration.source_eta_grid",
    )
    source_configuration = summary["configuration"]
    comparisons = {
        "source_eta_grid": "rho_grid",
        "cumulative_walk_weights": "cumulative_walk_weights",
        "source_region_cap_fraction": "source_region_cap_fraction",
        "endpoints_charged_per_component": "endpoints_charged_per_campaign_component",
    }
    for bundle_key, source_key in comparisons.items():
        if configuration.get(bundle_key) != source_configuration.get(source_key):
            raise SourcePowerBundleError(f"bundle configuration {bundle_key} mismatch")
    if configuration.get("source_regions_claimed_independent") is not False:
        raise SourcePowerBundleError("bundle must not claim source independence")
    input_identity = bundle["input_identity"]
    _require_exact_keys(
        input_identity,
        (
            "graph_bundle",
            "graphs",
            "source_inputs_record",
            "outcomes_consumed",
            "historical_inventory_consumed",
            "nomic_cache_consumed",
            "candidate_pool_consumed",
            "judge_responses_consumed",
        ),
        "bundle.input_identity",
    )
    source_inputs = summary["inputs"]
    if input_identity.get("graph_bundle") != source_inputs.get("graph_bundle"):
        raise SourcePowerBundleError("bundle graph-bundle identity mismatch")
    if input_identity.get("graphs") != source_inputs.get("graphs"):
        raise SourcePowerBundleError("bundle graph identity mismatch")
    if input_identity.get("source_inputs_record") != canonical_value_record(source_inputs):
        raise SourcePowerBundleError("bundle source-input record mismatch")
    for key in (
        "outcomes_consumed",
        "historical_inventory_consumed",
        "nomic_cache_consumed",
        "candidate_pool_consumed",
        "judge_responses_consumed",
    ):
        if input_identity.get(key) is not False:
            raise SourcePowerBundleError(f"bundle {key} must remain false")
    if set(bundle["designs"]) != set(EXPECTED_CORPORA):
        raise SourcePowerBundleError("bundle must contain exactly both corpus designs")
    design_count = allocation_count = 0
    for corpus in EXPECTED_CORPORA:
        expected_region_keys = {str(value) for value in EXPECTED_REGION_COUNTS}
        corpus_designs = bundle["designs"][corpus]
        if not isinstance(corpus_designs, dict) or set(corpus_designs) != expected_region_keys:
            raise SourcePowerBundleError(f"bundle {corpus} region grid mismatch")
        for region_count in EXPECTED_REGION_COUNTS:
            design_count += 1
            key = str(region_count)
            design = corpus_designs[key]
            required = {
                "region_count",
                "region_ids",
                "exposure_matrix",
                "exposure_matrix_record",
                "partition_assignment_record",
                "region_index_encoding",
                "allocations",
            }
            _require_exact_keys(design, required, f"bundle.designs.{corpus}.{key}")
            if design["region_count"] != region_count:
                raise SourcePowerBundleError("bundle design region count mismatch")
            region_ids = _validated_region_ids(
                design["region_ids"], region_count, f"bundle.{corpus}.{key}.region_ids"
            )
            matrix = _validated_matrix(
                design["exposure_matrix"], region_count, f"bundle.{corpus}.{key}.matrix"
            )
            summary_audit = summary["results"][corpus][key]
            if list(region_ids) != summary_audit["exposure"].get("region_ids"):
                raise SourcePowerBundleError("bundle region IDs mismatch reviewed summary")
            if summary_audit["exposure"].get("matrix_shape") != [
                region_count,
                region_count,
            ]:
                raise SourcePowerBundleError("reviewed matrix shape mismatch")
            expected_matrix_record = summary_audit["exposure"]["matrix_record"]
            if design["exposure_matrix_record"] != expected_matrix_record:
                raise SourcePowerBundleError("bundle matrix reviewed record mismatch")
            _require_record(matrix, design["exposure_matrix_record"], "bundle matrix record")
            if design["partition_assignment_record"] != summary_audit[
                "partition_assignment_record"
            ]:
                raise SourcePowerBundleError("bundle partition record mismatch")
            if design["region_index_encoding"] != REGION_INDEX_ENCODING:
                raise SourcePowerBundleError("bundle region-index encoding mismatch")
            allocations = design["allocations"]
            expected_size_keys = {str(value) for value in EXPECTED_COMPONENT_COUNTS}
            if not isinstance(allocations, dict) or set(allocations) != expected_size_keys:
                raise SourcePowerBundleError("bundle allocation grid mismatch")
            for components in EXPECTED_COMPONENT_COUNTS:
                allocation_count += 1
                size_key = str(components)
                allocation = allocations[size_key]
                if allocation.get("components_per_corpus") != components:
                    raise SourcePowerBundleError("bundle allocation component mismatch")
                _validate_bundle_allocation(
                    allocation,
                    summary_audit["registered_size_results"][size_key],
                    matrix,
                    region_ids,
                    f"bundle.{corpus}.{key}.{size_key}",
                )
    if design_count != 6 or allocation_count != 24:
        raise SourcePowerBundleError("bundle must contain six designs and 24 allocations")
    if bundle["authorization"] != AUTHORIZATION:
        raise SourcePowerBundleError("source bundle cannot unlock downstream work")
    if bundle["non_claims"] != NON_CLAIMS:
        raise SourcePowerBundleError("bundle non-claims mismatch")


def load_source_design_bundle(path, summary_path=DEFAULT_SUMMARY_PATH) -> dict:
    """Load and validate a canonical exact source-design bundle.

    The returned dictionary is safe to pass to the Stage-A scientific module.
    Its canonical fingerprint input is ``source_design_bundle_identity(result)``.
    """
    bundle_bytes = Path(path).read_bytes()
    summary_bytes = Path(summary_path).read_bytes()
    bundle = _loads(bundle_bytes, "source-design bundle")
    if "artifact_projection" in bundle:
        raise SourcePowerBundleError("a projection cannot be used as a source-design bundle")
    if canonical_json_bytes(bundle) != bundle_bytes:
        raise SourcePowerBundleError("source-design bundle bytes are not canonical")
    summary = _loads(summary_bytes, "reviewed source-dependence summary")
    _validate_bundle_against_summary(bundle, summary, summary_bytes)
    return bundle


def write_source_design_bundle(path, bundle) -> dict:
    """Atomically write canonical bundle bytes and return their identity record."""
    data = canonical_json_bytes(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return content_record(data)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, help="exact unprojected source payload")
    parser.add_argument(
        "--summary",
        default=str(DEFAULT_SUMMARY_PATH),
        help="reviewed tracked-summary-v1 artifact",
    )
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Iterable[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    full_bytes = Path(args.full).read_bytes()
    summary_bytes = Path(args.summary).read_bytes()
    bundle = build_source_design_bundle(full_bytes, summary_bytes)
    record = write_source_design_bundle(args.out, bundle)
    # Reload from disk so generation cannot bypass canonical loader validation.
    loaded = load_source_design_bundle(args.out, args.summary)
    if source_design_bundle_identity(loaded) != record:
        raise SourcePowerBundleError("written bundle identity changed on reload")
    print(
        json.dumps(
            {
                "bundle_record": record,
                "design_count": 6,
                "allocation_count": 24,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return bundle


if __name__ == "__main__":
    main()
