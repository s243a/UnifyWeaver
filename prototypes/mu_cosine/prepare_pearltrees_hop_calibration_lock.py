#!/usr/bin/env python3
"""Calibrate the frozen Pearltrees HOP plan and install its audit lock.

The command consumes only an accepted, fully reverified no-solve HOP plan and
the structural source chain that produced it.  Numerical work is restricted to
the eight registered calibration batches.  It freezes one global topology
leakage and one prospective audit mode; it never executes an audit solve.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import prepare_pearltrees_hop_plan as hop_plan  # noqa: E402
from unifyweaver.graph import local_diffusion as local  # noqa: E402
from unifyweaver.graph.bounded_diffusion_fidelity import (  # noqa: E402
    BoundedDomainSelection,
    BoundedFidelityResult,
    evaluate_nested_bounded_domain_fidelity,
)


SCHEMA = "pearltrees-hop-calibration-lock-v1"
ALGORITHM = "calibration-only-global-alpha-hop-lock-v1"
WORK_ORDER_SCHEMA = "pearltrees-hop-calibration-work-order-v1"
MARKER_NAME = "LOCAL_ONLY_DO_NOT_PUBLISH"
MARKER_BYTES = b"LOCAL ONLY - DO NOT PUBLISH HOP CALIBRATION ARTIFACTS\n"
MANIFEST_NAME = "manifest.json"
SCIENTIFIC_ARTIFACT_NAMES = (
    "calibration_work_order.json",
    "anchor_calibrations.jsonl",
    "batch_calibration.jsonl",
    "calibration_fidelity.jsonl",
    "selection.json",
)
OBSERVATIONAL_ARTIFACT_NAMES = ("execution.json",)
ARTIFACT_NAMES = SCIENTIFIC_ARTIFACT_NAMES + OBSERVATIONAL_ARTIFACT_NAMES
ALL_LOCK_FILES = frozenset(ARTIFACT_NAMES + (MANIFEST_NAME, MARKER_NAME))
ROLE_ORDER = ("S_256", "S_512", "S_1024", "R_top")
CANDIDATE_ROLES = ROLE_ORDER[:-1]
EXPECTED_CALIBRATION_BATCHES = 8
EXPECTED_CALIBRATION_ANCHORS = 32
EXPECTED_ANCHORS_PER_BATCH = 4
EXPECTED_QUARTILES = ("q1", "q2", "q3", "q4")
MAXIMUM_LOCK_MANIFEST_BYTES = 8 * 1024 * 1024
MAXIMUM_LOCK_ARTIFACT_BYTES = {
    "anchor_calibrations.jsonl": 8 * 1024 * 1024,
    "batch_calibration.jsonl": 8 * 1024 * 1024,
    "calibration_fidelity.jsonl": 16 * 1024 * 1024,
    "calibration_work_order.json": 128 * 1024 * 1024,
    "execution.json": 4 * 1024 * 1024,
    "selection.json": 4 * 1024 * 1024,
}
LOCK_DESIGN_PATH = HERE / "DESIGN_pearltrees_hop_calibration_lock.md"
AUDIT_DESIGN_PATH = HERE / "DESIGN_pearltrees_hop_audit.md"
AUDIT_RUNNER_PATH = HERE / "prepare_pearltrees_hop_audit.py"
PROTOCOL_PATH = HERE / "PROTOCOL_bounded_diffusion_fidelity.md"
PLAN_DESIGN_PATH = HERE / "DESIGN_pearltrees_hop_plan.md"
LOCAL_DIFFUSION_PATH = REPO_ROOT / "src/unifyweaver/graph/local_diffusion.py"
FIDELITY_PATH = REPO_ROOT / "src/unifyweaver/graph/bounded_diffusion_fidelity.py"


class CalibrationLockError(ValueError):
    """Fail-closed calibration integrity or operational error."""


def _canonical_json(value):
    try:
        return hop_plan._canonical_json(value)
    except hop_plan.HopPlanError as exc:
        raise CalibrationLockError("value is not canonical finite JSON") from exc


def _jsonl_bytes(records):
    return b"".join(_canonical_json(record) for record in records)


def _strict_json_bytes(data, label):
    try:
        return hop_plan._strict_json_bytes(data, label)
    except hop_plan.HopPlanError as exc:
        raise CalibrationLockError(f"invalid canonical JSON in {label}") from exc


def _strict_jsonl_bytes(data, label):
    try:
        return hop_plan._strict_jsonl_bytes(data, label)
    except hop_plan.HopPlanError as exc:
        raise CalibrationLockError(f"invalid canonical JSONL in {label}") from exc


def _content_record(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _content_record_is_valid(value):
    return (
        isinstance(value, dict)
        and set(value) == {"sha256", "size_bytes"}
        and isinstance(value["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None
        and isinstance(value["size_bytes"], int)
        and not isinstance(value["size_bytes"], bool)
        and value["size_bytes"] >= 0
    )


def _file_record(path):
    data = Path(path).read_bytes()
    return _content_record(data)


def _head_blob_record(repo_relative_path):
    try:
        data = subprocess.check_output(
            ["git", "show", f"HEAD:{repo_relative_path}"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CalibrationLockError(
            "required implementation is not present at HEAD"
        ) from exc
    return _content_record(data)


def _assert_records_at_head(records):
    for path, record in records.items():
        if _head_blob_record(path) != record:
            raise CalibrationLockError(
                "scientific implementation differs from repository HEAD"
            )


def _git_commit():
    try:
        value = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            .lower()
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CalibrationLockError("repository commit is unavailable") from exc
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise CalibrationLockError("repository commit is malformed")
    return value


def _portable_blas_runtime(records):
    identity_fields = (
        "user_api",
        "internal_api",
        "prefix",
        "version",
        "threading_layer",
        "architecture",
    )
    output = []
    for record in records:
        if record.get("user_api") != "blas":
            continue
        row = {field: record.get(field) for field in identity_fields}
        row["observed_num_threads"] = record.get("num_threads")
        output.append(row)
    return sorted(
        output,
        key=lambda row: tuple(str(row[field]) for field in identity_fields),
    )


@contextmanager
def _single_blas_thread():
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError as exc:
        raise CalibrationLockError(
            "threadpoolctl is required to prove the BLAS thread contract"
        ) from exc
    with threadpool_limits(limits=1, user_api="blas"):
        # Force both registered LAPACK paths to load before identity capture.
        np.linalg.cholesky(np.eye(2, dtype=np.float64))
        np.linalg.eigh(np.eye(2, dtype=np.float64))
        runtime = _portable_blas_runtime(threadpool_info())
        if not runtime:
            raise CalibrationLockError("actual BLAS identity is unavailable")
        if any(row["observed_num_threads"] != 1 for row in runtime):
            raise CalibrationLockError("BLAS thread contract is not one")
        if any("filepath" in row for row in runtime):
            raise CalibrationLockError("BLAS identity contains a machine path")
        yield runtime


def _read_bound_bytes(binding, name, expected, label, *, maximum_size=None):
    try:
        return hop_plan._read_bound_file(
            binding,
            name,
            expected,
            label,
            maximum_size=maximum_size,
        )
    except hop_plan.HopPlanError as exc:
        raise CalibrationLockError(f"{label} changed or is malformed") from exc


def _capture_verified_context(
    plan_dir,
    *,
    receipt_dir,
    attempt_a_dir,
    attempt_b_dir,
    source_spec,
    relation_policy,
):
    """Fully reverify upstream state and capture the exact calibration bytes."""

    try:
        manifest = hop_plan.verify_plan(
            plan_dir,
            receipt_dir=receipt_dir,
            attempt_a_dir=attempt_a_dir,
            attempt_b_dir=attempt_b_dir,
            source_spec=source_spec,
            relation_policy=relation_policy,
        )
    except hop_plan.HopPlanError as exc:
        raise CalibrationLockError("full HOP plan verification failed") from exc
    if (
        manifest.get("accepted") is not True
        or manifest.get("calibration_solve_authorized") is not True
        or manifest.get("audit_solve_authorized") is not False
        or manifest.get("diffusion_or_fidelity_metrics_computed") is not False
    ):
        raise CalibrationLockError("HOP plan does not authorize calibration")

    plan_binding = hop_plan._bind_directory(plan_dir, "HOP plan")
    attempt_binding = hop_plan._bind_directory(attempt_a_dir, "canonical attempt A")
    try:
        manifest_data = _read_bound_bytes(
            plan_binding,
            hop_plan.MANIFEST_NAME,
            None,
            "HOP plan manifest",
            maximum_size=hop_plan.MAXIMUM_PLAN_MANIFEST_BYTES,
        )
        if _strict_json_bytes(manifest_data, "HOP plan manifest") != manifest:
            raise CalibrationLockError("HOP plan manifest changed after verification")
        artifact_records = manifest["fingerprint_core"]["artifact_records"]
        required = (
            "batches.jsonl",
            "domains.jsonl",
            "boundaries.jsonl",
            "calibration_shells.jsonl",
        )
        plan_artifacts = {
            name: _read_bound_bytes(
                plan_binding,
                name,
                artifact_records[name],
                "HOP plan",
            )
            for name in required
        }
        bindings = manifest["fingerprint_core"]["input_bindings"]
        attempt_manifest_data = _read_bound_bytes(
            attempt_binding,
            "manifest.json",
            bindings["canonical_attempt_a_manifest"],
            "canonical attempt A",
        )
        planning_records = bindings["planning_graph_artifacts"]
        adjacency_data = _read_bound_bytes(
            attempt_binding,
            "adjacency.jsonl",
            planning_records["adjacency.jsonl"],
            "canonical attempt A",
        )
        eligibility_data = _read_bound_bytes(
            attempt_binding,
            "anchor_eligibility.jsonl",
            planning_records["anchor_eligibility.jsonl"],
            "canonical attempt A",
        )
        try:
            _attempt_manifest, adjacency, _eligible = hop_plan._load_graph(
                attempt_manifest_data,
                adjacency_data,
                eligibility_data,
            )
        except hop_plan.HopPlanError as exc:
            raise CalibrationLockError(
                "canonical calibration adjacency failed verification"
            ) from exc
        captured_records = {
            "plan_manifest": _content_record(manifest_data),
            "plan_artifacts": {
                name: _content_record(data) for name, data in plan_artifacts.items()
            },
            "attempt_a_manifest": _content_record(attempt_manifest_data),
            "adjacency": _content_record(adjacency_data),
            "eligibility": _content_record(eligibility_data),
        }
        capture_fingerprint = hashlib.sha256(
            _canonical_json(captured_records)
        ).hexdigest()
        hop_plan._assert_private_directory_files(plan_binding, "HOP plan")
        hop_plan._assert_private_directory_files(
            attempt_binding, "canonical attempt A"
        )
        return {
            "adjacency": adjacency,
            "capture_fingerprint": capture_fingerprint,
            "captured_records": captured_records,
            "manifest": manifest,
            "manifest_data": manifest_data,
            "plan_artifacts": plan_artifacts,
        }
    finally:
        plan_binding.close()
        attempt_binding.close()


def _unique_index(records, keys, label):
    output = {}
    for record in records:
        try:
            key = tuple(record[name] for name in keys)
        except (KeyError, TypeError) as exc:
            raise CalibrationLockError(f"{label} row is malformed") from exc
        if key in output:
            raise CalibrationLockError(f"{label} contains duplicate rows")
        output[key] = record
    return output


def _derive_calibration_work_order(context):
    """Produce the only node-level object supplied to numerical calibration."""

    artifacts = context["plan_artifacts"]
    batches = _strict_jsonl_bytes(artifacts["batches.jsonl"], "batches")
    domains = _strict_jsonl_bytes(artifacts["domains.jsonl"], "domains")
    boundaries = _strict_jsonl_bytes(
        artifacts["boundaries.jsonl"], "boundaries"
    )
    shells = _strict_jsonl_bytes(
        artifacts["calibration_shells.jsonl"], "calibration shells"
    )
    calibration_batches = [row for row in batches if row.get("split") == "calibration"]
    if len(calibration_batches) != EXPECTED_CALIBRATION_BATCHES:
        raise CalibrationLockError("calibration work order must contain eight batches")
    if any(row.get("split") not in {"calibration", "audit"} for row in batches):
        raise CalibrationLockError("HOP plan contains an unknown split")

    domain_index = _unique_index(domains, ("batch_id", "role"), "domains")
    boundary_index = _unique_index(
        boundaries, ("batch_id", "role"), "boundaries"
    )
    shell_index = _unique_index(
        shells, ("batch_id", "anchor_node_id"), "calibration shells"
    )
    adjacency = context["adjacency"]
    work_batches = []
    all_anchors = set()
    for expected_index, batch in enumerate(
        sorted(calibration_batches, key=lambda row: row.get("batch_index", -1))
    ):
        if (
            batch.get("batch_index") != expected_index
            or batch.get("batch_id") != f"calibration-{expected_index:02d}"
            or batch.get("split") != "calibration"
        ):
            raise CalibrationLockError("calibration batch identity mismatch")
        anchors_by_quartile = batch.get("anchors_by_quartile")
        if not isinstance(anchors_by_quartile, list) or len(anchors_by_quartile) != 4:
            raise CalibrationLockError("calibration batch must contain four anchors")
        quartiles = tuple(row.get("quartile_id") for row in anchors_by_quartile)
        anchors = tuple(row.get("node_id") for row in anchors_by_quartile)
        if quartiles != EXPECTED_QUARTILES or len(set(anchors)) != 4:
            raise CalibrationLockError("calibration batch is not quartile-balanced")
        if any(not isinstance(node, str) or node not in adjacency for node in anchors):
            raise CalibrationLockError("calibration anchor is absent from adjacency")
        if all_anchors.intersection(anchors):
            raise CalibrationLockError("calibration anchors must be globally unique")
        all_anchors.update(anchors)

        role_rows = []
        previous_nodes = None
        for role in ROLE_ORDER:
            key = (batch["batch_id"], role)
            try:
                domain = domain_index[key]
                boundary = boundary_index[key]
            except KeyError as exc:
                raise CalibrationLockError("calibration role is missing") from exc
            nodes = domain.get("nodes")
            if not isinstance(nodes, list) or not nodes:
                raise CalibrationLockError("calibration domain is malformed")
            node_ids = tuple(row.get("node_id") for row in nodes)
            if len(node_ids) != len(set(node_ids)) or any(
                node not in adjacency for node in node_ids
            ):
                raise CalibrationLockError("calibration domain nodes are malformed")
            if previous_nodes is not None and tuple(node_ids[: len(previous_nodes)]) != previous_nodes:
                raise CalibrationLockError("calibration HOP domains are not nested")
            previous_nodes = node_ids
            expected_boundary = hop_plan._boundary_record(
                batch["batch_id"],
                role,
                domain["requested_nodes"],
                node_ids,
                adjacency,
            )
            if boundary != expected_boundary:
                raise CalibrationLockError("calibration boundary ledger mismatch")
            role_rows.append({"boundary": boundary, "domain": domain, "role": role})

        shell_rows = []
        for quartile_id, anchor in zip(quartiles, anchors):
            try:
                shell = shell_index[(batch["batch_id"], anchor)]
            except KeyError as exc:
                raise CalibrationLockError("calibration shell is missing") from exc
            if (
                shell.get("strictly_interior_pass") is not True
                or shell.get("reasons") != []
                or shell.get("hop_radius") != 3
                or shell.get("target_attenuation") != "exp(-1)"
                or not isinstance(shell.get("shell_nodes"), list)
                or not shell["shell_nodes"]
            ):
                raise CalibrationLockError("calibration shell contract mismatch")
            reference_nodes = {row["node_id"] for row in role_rows[-1]["domain"]["nodes"]}
            if any(node not in reference_nodes for node in shell["shell_nodes"]):
                raise CalibrationLockError("calibration shell leaves its reference")
            shell_rows.append({**shell, "quartile_id": quartile_id})

        work_batches.append(
            {
                "anchors_by_quartile": anchors_by_quartile,
                "batch_id": batch["batch_id"],
                "batch_index": batch["batch_index"],
                "protected_nodes": batch["protected_nodes"],
                "roles": role_rows,
                "shells": shell_rows,
                "split": "calibration",
            }
        )

    if len(all_anchors) != EXPECTED_CALIBRATION_ANCHORS:
        raise CalibrationLockError("calibration work order must contain 32 anchors")
    calibration_nodes = {
        row["node_id"]
        for batch in work_batches
        for role in batch["roles"]
        for row in role["domain"]["nodes"]
    }
    incident_adjacency = [
        {"neighbors": list(adjacency[node]), "node_id": node}
        for node in sorted(calibration_nodes, key=hop_plan._typed_id_key)
    ]
    work_order = {
        "algorithm": "frozen-plan-calibration-only-projection-v1",
        "batch_count": len(work_batches),
        "batches": work_batches,
        "calibration_anchor_count": len(all_anchors),
        "contains_audit_metrics_or_responses": False,
        "incident_adjacency": incident_adjacency,
        "plan_fingerprint": context["manifest"]["plan_fingerprint"],
        "plan_manifest_record": context["captured_records"]["plan_manifest"],
        "schema": WORK_ORDER_SCHEMA,
    }
    if any(batch["split"] != "calibration" for batch in work_order["batches"]):
        raise CalibrationLockError("audit row entered the numerical work order")
    return work_order


def _adjacency_from_work_order(work_order):
    rows = work_order.get("incident_adjacency")
    if not isinstance(rows, list) or not rows:
        raise CalibrationLockError("calibration incident adjacency is missing")
    adjacency = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"neighbors", "node_id"}:
            raise CalibrationLockError("calibration incident adjacency is malformed")
        node = row["node_id"]
        neighbors = row["neighbors"]
        if (
            not isinstance(node, str)
            or node in adjacency
            or not isinstance(neighbors, list)
            or len(neighbors) != len(set(neighbors))
        ):
            raise CalibrationLockError("calibration incident adjacency is malformed")
        adjacency[node] = tuple(neighbors)
    required = {
        item["node_id"]
        for batch in work_order["batches"]
        for role in batch["roles"]
        for item in role["domain"]["nodes"]
    }
    if set(adjacency) != required:
        raise CalibrationLockError("calibration adjacency projection is not exact")
    return adjacency


def _role_item(batch, role):
    matches = [item for item in batch["roles"] if item["role"] == role]
    if len(matches) != 1:
        raise CalibrationLockError("calibration role lookup is ambiguous")
    return matches[0]


def _domain_from_role(batch, role, adjacency):
    item = _role_item(batch, role)
    record = item["domain"]
    anchors = tuple(
        sorted(
            (row["node_id"] for row in batch["anchors_by_quartile"]),
            key=hop_plan._typed_id_key,
        )
    )
    nodes = tuple(row["node_id"] for row in record["nodes"])
    distances = np.asarray(
        [row["hop_distance"] for row in record["nodes"]], dtype=np.int64
    )
    return local.LocalDiffusionDomain(
        nodes=nodes,
        anchors=anchors,
        hop_distance=distances,
        neighbors=tuple(adjacency[node] for node in nodes),
        maximum_nodes=record["requested_nodes"],
        complete_distance_shell=(record["truncated_final_shell_nodes"] == 0),
        truncated_tie_count=record["truncated_final_shell_nodes"],
        selection_metric="frozen_typed_id_hop_prefix",
    )


def _selection_from_role(batch, role, adjacency, plan_fingerprint):
    item = _role_item(batch, role)
    domain = _domain_from_role(batch, role, adjacency)
    fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "boundary": item["boundary"],
                "domain": item["domain"],
                "plan_fingerprint": plan_fingerprint,
            }
        )
    ).hexdigest()
    return BoundedDomainSelection(
        domain=domain,
        strategy=f"frozen_hop_{role}",
        requested_nodes=item["domain"]["requested_nodes"],
        selection_distance=np.asarray(domain.hop_distance, dtype=float),
        selector_parameters=(("frozen_role", role),),
        provider_calls=0,
        maximum_touched_degree=max(len(adjacency[node]) for node in domain.nodes),
        selection_seconds=0.0,
        selection_fingerprint=fingerprint,
    )


def _relative_residual(precision, response, rhs):
    numerator = float(np.linalg.norm(precision @ response - rhs))
    denominator = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    return numerator / denominator


def _zero_alpha_precheck(
    domain,
    *,
    minimum_rcond,
    solve_tolerance,
    cholesky_rtol,
    cholesky_atol,
    m_matrix_tolerance,
    maximum_principle_tolerance,
):
    """Prove the reference needs no numerical repair before calibration."""

    try:
        model = local.build_local_grounded_semantic_diffusion(
            domain,
            intrinsic_leakage_conductance=0.0,
            bath_temperature=0.0,
            minimum_reciprocal_condition=minimum_rcond,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise np.linalg.LinAlgError(
            "zero-alpha reference is not numerically admissible"
        ) from exc
    precision = np.asarray(model.precision)
    off_diagonal = precision - np.diag(np.diag(precision))
    if float(np.max(off_diagonal, initial=0.0)) > m_matrix_tolerance:
        raise np.linalg.LinAlgError("zero-alpha precision is not an M-matrix")
    reconstructed = model.model.precision_root.T @ model.model.precision_root
    if not np.allclose(
        reconstructed,
        precision,
        rtol=cholesky_rtol,
        atol=cholesky_atol,
    ):
        raise np.linalg.LinAlgError("zero-alpha Cholesky reconstruction failed")
    reconstruction_error = float(
        np.linalg.norm(reconstructed - precision)
        / max(float(np.linalg.norm(precision)), np.finfo(float).tiny)
    )
    index = {node: row for row, node in enumerate(domain.nodes)}
    rhs = np.zeros((len(domain.nodes), len(domain.anchors)), dtype=float)
    for column, anchor in enumerate(domain.anchors):
        rhs[index[anchor], column] = 1.0
    response = model.equilibrium_response(rhs)
    residual = _relative_residual(precision, response, rhs)
    if not math.isfinite(residual) or residual > solve_tolerance:
        raise np.linalg.LinAlgError("zero-alpha solve residual exceeds tolerance")
    maximum_kirchhoff_error = 0.0
    maximum_principle_violation = 0.0
    for column, anchor in enumerate(domain.anchors):
        diagonal = float(response[index[anchor], column])
        if diagonal <= 0.0:
            raise np.linalg.LinAlgError("zero-alpha maximum principle failed")
        normalized = response[:, column] / diagonal
        violation = max(
            0.0,
            -float(np.min(normalized)),
            float(np.max(normalized)) - 1.0,
        )
        maximum_principle_violation = max(
            maximum_principle_violation, violation
        )
        if violation > maximum_principle_tolerance:
            raise np.linalg.LinAlgError("zero-alpha maximum principle failed")
        source = rhs[:, column]
        cut_fraction = model.cut_current_fraction(source)
        if not 0.0 <= cut_fraction <= 1.0:
            raise np.linalg.LinAlgError("zero-alpha cut current left [0, 1]")
        balance = float(model.cut_conductance @ response[:, column])
        kirchhoff_error = abs(balance - 1.0) / max(1.0, abs(balance))
        maximum_kirchhoff_error = max(maximum_kirchhoff_error, kirchhoff_error)
        if kirchhoff_error > solve_tolerance:
            raise np.linalg.LinAlgError("zero-alpha Kirchhoff balance failed")
    boundary_harmonic = model.boundary_harmonic_measure()
    return {
        "boundary_harmonic_max": float(
            np.max(boundary_harmonic, initial=0.0)
        ).hex(),
        "maximum_precision_eigenvalue": float(
            model.model.maximum_precision_eigenvalue
        ).hex(),
        "maximum_kirchhoff_relative_error": float(
            maximum_kirchhoff_error
        ).hex(),
        "maximum_principle_violation": float(
            maximum_principle_violation
        ).hex(),
        "minimum_precision_eigenvalue": float(
            model.model.minimum_precision_eigenvalue
        ).hex(),
        "reciprocal_condition": float(
            model.model.reciprocal_condition_number
        ).hex(),
        "solve_relative_residual": float(residual).hex(),
        "cholesky_reconstruction_relative_error": float(
            reconstruction_error
        ).hex(),
        "status": "pass_without_added_leakage",
    }


def _higher(values, q):
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) for value in values):
        raise CalibrationLockError("upper-tail values must be finite and nonempty")
    return float(np.quantile(values, q, method="higher"))


def _lower(values, q):
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) for value in values):
        raise CalibrationLockError("lower-tail values must be finite and nonempty")
    return float(np.quantile(values, q, method="lower"))


def _float_or_none_hex(value):
    return None if value is None else float(value).hex()


def _strip_timing_fields(value):
    return {
        key: item
        for key, item in value.items()
        if not key.endswith("_seconds")
    }


def _hexify_scientific(value):
    """Encode binary64 decisions exactly while retaining canonical structure."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            raise CalibrationLockError("scientific output contains a nonfinite float")
        return value.hex()
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {
            str(key): _hexify_scientific(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_hexify_scientific(item) for item in value]
    if value is None or isinstance(value, str):
        return value
    raise CalibrationLockError(
        f"unsupported scientific output type: {type(value).__name__}"
    )


def _require_exact_mapping(value, expected, label):
    if not isinstance(value, dict):
        raise CalibrationLockError(f"{label} is malformed")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise CalibrationLockError(f"{label} changed at {key}")


def _validated_bound_contract(manifest):
    """Extract only the prospective contract that this runner implements."""

    try:
        core = manifest["fingerprint_core"]
        calibration = core["calibration_contract"]
        numeric = core["numeric_backend_contract"]
        statistics = core["statistical_contract"]
        resources = core["resource_contract"]
    except (KeyError, TypeError) as exc:
        raise CalibrationLockError("HOP plan contract is incomplete") from exc
    if core.get("repository_commit") != _git_commit():
        raise CalibrationLockError(
            "HOP plan was not generated at this implementation commit"
        )
    expected_plan_records = {
        "design": _file_record(PLAN_DESIGN_PATH),
        "planner": _file_record(Path(hop_plan.__file__).resolve()),
        "protocol": _file_record(PROTOCOL_PATH),
    }
    if core.get("implementation_records") != expected_plan_records:
        raise CalibrationLockError("HOP plan implementation binding changed")
    _assert_records_at_head(
        {
            str(PLAN_DESIGN_PATH.relative_to(REPO_ROOT)): expected_plan_records[
                "design"
            ],
            str(Path(hop_plan.__file__).resolve().relative_to(REPO_ROOT)): (
                expected_plan_records["planner"]
            ),
            str(PROTOCOL_PATH.relative_to(REPO_ROOT)): expected_plan_records[
                "protocol"
            ],
        }
    )
    _require_exact_mapping(
        calibration,
        {
            "alpha_zero_evaluation_required": True,
            "alpha_zero_numerical_admissibility_required": True,
            "alpha_status": "unfrozen",
            "base_intrinsic_leakage_conductance_hex": float(0.0).hex(),
            "bath_temperature_hex": float(0.0).hex(),
            "bisection_relative_tolerance_hex": float(1e-8).hex(),
            "bracket_seed_hop_radius": 3,
            "calibration_anchor_count": EXPECTED_CALIBRATION_ANCHORS,
            "calibration_batch_count": EXPECTED_CALIBRATION_BATCHES,
            "calibration_split_only": True,
            "finite_result_required": True,
            "hidden_floor_or_jitter": False,
            "hidden_maximum_alpha_cap": False,
            "maximum_function_evaluations_per_anchor": 80,
            "maximum_leakage_conductance": None,
            "radius": 3,
            "required_numerical_minimum_added_leakage_hex": float(0.0).hex(),
            "target_attenuation": "exp(-1)",
            "zero_alpha_allowed": True,
        },
        "calibration contract",
    )
    _require_exact_mapping(
        numeric,
        {
            "actual_blas_identity_absolute_paths_prohibited": True,
            "actual_blas_identity_nonempty_lock_requirement": True,
            "alpha_calibration_backend": "numpy.linalg.eigh",
            "blas_threads_observed_lock_requirement": 1,
            "blas_threads_requested": 1,
            "condition_estimation_backend": "numpy.linalg.eigvalsh",
            "decision_dtype": "float64",
            "decision_factorization_backend": "numpy.linalg.cholesky",
            "device_class": "cpu",
            "hidden_jitter": False,
            "m_matrix_off_diagonal_tolerance_hex": float(1e-12).hex(),
            "maximum_principle_relative_tolerance_hex": float(1e-10).hex(),
            "triangular_solve_backend": "numpy.linalg.solve",
        },
        "numeric backend contract",
    )
    if numeric.get("numpy_version") != np.__version__:
        raise CalibrationLockError("NumPy version differs from the frozen plan")
    current_python = ".".join(map(str, sys.version_info[:3]))
    if (
        numeric.get("python_implementation") != sys.implementation.name
        or numeric.get("python_version") != current_python
    ):
        raise CalibrationLockError("Python runtime differs from the frozen plan")
    allowed_modes = statistics.get("allowed_calibration_lock_modes")
    if allowed_modes != [
        "absolute_only",
        "blocked",
        "finite_contrast",
        "right_censored_diagnostics",
    ]:
        raise CalibrationLockError("calibration lock modes changed")
    if statistics.get("audit_model_role_rules") != {
        "decision_roles_field": "frozen_audit_roles",
        "required_roles_field": "required_audit_model_roles",
        "required_roles_rule": (
            "frozen-order union of decision roles with S_1024 and R_top"
        ),
        "s1024_support_use": (
            "audit-reference-adequacy-only-when-not-a-decision-role"
        ),
    }:
        raise CalibrationLockError("audit model-role contract changed")
    if resources.get("candidate_budgets") != [256, 512, 1024]:
        raise CalibrationLockError("candidate budgets changed")
    if resources.get("peak_rss_scope") != (
        "process_high_water_through_scientific_payload_serialization_"
        "before_staging"
    ):
        raise CalibrationLockError("audit peak-RSS scope changed")
    return {
        "calibration": calibration,
        "cholesky_atol": float.fromhex(
            numeric["cholesky_reconstruction_atol_hex"]
        ),
        "cholesky_rtol": float.fromhex(
            numeric["cholesky_reconstruction_rtol_hex"]
        ),
        "effective_resistance": resources.get("effective_resistance_arm") == "enabled",
        "minimum_rcond": float.fromhex(
            numeric["minimum_reciprocal_condition_hex"]
        ),
        "m_matrix_tolerance": float.fromhex(
            numeric["m_matrix_off_diagonal_tolerance_hex"]
        ),
        "maximum_principle_tolerance": float.fromhex(
            numeric["maximum_principle_relative_tolerance_hex"]
        ),
        "numeric": numeric,
        "peak_rss_ceiling_bytes": resources["study_peak_rss_ceiling_bytes"],
        "peak_rss_scope": resources["peak_rss_scope"],
        "solve_residual_tolerance": float.fromhex(
            numeric["solve_residual_relative_tolerance_hex"]
        ),
        "per_batch_elapsed_ceiling_seconds": resources[
            "per_batch_elapsed_ceiling_seconds"
        ],
        "statistics": statistics,
    }


def _node_content_hash(domain):
    return hashlib.sha256(_canonical_json(list(domain.nodes))).hexdigest()


def _anchor_quartiles(batch):
    return {
        row["node_id"]: row["quartile_id"]
        for row in batch["anchors_by_quartile"]
    }


def _calibration_record(batch, record, *, screening_at_global):
    certificate = record.minimality_certificate
    if record.numerical_minimum_added_leakage != 0.0:
        raise np.linalg.LinAlgError(
            "spectral calibration required numerical repair"
        )
    if record.iterations > 80:
        raise np.linalg.LinAlgError("anchor calibration exceeded its evaluation cap")
    if certificate.upper_added_leakage_conductance != (
        record.added_leakage_conductance
    ):
        raise np.linalg.LinAlgError("minimality certificate lost its selected endpoint")
    return _hexify_scientific(
        {
            "added_leakage_conductance": record.added_leakage_conductance,
            "anchor_node_id": record.anchor,
            "batch_id": batch["batch_id"],
            "iterations": record.iterations,
            "minimality_certificate": asdict(certificate),
            "numerical_minimum_added_leakage": (
                record.numerical_minimum_added_leakage
            ),
            "quartile_id": _anchor_quartiles(batch)[record.anchor],
            "screening_at_batch_leakage": asdict(
                record.screening_at_study_leakage
            ),
            "screening_at_global_leakage": screening_at_global,
            "screening_at_selected_leakage": asdict(
                record.screening_at_selected_leakage
            ),
            "shell_nodes": list(record.shell_nodes),
        }
    )


def _run_calibration_pass(work_order, adjacency, contract):
    """Calibrate all 32 anchors without ever touching an audit row."""

    raw_batches = []
    for batch in work_order["batches"]:
        batch_started = time.perf_counter()
        reference = _domain_from_role(batch, "R_top", adjacency)
        precheck = _zero_alpha_precheck(
            reference,
            minimum_rcond=contract["minimum_rcond"],
            solve_tolerance=contract["solve_residual_tolerance"],
            cholesky_rtol=contract["cholesky_rtol"],
            cholesky_atol=contract["cholesky_atol"],
            m_matrix_tolerance=contract["m_matrix_tolerance"],
            maximum_principle_tolerance=contract[
                "maximum_principle_tolerance"
            ],
        )
        anchors = tuple(reference.anchors)
        shells = {
            row["anchor_node_id"]: tuple(row["shell_nodes"])
            for row in batch["shells"]
        }
        calibrated = local.calibrate_uniform_leakage_per_anchor(
            reference,
            anchors=anchors,
            shell_nodes_by_anchor=shells,
            target_attenuation=math.exp(-1.0),
            intrinsic_leakage_conductance=0.0,
            bath_temperature=0.0,
            minimum_reciprocal_condition=contract["minimum_rcond"],
            maximum_leakage_conductance=None,
            relative_tolerance=1e-8,
            maximum_iterations=80,
            bracket_seed_radius=3,
        )
        if calibrated.numerical_minimum_added_leakage != 0.0:
            raise np.linalg.LinAlgError(
                "batch calibration required conditioning-derived leakage"
            )
        if calibrated.eigendecomposition_count != 1:
            raise np.linalg.LinAlgError(
                "batch calibration did not share one eigendecomposition"
            )
        batch_elapsed = time.perf_counter() - batch_started
        if batch_elapsed > contract["per_batch_elapsed_ceiling_seconds"]:
            raise np.linalg.LinAlgError(
                "calibration batch exceeded its elapsed ceiling"
            )
        raw_batches.append(
            {
                "batch": batch,
                "calibration_seconds": batch_elapsed,
                "eigendecomposition_count": calibrated.eigendecomposition_count,
                "numerical_minimum_added_leakage": (
                    calibrated.numerical_minimum_added_leakage
                ),
                "per_anchor": calibrated.per_anchor,
                "precheck": precheck,
                "study_added_leakage_conductance": (
                    calibrated.study_added_leakage_conductance
                ),
                "total_evaluations": calibrated.total_evaluations,
            }
        )
    if len(raw_batches) != EXPECTED_CALIBRATION_BATCHES:
        raise CalibrationLockError("calibration pass changed its batch count")
    anchor_values = [
        record.added_leakage_conductance
        for row in raw_batches
        for record in row["per_anchor"]
    ]
    if len(anchor_values) != EXPECTED_CALIBRATION_ANCHORS:
        raise CalibrationLockError("calibration pass changed its anchor count")
    alpha_top = max(anchor_values)
    if not math.isfinite(alpha_top) or alpha_top < 0.0:
        raise np.linalg.LinAlgError("global leakage is nonfinite or negative")
    return raw_batches, alpha_top


def _run_fidelity_pass(work_order, adjacency, contract, alpha_top):
    """Rebuild each role at one alpha and reuse one reference factor per batch."""

    fidelity_rows = []
    batch_provenance = []
    global_screening = {}
    observational_timings = []
    for batch in work_order["batches"]:
        candidates = tuple(
            _selection_from_role(
                batch, role, adjacency, work_order["plan_fingerprint"]
            )
            for role in CANDIDATE_ROLES
        )
        reference = _selection_from_role(
            batch, "R_top", adjacency, work_order["plan_fingerprint"]
        )
        shells = {
            row["anchor_node_id"]: tuple(row["shell_nodes"])
            for row in batch["shells"]
        }
        started = time.perf_counter()
        evaluated = evaluate_nested_bounded_domain_fidelity(
            candidates,
            reference,
            protected_nodes=tuple(batch["protected_nodes"]),
            intrinsic_leakage_conductance=alpha_top,
            rank_top_k=8,
            minimum_reciprocal_condition=contract["minimum_rcond"],
            include_effective_resistance=contract["effective_resistance"],
            screening_shell_nodes_by_anchor=shells,
            screening_attenuation_threshold=math.exp(-1.0),
        )
        elapsed = time.perf_counter() - started
        if elapsed > contract["per_batch_elapsed_ceiling_seconds"]:
            raise np.linalg.LinAlgError(
                "fidelity batch exceeded its elapsed ceiling"
            )
        if (
            evaluated.reference_build_count != 1
            or evaluated.reference_factorization_count != 1
            or evaluated.candidate_build_count
            != evaluated.candidate_unique_model_count
            or evaluated.candidate_factorization_count
            != evaluated.candidate_unique_model_count
            or evaluated.candidate_requested_result_count != len(CANDIDATE_ROLES)
        ):
            raise np.linalg.LinAlgError("shared-factor provenance contract failed")
        if evaluated.candidate_reference_reuse_count != sum(
            index == -1 for index in evaluated.candidate_model_index
        ):
            raise np.linalg.LinAlgError("reference-reuse provenance is inconsistent")
        if evaluated.reference_anchor_screening is None:
            raise np.linalg.LinAlgError("global-alpha screening provenance is missing")
        for screening in evaluated.reference_anchor_screening:
            key = (batch["batch_id"], screening.anchor)
            if key in global_screening:
                raise CalibrationLockError("global screening identity duplicated")
            global_screening[key] = asdict(screening)
        for index, (role, result) in enumerate(zip(CANDIDATE_ROLES, evaluated)):
            diagnostic_index = evaluated.candidate_model_index[index]
            diagnostic = (
                evaluated.reference_model_diagnostics
                if diagnostic_index == -1
                else evaluated.candidate_model_diagnostics[diagnostic_index]
            )
            fidelity_rows.append(
                {
                    "batch_id": batch["batch_id"],
                    "candidate_model_index": diagnostic_index,
                    "candidate_safety": diagnostic.as_dict(),
                    "reference_safety": (
                        evaluated.reference_model_diagnostics.as_dict()
                    ),
                    "result": result,
                    "role": role,
                    "scientific_result": _hexify_scientific(
                        _strip_timing_fields(result.as_dict())
                    ),
                }
            )
        batch_provenance.append(
            {
                "batch_id": batch["batch_id"],
                "candidate_build_count": evaluated.candidate_build_count,
                "candidate_factorization_count": (
                    evaluated.candidate_factorization_count
                ),
                "candidate_model_index": list(evaluated.candidate_model_index),
                "candidate_requested_result_count": (
                    evaluated.candidate_requested_result_count
                ),
                "candidate_reference_reuse_count": (
                    evaluated.candidate_reference_reuse_count
                ),
                "candidate_unique_model_count": evaluated.candidate_unique_model_count,
                "reference_build_count": evaluated.reference_build_count,
                "reference_factorization_count": (
                    evaluated.reference_factorization_count
                ),
            }
        )
        observational_timings.append(
            {"batch_id": batch["batch_id"], "fidelity_seconds": elapsed}
        )
    if len(global_screening) != EXPECTED_CALIBRATION_ANCHORS:
        raise CalibrationLockError("global screening does not cover all anchors")
    return fidelity_rows, batch_provenance, global_screening, observational_timings


def _metric_aggregate(fidelity_rows, role, *, effective_resistance):
    selected = [row for row in fidelity_rows if row["role"] == role]
    if len(selected) != EXPECTED_CALIBRATION_BATCHES:
        raise CalibrationLockError(f"{role} is incomplete across calibration batches")
    results = [row["result"] for row in selected]

    def flattened(name):
        return [value for result in results for value in getattr(result, name)]

    raw = flattened("per_anchor_raw_relative_l2_error")
    h_error = flattened("per_anchor_maximum_h_absolute_error")
    inversion = flattened("per_anchor_rank_inversion_fraction")
    overlap = flattened("per_anchor_top_k_overlap")
    diagonal = flattened("per_anchor_source_diagonal_relative_error")
    if any(len(values) != EXPECTED_CALIBRATION_ANCHORS for values in (
        raw, h_error, inversion, overlap, diagonal
    )):
        raise CalibrationLockError(f"{role} does not cover all 32 anchors")
    boundary = [result.candidate_boundary_harmonic_max for result in results]
    resistance = None
    if effective_resistance:
        resistance = flattened("per_anchor_effective_resistance_relative_error")
        if len(resistance) != EXPECTED_CALIBRATION_ANCHORS:
            raise CalibrationLockError(
                f"{role} effective-resistance endpoint is incomplete"
            )
    elif any(result.effective_resistance_evaluated for result in results):
        raise CalibrationLockError("omitted effective-resistance arm was evaluated")
    summary = {
        "boundary_harmonic_q90": _higher(boundary, 0.9),
        "maximum_h_absolute_error_q90": _higher(h_error, 0.9),
        "rank_inversion_fraction_q90": _higher(inversion, 0.9),
        "raw_relative_l2_error_q90": _higher(raw, 0.9),
        "source_diagonal_relative_error_q90": _higher(diagonal, 0.9),
        "top8_overlap_q10": _lower(overlap, 0.1),
    }
    if resistance is not None:
        summary["effective_resistance_relative_error_q90"] = _higher(
            resistance, 0.9
        )
    return summary


def _passes_absolute_adequacy(summary, frozen):
    limits = frozen["absolute_adequacy"]
    checks = {
        "boundary_harmonic_q90": summary["boundary_harmonic_q90"]
        <= limits["boundary_harmonic_q90_max"],
        "maximum_h_absolute_error_q90": summary[
            "maximum_h_absolute_error_q90"
        ]
        <= limits["maximum_h_absolute_error_q90_max"],
        "rank_inversion_fraction_q90": summary[
            "rank_inversion_fraction_q90"
        ]
        <= limits["rank_inversion_fraction_q90_max"],
        "raw_relative_l2_error_q90": summary["raw_relative_l2_error_q90"]
        <= limits["raw_relative_l2_error_q90_max"],
        "top8_overlap_q10": summary["top8_overlap_q10"]
        >= limits["top8_overlap_q10_min"],
    }
    if "effective_resistance_relative_error_q90_max" in limits:
        checks["effective_resistance_relative_error_q90"] = summary[
            "effective_resistance_relative_error_q90"
        ] <= limits["effective_resistance_relative_error_q90_max"]
    return checks, all(checks.values())


def _passes_reference_adequacy(summary, frozen):
    limits = frozen["reference_adequacy"]
    checks = {
        "maximum_h_absolute_error_q90": summary[
            "maximum_h_absolute_error_q90"
        ] <= limits["maximum_h_absolute_error_q90_max"],
        "raw_relative_l2_error_q90": summary[
            "raw_relative_l2_error_q90"
        ] <= limits["raw_relative_l2_error_q90_max"],
        "top8_overlap_q10": summary["top8_overlap_q10"]
        >= limits["top8_overlap_q10_min"],
    }
    return checks, all(checks.values())


def _endpoint_inventory(work_order, adjacency):
    rows = []
    for batch in work_order["batches"]:
        for role in ROLE_ORDER:
            domain = _domain_from_role(batch, role, adjacency)
            rows.append(
                {
                    "batch_id": batch["batch_id"],
                    "node_content_sha256": _node_content_hash(domain),
                    "realized_nodes": len(domain.nodes),
                    "role": role,
                }
            )
    return rows


def _roles_identical_across_batches(endpoint_rows, left, right):
    index = {
        (row["batch_id"], row["role"]): row["node_content_sha256"]
        for row in endpoint_rows
    }
    batches = sorted({row["batch_id"] for row in endpoint_rows})
    return all(index[(batch, left)] == index[(batch, right)] for batch in batches)


def _select_lock_mode(
    fidelity_rows,
    work_order,
    adjacency,
    contract,
    *,
    alpha_top,
):
    statistics = contract["statistics"]
    endpoint_rows = _endpoint_inventory(work_order, adjacency)
    summaries = {}
    candidate_checks = {}
    adequate_roles = []
    for role in CANDIDATE_ROLES:
        summary = _metric_aggregate(
            fidelity_rows,
            role,
            effective_resistance=contract["effective_resistance"],
        )
        checks, passed = _passes_absolute_adequacy(summary, statistics)
        summaries[role] = summary
        candidate_checks[role] = {"checks": checks, "passed": passed}
        if passed:
            adequate_roles.append(role)

    reference_checks, reference_adequate = _passes_reference_adequacy(
        summaries["S_1024"], statistics
    )
    base = {
        "alpha_top": alpha_top,
        "audit_solve_authorized": False,
        "candidate_adequacy": candidate_checks,
        "candidate_summaries": summaries,
        "confirmatory_claim_authorized": False,
        "efficacy_or_resource_claim_authorized": False,
        "endpoint_inventory": endpoint_rows,
        "frozen_audit_roles": [],
        "required_audit_model_roles": [],
        "k_high": None,
        "k_low": None,
        "reference_adequacy": {
            "checks": reference_checks,
            "passed": reference_adequate,
            "u_role": "S_1024",
            "reference_role": "R_top",
        },
    }
    if not reference_adequate:
        return {
            **base,
            "lock_mode": "blocked",
            "reason": "calibration_reference_inadequate",
        }
    if not adequate_roles:
        return {
            **base,
            "audit_solve_authorized": True,
            "frozen_audit_roles": list(ROLE_ORDER),
            "required_audit_model_roles": list(ROLE_ORDER),
            "lock_mode": "right_censored_diagnostics",
            "reason": "no_calibration_candidate_adequate",
        }

    low = adequate_roles[0]
    high = ROLE_ORDER[ROLE_ORDER.index(low) + 1]
    exhausted_identity = _roles_identical_across_batches(
        endpoint_rows, low, high
    )
    if exhausted_identity:
        return {
            **base,
            "endpoint_node_identical_across_all_batches": True,
            "lock_mode": "blocked",
            "reason": "node_identical_exhaustion_requires_gauge_aware_amendment",
        }
    if low == "S_1024":
        return {
            **base,
            "audit_solve_authorized": True,
            "endpoint_node_identical_across_all_batches": False,
            "frozen_audit_roles": [low, high],
            "required_audit_model_roles": ["S_1024", "R_top"],
            "k_high": high,
            "k_low": low,
            "lock_mode": "absolute_only",
            "reason": "selected_low_is_1024_absolute_endpoint",
        }
    required_roles = [
        role
        for role in ROLE_ORDER
        if role in {low, high, "S_1024", "R_top"}
    ]
    return {
        **base,
        "audit_solve_authorized": True,
        "efficacy_or_resource_claim_authorized": True,
        "endpoint_node_identical_across_all_batches": False,
        "frozen_audit_roles": [low, high],
        "required_audit_model_roles": required_roles,
        "k_high": high,
        "k_low": low,
        "lock_mode": "finite_contrast",
        "reason": "smallest_adequate_candidate_with_distinct_next_endpoint",
    }


def _blocked_selection(reason, *, alpha_top=None, details=None):
    return {
        "alpha_top": alpha_top,
        "audit_solve_authorized": False,
        "confirmatory_claim_authorized": False,
        "details": {} if details is None else details,
        "efficacy_or_resource_claim_authorized": False,
        "frozen_audit_roles": [],
        "required_audit_model_roles": [],
        "k_high": None,
        "k_low": None,
        "lock_mode": "blocked",
        "reason": reason,
    }


def _derive_lock_payloads(context, runtime_blas):
    started_all = time.perf_counter()
    contract = _validated_bound_contract(context["manifest"])
    work_order = _derive_calibration_work_order(context)
    calibration_adjacency = _adjacency_from_work_order(work_order)
    calibration_seconds = None
    fidelity_seconds = None
    batch_timings = []
    alpha_top = None
    calibration_completed = False
    anchor_rows = []
    batch_rows = []
    public_fidelity_rows = []

    try:
        started = time.perf_counter()
        raw_batches, alpha_top = _run_calibration_pass(
            work_order, calibration_adjacency, contract
        )
        calibration_seconds = time.perf_counter() - started
        started = time.perf_counter()
        (
            fidelity_rows,
            factor_provenance,
            global_screening,
            fidelity_batch_timings,
        ) = _run_fidelity_pass(
            work_order, calibration_adjacency, contract, alpha_top
        )
        fidelity_seconds = time.perf_counter() - started
        fidelity_timing_by_batch = {
            row["batch_id"]: row["fidelity_seconds"]
            for row in fidelity_batch_timings
        }
        batch_timings = [
            {
                "batch_id": raw["batch"]["batch_id"],
                "calibration_seconds": raw["calibration_seconds"],
                "fidelity_seconds": fidelity_timing_by_batch[
                    raw["batch"]["batch_id"]
                ],
            }
            for raw in raw_batches
        ]

        for raw, factor in zip(raw_batches, factor_provenance):
            batch = raw["batch"]
            batch_maximum = max(
                record.added_leakage_conductance
                for record in raw["per_anchor"]
            )
            if raw["study_added_leakage_conductance"] != batch_maximum:
                raise np.linalg.LinAlgError("batch leakage maximum is inconsistent")
            batch_rows.append(
                _hexify_scientific(
                    {
                        "batch_id": batch["batch_id"],
                        "batch_maximum_added_leakage": batch_maximum,
                        "eigendecomposition_count": (
                            raw["eigendecomposition_count"]
                        ),
                        "factor_provenance": factor,
                        "global_alpha_top": alpha_top,
                        "numerical_minimum_added_leakage": (
                            raw["numerical_minimum_added_leakage"]
                        ),
                        "total_attenuation_evaluations": (
                            raw["total_evaluations"]
                        ),
                        "zero_alpha_precheck": raw["precheck"],
                    }
                )
            )
            for record in raw["per_anchor"]:
                screening = global_screening[(batch["batch_id"], record.anchor)]
                if screening["shell_attenuation"] > math.exp(-1.0) * (
                    1.0 + 1e-8
                ):
                    raise np.linalg.LinAlgError(
                        "global leakage missed a registered anchor target"
                    )
                anchor_rows.append(
                    _calibration_record(
                        batch, record, screening_at_global=screening
                    )
                )
        observed_anchor_maximum = max(
            float.fromhex(row["added_leakage_conductance"])
            for row in anchor_rows
        )
        observed_batch_maximum = max(
            float.fromhex(row["batch_maximum_added_leakage"])
            for row in batch_rows
        )
        if alpha_top != observed_anchor_maximum or alpha_top != observed_batch_maximum:
            raise np.linalg.LinAlgError("global alpha is not both registered maxima")

        selection = _select_lock_mode(
            fidelity_rows,
            work_order,
            calibration_adjacency,
            contract,
            alpha_top=alpha_top,
        )
        for row in fidelity_rows:
            public_fidelity_rows.append(
                {
                    "batch_id": row["batch_id"],
                    "candidate_model_index": row["candidate_model_index"],
                    "candidate_safety": _hexify_scientific(
                        row["candidate_safety"]
                    ),
                    "reference_safety": _hexify_scientific(
                        row["reference_safety"]
                    ),
                    "result": row["scientific_result"],
                    "role": row["role"],
                }
            )
        calibration_completed = True
    except (
        CalibrationLockError,
        np.linalg.LinAlgError,
        FloatingPointError,
        ValueError,
    ) as exc:
        # A numerical scientific failure is itself a valid fail-closed outcome.
        # Partial rows are deliberately discarded so no survivor-only maximum
        # can be mistaken for the registered 32-anchor calibration.
        selection = _blocked_selection(
            "calibration_numerical_contract_failed",
            details={"failure_class": type(exc).__name__},
        )
        alpha_top = None
        anchor_rows = []
        batch_rows = []
        public_fidelity_rows = []

    peak_rss = _peak_rss_bytes()
    if peak_rss > contract["peak_rss_ceiling_bytes"]:
        selection = _resource_blocked_selection(
            selection, contract["peak_rss_ceiling_bytes"]
        )
    total_seconds = time.perf_counter() - started_all
    execution = {
        "actual_blas_identity": runtime_blas,
        "audit_metrics_materialized": False,
        "audit_solves_executed": 0,
        "batch_phase_timings": batch_timings,
        "calibration_only_work_order": True,
        "calibration_seconds": calibration_seconds,
        "fidelity_seconds": fidelity_seconds,
        "observed_peak_rss_bytes": peak_rss,
        "peak_rss_ceiling_bytes": contract["peak_rss_ceiling_bytes"],
        "timing_and_rss_outside_scientific_fingerprint": True,
        "total_seconds": total_seconds,
    }
    payloads = {
        "anchor_calibrations.jsonl": _jsonl_bytes(anchor_rows),
        "batch_calibration.jsonl": _jsonl_bytes(batch_rows),
        "calibration_fidelity.jsonl": _jsonl_bytes(public_fidelity_rows),
        "calibration_work_order.json": _canonical_json(work_order),
        "execution.json": _canonical_json(execution),
        "selection.json": _canonical_json(_hexify_scientific(selection)),
    }
    return payloads, selection, contract, calibration_completed


def _implementation_records():
    paths = (
        Path(__file__).resolve(),
        LOCK_DESIGN_PATH,
        AUDIT_DESIGN_PATH,
        AUDIT_RUNNER_PATH,
        PROTOCOL_PATH,
        PLAN_DESIGN_PATH,
        LOCAL_DIFFUSION_PATH,
        FIDELITY_PATH,
    )
    records = {
        str(path.relative_to(REPO_ROOT)): _file_record(path) for path in paths
    }
    _assert_records_at_head(records)
    return records


def _peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; this frozen runner is Linux/WSL only.
    return value * 1024


def _manifest_without_seal(manifest):
    return {
        key: value
        for key, value in manifest.items()
        if key != "manifest_integrity_seal"
    }


def _build_lock_manifest(
    payloads,
    *,
    context,
    contract,
    runtime_blas,
    selection,
    calibration_completed,
):
    if set(payloads) != set(ARTIFACT_NAMES):
        raise CalibrationLockError("lock payload inventory mismatch")
    oversized = [
        name
        for name, data in payloads.items()
        if len(data) > MAXIMUM_LOCK_ARTIFACT_BYTES[name]
    ]
    if oversized:
        raise CalibrationLockError(
            "generated calibration artifact exceeds its frozen size ceiling"
        )
    scientific_records = {
        name: _content_record(payloads[name]) for name in SCIENTIFIC_ARTIFACT_NAMES
    }
    observational_records = {
        name: _content_record(payloads[name])
        for name in OBSERVATIONAL_ARTIFACT_NAMES
    }
    alpha = selection.get("alpha_top")
    scientific_core = {
        "actual_blas_identity": runtime_blas,
        "algorithm": ALGORITHM,
        "alpha_top_hex": _float_or_none_hex(alpha),
        "audit_solve_authorized": selection["audit_solve_authorized"],
        "audit_solves_executed": 0,
        "calibration_completed": calibration_completed,
        "confirmatory_claim_authorized": False,
        "effective_resistance_arm": (
            "enabled" if contract["effective_resistance"] else "omitted"
        ),
        "implementation_records": _implementation_records(),
        "lock_mode": selection["lock_mode"],
        "numeric_backend_contract": contract["numeric"],
        "plan_fingerprint": context["manifest"]["plan_fingerprint"],
        "plan_manifest_record": context["captured_records"]["plan_manifest"],
        "repository_commit": _git_commit(),
        "schema": SCHEMA,
        "scientific_artifact_records": scientific_records,
        "selection_reason": selection["reason"],
    }
    lock_fingerprint = hashlib.sha256(
        _canonical_json(scientific_core)
    ).hexdigest()
    manifest = {
        "accepted": selection["lock_mode"] != "blocked",
        "algorithm": ALGORITHM,
        "audit_solve_authorized": selection["audit_solve_authorized"],
        "audit_solves_executed": 0,
        "calibration_completed": calibration_completed,
        "confirmatory_claim_authorized": False,
        "lock_fingerprint": lock_fingerprint,
        "lock_mode": selection["lock_mode"],
        "manifest_integrity_contract": (
            "sha256-canonical-manifest-without-manifest_integrity_seal-v1"
        ),
        "marker_record": _content_record(MARKER_BYTES),
        "observational_artifact_records": observational_records,
        "plan_manifest_record": context["captured_records"]["plan_manifest"],
        "reason": selection["reason"],
        "schema": SCHEMA,
        "scientific_core": scientific_core,
    }
    manifest["manifest_integrity_seal"] = hashlib.sha256(
        _canonical_json(_manifest_without_seal(manifest))
    ).hexdigest()
    return manifest


def _validate_manifest_invariants(manifest):
    required = {
        "accepted",
        "algorithm",
        "audit_solve_authorized",
        "audit_solves_executed",
        "calibration_completed",
        "confirmatory_claim_authorized",
        "lock_fingerprint",
        "lock_mode",
        "manifest_integrity_contract",
        "manifest_integrity_seal",
        "marker_record",
        "observational_artifact_records",
        "plan_manifest_record",
        "reason",
        "schema",
        "scientific_core",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise CalibrationLockError("calibration lock manifest shape mismatch")
    if manifest["schema"] != SCHEMA or manifest["algorithm"] != ALGORITHM:
        raise CalibrationLockError("calibration lock schema mismatch")
    mode = manifest["lock_mode"]
    if mode not in {
        "absolute_only",
        "blocked",
        "finite_contrast",
        "right_censored_diagnostics",
    }:
        raise CalibrationLockError("calibration lock mode is invalid")
    if not isinstance(manifest["calibration_completed"], bool):
        raise CalibrationLockError("calibration completion flag is malformed")
    if mode != "blocked" and manifest["calibration_completed"] is not True:
        raise CalibrationLockError("an incomplete calibration cannot authorize audit")
    expected_authorized = mode != "blocked"
    if (
        manifest["accepted"] is not expected_authorized
        or manifest["audit_solve_authorized"] is not expected_authorized
        or manifest["audit_solves_executed"] != 0
        or manifest["confirmatory_claim_authorized"] is not False
    ):
        raise CalibrationLockError("calibration lock authorization is inconsistent")
    if re.fullmatch(r"[0-9a-f]{64}", manifest["lock_fingerprint"]) is None:
        raise CalibrationLockError("calibration lock fingerprint is malformed")
    if re.fullmatch(r"[0-9a-f]{64}", manifest["manifest_integrity_seal"]) is None:
        raise CalibrationLockError("calibration manifest seal is malformed")
    expected_seal = hashlib.sha256(
        _canonical_json(_manifest_without_seal(manifest))
    ).hexdigest()
    if manifest["manifest_integrity_seal"] != expected_seal:
        raise CalibrationLockError("calibration manifest seal mismatch")
    core = manifest["scientific_core"]
    if not isinstance(core, dict):
        raise CalibrationLockError("calibration scientific core is malformed")
    expected_core_fields = {
        "actual_blas_identity",
        "algorithm",
        "alpha_top_hex",
        "audit_solve_authorized",
        "audit_solves_executed",
        "calibration_completed",
        "confirmatory_claim_authorized",
        "effective_resistance_arm",
        "implementation_records",
        "lock_mode",
        "numeric_backend_contract",
        "plan_fingerprint",
        "plan_manifest_record",
        "repository_commit",
        "schema",
        "scientific_artifact_records",
        "selection_reason",
    }
    if set(core) != expected_core_fields:
        raise CalibrationLockError("calibration scientific core shape mismatch")
    scientific_records = core.get("scientific_artifact_records")
    observational_records = manifest.get("observational_artifact_records")
    if (
        not isinstance(scientific_records, dict)
        or set(scientific_records) != set(SCIENTIFIC_ARTIFACT_NAMES)
        or not all(
            _content_record_is_valid(record)
            for record in scientific_records.values()
        )
        or not isinstance(observational_records, dict)
        or set(observational_records) != set(OBSERVATIONAL_ARTIFACT_NAMES)
        or not all(
            _content_record_is_valid(record)
            for record in observational_records.values()
        )
        or not _content_record_is_valid(manifest.get("marker_record"))
        or not _content_record_is_valid(manifest.get("plan_manifest_record"))
    ):
        raise CalibrationLockError("calibration content-record inventory is invalid")
    expected_lock = hashlib.sha256(_canonical_json(core)).hexdigest()
    if manifest["lock_fingerprint"] != expected_lock:
        raise CalibrationLockError("calibration scientific fingerprint mismatch")
    if (
        core.get("schema") != SCHEMA
        or core.get("algorithm") != ALGORITHM
        or core.get("lock_mode") != mode
        or core.get("audit_solve_authorized") is not expected_authorized
        or core.get("audit_solves_executed") != 0
        or core.get("confirmatory_claim_authorized") is not False
        or core.get("calibration_completed") is not manifest["calibration_completed"]
        or core.get("selection_reason") != manifest["reason"]
        or core.get("plan_manifest_record") != manifest["plan_manifest_record"]
    ):
        raise CalibrationLockError("calibration scientific core is inconsistent")
    alpha_hex = core.get("alpha_top_hex")
    if manifest["calibration_completed"]:
        if not isinstance(alpha_hex, str):
            raise CalibrationLockError("completed calibration must freeze alpha")
        try:
            alpha_value = float.fromhex(alpha_hex)
        except ValueError as exc:
            raise CalibrationLockError("frozen alpha is malformed") from exc
        if not math.isfinite(alpha_value) or alpha_value < 0.0:
            raise CalibrationLockError("frozen alpha must be finite and nonnegative")
    elif alpha_hex is not None:
        raise CalibrationLockError("incomplete calibration cannot freeze alpha")
    runtime = core.get("actual_blas_identity")
    if not isinstance(runtime, list) or not runtime:
        raise CalibrationLockError("calibration BLAS identity is empty")
    for row in runtime:
        if not isinstance(row, dict) or "filepath" in row:
            raise CalibrationLockError("calibration BLAS identity is not portable")
        if row.get("observed_num_threads") != 1:
            raise CalibrationLockError("calibration BLAS thread identity is invalid")


def _lock_manifest_from_binding(binding):
    data = _read_bound_bytes(
        binding,
        MANIFEST_NAME,
        None,
        "calibration lock manifest",
        maximum_size=MAXIMUM_LOCK_MANIFEST_BYTES,
    )
    manifest = _strict_json_bytes(data, "calibration lock manifest")
    if _canonical_json(manifest) != data:
        raise CalibrationLockError("calibration lock manifest is not canonical")
    _validate_manifest_invariants(manifest)
    return manifest, data


def _read_lock_payloads(binding, manifest):
    core = manifest["scientific_core"]
    scientific_records = core.get("scientific_artifact_records")
    observational_records = manifest.get("observational_artifact_records")
    if (
        not isinstance(scientific_records, dict)
        or set(scientific_records) != set(SCIENTIFIC_ARTIFACT_NAMES)
        or not isinstance(observational_records, dict)
        or set(observational_records) != set(OBSERVATIONAL_ARTIFACT_NAMES)
    ):
        raise CalibrationLockError("calibration artifact record inventory mismatch")
    for name, record in scientific_records.items():
        if record["size_bytes"] > MAXIMUM_LOCK_ARTIFACT_BYTES[name]:
            raise CalibrationLockError(
                "calibration scientific artifact exceeds its frozen read ceiling"
            )
    for name, record in observational_records.items():
        if record["size_bytes"] > MAXIMUM_LOCK_ARTIFACT_BYTES[name]:
            raise CalibrationLockError(
                "calibration observational artifact exceeds its frozen read ceiling"
            )
    if manifest["marker_record"]["size_bytes"] > len(MARKER_BYTES):
        raise CalibrationLockError("calibration marker exceeds its read ceiling")
    payloads = {}
    for name in SCIENTIFIC_ARTIFACT_NAMES:
        maximum_size = MAXIMUM_LOCK_ARTIFACT_BYTES[name]
        payloads[name] = _read_bound_bytes(
            binding,
            name,
            scientific_records[name],
            "calibration lock",
            maximum_size=maximum_size,
        )
    for name in OBSERVATIONAL_ARTIFACT_NAMES:
        maximum_size = MAXIMUM_LOCK_ARTIFACT_BYTES[name]
        payloads[name] = _read_bound_bytes(
            binding,
            name,
            observational_records[name],
            "calibration lock",
            maximum_size=maximum_size,
        )
    marker = _read_bound_bytes(
        binding,
        MARKER_NAME,
        manifest["marker_record"],
        "calibration lock",
        maximum_size=len(MARKER_BYTES),
    )
    if marker != MARKER_BYTES:
        raise CalibrationLockError("calibration local-only marker mismatch")
    return payloads


def _scalar_alpha_fingerprint(nodes, alpha):
    payload = [
        [
            [type(node).__module__, type(node).__qualname__, repr(node)],
            float(alpha).hex(),
        ]
        for node in sorted(nodes, key=local._stable_key)
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decoded_fidelity_rows(
    rows, contract, expected_work_order, adjacency, alpha_top
):
    output = []
    expected_result_fields = {
        field.name
        for field in fields(BoundedFidelityResult)
        if not field.name.endswith("_seconds")
    }
    required_float_fields = {
        "candidate_boundary_harmonic_max",
        "candidate_cut_current_fraction_max",
        "candidate_reciprocal_condition",
        "closure_mass_fraction",
        "closure_total_self_return_mass",
        "closure_total_transfer_mass",
        "h_root_mean_square_error",
        "maximum_h_absolute_error",
        "maximum_h_absolute_error_90th_percentile",
        "maximum_raw_absolute_error",
        "maximum_raw_relative_error",
        "mean_kendall_rank_agreement",
        "mean_top_k_overlap",
        "minimum_kendall_rank_agreement",
        "minimum_top_k_overlap",
        "protected_candidate_fraction",
        "protected_reference_fraction",
        "rank_inversion_fraction_90th_percentile",
        "raw_relative_frobenius_error",
        "raw_relative_l2_error_90th_percentile",
        "reference_boundary_harmonic_max",
        "reference_cut_current_fraction_max",
        "reference_reciprocal_condition",
        "top_k_overlap_10th_percentile",
    }
    optional_float_fields = {
        "effective_resistance_relative_error_90th_percentile",
        "maximum_effective_resistance_absolute_error",
        "maximum_effective_resistance_relative_error",
    }
    vector_fields = (
        "per_anchor_raw_relative_l2_error",
        "per_anchor_maximum_h_absolute_error",
        "per_anchor_rank_inversion_fraction",
        "per_anchor_top_k_overlap",
        "per_anchor_source_diagonal_relative_error",
    )
    batch_index = {
        batch["batch_id"]: batch for batch in expected_work_order["batches"]
    }
    row_index = {
        (row.get("batch_id"), row.get("role")): row for row in rows
    }
    model_owner_role = {}
    for batch_id, batch in batch_index.items():
        reference = _selection_from_role(
            batch, "R_top", adjacency, expected_work_order["plan_fingerprint"]
        )
        reference_key = (
            reference.domain.nodes,
            reference.domain.neighbors,
        )
        operator_index = {}
        for role in CANDIDATE_ROLES:
            candidate = _selection_from_role(
                batch,
                role,
                adjacency,
                expected_work_order["plan_fingerprint"],
            )
            key = (candidate.domain.nodes, candidate.domain.neighbors)
            if key == reference_key:
                model_index = -1
            else:
                model_index = operator_index.setdefault(key, len(operator_index))
            if (
                row_index[(batch_id, role)].get("candidate_model_index")
                != model_index
            ):
                raise CalibrationLockError(
                    "factor index does not match the frozen candidate operator"
                )
            if model_index >= 0:
                model_owner_role.setdefault((batch_id, model_index), role)

    for row in rows:
        expected_row_fields = {
            "batch_id",
            "candidate_model_index",
            "candidate_safety",
            "reference_safety",
            "result",
            "role",
        }
        if (
            not isinstance(row, dict)
            or set(row) != expected_row_fields
            or row.get("role") not in CANDIDATE_ROLES
            or row.get("batch_id") not in batch_index
        ):
            raise CalibrationLockError("calibration fidelity row is malformed")
        result = row.get("result")
        if not isinstance(result, dict) or set(result) != expected_result_fields:
            raise CalibrationLockError("calibration fidelity result shape changed")

        batch = batch_index[row["batch_id"]]
        role = row["role"]
        candidate = _selection_from_role(
            batch, role, adjacency, expected_work_order["plan_fingerprint"]
        )
        reference = _selection_from_role(
            batch, "R_top", adjacency, expected_work_order["plan_fingerprint"]
        )
        protected = tuple(
            sorted(set(batch["protected_nodes"]), key=local._stable_key)
        )
        structural_expected = {
            "alpha_fingerprint": _scalar_alpha_fingerprint(
                reference.domain.nodes, alpha_top
            ),
            "candidate_nodes": candidate.realized_nodes,
            "candidate_selection_fingerprint": candidate.selection_fingerprint,
            "candidate_strategy": candidate.strategy,
            "protected_nodes": list(protected),
            "protected_nodes_count": len(protected),
            "rank_excludes_source": True,
            "rank_top_k": 8,
            "reference_nodes": reference.realized_nodes,
            "reference_selection_fingerprint": reference.selection_fingerprint,
            "reference_strategy": reference.strategy,
            "source_nodes": list(reference.domain.anchors),
        }
        if any(result.get(name) != value for name, value in structural_expected.items()):
            raise CalibrationLockError(
                "calibration fidelity provenance disagrees with the work order"
            )
        integer_fields = (
            "candidate_nodes",
            "closure_edges",
            "closure_filtered_pairs",
            "closure_realized_pairs",
            "closure_supplied_pairs",
            "protected_nodes_count",
            "rank_top_k",
            "reference_nodes",
        )
        if any(
            not isinstance(result[name], int) or isinstance(result[name], bool)
            for name in integer_fields
        ):
            raise CalibrationLockError("calibration fidelity count is malformed")
        if (
            result["closure_edges"] != 0
            or result["closure_filtered_pairs"] != 0
            or result["closure_realized_pairs"] != 0
            or result["closure_supplied_pairs"] != 0
            or result["closure_ledger_fingerprint"] is not None
            or result["closure_pair_source"] is not None
            or result["closure_approximation_limits_apply"] is not False
            or result["closure_policy"] != "none_full_dirichlet_beta"
        ):
            raise CalibrationLockError("calibration fidelity used boundary closure")

        decoded = {}
        try:
            for name in required_float_fields:
                decoded[name] = float.fromhex(result[name])
            for name in vector_fields:
                decoded[name] = tuple(float.fromhex(value) for value in result[name])
            for name in optional_float_fields:
                decoded[name] = (
                    None if result[name] is None else float.fromhex(result[name])
                )
            decoded["effective_resistance_evaluated"] = result[
                "effective_resistance_evaluated"
            ]
            resistance = result["per_anchor_effective_resistance_relative_error"]
            decoded["per_anchor_effective_resistance_relative_error"] = (
                None
                if resistance is None
                else tuple(float.fromhex(value) for value in resistance)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationLockError(
                "calibration fidelity metric encoding is malformed"
            ) from exc
        if any(
            not math.isfinite(decoded[name]) for name in required_float_fields
        ) or any(
            not math.isfinite(value)
            for name in vector_fields
            for value in decoded[name]
        ) or any(
            decoded[name] is not None and not math.isfinite(decoded[name])
            for name in optional_float_fields
        ):
            raise CalibrationLockError("calibration fidelity metric is nonfinite")

        nonnegative_float_fields = required_float_fields - {
            "mean_kendall_rank_agreement",
            "minimum_kendall_rank_agreement",
        }
        if any(
            decoded[name] < 0.0 for name in nonnegative_float_fields
        ) or any(
            decoded[name] is not None and decoded[name] < 0.0
            for name in optional_float_fields
        ):
            raise CalibrationLockError("calibration fidelity metric is negative")
        for name in vector_fields:
            values = decoded[name]
            if len(values) != EXPECTED_ANCHORS_PER_BATCH or any(
                value < 0.0 for value in values
            ):
                raise CalibrationLockError(
                    "calibration fidelity vector has an invalid domain"
                )
        unit_interval_fields = (
            "candidate_cut_current_fraction_max",
            "candidate_reciprocal_condition",
            "closure_mass_fraction",
            "mean_top_k_overlap",
            "minimum_top_k_overlap",
            "protected_candidate_fraction",
            "protected_reference_fraction",
            "rank_inversion_fraction_90th_percentile",
            "reference_cut_current_fraction_max",
            "reference_reciprocal_condition",
            "top_k_overlap_10th_percentile",
        )
        if any(decoded[name] > 1.0 for name in unit_interval_fields) or any(
            not -1.0 <= decoded[name] <= 1.0
            for name in (
                "mean_kendall_rank_agreement",
                "minimum_kendall_rank_agreement",
            )
        ):
            raise CalibrationLockError(
                "calibration fidelity fraction is outside its range"
            )
        if (
            decoded["minimum_kendall_rank_agreement"]
            > decoded["mean_kendall_rank_agreement"]
            or decoded["minimum_top_k_overlap"]
            > decoded["mean_top_k_overlap"]
        ):
            raise CalibrationLockError(
                "calibration fidelity minimum exceeds its mean"
            )
        for name in (
            "per_anchor_rank_inversion_fraction",
            "per_anchor_top_k_overlap",
        ):
            if any(value > 1.0 for value in decoded[name]):
                raise CalibrationLockError(
                    "calibration fidelity fraction is outside [0, 1]"
                )
        if (
            decoded["closure_mass_fraction"] != 0.0
            or decoded["closure_total_self_return_mass"] != 0.0
            or decoded["closure_total_transfer_mass"] != 0.0
            or decoded["protected_candidate_fraction"]
            != len(protected) / candidate.realized_nodes
            or decoded["protected_reference_fraction"]
            != len(protected) / reference.realized_nodes
        ):
            raise CalibrationLockError(
                "calibration fidelity structural summaries disagree"
            )

        summary_expected = {
            "maximum_h_absolute_error_90th_percentile": _higher(
                decoded["per_anchor_maximum_h_absolute_error"], 0.9
            ),
            "rank_inversion_fraction_90th_percentile": _higher(
                decoded["per_anchor_rank_inversion_fraction"], 0.9
            ),
            "raw_relative_l2_error_90th_percentile": _higher(
                decoded["per_anchor_raw_relative_l2_error"], 0.9
            ),
            "top_k_overlap_10th_percentile": _lower(
                decoded["per_anchor_top_k_overlap"], 0.1
            ),
        }
        if any(decoded[name] != value for name, value in summary_expected.items()):
            raise CalibrationLockError(
                "calibration fidelity order-statistic summary disagrees"
            )
        kendall = tuple(
            1.0 - 2.0 * value
            for value in decoded["per_anchor_rank_inversion_fraction"]
        )
        descriptive_expected = {
            "maximum_h_absolute_error": max(
                decoded["per_anchor_maximum_h_absolute_error"]
            ),
            "mean_kendall_rank_agreement": float(np.mean(kendall)),
            "mean_top_k_overlap": float(
                np.mean(decoded["per_anchor_top_k_overlap"])
            ),
            "minimum_kendall_rank_agreement": min(kendall),
            "minimum_top_k_overlap": min(decoded["per_anchor_top_k_overlap"]),
        }
        if any(
            not math.isclose(
                decoded[name],
                value,
                rel_tol=0.0,
                abs_tol=64.0 * np.finfo(float).eps,
            )
            for name, value in descriptive_expected.items()
        ):
            raise CalibrationLockError(
                "calibration fidelity descriptive summary disagrees"
            )
        h_fields = (
            "candidate_boundary_harmonic_max",
            "maximum_h_absolute_error",
            "maximum_h_absolute_error_90th_percentile",
            "reference_boundary_harmonic_max",
        )
        if any(decoded[name] > 1.0 + 1e-10 for name in h_fields) or any(
            value > 1.0 + 1e-10
            for value in decoded["per_anchor_maximum_h_absolute_error"]
        ):
            raise CalibrationLockError("calibration h metric exceeds [0, 1]")

        effective_resistance = decoded["effective_resistance_evaluated"]
        resistance_values = decoded[
            "per_anchor_effective_resistance_relative_error"
        ]
        if not isinstance(effective_resistance, bool):
            raise CalibrationLockError(
                "effective-resistance evaluation flag is not boolean"
            )
        if effective_resistance:
            if (
                resistance_values is None
                or len(resistance_values) != EXPECTED_ANCHORS_PER_BATCH
                or any(
                    not math.isfinite(value) or value < 0.0
                    for value in resistance_values
                )
                or any(decoded[name] is None for name in optional_float_fields)
                or decoded["effective_resistance_relative_error_90th_percentile"]
                != _higher(resistance_values, 0.9)
                or decoded["maximum_effective_resistance_relative_error"]
                != max(resistance_values)
            ):
                raise CalibrationLockError(
                    "effective-resistance fidelity vector is invalid"
                )
        elif resistance_values is not None or any(
            decoded[name] is not None for name in optional_float_fields
        ):
            raise CalibrationLockError(
                "omitted effective-resistance arm contains measurements"
            )
        if effective_resistance is not contract["effective_resistance"]:
            raise CalibrationLockError(
                "effective-resistance result disagrees with the frozen arm"
            )

        reference_safety = _validate_safety_diagnostics(
            row["reference_safety"], contract, "reference", reference
        )
        model_index = row["candidate_model_index"]
        if model_index == -1:
            safety_selection = reference
        else:
            owner_role = model_owner_role[(row["batch_id"], model_index)]
            safety_selection = _selection_from_role(
                batch,
                owner_role,
                adjacency,
                expected_work_order["plan_fingerprint"],
            )
        candidate_safety = _validate_safety_diagnostics(
            row["candidate_safety"], contract, "candidate", safety_selection
        )
        if (
            decoded["candidate_reciprocal_condition"]
            != candidate_safety["reciprocal_condition"]
            or decoded["reference_reciprocal_condition"]
            != reference_safety["reciprocal_condition"]
        ):
            raise CalibrationLockError(
                "calibration fidelity and safety condition numbers disagree"
            )
        output.append(
            {
                "batch_id": row["batch_id"],
                "result": SimpleNamespace(**decoded),
                "role": role,
            }
        )
    return output


def _resource_blocked_selection(selection, ceiling):
    return {
        **selection,
        "audit_solve_authorized": False,
        "confirmatory_claim_authorized": False,
        "details": {"ceiling_bytes": ceiling},
        "efficacy_or_resource_claim_authorized": False,
        "frozen_audit_roles": [],
        "required_audit_model_roles": [],
        "k_high": None,
        "k_low": None,
        "lock_mode": "blocked",
        "reason": "calibration_resource_ceiling_exceeded",
    }


def _finite_hex(value, label, *, nonnegative=False):
    if not isinstance(value, str):
        raise CalibrationLockError(f"{label} is not a hexadecimal float")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise CalibrationLockError(f"{label} is malformed") from exc
    if not math.isfinite(parsed) or (nonnegative and parsed < 0.0):
        raise CalibrationLockError(f"{label} is outside its finite range")
    return parsed


def _validate_screening_record(record, anchor, target, label):
    expected = {
        "anchor",
        "attenuation_threshold",
        "distance_metric",
        "maximum_observed_radius",
        "radius_lower",
        "radius_upper",
        "right_censored",
        "shell_attenuation",
    }
    if not isinstance(record, dict) or set(record) != expected:
        raise CalibrationLockError(f"{label} screening shape mismatch")
    if record["anchor"] != anchor or record["distance_metric"] == "":
        raise CalibrationLockError(f"{label} screening identity mismatch")
    threshold = _finite_hex(record["attenuation_threshold"], label)
    attenuation = _finite_hex(
        record["shell_attenuation"], label, nonnegative=True
    )
    lower = _finite_hex(record["radius_lower"], label, nonnegative=True)
    maximum = _finite_hex(
        record["maximum_observed_radius"], label, nonnegative=True
    )
    if threshold != target or attenuation > 1.0 + 1e-8 or lower > maximum:
        raise CalibrationLockError(f"{label} screening values mismatch")
    censored = record["right_censored"]
    if not isinstance(censored, bool):
        raise CalibrationLockError(f"{label} screening censor flag is malformed")
    if censored:
        if record["radius_upper"] is not None or lower != maximum:
            raise CalibrationLockError(f"{label} censored radius mismatch")
    else:
        upper = _finite_hex(record["radius_upper"], label, nonnegative=True)
        if not lower < upper <= maximum:
            raise CalibrationLockError(f"{label} radius bracket mismatch")
    return attenuation


def _validate_zero_alpha_precheck(record, contract):
    expected = {
        "boundary_harmonic_max",
        "cholesky_reconstruction_relative_error",
        "maximum_kirchhoff_relative_error",
        "maximum_precision_eigenvalue",
        "maximum_principle_violation",
        "minimum_precision_eigenvalue",
        "reciprocal_condition",
        "solve_relative_residual",
        "status",
    }
    if not isinstance(record, dict) or set(record) != expected:
        raise CalibrationLockError("zero-alpha precheck shape mismatch")
    values = {
        name: _finite_hex(record[name], f"zero-alpha {name}", nonnegative=True)
        for name in expected - {"status"}
    }
    if (
        record["status"] != "pass_without_added_leakage"
        or values["minimum_precision_eigenvalue"] <= 0.0
        or values["maximum_precision_eigenvalue"]
        < values["minimum_precision_eigenvalue"]
        or values["reciprocal_condition"] < contract["minimum_rcond"]
        or values["solve_relative_residual"]
        > contract["solve_residual_tolerance"]
        or values["maximum_kirchhoff_relative_error"]
        > contract["solve_residual_tolerance"]
        or values["maximum_principle_violation"]
        > contract["maximum_principle_tolerance"]
    ):
        raise CalibrationLockError("zero-alpha precheck evidence failed")


def _validate_safety_diagnostics(record, contract, label, expected_selection):
    expected = {
        "checks_passed",
        "cholesky_reconstruction_relative_error",
        "maximum_kirchhoff_relative_error",
        "maximum_normalized_source_response",
        "maximum_positive_off_diagonal",
        "maximum_principle_violation",
        "minimum_source_response",
        "multi_rhs_solve_relative_residual",
        "nodes",
        "reciprocal_condition",
        "selection_fingerprint",
        "strategy",
    }
    if not isinstance(record, dict) or set(record) != expected:
        raise CalibrationLockError(f"{label} safety-diagnostic shape mismatch")
    if (
        record["checks_passed"] is not True
        or not isinstance(record["nodes"], int)
        or isinstance(record["nodes"], bool)
        or record["nodes"] <= 0
        or not isinstance(record["strategy"], str)
        or not record["strategy"]
        or not isinstance(record["selection_fingerprint"], str)
        or re.fullmatch(r"[0-9a-f]{64}", record["selection_fingerprint"])
        is None
    ):
        raise CalibrationLockError(f"{label} safety identity is malformed")
    if (
        record["nodes"] != expected_selection.realized_nodes
        or record["strategy"] != expected_selection.strategy
        or record["selection_fingerprint"]
        != expected_selection.selection_fingerprint
    ):
        raise CalibrationLockError(
            f"{label} safety identity disagrees with the frozen selection"
        )
    numeric_names = expected - {
        "checks_passed",
        "nodes",
        "selection_fingerprint",
        "strategy",
    }
    values = {
        name: _finite_hex(record[name], f"{label} {name}")
        for name in numeric_names
    }
    nonnegative_names = numeric_names - {
        "minimum_source_response",
        "maximum_normalized_source_response",
    }
    if any(values[name] < 0.0 for name in nonnegative_names):
        raise CalibrationLockError(f"{label} safety diagnostic is negative")
    if (
        values["reciprocal_condition"] < contract["minimum_rcond"]
        or values["maximum_positive_off_diagonal"]
        > contract["m_matrix_tolerance"]
        or values["multi_rhs_solve_relative_residual"]
        > contract["solve_residual_tolerance"]
        or values["maximum_principle_violation"]
        > contract["maximum_principle_tolerance"]
        or values["maximum_kirchhoff_relative_error"]
        > contract["solve_residual_tolerance"]
        or values["maximum_normalized_source_response"]
        < 1.0 - contract["maximum_principle_tolerance"]
        or values["maximum_normalized_source_response"]
        > 1.0 + contract["maximum_principle_tolerance"]
    ):
        raise CalibrationLockError(f"{label} stored numerical gate failed")
    return values


def _validate_calibration_evidence(
    anchors, batches, fidelity, expected_work_order, alpha_top, contract
):
    expected = {
        (batch["batch_id"], anchor["node_id"]): (
            anchor["quartile_id"],
            next(
                shell["shell_nodes"]
                for shell in batch["shells"]
                if shell["anchor_node_id"] == anchor["node_id"]
            ),
        )
        for batch in expected_work_order["batches"]
        for anchor in batch["anchors_by_quartile"]
    }
    by_batch = {batch["batch_id"]: [] for batch in expected_work_order["batches"]}
    target = math.exp(-1.0)
    tolerance = float.fromhex(
        contract["calibration"]["bisection_relative_tolerance_hex"]
    )
    iteration_cap = contract["calibration"][
        "maximum_function_evaluations_per_anchor"
    ]
    for row in anchors:
        expected_anchor_fields = {
            "added_leakage_conductance",
            "anchor_node_id",
            "batch_id",
            "iterations",
            "minimality_certificate",
            "numerical_minimum_added_leakage",
            "quartile_id",
            "screening_at_batch_leakage",
            "screening_at_global_leakage",
            "screening_at_selected_leakage",
            "shell_nodes",
        }
        if not isinstance(row, dict) or set(row) != expected_anchor_fields:
            raise CalibrationLockError("anchor calibration shape mismatch")
        identity = (row.get("batch_id"), row.get("anchor_node_id"))
        if identity not in expected:
            raise CalibrationLockError("anchor calibration identity mismatch")
        quartile, shell = expected[identity]
        if row.get("quartile_id") != quartile or row.get("shell_nodes") != shell:
            raise CalibrationLockError("anchor calibration shell binding mismatch")
        added = _finite_hex(
            row.get("added_leakage_conductance"),
            "anchor added leakage",
            nonnegative=True,
        )
        numerical = _finite_hex(
            row.get("numerical_minimum_added_leakage"),
            "anchor numerical minimum",
            nonnegative=True,
        )
        iterations = row.get("iterations")
        if (
            numerical != 0.0
            or not isinstance(iterations, int)
            or isinstance(iterations, bool)
            or not 1 <= iterations <= iteration_cap
        ):
            raise CalibrationLockError("anchor calibration numerical contract mismatch")
        certificate = row.get("minimality_certificate")
        certificate_fields = {
            "attenuation_at_lower",
            "attenuation_at_upper",
            "bracket_seed_radius",
            "initial_lower_passed",
            "lower_added_leakage_conductance",
            "relative_tolerance",
            "target_attenuation",
            "upper_added_leakage_conductance",
        }
        if not isinstance(certificate, dict) or set(certificate) != certificate_fields:
            raise CalibrationLockError("minimality certificate is missing")
        lower = _finite_hex(
            certificate.get("lower_added_leakage_conductance"),
            "minimality lower leakage",
            nonnegative=True,
        )
        upper = _finite_hex(
            certificate.get("upper_added_leakage_conductance"),
            "minimality upper leakage",
            nonnegative=True,
        )
        lower_attenuation = _finite_hex(
            certificate.get("attenuation_at_lower"),
            "minimality lower attenuation",
            nonnegative=True,
        )
        upper_attenuation = _finite_hex(
            certificate.get("attenuation_at_upper"),
            "minimality upper attenuation",
            nonnegative=True,
        )
        certificate_target = _finite_hex(
            certificate.get("target_attenuation"), "minimality target"
        )
        certificate_tolerance = _finite_hex(
            certificate.get("relative_tolerance"), "minimality tolerance"
        )
        initial_passed = certificate.get("initial_lower_passed")
        attenuation_scale = max(
            abs(lower_attenuation),
            abs(upper_attenuation),
            np.finfo(float).tiny,
        )
        if (
            upper != added
            or lower > upper
            or certificate_target != target
            or certificate_tolerance != tolerance
            or certificate.get("bracket_seed_radius") != 3
            or not isinstance(initial_passed, bool)
            or upper_attenuation > target
            or upper_attenuation
            > lower_attenuation
            + 64.0 * np.finfo(float).eps * attenuation_scale
        ):
            raise CalibrationLockError("minimality certificate contract mismatch")
        if lower < numerical:
            raise CalibrationLockError(
                "minimality bracket falls below the numerical minimum"
            )
        relative_width = (upper - lower) / max(abs(upper), np.finfo(float).tiny)
        if relative_width > tolerance:
            raise CalibrationLockError("minimality bracket is too wide")
        if initial_passed:
            if (
                lower != upper
                or lower_attenuation != upper_attenuation
                or lower != numerical
            ):
                raise CalibrationLockError("collapsed minimality bracket mismatch")
        elif lower_attenuation <= target:
            raise CalibrationLockError("minimality lower endpoint does not miss")
        selected_attenuation = _validate_screening_record(
            row.get("screening_at_selected_leakage"), identity[1], target, "selected"
        )
        batch_attenuation = _validate_screening_record(
            row.get("screening_at_batch_leakage"), identity[1], target, "batch"
        )
        global_attenuation = _validate_screening_record(
            row.get("screening_at_global_leakage"), identity[1], target, "global"
        )
        selected_scale = max(
            abs(selected_attenuation),
            abs(upper_attenuation),
            np.finfo(float).tiny,
        )
        if (
            abs(selected_attenuation - upper_attenuation)
            > 128.0 * np.finfo(float).eps * selected_scale
            or batch_attenuation > selected_attenuation + 1e-8
            or batch_attenuation > target * (1.0 + tolerance)
            or global_attenuation > selected_attenuation + 1e-8
            or global_attenuation > target * (1.0 + tolerance)
        ):
            raise CalibrationLockError("anchor screening and minimality disagree")
        by_batch[identity[0]].append((added, iterations))
    batch_index = {row.get("batch_id"): row for row in batches}
    fidelity_index = {
        (row.get("batch_id"), row.get("role")): row for row in fidelity
    }
    observed_batch_maxima = []
    for batch_id, evidence in by_batch.items():
        row = batch_index[batch_id]
        expected_batch_fields = {
            "batch_id",
            "batch_maximum_added_leakage",
            "eigendecomposition_count",
            "factor_provenance",
            "global_alpha_top",
            "numerical_minimum_added_leakage",
            "total_attenuation_evaluations",
            "zero_alpha_precheck",
        }
        if not isinstance(row, dict) or set(row) != expected_batch_fields:
            raise CalibrationLockError("batch calibration shape mismatch")
        batch_maximum = _finite_hex(
            row.get("batch_maximum_added_leakage"),
            "batch maximum leakage",
            nonnegative=True,
        )
        global_alpha = _finite_hex(
            row.get("global_alpha_top"), "batch global alpha", nonnegative=True
        )
        numeric = _finite_hex(
            row.get("numerical_minimum_added_leakage"),
            "batch numerical minimum",
            nonnegative=True,
        )
        if (
            len(evidence) != EXPECTED_ANCHORS_PER_BATCH
            or batch_maximum != max(value for value, _ in evidence)
            or global_alpha != alpha_top
            or numeric != 0.0
            or not isinstance(row.get("eigendecomposition_count"), int)
            or isinstance(row.get("eigendecomposition_count"), bool)
            or row.get("eigendecomposition_count") != 1
            or not isinstance(row.get("total_attenuation_evaluations"), int)
            or isinstance(row.get("total_attenuation_evaluations"), bool)
            or row.get("total_attenuation_evaluations")
            != sum(count for _, count in evidence)
        ):
            raise CalibrationLockError("batch calibration evidence mismatch")
        _validate_zero_alpha_precheck(row.get("zero_alpha_precheck"), contract)
        factor = row.get("factor_provenance")
        factor_fields = {
            "batch_id",
            "candidate_build_count",
            "candidate_factorization_count",
            "candidate_model_index",
            "candidate_reference_reuse_count",
            "candidate_requested_result_count",
            "candidate_unique_model_count",
            "reference_build_count",
            "reference_factorization_count",
        }
        if not isinstance(factor, dict) or set(factor) != factor_fields:
            raise CalibrationLockError("batch factor provenance is missing")
        model_index = factor.get("candidate_model_index")
        unique = factor.get("candidate_unique_model_count")
        reuse = factor.get("candidate_reference_reuse_count")
        count_fields = (
            "candidate_build_count",
            "candidate_factorization_count",
            "candidate_reference_reuse_count",
            "candidate_requested_result_count",
            "candidate_unique_model_count",
            "reference_build_count",
            "reference_factorization_count",
        )
        if (
            factor.get("batch_id") != batch_id
            or any(
                not isinstance(factor.get(name), int)
                or isinstance(factor.get(name), bool)
                for name in count_fields
            )
            or factor.get("candidate_requested_result_count")
            != len(CANDIDATE_ROLES)
            or not 0 <= unique <= len(CANDIDATE_ROLES)
            or factor.get("candidate_build_count") != unique
            or factor.get("candidate_factorization_count") != unique
            or factor.get("reference_build_count") != 1
            or factor.get("reference_factorization_count") != 1
            or not isinstance(model_index, list)
            or len(model_index) != len(CANDIDATE_ROLES)
            or reuse != sum(index == -1 for index in model_index)
            or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < -1
                or index >= unique
                for index in model_index
            )
            or {index for index in model_index if index >= 0}
            != set(range(unique))
        ):
            raise CalibrationLockError("batch factor-count provenance mismatch")
        for role_index, role in enumerate(CANDIDATE_ROLES):
            fidelity_row = fidelity_index[(batch_id, role)]
            if fidelity_row.get("candidate_model_index") != model_index[role_index]:
                raise CalibrationLockError(
                    "fidelity row disagrees with factor-count provenance"
                )
        reference_safety = [
            fidelity_index[(batch_id, role)].get("reference_safety")
            for role in CANDIDATE_ROLES
        ]
        if any(value != reference_safety[0] for value in reference_safety[1:]):
            raise CalibrationLockError(
                "shared reference safety diagnostics disagree within a batch"
            )
        candidate_safety_by_index = {}
        for role_index, role in enumerate(CANDIDATE_ROLES):
            index = model_index[role_index]
            if index < 0:
                expected_safety = reference_safety[0]
            else:
                expected_safety = candidate_safety_by_index.setdefault(
                    index,
                    fidelity_index[(batch_id, role)].get("candidate_safety"),
                )
            if (
                fidelity_index[(batch_id, role)].get("candidate_safety")
                != expected_safety
            ):
                raise CalibrationLockError(
                    "reused model safety diagnostics disagree within a batch"
                )
        observed_batch_maxima.append(batch_maximum)
    observed_anchor_maximum = max(
        value for evidence in by_batch.values() for value, _ in evidence
    )
    if alpha_top != observed_anchor_maximum or alpha_top != max(observed_batch_maxima):
        raise CalibrationLockError("global alpha does not equal both registered maxima")


def _validate_execution_provenance(
    execution, core, contract, completed, expected_work_order
):
    expected_fields = {
        "actual_blas_identity",
        "audit_metrics_materialized",
        "audit_solves_executed",
        "batch_phase_timings",
        "calibration_only_work_order",
        "calibration_seconds",
        "fidelity_seconds",
        "observed_peak_rss_bytes",
        "peak_rss_ceiling_bytes",
        "timing_and_rss_outside_scientific_fingerprint",
        "total_seconds",
    }
    if not isinstance(execution, dict) or set(execution) != expected_fields:
        raise CalibrationLockError("execution provenance shape mismatch")
    if (
        execution["actual_blas_identity"] != core.get("actual_blas_identity")
        or execution["audit_solves_executed"] != 0
        or execution["audit_metrics_materialized"] is not False
        or execution["calibration_only_work_order"] is not True
        or execution["timing_and_rss_outside_scientific_fingerprint"]
        is not True
    ):
        raise CalibrationLockError("execution provenance violates audit isolation")

    def optional_elapsed(value, label):
        if value is None:
            return None
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise CalibrationLockError(f"{label} elapsed time is malformed")
        return float(value)

    total = optional_elapsed(execution["total_seconds"], "total")
    calibration = optional_elapsed(
        execution["calibration_seconds"], "calibration"
    )
    fidelity = optional_elapsed(execution["fidelity_seconds"], "fidelity")
    if total is None:
        raise CalibrationLockError("total elapsed time is missing")
    if completed and (calibration is None or fidelity is None):
        raise CalibrationLockError("completed calibration timings are missing")

    timings = execution["batch_phase_timings"]
    if not isinstance(timings, list):
        raise CalibrationLockError("batch timing provenance is malformed")
    expected_ids = {
        batch["batch_id"] for batch in expected_work_order["batches"]
    }
    if timings and len(timings) != len(expected_ids):
        raise CalibrationLockError("batch timing provenance is incomplete")
    if completed and len(timings) != EXPECTED_CALIBRATION_BATCHES:
        raise CalibrationLockError("completed calibration batch timings are missing")
    observed_ids = set()
    for row in timings:
        if (
            not isinstance(row, dict)
            or set(row)
            != {"batch_id", "calibration_seconds", "fidelity_seconds"}
            or row["batch_id"] in observed_ids
            or row["batch_id"] not in expected_ids
        ):
            raise CalibrationLockError("batch timing identity is malformed")
        observed_ids.add(row["batch_id"])
        for phase in ("calibration_seconds", "fidelity_seconds"):
            elapsed = optional_elapsed(row[phase], f"batch {phase}")
            if elapsed is None or elapsed > contract[
                "per_batch_elapsed_ceiling_seconds"
            ]:
                raise CalibrationLockError(
                    "batch elapsed time exceeds its frozen ceiling"
                )
    if timings and observed_ids != expected_ids:
        raise CalibrationLockError("batch timing identities are incomplete")

    observed_rss = execution["observed_peak_rss_bytes"]
    ceiling = execution["peak_rss_ceiling_bytes"]
    if (
        not isinstance(observed_rss, int)
        or isinstance(observed_rss, bool)
        or observed_rss < 0
        or ceiling != contract["peak_rss_ceiling_bytes"]
    ):
        raise CalibrationLockError("execution resource provenance is malformed")
    return observed_rss, ceiling


def _validate_lock_payload_structure(
    payloads, manifest, expected_work_order, contract
):
    work_order = _strict_json_bytes(
        payloads["calibration_work_order.json"], "calibration work order"
    )
    if work_order != expected_work_order:
        raise CalibrationLockError("calibration work order does not rederive")
    anchors = _strict_jsonl_bytes(
        payloads["anchor_calibrations.jsonl"], "anchor calibrations"
    )
    batches = _strict_jsonl_bytes(
        payloads["batch_calibration.jsonl"], "batch calibration"
    )
    fidelity = _strict_jsonl_bytes(
        payloads["calibration_fidelity.jsonl"], "calibration fidelity"
    )
    selection = _strict_json_bytes(payloads["selection.json"], "selection")
    execution = _strict_json_bytes(payloads["execution.json"], "execution")
    core = manifest["scientific_core"]
    if (
        not isinstance(selection, dict)
        or selection.get("lock_mode") != manifest["lock_mode"]
        or selection.get("reason") != manifest["reason"]
        or selection.get("audit_solve_authorized")
        is not manifest["audit_solve_authorized"]
        or selection.get("confirmatory_claim_authorized") is not False
    ):
        raise CalibrationLockError("selection and manifest disagree")
    if selection.get("alpha_top") != core.get("alpha_top_hex"):
        raise CalibrationLockError("selection and manifest alpha disagree")
    completed = manifest["calibration_completed"]
    observed_rss, ceiling = _validate_execution_provenance(
        execution, core, contract, completed, expected_work_order
    )
    resource_exceeded = observed_rss > ceiling
    resource_blocked = (
        selection.get("reason") == "calibration_resource_ceiling_exceeded"
    )
    if resource_exceeded is not resource_blocked:
        raise CalibrationLockError(
            "execution resource evidence disagrees with the lock decision"
        )
    if completed:
        if (
            len(anchors) != EXPECTED_CALIBRATION_ANCHORS
            or len(batches) != EXPECTED_CALIBRATION_BATCHES
            or len(fidelity)
            != EXPECTED_CALIBRATION_BATCHES * len(CANDIDATE_ROLES)
        ):
            raise CalibrationLockError("completed calibration artifacts are incomplete")
        identities = {
            (row.get("batch_id"), row.get("anchor_node_id")) for row in anchors
        }
        expected_anchor_identities = {
            (batch["batch_id"], anchor["node_id"])
            for batch in expected_work_order["batches"]
            for anchor in batch["anchors_by_quartile"]
        }
        if identities != expected_anchor_identities:
            raise CalibrationLockError("anchor calibration identities are not unique")
        expected_batch_ids = {
            batch["batch_id"] for batch in expected_work_order["batches"]
        }
        if {row.get("batch_id") for row in batches} != expected_batch_ids:
            raise CalibrationLockError("batch calibration identities are incomplete")
        fidelity_identities = {
            (row.get("batch_id"), row.get("role")) for row in fidelity
        }
        expected_fidelity_identities = {
            (batch_id, role)
            for batch_id in expected_batch_ids
            for role in CANDIDATE_ROLES
        }
        if fidelity_identities != expected_fidelity_identities:
            raise CalibrationLockError("fidelity identities are not unique")
        try:
            alpha_top = float.fromhex(selection["alpha_top"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationLockError("selection alpha is malformed") from exc
        _validate_calibration_evidence(
            anchors,
            batches,
            fidelity,
            expected_work_order,
            alpha_top,
            contract,
        )
        adjacency = _adjacency_from_work_order(expected_work_order)
        decoded_rows = _decoded_fidelity_rows(
            fidelity,
            contract,
            expected_work_order,
            adjacency,
            alpha_top,
        )
        expected_selection = _select_lock_mode(
            decoded_rows,
            expected_work_order,
            adjacency,
            contract,
            alpha_top=alpha_top,
        )
        if selection.get("reason") == "calibration_resource_ceiling_exceeded":
            expected_selection = _resource_blocked_selection(
                expected_selection, contract["peak_rss_ceiling_bytes"]
            )
        if selection != _hexify_scientific(expected_selection):
            raise CalibrationLockError("selection does not rederive from fidelity")
    elif anchors or batches or fidelity:
        raise CalibrationLockError("failed calibration must not contain partial results")
    elif (
        selection.get("lock_mode") != "blocked"
        or selection.get("k_low") is not None
        or selection.get("k_high") is not None
        or selection.get("frozen_audit_roles") != []
        or selection.get("required_audit_model_roles") != []
        or selection.get("efficacy_or_resource_claim_authorized") is not False
    ):
        raise CalibrationLockError("incomplete calibration selection is unsafe")
    return {
        "anchors": anchors,
        "batches": batches,
        "execution": execution,
        "fidelity": fidelity,
        "selection": selection,
        "work_order": work_order,
    }


def _assert_lock_path_and_overlap(
    lock_dir,
    *,
    plan_dir,
    receipt_dir,
    attempt_a_dir,
    attempt_b_dir,
    source_spec,
    relation_policy,
    input_ceiling,
):
    raw = Path(lock_dir)
    if hop_plan._has_symlink_ancestor(raw):
        raise CalibrationLockError("calibration lock path cannot contain a symlink")
    resolved = raw.resolve()
    if (
        hop_plan._path_is_within(resolved, REPO_ROOT.resolve())
        or hop_plan._has_git_ancestor(resolved.parent)
    ):
        raise CalibrationLockError("calibration lock cannot be inside a Git worktree")
    try:
        hop_plan._assert_no_output_input_overlap(
            resolved,
            (plan_dir, receipt_dir, attempt_a_dir, attempt_b_dir),
            (source_spec, relation_policy),
            source_spec,
            source_spec_maximum_size=input_ceiling,
        )
    except hop_plan.HopPlanError as exc:
        raise CalibrationLockError("calibration lock overlaps a verified input") from exc
    return resolved


def _cleanup_lock_staging(parent_binding, staging_binding, staging_leaf):
    """Remove only this runner's known, still-bound staging leaves."""

    try:
        hop_plan._assert_bound_directory(parent_binding, "lock output parent")
        hop_plan._assert_bound_directory(staging_binding, "calibration staging")
        entries = set(
            hop_plan._bounded_directory_names(
                staging_binding, len(ALL_LOCK_FILES), "calibration staging"
            )
        )
        if not entries.issubset(ALL_LOCK_FILES):
            raise CalibrationLockError("calibration staging inventory changed")
        for name in sorted(entries):
            observed = os.stat(
                name, dir_fd=staging_binding.fd, follow_symlinks=False
            )
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise CalibrationLockError("calibration staging artifact changed")
            os.unlink(name, dir_fd=staging_binding.fd)
        os.fsync(staging_binding.fd)
        hop_plan._assert_bound_directory(staging_binding, "calibration staging")
        hop_plan._assert_bound_directory(parent_binding, "lock output parent")
        os.rmdir(staging_leaf, dir_fd=parent_binding.fd)
        os.fsync(parent_binding.fd)
        hop_plan._assert_bound_directory(parent_binding, "lock output parent")
    except (OSError, hop_plan.HopPlanError) as exc:
        raise CalibrationLockError("calibration staging cleanup failed") from exc


def verify_lock(
    lock_dir,
    *,
    plan_dir,
    receipt_dir,
    attempt_a_dir,
    attempt_b_dir,
    source_spec,
    relation_policy,
):
    context = _capture_verified_context(
        plan_dir,
        receipt_dir=receipt_dir,
        attempt_a_dir=attempt_a_dir,
        attempt_b_dir=attempt_b_dir,
        source_spec=source_spec,
        relation_policy=relation_policy,
    )
    contract = _validated_bound_contract(context["manifest"])
    resolved = _assert_lock_path_and_overlap(
        lock_dir,
        plan_dir=plan_dir,
        receipt_dir=receipt_dir,
        attempt_a_dir=attempt_a_dir,
        attempt_b_dir=attempt_b_dir,
        source_spec=source_spec,
        relation_policy=relation_policy,
        input_ceiling=context["manifest"]["fingerprint_core"]["resource_contract"]
        ["planner_input_ceiling_bytes"],
    )
    try:
        binding = hop_plan._bind_directory(resolved, "calibration lock")
    except hop_plan.HopPlanError as exc:
        raise CalibrationLockError("calibration lock is unavailable") from exc
    try:
        if stat.S_IMODE(os.fstat(binding.fd).st_mode) != 0o700:
            raise CalibrationLockError("calibration lock directory mode mismatch")
        try:
            names = hop_plan._bounded_directory_names(
                binding, len(ALL_LOCK_FILES), "calibration lock"
            )
        except hop_plan.HopPlanError as exc:
            raise CalibrationLockError("calibration lock inventory is unavailable") from exc
        if set(names) != ALL_LOCK_FILES or len(names) != len(ALL_LOCK_FILES):
            raise CalibrationLockError("calibration lock inventory mismatch")
        manifest, manifest_data = _lock_manifest_from_binding(binding)
        if manifest["plan_manifest_record"] != context["captured_records"][
            "plan_manifest"
        ]:
            raise CalibrationLockError("calibration lock binds a different plan")
        core = manifest["scientific_core"]
        if (
            core.get("repository_commit") != _git_commit()
            or core.get("plan_fingerprint")
            != context["manifest"]["plan_fingerprint"]
            or core.get("implementation_records") != _implementation_records()
            or core.get("numeric_backend_contract") != contract["numeric"]
            or core.get("effective_resistance_arm")
            != ("enabled" if contract["effective_resistance"] else "omitted")
        ):
            raise CalibrationLockError("calibration implementation binding changed")
        payloads = _read_lock_payloads(binding, manifest)
        expected_work_order = _derive_calibration_work_order(context)
        _validate_lock_payload_structure(
            payloads, manifest, expected_work_order, contract
        )
        if _canonical_json(manifest) != manifest_data:
            raise CalibrationLockError("calibration manifest changed during verification")
        try:
            hop_plan._assert_private_directory_files(binding, "calibration lock")
        except hop_plan.HopPlanError as exc:
            raise CalibrationLockError("calibration lock privacy envelope failed") from exc
        final_context = _capture_verified_context(
            plan_dir,
            receipt_dir=receipt_dir,
            attempt_a_dir=attempt_a_dir,
            attempt_b_dir=attempt_b_dir,
            source_spec=source_spec,
            relation_policy=relation_policy,
        )
        if (
            final_context["captured_records"] != context["captured_records"]
            or final_context["manifest"] != context["manifest"]
        ):
            raise CalibrationLockError("verified inputs changed during lock verification")
        return manifest
    finally:
        binding.close()


def prepare_lock(
    *,
    lock_dir,
    local_root,
    plan_dir,
    receipt_dir,
    attempt_a_dir,
    attempt_b_dir,
    source_spec,
    relation_policy,
):
    context = _capture_verified_context(
        plan_dir,
        receipt_dir=receipt_dir,
        attempt_a_dir=attempt_a_dir,
        attempt_b_dir=attempt_b_dir,
        source_spec=source_spec,
        relation_policy=relation_policy,
    )
    contract = _validated_bound_contract(context["manifest"])
    try:
        target, parent_binding = hop_plan._validate_output_paths(
            lock_dir,
            local_root,
            (plan_dir, receipt_dir, attempt_a_dir, attempt_b_dir),
            (source_spec, relation_policy),
            source_spec,
            context["manifest"]["fingerprint_core"]["resource_contract"]
            ["planner_input_ceiling_bytes"],
        )
    except hop_plan.HopPlanError as exc:
        raise CalibrationLockError("calibration output path is not admissible") from exc

    staging_binding = None
    staging_leaf = None
    committed = False
    try:
        with _single_blas_thread() as runtime_blas:
            payloads, selection, contract_again, calibration_completed = (
                _derive_lock_payloads(context, runtime_blas)
            )
            if contract_again != contract:
                raise CalibrationLockError("calibration contract changed in memory")
            manifest = _build_lock_manifest(
                payloads,
                context=context,
                contract=contract,
                runtime_blas=runtime_blas,
                selection=selection,
                calibration_completed=calibration_completed,
            )

        # Reverify upstream bytes after all numerical work and before creating
        # the install transaction. This does not expose audit rows to a solve.
        final_context = _capture_verified_context(
            plan_dir,
            receipt_dir=receipt_dir,
            attempt_a_dir=attempt_a_dir,
            attempt_b_dir=attempt_b_dir,
            source_spec=source_spec,
            relation_policy=relation_policy,
        )
        if (
            final_context["captured_records"] != context["captured_records"]
            or final_context["manifest"] != context["manifest"]
        ):
            raise CalibrationLockError("verified inputs changed during calibration")

        try:
            staging_leaf, staging_binding = hop_plan._create_bound_staging(
                parent_binding, target.name
            )
            for name in ARTIFACT_NAMES:
                hop_plan._write_bound_bytes(staging_binding, name, payloads[name])
            hop_plan._write_bound_bytes(staging_binding, MARKER_NAME, MARKER_BYTES)
            hop_plan._write_bound_bytes(
                staging_binding, MANIFEST_NAME, _canonical_json(manifest)
            )
            os.fsync(staging_binding.fd)
        except hop_plan.HopPlanError as exc:
            raise CalibrationLockError("calibration staging transaction failed") from exc

        staged = verify_lock(
            staging_binding.path,
            plan_dir=plan_dir,
            receipt_dir=receipt_dir,
            attempt_a_dir=attempt_a_dir,
            attempt_b_dir=attempt_b_dir,
            source_spec=source_spec,
            relation_policy=relation_policy,
        )
        if staged != manifest:
            raise CalibrationLockError("staged calibration lock changed")
        try:
            hop_plan._assert_bound_directory(parent_binding, "lock output parent")
            hop_plan._assert_bound_directory(staging_binding, "calibration staging")
            hop_plan._rename_directory_noreplace(
                parent_binding, staging_leaf, target.name
            )
            staging_binding.path = target
            staging_leaf = target.name
            hop_plan._assert_bound_directory(staging_binding, "installed calibration lock")
            os.fsync(parent_binding.fd)
        except hop_plan.HopPlanError as exc:
            raise CalibrationLockError("calibration lock installation failed") from exc
        installed = verify_lock(
            target,
            plan_dir=plan_dir,
            receipt_dir=receipt_dir,
            attempt_a_dir=attempt_a_dir,
            attempt_b_dir=attempt_b_dir,
            source_spec=source_spec,
            relation_policy=relation_policy,
        )
        if installed != manifest:
            raise CalibrationLockError("installed calibration lock changed")
        os.fsync(parent_binding.fd)
        committed = True
        return manifest, (0 if manifest["accepted"] else 2)
    finally:
        try:
            if not committed and staging_binding is not None:
                _cleanup_lock_staging(
                    parent_binding, staging_binding, staging_leaf
                )
        finally:
            if staging_binding is not None:
                staging_binding.close()
            parent_binding.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--receipt-dir", required=True)
        command.add_argument("--attempt-a-dir", required=True)
        command.add_argument("--attempt-b-dir", required=True)
        command.add_argument("--source-spec", required=True)
        command.add_argument("--relation-policy", required=True)
        command.add_argument("--plan-dir", required=True)
        command.add_argument("--lock-dir", required=True)
        if name == "prepare":
            command.add_argument("--local-root", required=True)
            command.add_argument("--local-only", action="store_true", required=True)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    common = {
        "attempt_a_dir": args.attempt_a_dir,
        "attempt_b_dir": args.attempt_b_dir,
        "lock_dir": args.lock_dir,
        "plan_dir": args.plan_dir,
        "receipt_dir": args.receipt_dir,
        "relation_policy": args.relation_policy,
        "source_spec": args.source_spec,
    }
    try:
        if args.command == "prepare":
            manifest, exit_code = prepare_lock(
                **common,
                local_root=args.local_root,
            )
        else:
            manifest = verify_lock(**common)
            exit_code = 0 if manifest["accepted"] else 2
        print(
            json.dumps(
                {
                    "accepted": manifest["accepted"],
                    "audit_solve_authorized": manifest[
                        "audit_solve_authorized"
                    ],
                    "audit_solves_executed": 0,
                    "lock_mode": manifest["lock_mode"],
                    "reason": manifest["reason"],
                    "verified": args.command == "verify",
                },
                sort_keys=True,
            )
        )
        return exit_code
    except Exception:
        print(
            json.dumps({"error": "HOP calibration lock failed closed"}, sort_keys=True),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
