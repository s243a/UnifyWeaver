#!/usr/bin/env python3
"""Execute and seal the untouched Pearltrees HOP fidelity audit.

This runner is intentionally downstream of the immutable no-solve plan and
calibration-only lock.  It accepts no selector, alpha, endpoint, bootstrap, or
threshold override.  The only numerical work is the audit projection already
authorized by the lock; calibration rows never enter the work order.
"""

from __future__ import annotations

import argparse
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

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import prepare_pearltrees_hop_plan as hop_plan  # noqa: E402
import prepare_pearltrees_hop_calibration_lock as hop_lock  # noqa: E402
from unifyweaver.graph import bounded_diffusion_fidelity as fidelity_module  # noqa: E402
from unifyweaver.graph.bounded_diffusion_fidelity import (  # noqa: E402
    ProtectedSetCoverageError,
    evaluate_nested_bounded_domain_fidelity,
)


SCHEMA = "pearltrees-hop-untouched-audit-v1"
ALGORITHM = "locked-hop-untouched-audit-v1"
WORK_ORDER_SCHEMA = "pearltrees-hop-audit-work-order-v1"
MARKER_NAME = "LOCAL_ONLY_DO_NOT_PUBLISH"
MARKER_BYTES = b"LOCAL ONLY - DO NOT PUBLISH HOP AUDIT ARTIFACTS\n"
MANIFEST_NAME = "manifest.json"
SCIENTIFIC_ARTIFACT_NAMES = (
    "audit_work_order.json",
    "audit_fidelity.jsonl",
    "batch_status.jsonl",
    "bootstrap_statistics.jsonl",
    "decision.json",
)
OBSERVATIONAL_ARTIFACT_NAMES = ("execution.json",)
ARTIFACT_NAMES = SCIENTIFIC_ARTIFACT_NAMES + OBSERVATIONAL_ARTIFACT_NAMES
ALL_AUDIT_FILES = frozenset(ARTIFACT_NAMES + (MANIFEST_NAME, MARKER_NAME))
ROLE_ORDER = ("S_256", "S_512", "S_1024", "R_top")
EXPECTED_AUDIT_BATCHES = 24
EXPECTED_AUDIT_ANCHORS = 96
EXPECTED_ANCHORS_PER_BATCH = 4
EXPECTED_QUARTILES = ("q1", "q2", "q3", "q4")
MINIMUM_COMPLETE_BATCHES = 18
EXPECTED_BOOTSTRAP_REPLICATES = 9999
MAXIMUM_AUDIT_MANIFEST_BYTES = 8 * 1024 * 1024
MAXIMUM_AUDIT_ARTIFACT_BYTES = {
    "audit_work_order.json": 192 * 1024 * 1024,
    "audit_fidelity.jsonl": 96 * 1024 * 1024,
    "batch_status.jsonl": 8 * 1024 * 1024,
    "bootstrap_statistics.jsonl": 64 * 1024 * 1024,
    "decision.json": 8 * 1024 * 1024,
    "execution.json": 8 * 1024 * 1024,
}
AUDIT_DESIGN_PATH = HERE / "DESIGN_pearltrees_hop_audit.md"
LOCK_DESIGN_PATH = HERE / "DESIGN_pearltrees_hop_calibration_lock.md"
PROTOCOL_PATH = HERE / "PROTOCOL_bounded_diffusion_fidelity.md"
PLAN_DESIGN_PATH = HERE / "DESIGN_pearltrees_hop_plan.md"
LOCAL_DIFFUSION_PATH = REPO_ROOT / "src/unifyweaver/graph/local_diffusion.py"
FIDELITY_PATH = REPO_ROOT / "src/unifyweaver/graph/bounded_diffusion_fidelity.py"


class HopAuditError(ValueError):
    """Fail-closed audit integrity or operational error."""


def _canonical_json(value):
    try:
        return hop_plan._canonical_json(value)
    except hop_plan.HopPlanError as exc:
        raise HopAuditError("value is not canonical finite JSON") from exc


def _jsonl_bytes(records):
    return b"".join(_canonical_json(record) for record in records)


def _strict_json_bytes(data, label):
    try:
        return hop_plan._strict_json_bytes(data, label)
    except hop_plan.HopPlanError as exc:
        raise HopAuditError(f"invalid canonical JSON in {label}") from exc


def _strict_jsonl_bytes(data, label):
    try:
        return hop_plan._strict_jsonl_bytes(data, label)
    except hop_plan.HopPlanError as exc:
        raise HopAuditError(f"invalid canonical JSONL in {label}") from exc


def _content_record(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _file_record(path):
    return _content_record(Path(path).read_bytes())


def _head_blob_record(repo_relative_path):
    try:
        data = subprocess.check_output(
            ["git", "show", f"HEAD:{repo_relative_path}"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HopAuditError("required implementation is not present at HEAD") from exc
    return _content_record(data)


def _assert_records_at_head(records):
    for path, record in records.items():
        if _head_blob_record(path) != record:
            raise HopAuditError("scientific implementation differs from repository HEAD")


def _git_commit():
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip().lower()
    except (OSError, subprocess.SubprocessError) as exc:
        raise HopAuditError("repository commit is unavailable") from exc
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise HopAuditError("repository commit is malformed")
    return value


def _implementation_records():
    paths = (
        Path(__file__).resolve(),
        AUDIT_DESIGN_PATH,
        Path(hop_lock.__file__).resolve(),
        LOCK_DESIGN_PATH,
        Path(hop_plan.__file__).resolve(),
        PLAN_DESIGN_PATH,
        PROTOCOL_PATH,
        LOCAL_DIFFUSION_PATH,
        FIDELITY_PATH,
    )
    records = {
        str(path.relative_to(REPO_ROOT)): _file_record(path) for path in paths
    }
    _assert_records_at_head(records)
    return records


def _peak_rss_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _finite_hex(value):
    value = float(value)
    if not math.isfinite(value):
        raise HopAuditError("scientific evidence contains a nonfinite finite field")
    return value.hex()


def _decode_hex(value, label):
    if not isinstance(value, str):
        raise HopAuditError(f"{label} is not a hexadecimal float")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise HopAuditError(f"{label} is not a hexadecimal float") from exc
    if not math.isfinite(result):
        raise HopAuditError(f"{label} must be finite")
    return result


def _encode_extended(value):
    value = float(value)
    if math.isnan(value):
        raise HopAuditError("extended-real statistic is NaN")
    if value == math.inf:
        return "+inf"
    if value == -math.inf:
        return "-inf"
    return value.hex()


def _decode_extended(value, label):
    if value == "+inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    return _decode_hex(value, label)


def _manifest_without_seal(manifest):
    return {
        key: value
        for key, value in manifest.items()
        if key != "manifest_integrity_seal"
    }


def _selection_payload(lock_dir, lock_manifest):
    try:
        binding = hop_plan._bind_directory(Path(lock_dir).resolve(strict=True), "calibration lock")
    except (OSError, hop_plan.HopPlanError) as exc:
        raise HopAuditError("calibration lock is unavailable") from exc
    try:
        payloads = hop_lock._read_lock_payloads(binding, lock_manifest)
        selection = _strict_json_bytes(payloads["selection.json"], "lock selection")
        execution = _strict_json_bytes(payloads["execution.json"], "lock execution")
        return selection, execution, _content_record(
            hop_lock._read_bound_bytes(
                binding,
                MANIFEST_NAME,
                _content_record(_canonical_json(lock_manifest)),
                "calibration lock",
                maximum_size=hop_lock.MAXIMUM_LOCK_MANIFEST_BYTES,
            )
        )
    finally:
        binding.close()


def _capture_verified_context(
    *,
    lock_dir,
    plan_dir,
    receipt_dir,
    attempt_a_dir,
    attempt_b_dir,
    source_spec,
    relation_policy,
):
    lock_manifest = hop_lock.verify_lock(
        lock_dir,
        plan_dir=plan_dir,
        receipt_dir=receipt_dir,
        attempt_a_dir=attempt_a_dir,
        attempt_b_dir=attempt_b_dir,
        source_spec=source_spec,
        relation_policy=relation_policy,
    )
    if (
        lock_manifest.get("accepted") is not True
        or lock_manifest.get("calibration_completed") is not True
        or lock_manifest.get("audit_solve_authorized") is not True
        or lock_manifest.get("audit_solves_executed") != 0
        or lock_manifest.get("confirmatory_claim_authorized") is not False
        or lock_manifest.get("lock_mode") == "blocked"
    ):
        raise HopAuditError("calibration lock does not authorize an audit solve")
    plan_context = hop_lock._capture_verified_context(
        plan_dir,
        receipt_dir=receipt_dir,
        attempt_a_dir=attempt_a_dir,
        attempt_b_dir=attempt_b_dir,
        source_spec=source_spec,
        relation_policy=relation_policy,
    )
    contract = hop_lock._validated_bound_contract(plan_context["manifest"])
    selection, lock_execution, lock_manifest_record = _selection_payload(
        lock_dir, lock_manifest
    )
    core = lock_manifest["scientific_core"]
    if (
        core.get("repository_commit") != _git_commit()
        or core.get("plan_fingerprint") != plan_context["manifest"]["plan_fingerprint"]
        or core.get("actual_blas_identity") != lock_execution.get("actual_blas_identity")
    ):
        raise HopAuditError("calibration lock and plan bindings disagree")
    return {
        "contract": contract,
        "lock_execution": lock_execution,
        "lock_manifest": lock_manifest,
        "lock_manifest_record": lock_manifest_record,
        "plan_context": plan_context,
        "selection": selection,
    }


def _unique_index(records, keys, label):
    output = {}
    for record in records:
        try:
            key = tuple(record[name] for name in keys)
        except (KeyError, TypeError) as exc:
            raise HopAuditError(f"{label} row is malformed") from exc
        if key in output:
            raise HopAuditError(f"{label} contains duplicate rows")
        output[key] = record
    return output


def _derive_audit_work_order(context):
    """Project the immutable plan onto the 24 audit batches only."""

    plan_context = context["plan_context"]
    artifacts = plan_context["plan_artifacts"]
    batches = _strict_jsonl_bytes(artifacts["batches.jsonl"], "batches")
    domains = _strict_jsonl_bytes(artifacts["domains.jsonl"], "domains")
    boundaries = _strict_jsonl_bytes(artifacts["boundaries.jsonl"], "boundaries")
    if "audit_shells.jsonl" not in artifacts:
        raise HopAuditError("plan does not freeze audit screening shells")
    shells = _strict_jsonl_bytes(artifacts["audit_shells.jsonl"], "audit shells")
    bootstrap = _strict_jsonl_bytes(
        artifacts["bootstrap_multiplicities.jsonl"], "bootstrap multiplicities"
    )
    if len(bootstrap) != EXPECTED_BOOTSTRAP_REPLICATES:
        raise HopAuditError("bootstrap schedule must contain 9999 replicates")
    for expected, row in enumerate(bootstrap):
        counts = row.get("multiplicities")
        if (
            row.get("replicate_index") != expected
            or not isinstance(counts, list)
            or len(counts) != EXPECTED_AUDIT_BATCHES
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts)
            or sum(counts) != EXPECTED_AUDIT_BATCHES
            or sum(value > 0 for value in counts) < 10
        ):
            raise HopAuditError("bootstrap schedule row is malformed")

    selection = context["selection"]
    mode = selection.get("lock_mode")
    roles = selection.get("frozen_audit_roles")
    if mode not in {"finite_contrast", "absolute_only", "right_censored_diagnostics"}:
        raise HopAuditError("lock mode cannot enter the audit work order")
    if not isinstance(roles, list) or not roles:
        raise HopAuditError("lock did not freeze audit roles")
    if any(role not in ROLE_ORDER for role in roles) or len(set(roles)) != len(roles):
        raise HopAuditError("lock audit roles are malformed")
    if mode == "right_censored_diagnostics" and roles != list(ROLE_ORDER):
        raise HopAuditError("right-censored mode must freeze all diagnostic roles")
    if mode in {"finite_contrast", "absolute_only"}:
        if roles != [selection.get("k_low"), selection.get("k_high")]:
            raise HopAuditError("locked contrast roles disagree with selected endpoints")
    required_roles = selection.get("required_audit_model_roles")
    if not isinstance(required_roles, list):
        raise HopAuditError("lock did not freeze required audit model roles")
    expected_required = [
        role
        for role in ROLE_ORDER
        if role in set(roles).union({"S_1024", "R_top"})
    ]
    if required_roles != expected_required:
        raise HopAuditError("required audit model roles changed")
    candidate_roles = [role for role in required_roles if role != "R_top"]
    if not candidate_roles:
        raise HopAuditError("audit requires at least one bounded candidate")

    audit_batches = [row for row in batches if row.get("split") == "audit"]
    if len(audit_batches) != EXPECTED_AUDIT_BATCHES:
        raise HopAuditError("audit work order must contain 24 batches")
    if any(row.get("split") not in {"calibration", "audit"} for row in batches):
        raise HopAuditError("HOP plan contains an unknown split")
    domain_index = _unique_index(domains, ("batch_id", "role"), "domains")
    boundary_index = _unique_index(boundaries, ("batch_id", "role"), "boundaries")
    shell_index = _unique_index(shells, ("batch_id", "anchor_node_id"), "audit shells")
    adjacency = plan_context["adjacency"]
    all_anchors = set()
    work_batches = []
    required_roles = tuple(required_roles)
    for expected_index, batch in enumerate(
        sorted(audit_batches, key=lambda row: row.get("batch_index", -1))
    ):
        if (
            batch.get("batch_index") != expected_index
            or batch.get("batch_id") != f"audit-{expected_index:02d}"
            or batch.get("split") != "audit"
        ):
            raise HopAuditError("audit batch identity mismatch")
        anchors_by_quartile = batch.get("anchors_by_quartile")
        if not isinstance(anchors_by_quartile, list) or len(anchors_by_quartile) != 4:
            raise HopAuditError("audit batch must contain four anchors")
        quartiles = tuple(row.get("quartile_id") for row in anchors_by_quartile)
        anchors = tuple(row.get("node_id") for row in anchors_by_quartile)
        if quartiles != EXPECTED_QUARTILES or len(set(anchors)) != 4:
            raise HopAuditError("audit batch is not quartile-balanced")
        if any(not isinstance(node, str) or node not in adjacency for node in anchors):
            raise HopAuditError("audit anchor is absent from adjacency")
        if all_anchors.intersection(anchors):
            raise HopAuditError("audit anchors must be globally unique")
        all_anchors.update(anchors)

        role_rows = []
        previous_nodes = None
        for role in required_roles:
            key = (batch["batch_id"], role)
            try:
                domain = domain_index[key]
                boundary = boundary_index[key]
            except KeyError as exc:
                raise HopAuditError("audit role is missing") from exc
            nodes = domain.get("nodes")
            if not isinstance(nodes, list) or not nodes:
                raise HopAuditError("audit domain is malformed")
            node_ids = tuple(row.get("node_id") for row in nodes)
            if len(node_ids) != len(set(node_ids)) or any(node not in adjacency for node in node_ids):
                raise HopAuditError("audit domain nodes are malformed")
            if previous_nodes is not None and tuple(node_ids[: len(previous_nodes)]) != previous_nodes:
                raise HopAuditError("audit HOP domains are not nested")
            previous_nodes = node_ids
            expected_boundary = hop_plan._boundary_record(
                batch["batch_id"], role, domain["requested_nodes"], node_ids, adjacency
            )
            if boundary != expected_boundary:
                raise HopAuditError("audit boundary ledger mismatch")
            role_rows.append({"boundary": boundary, "domain": domain, "role": role})

        reference_nodes = {row["node_id"] for row in role_rows[-1]["domain"]["nodes"]}
        shell_rows = []
        for quartile_id, anchor in zip(quartiles, anchors):
            try:
                shell = shell_index[(batch["batch_id"], anchor)]
            except KeyError as exc:
                raise HopAuditError("audit shell is missing") from exc
            if (
                shell.get("strictly_interior_pass") is not True
                or shell.get("reasons") != []
                or shell.get("hop_radius") != 3
                or shell.get("target_attenuation") != "exp(-1)"
                or not isinstance(shell.get("shell_nodes"), list)
                or not shell["shell_nodes"]
                or any(node not in reference_nodes for node in shell["shell_nodes"])
            ):
                raise HopAuditError("audit shell contract mismatch")
            shell_rows.append({**shell, "quartile_id": quartile_id})

        work_batches.append(
            {
                "anchors_by_quartile": anchors_by_quartile,
                "batch_id": batch["batch_id"],
                "batch_index": batch["batch_index"],
                "protected_nodes": batch["protected_nodes"],
                "roles": role_rows,
                "shells": shell_rows,
                "split": "audit",
            }
        )
    if len(all_anchors) != EXPECTED_AUDIT_ANCHORS:
        raise HopAuditError("audit work order must contain 96 unique anchors")
    calibration_anchors = {
        row.get("node_id")
        for batch in batches
        if batch.get("split") == "calibration"
        for row in batch.get("anchors_by_quartile", [])
    }
    if all_anchors.intersection(calibration_anchors):
        raise HopAuditError("audit and calibration anchor identities overlap")
    audit_nodes = {
        row["node_id"]
        for batch in work_batches
        for role in batch["roles"]
        for row in role["domain"]["nodes"]
    }
    incident_adjacency = [
        {"neighbors": list(adjacency[node]), "node_id": node}
        for node in sorted(audit_nodes, key=hop_plan._typed_id_key)
    ]
    alpha_top = _decode_hex(
        context["lock_manifest"]["scientific_core"].get("alpha_top_hex"), "alpha_top"
    )
    work_order = {
        "algorithm": "frozen-plan-audit-only-projection-v1",
        "alpha_top_hex": alpha_top.hex(),
        "audit_anchor_count": len(all_anchors),
        "batch_count": len(work_batches),
        "batches": work_batches,
        "bootstrap_schedule_record": plan_context["captured_records"]["plan_artifacts"][
            "bootstrap_multiplicities.jsonl"
        ],
        "candidate_roles": candidate_roles,
        "contains_calibration_metrics_or_responses": False,
        "decision_roles": roles,
        "incident_adjacency": incident_adjacency,
        "lock_fingerprint": context["lock_manifest"]["lock_fingerprint"],
        "lock_manifest_record": context["lock_manifest_record"],
        "lock_mode": mode,
        "plan_fingerprint": plan_context["manifest"]["plan_fingerprint"],
        "plan_manifest_record": plan_context["captured_records"]["plan_manifest"],
        "reference_role": "R_top",
        "schema": WORK_ORDER_SCHEMA,
    }
    if any(batch["split"] != "audit" for batch in work_order["batches"]):
        raise HopAuditError("calibration row entered the numerical work order")
    return work_order, bootstrap


def _adjacency_from_work_order(work_order):
    rows = work_order.get("incident_adjacency")
    if not isinstance(rows, list) or not rows:
        raise HopAuditError("audit incident adjacency is missing")
    adjacency = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"neighbors", "node_id"}:
            raise HopAuditError("audit incident adjacency is malformed")
        node = row.get("node_id")
        neighbors = row.get("neighbors")
        if (
            node in adjacency
            or not isinstance(node, str)
            or not isinstance(neighbors, list)
            or len(neighbors) != len(set(neighbors))
        ):
            raise HopAuditError("audit incident adjacency is malformed")
        adjacency[node] = tuple(neighbors)
    required = {
        item["node_id"]
        for batch in work_order["batches"]
        for role in batch["roles"]
        for item in role["domain"]["nodes"]
    }
    if set(adjacency) != required:
        raise HopAuditError("audit adjacency projection is not exact")
    return adjacency


def _metric_evidence(
    batch,
    role,
    result,
    candidate_safety,
    reference_safety,
    screening,
    candidate_model_index=None,
):
    anchors = tuple(row["node_id"] for row in batch["anchors_by_quartile"])
    source_index = {node: index for index, node in enumerate(result.source_nodes)}
    if set(source_index) != set(anchors) or len(source_index) != len(anchors):
        raise HopAuditError("fidelity sources differ from the frozen batch anchors")
    vectors = {
        "maximum_h_absolute_error": result.per_anchor_maximum_h_absolute_error,
        "rank_inversion_fraction": result.per_anchor_rank_inversion_fraction,
        "raw_relative_l2_error": result.per_anchor_raw_relative_l2_error,
        "source_diagonal_relative_error": result.per_anchor_source_diagonal_relative_error,
        "top8_overlap": result.per_anchor_top_k_overlap,
    }
    if result.effective_resistance_evaluated:
        vectors["effective_resistance_relative_error"] = (
            result.per_anchor_effective_resistance_relative_error
        )
    anchor_rows = []
    candidate_cut = result.per_anchor_candidate_cut_current_fraction
    reference_cut = result.per_anchor_reference_cut_current_fraction
    screening_by_anchor = {}
    for item in screening:
        anchor = item.get("anchor") if isinstance(item, dict) else None
        if anchor in screening_by_anchor:
            raise HopAuditError("audit screening contains duplicate anchors")
        screening_by_anchor[anchor] = item
    if set(screening_by_anchor) != set(anchors):
        raise HopAuditError("audit screening differs from the frozen batch anchors")
    ordered_screening = [screening_by_anchor[anchor] for anchor in anchors]
    for quartile_id, anchor in zip(EXPECTED_QUARTILES, anchors):
        index = source_index[anchor]
        anchor_rows.append(
            {
                "anchor_node_id": anchor,
                "candidate_cut_current_fraction": _finite_hex(candidate_cut[index]),
                "metrics": {
                    name: _finite_hex(values[index]) for name, values in vectors.items()
                },
                "quartile_id": quartile_id,
                "reference_cut_current_fraction": _finite_hex(reference_cut[index]),
            }
        )
    full_result = hop_lock._hexify_scientific(
        hop_lock._strip_timing_fields(result.as_dict())
    )
    return {
        "anchors": anchor_rows,
        "batch_id": batch["batch_id"],
        "boundary_harmonic_max_hex": _finite_hex(result.candidate_boundary_harmonic_max),
        "candidate_nodes": result.candidate_nodes,
        "candidate_model_index": candidate_model_index,
        "candidate_safety": hop_lock._hexify_scientific(candidate_safety.as_dict()),
        "effective_resistance_evaluated": bool(result.effective_resistance_evaluated),
        "full_timing_free_result": full_result,
        "protected_candidate_fraction_hex": _finite_hex(result.protected_candidate_fraction),
        "protected_reference_fraction_hex": _finite_hex(result.protected_reference_fraction),
        "reference_nodes": result.reference_nodes,
        "reference_safety": hop_lock._hexify_scientific(reference_safety.as_dict()),
        "role": role,
        "screening": hop_lock._hexify_scientific(ordered_screening),
    }


def _run_audit_batches(work_order, contract):
    adjacency = _adjacency_from_work_order(work_order)
    alpha = _decode_hex(work_order["alpha_top_hex"], "audit alpha")
    candidate_roles = tuple(work_order["candidate_roles"])
    fidelity_rows = []
    statuses = []
    timings = []
    evaluator_calls = 0
    for batch_position, batch in enumerate(work_order["batches"]):
        started = time.perf_counter()
        try:
            evaluator_calls += 1
            candidates = tuple(
                hop_lock._selection_from_role(
                    batch, role, adjacency, work_order["plan_fingerprint"]
                )
                for role in candidate_roles
            )
            reference = hop_lock._selection_from_role(
                batch, "R_top", adjacency, work_order["plan_fingerprint"]
            )
            shells = {
                row["anchor_node_id"]: tuple(row["shell_nodes"])
                for row in batch["shells"]
            }
            evaluated = evaluate_nested_bounded_domain_fidelity(
                candidates,
                reference,
                protected_nodes=tuple(batch["protected_nodes"]),
                intrinsic_leakage_conductance=alpha,
                rank_top_k=8,
                minimum_reciprocal_condition=contract["minimum_rcond"],
                include_effective_resistance=contract["effective_resistance"],
                screening_shell_nodes_by_anchor=shells,
                screening_attenuation_threshold=math.exp(-1.0),
            )
            elapsed = time.perf_counter() - started
            if elapsed > contract["per_batch_elapsed_ceiling_seconds"]:
                raise np.linalg.LinAlgError("audit batch exceeded its elapsed ceiling")
            if _peak_rss_bytes() > contract["peak_rss_ceiling_bytes"]:
                raise np.linalg.LinAlgError("audit process exceeded its peak RSS ceiling")
            if (
                evaluated.reference_build_count != 1
                or evaluated.reference_factorization_count != 1
                or evaluated.candidate_requested_result_count != len(candidate_roles)
                or evaluated.candidate_build_count != evaluated.candidate_unique_model_count
                or evaluated.candidate_factorization_count != evaluated.candidate_unique_model_count
                or evaluated.reference_anchor_screening is None
            ):
                raise np.linalg.LinAlgError("audit shared-factor provenance contract failed")
            screening = [item.__dict__ for item in evaluated.reference_anchor_screening]
            pending_rows = []
            role_timings = []
            for index, (role, result) in enumerate(zip(candidate_roles, evaluated)):
                diagnostic_index = evaluated.candidate_model_index[index]
                candidate_safety = (
                    evaluated.reference_model_diagnostics
                    if diagnostic_index == -1
                    else evaluated.candidate_model_diagnostics[diagnostic_index]
                )
                pending_rows.append(
                    _metric_evidence(
                        batch,
                        role,
                        result,
                        candidate_safety,
                        evaluated.reference_model_diagnostics,
                        screening,
                        candidate_model_index=diagnostic_index,
                    )
                )
                role_timings.append(
                    {
                        "candidate_build_seconds": result.candidate_build_seconds,
                        "candidate_selection_seconds": result.candidate_selection_seconds,
                        "reference_build_seconds": result.reference_build_seconds,
                        "reference_selection_seconds": result.reference_selection_seconds,
                        "role": role,
                        "solve_seconds": result.solve_seconds,
                    }
                )
            fidelity_rows.extend(pending_rows)
            statuses.append(
                {
                    "batch_id": batch["batch_id"],
                    "batch_index": batch["batch_index"],
                    "complete": True,
                    "failure_class": None,
                    "failure_details": None,
                    "failure_kind": None,
                    "factor_provenance": {
                        "candidate_build_count": evaluated.candidate_build_count,
                        "candidate_factorization_count": evaluated.candidate_factorization_count,
                        "candidate_model_index": list(evaluated.candidate_model_index),
                        "candidate_reference_reuse_count": evaluated.candidate_reference_reuse_count,
                        "candidate_requested_result_count": evaluated.candidate_requested_result_count,
                        "candidate_unique_model_count": evaluated.candidate_unique_model_count,
                        "reference_build_count": evaluated.reference_build_count,
                        "reference_factorization_count": evaluated.reference_factorization_count,
                    },
                    "roles_completed": list(candidate_roles),
                }
            )
            timings.append(
                {
                    "batch_id": batch["batch_id"],
                    "fidelity_seconds": elapsed,
                    "role_timings": role_timings,
                }
            )
        except ProtectedSetCoverageError as exc:
            elapsed = time.perf_counter() - started
            statuses.append(
                {
                    "batch_id": batch["batch_id"],
                    "batch_index": batch["batch_index"],
                    "complete": False,
                    "failure_class": type(exc).__name__,
                    "failure_details": {
                        "missing_candidate_nodes": list(exc.missing_candidate),
                        "missing_reference_nodes": list(exc.missing_reference),
                    },
                    "failure_kind": "registered_coverage_failure",
                    "factor_provenance": None,
                    "roles_completed": [],
                }
            )
            timings.append({"batch_id": batch["batch_id"], "fidelity_seconds": elapsed, "role_timings": []})
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            elapsed = time.perf_counter() - started
            statuses.append(
                {
                    "batch_id": batch["batch_id"],
                    "batch_index": batch["batch_index"],
                    "complete": False,
                    "failure_class": type(exc).__name__,
                    "failure_details": None,
                    "failure_kind": "deterministic_safety_failure",
                    "factor_provenance": None,
                    "roles_completed": [],
                }
            )
            timings.append({"batch_id": batch["batch_id"], "fidelity_seconds": elapsed, "role_timings": []})
            for remaining in work_order["batches"][batch_position + 1 :]:
                statuses.append(
                    {
                        "batch_id": remaining["batch_id"],
                        "batch_index": remaining["batch_index"],
                        "complete": False,
                        "failure_class": None,
                        "failure_details": None,
                        "failure_kind": "not_attempted_after_global_block",
                        "factor_provenance": None,
                        "roles_completed": [],
                    }
                )
                timings.append(
                    {
                        "batch_id": remaining["batch_id"],
                        "fidelity_seconds": 0.0,
                        "role_timings": [],
                    }
                )
            break
    return fidelity_rows, statuses, timings, evaluator_calls


def _observed_order(values, q, *, upper):
    if not values or any(math.isnan(float(value)) for value in values):
        raise HopAuditError("order statistic requires nonempty non-NaN values")
    ordered = sorted(float(value) for value in values)
    position = q * (len(ordered) - 1)
    index = math.ceil(position) if upper else math.floor(position)
    return ordered[index]


def _extended_log_ratio(numerator, denominator):
    numerator = float(numerator)
    denominator = float(denominator)
    if numerator < 0.0 or denominator < 0.0 or not (
        math.isfinite(numerator) and math.isfinite(denominator)
    ):
        raise HopAuditError("error ratios require finite nonnegative operands")
    if numerator == 0.0 and denominator == 0.0:
        return 0.0
    if numerator == 0.0:
        return -math.inf
    if denominator == 0.0:
        return math.inf
    return math.log(numerator) - math.log(denominator)


def _mean_extended(values, weights=None):
    values = tuple(float(value) for value in values)
    if not values or any(math.isnan(value) for value in values):
        raise HopAuditError("extended mean requires nonempty non-NaN values")
    if weights is None:
        weights = (1,) * len(values)
    if len(weights) != len(values):
        raise HopAuditError("extended mean weights are misaligned")
    active = [(value, int(weight)) for value, weight in zip(values, weights) if int(weight) > 0]
    if not active:
        raise HopAuditError("bootstrap replicate has zero retained multiplicity")
    signs = {math.copysign(1.0, value) for value, weight in active if math.isinf(value)}
    if len(signs) > 1:
        raise HopAuditError("extended mean mixes positive and negative infinity")
    if signs:
        return math.inf if 1.0 in signs else -math.inf
    total_weight = sum(weight for _value, weight in active)
    return sum(value * weight for value, weight in active) / total_weight


def _evidence_index(fidelity_rows):
    index = {}
    for row in fidelity_rows:
        key = (row.get("batch_id"), row.get("role"))
        if key in index:
            raise HopAuditError("audit fidelity contains duplicate rows")
        anchors = row.get("anchors")
        if not isinstance(anchors, list) or len(anchors) != 4:
            raise HopAuditError("audit fidelity anchor evidence is malformed")
        if tuple(item.get("quartile_id") for item in anchors) != EXPECTED_QUARTILES:
            raise HopAuditError("audit fidelity lost quartile ordering")
        index[key] = row
    return index


def _expected_factor_provenance(batch, candidate_roles, adjacency, plan_fingerprint):
    reference = hop_lock._selection_from_role(
        batch, "R_top", adjacency, plan_fingerprint
    )
    reference_key = (reference.domain.nodes, reference.domain.neighbors)
    operator_index = {}
    owner_by_index = {}
    indices = []
    for role in candidate_roles:
        selection = hop_lock._selection_from_role(
            batch, role, adjacency, plan_fingerprint
        )
        key = (selection.domain.nodes, selection.domain.neighbors)
        if key == reference_key:
            indices.append(-1)
            continue
        index = operator_index.get(key)
        if index is None:
            index = len(operator_index)
            operator_index[key] = index
            owner_by_index[index] = role
        indices.append(index)
    unique = len(operator_index)
    return {
        "candidate_build_count": unique,
        "candidate_factorization_count": unique,
        "candidate_model_index": indices,
        "candidate_reference_reuse_count": indices.count(-1),
        "candidate_requested_result_count": len(candidate_roles),
        "candidate_unique_model_count": unique,
        "reference_build_count": 1,
        "reference_factorization_count": 1,
    }, owner_by_index


_FULL_RESULT_TIMING_FIELDS = frozenset(
    {
        "candidate_build_seconds",
        "candidate_selection_seconds",
        "reference_build_seconds",
        "reference_selection_seconds",
        "solve_seconds",
    }
)


def _decode_full_fidelity_result(record, contract, label):
    """Reconstruct the typed result so its native invariants run on replay."""

    fields = tuple(
        fidelity_module.BoundedFidelityResult.__dataclass_fields__.values()
    )
    expected = {
        field.name for field in fields if field.name not in _FULL_RESULT_TIMING_FIELDS
    }
    if not isinstance(record, dict) or set(record) != expected:
        raise HopAuditError(f"{label} full fidelity-result shape mismatch")
    values = {}
    try:
        for field in fields:
            name = field.name
            annotation = str(field.type)
            if name in _FULL_RESULT_TIMING_FIELDS:
                values[name] = 0.0
                continue
            value = record[name]
            if annotation == "float":
                values[name] = _decode_hex(value, f"{label} {name}")
            elif annotation == "float | None":
                values[name] = (
                    None if value is None else _decode_hex(value, f"{label} {name}")
                )
            elif annotation == "tuple":
                if not isinstance(value, list):
                    raise HopAuditError(f"{label} {name} is not a list")
                if name in {"protected_nodes", "source_nodes"}:
                    values[name] = tuple(value)
                else:
                    values[name] = tuple(
                        _decode_hex(item, f"{label} {name}") for item in value
                    )
            elif annotation == "tuple | None":
                if value is None:
                    values[name] = None
                elif isinstance(value, list):
                    values[name] = tuple(
                        _decode_hex(item, f"{label} {name}") for item in value
                    )
                else:
                    raise HopAuditError(f"{label} {name} is not a list or null")
            elif annotation == "int":
                if not isinstance(value, int) or isinstance(value, bool):
                    raise HopAuditError(f"{label} {name} is not an integer")
                values[name] = value
            elif annotation == "bool":
                if not isinstance(value, bool):
                    raise HopAuditError(f"{label} {name} is not boolean")
                values[name] = value
            elif annotation == "str":
                if not isinstance(value, str):
                    raise HopAuditError(f"{label} {name} is not text")
                values[name] = value
            elif annotation == "str | None":
                if value is not None and not isinstance(value, str):
                    raise HopAuditError(f"{label} {name} is not text or null")
                values[name] = value
            else:  # A new result field must receive an explicit replay contract.
                raise HopAuditError(
                    f"{label} has an unsupported result field type for {name}"
                )
        result = fidelity_module.BoundedFidelityResult(**values)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, HopAuditError):
            raise
        raise HopAuditError(f"{label} full fidelity-result invariant failed") from exc

    nonnegative = (
        "raw_relative_l2_error_90th_percentile",
        "maximum_h_absolute_error_90th_percentile",
        "rank_inversion_fraction_90th_percentile",
        "top_k_overlap_10th_percentile",
        "protected_candidate_fraction",
        "protected_reference_fraction",
        "maximum_raw_absolute_error",
        "maximum_raw_relative_error",
        "raw_relative_frobenius_error",
        "maximum_h_absolute_error",
        "h_root_mean_square_error",
        "mean_top_k_overlap",
        "minimum_top_k_overlap",
        "candidate_boundary_harmonic_max",
        "reference_boundary_harmonic_max",
        "candidate_cut_current_fraction_max",
        "reference_cut_current_fraction_max",
        "candidate_reciprocal_condition",
        "reference_reciprocal_condition",
        "closure_mass_fraction",
        "closure_total_transfer_mass",
        "closure_total_self_return_mass",
    )
    if any(getattr(result, name) < 0.0 for name in nonnegative):
        raise HopAuditError(f"{label} full fidelity result contains a negative metric")
    optional_nonnegative = (
        "maximum_effective_resistance_absolute_error",
        "maximum_effective_resistance_relative_error",
        "effective_resistance_relative_error_90th_percentile",
    )
    if any(
        value is not None and value < 0.0
        for value in (getattr(result, name) for name in optional_nonnegative)
    ):
        raise HopAuditError(f"{label} full resistance result contains a negative metric")
    tolerance = contract["maximum_principle_tolerance"]
    unit_interval = (
        result.top_k_overlap_10th_percentile,
        result.rank_inversion_fraction_90th_percentile,
        result.protected_candidate_fraction,
        result.protected_reference_fraction,
        result.mean_top_k_overlap,
        result.minimum_top_k_overlap,
        result.candidate_boundary_harmonic_max,
        result.reference_boundary_harmonic_max,
        result.candidate_cut_current_fraction_max,
        result.reference_cut_current_fraction_max,
        result.closure_mass_fraction,
    )
    if any(value > 1.0 + tolerance for value in unit_interval):
        raise HopAuditError(f"{label} full fidelity result exceeds a unit interval")
    if not (
        -1.0 <= result.mean_kendall_rank_agreement <= 1.0
        and -1.0 <= result.minimum_kendall_rank_agreement <= 1.0
    ):
        raise HopAuditError(f"{label} Kendall result is outside [-1, 1]")
    if (
        result.candidate_reciprocal_condition < contract["minimum_rcond"]
        or result.reference_reciprocal_condition < contract["minimum_rcond"]
        or result.candidate_cut_current_fraction_max
        != max(result.per_anchor_candidate_cut_current_fraction)
        or result.reference_cut_current_fraction_max
        != max(result.per_anchor_reference_cut_current_fraction)
    ):
        raise HopAuditError(f"{label} full fidelity numerical summary is inconsistent")
    return result


def _validate_evidence_against_work_order(work_order, fidelity_rows, statuses, contract):
    index = _evidence_index(fidelity_rows)
    complete_ids = {
        row["batch_id"] for row in statuses if row.get("complete") is True
    }
    expected = {
        (batch_id, role)
        for batch_id in complete_ids
        for role in work_order["candidate_roles"]
    }
    if set(index) != expected:
        raise HopAuditError("audit fidelity evidence is not the complete-batch rectangle")
    batches = {row["batch_id"]: row for row in work_order["batches"]}
    adjacency = _adjacency_from_work_order(work_order)
    factor_by_batch = {}
    owner_by_batch = {}
    for batch_id, batch in batches.items():
        factor_by_batch[batch_id], owner_by_batch[batch_id] = (
            _expected_factor_provenance(
                batch,
                work_order["candidate_roles"],
                adjacency,
                work_order["plan_fingerprint"],
            )
        )
    alpha_by_batch = {}
    screening_by_batch = {}
    alpha_value = _decode_hex(work_order["alpha_top_hex"], "audit alpha")
    for (batch_id, role), row in index.items():
        batch = batches[batch_id]
        candidate = hop_lock._selection_from_role(
            batch, role, adjacency, work_order["plan_fingerprint"]
        )
        reference = hop_lock._selection_from_role(
            batch, "R_top", adjacency, work_order["plan_fingerprint"]
        )
        role_position = work_order["candidate_roles"].index(role)
        expected_model_index = factor_by_batch[batch_id]["candidate_model_index"][
            role_position
        ]
        expected_row_fields = {
            "anchors",
            "batch_id",
            "boundary_harmonic_max_hex",
            "candidate_model_index",
            "candidate_nodes",
            "candidate_safety",
            "effective_resistance_evaluated",
            "full_timing_free_result",
            "protected_candidate_fraction_hex",
            "protected_reference_fraction_hex",
            "reference_nodes",
            "reference_safety",
            "role",
            "screening",
        }
        if not isinstance(row, dict) or set(row) != expected_row_fields:
            raise HopAuditError("audit fidelity row shape mismatch")
        full = row.get("full_timing_free_result")
        full_result = _decode_full_fidelity_result(
            full, contract, f"audit {batch_id} {role}"
        )
        expected_sources = sorted(
            (item["node_id"] for item in batch["anchors_by_quartile"]),
            key=hop_plan._typed_id_key,
        )
        expected_anchor_sequence = [
            (item["node_id"], item["quartile_id"])
            for item in batch["anchors_by_quartile"]
        ]
        expected_protected = sorted(
            batch["protected_nodes"], key=fidelity_module._stable_key
        )
        if (
            full_result.candidate_strategy != f"frozen_hop_{role}"
            or full_result.reference_strategy != "frozen_hop_R_top"
            or full_result.candidate_selection_fingerprint
            != candidate.selection_fingerprint
            or full_result.reference_selection_fingerprint
            != reference.selection_fingerprint
            or list(full_result.source_nodes) != expected_sources
            or list(full_result.protected_nodes) != expected_protected
            or full_result.protected_nodes_count != len(expected_protected)
            or full_result.rank_top_k != 8
            or full_result.rank_excludes_source is not True
            or full_result.candidate_nodes != row.get("candidate_nodes")
            or full_result.reference_nodes != row.get("reference_nodes")
            or row.get("candidate_nodes") != candidate.realized_nodes
            or row.get("reference_nodes") != reference.realized_nodes
            or full_result.protected_candidate_fraction
            != len(expected_protected) / candidate.realized_nodes
            or full_result.protected_reference_fraction
            != len(expected_protected) / reference.realized_nodes
            or full_result.effective_resistance_evaluated
            is not contract["effective_resistance"]
            or row.get("effective_resistance_evaluated")
            is not contract["effective_resistance"]
            or full_result.closure_policy != "none_full_dirichlet_beta"
            or full_result.closure_edges != 0
            or full_result.closure_approximation_limits_apply is not False
            or row.get("candidate_model_index") != expected_model_index
            or [
                (item.get("anchor_node_id"), item.get("quartile_id"))
                for item in row["anchors"]
            ]
            != expected_anchor_sequence
        ):
            raise HopAuditError("audit fidelity structural provenance changed")
        alpha = full_result.alpha_fingerprint
        if not isinstance(alpha, str) or re.fullmatch(r"[0-9a-f]{64}", alpha) is None:
            raise HopAuditError("audit alpha fingerprint is malformed")
        expected_alpha = fidelity_module._fingerprint(
            [
                [fidelity_module._stable_node_token(node), alpha_value.hex()]
                for node in sorted(
                    reference.domain.nodes,
                    key=fidelity_module._stable_key,
                )
            ]
        )
        if alpha != expected_alpha:
            raise HopAuditError("audit alpha fingerprint differs from the frozen scalar")
        previous_alpha = alpha_by_batch.setdefault(batch_id, alpha)
        if previous_alpha != alpha:
            raise HopAuditError("audit roles did not share one batch alpha fingerprint")
        vector_map = {
            "maximum_h_absolute_error": "per_anchor_maximum_h_absolute_error",
            "rank_inversion_fraction": "per_anchor_rank_inversion_fraction",
            "raw_relative_l2_error": "per_anchor_raw_relative_l2_error",
            "source_diagonal_relative_error": "per_anchor_source_diagonal_relative_error",
            "top8_overlap": "per_anchor_top_k_overlap",
        }
        if contract["effective_resistance"]:
            vector_map["effective_resistance_relative_error"] = (
                "per_anchor_effective_resistance_relative_error"
            )
        full_source_index = {
            node: i for i, node in enumerate(full["source_nodes"])
        }
        expected_metric_names = set(vector_map)
        for anchor in row["anchors"]:
            if (
                not isinstance(anchor, dict)
                or set(anchor)
                != {
                    "anchor_node_id",
                    "candidate_cut_current_fraction",
                    "metrics",
                    "quartile_id",
                    "reference_cut_current_fraction",
                }
                or not isinstance(anchor.get("metrics"), dict)
                or set(anchor["metrics"]) != expected_metric_names
            ):
                raise HopAuditError("compact audit anchor evidence is malformed")
            position = full_source_index[anchor["anchor_node_id"]]
            for compact_name, full_name in vector_map.items():
                compact = _decode_hex(anchor["metrics"][compact_name], compact_name)
                expanded = _decode_hex(full[full_name][position], full_name)
                if compact != expanded:
                    raise HopAuditError("compact audit metric differs from full evidence")
            candidate_cut = _decode_hex(
                anchor["candidate_cut_current_fraction"], "candidate cut current"
            )
            reference_cut = _decode_hex(
                anchor["reference_cut_current_fraction"], "reference cut current"
            )
            if (
                candidate_cut
                != _decode_hex(
                    full["per_anchor_candidate_cut_current_fraction"][position],
                    "full candidate cut current",
                )
                or reference_cut
                != _decode_hex(
                    full["per_anchor_reference_cut_current_fraction"][position],
                    "full reference cut current",
                )
            ):
                raise HopAuditError("per-anchor cut-current evidence changed")
        expected_safety_selection = reference
        if expected_model_index != -1:
            expected_safety_selection = hop_lock._selection_from_role(
                batch,
                owner_by_batch[batch_id][expected_model_index],
                adjacency,
                work_order["plan_fingerprint"],
            )
        try:
            candidate_safety = hop_lock._validate_safety_diagnostics(
                row.get("candidate_safety"),
                contract,
                f"audit {batch_id} {role} candidate",
                expected_safety_selection,
            )
            reference_safety = hop_lock._validate_safety_diagnostics(
                row.get("reference_safety"),
                contract,
                f"audit {batch_id} {role} reference",
                reference,
            )
        except (hop_lock.CalibrationLockError, TypeError, ValueError) as exc:
            raise HopAuditError("audit stored numerical safety gate failed") from exc
        if (
            _decode_hex(row["boundary_harmonic_max_hex"], "boundary harmonic")
            != full_result.candidate_boundary_harmonic_max
            or _decode_hex(
                row["protected_candidate_fraction_hex"],
                "protected candidate fraction",
            )
            != full_result.protected_candidate_fraction
            or _decode_hex(
                row["protected_reference_fraction_hex"],
                "protected reference fraction",
            )
            != full_result.protected_reference_fraction
            or full_result.candidate_reciprocal_condition
            != candidate_safety["reciprocal_condition"]
            or full_result.reference_reciprocal_condition
            != reference_safety["reciprocal_condition"]
        ):
            raise HopAuditError("audit safety or boundary evidence changed")
        screening = row.get("screening")
        if (
            not isinstance(screening, list)
            or [item.get("anchor") for item in screening]
            != [item[0] for item in expected_anchor_sequence]
        ):
            raise HopAuditError("audit reference screening provenance is incomplete")
        try:
            for screening_row, (anchor, _quartile) in zip(
                screening, expected_anchor_sequence
            ):
                hop_lock._validate_screening_record(
                    screening_row,
                    anchor,
                    math.exp(-1.0),
                    f"audit {batch_id} reference",
                )
        except (hop_lock.CalibrationLockError, TypeError, ValueError) as exc:
            raise HopAuditError("audit reference screening provenance failed") from exc
        previous_screening = screening_by_batch.setdefault(batch_id, screening)
        if previous_screening != screening:
            raise HopAuditError("audit roles did not share identical reference screening")
    return index


def _anchor_metric(row, name):
    return [
        _decode_hex(anchor["metrics"].get(name), f"{name} evidence")
        for anchor in row["anchors"]
    ]


def _absolute_summary(rows, role, complete_ids, *, effective_resistance):
    selected = [rows[(batch_id, role)] for batch_id in complete_ids]
    if len(selected) != len(complete_ids):
        raise HopAuditError(f"{role} evidence is incomplete")
    def flattened(name):
        return [value for row in selected for value in _anchor_metric(row, name)]
    summary = {
        "boundary_harmonic_q90": _observed_order(
            [_decode_hex(row["boundary_harmonic_max_hex"], "boundary harmonic") for row in selected],
            0.9,
            upper=True,
        ),
        "maximum_h_absolute_error_q90": _observed_order(flattened("maximum_h_absolute_error"), 0.9, upper=True),
        "rank_inversion_fraction_q90": _observed_order(flattened("rank_inversion_fraction"), 0.9, upper=True),
        "raw_relative_l2_error_q90": _observed_order(flattened("raw_relative_l2_error"), 0.9, upper=True),
        "source_diagonal_relative_error_q90": _observed_order(flattened("source_diagonal_relative_error"), 0.9, upper=True),
        "top8_overlap_q10": _observed_order(flattened("top8_overlap"), 0.1, upper=False),
    }
    if effective_resistance:
        summary["effective_resistance_relative_error_q90"] = _observed_order(
            flattened("effective_resistance_relative_error"), 0.9, upper=True
        )
    return summary


def _absolute_checks(summary, limits):
    checks = {
        "boundary_harmonic_q90": summary["boundary_harmonic_q90"] <= limits["boundary_harmonic_q90_max"],
        "maximum_h_absolute_error_q90": summary["maximum_h_absolute_error_q90"] <= limits["maximum_h_absolute_error_q90_max"],
        "rank_inversion_fraction_q90": summary["rank_inversion_fraction_q90"] <= limits["rank_inversion_fraction_q90_max"],
        "raw_relative_l2_error_q90": summary["raw_relative_l2_error_q90"] <= limits["raw_relative_l2_error_q90_max"],
        "top8_overlap_q10": summary["top8_overlap_q10"] >= limits["top8_overlap_q10_min"],
    }
    if "effective_resistance_relative_error_q90_max" in limits:
        checks["effective_resistance_relative_error_q90"] = (
            summary["effective_resistance_relative_error_q90"]
            <= limits["effective_resistance_relative_error_q90_max"]
        )
    return checks, all(checks.values())


def _reference_checks(summary, limits):
    checks = {
        "maximum_h_absolute_error_q90": (
            summary["maximum_h_absolute_error_q90"]
            <= limits["maximum_h_absolute_error_q90_max"]
        ),
        "raw_relative_l2_error_q90": (
            summary["raw_relative_l2_error_q90"]
            <= limits["raw_relative_l2_error_q90_max"]
        ),
        "top8_overlap_q10": (
            summary["top8_overlap_q10"] >= limits["top8_overlap_q10_min"]
        ),
    }
    return checks, all(checks.values())


def _batch_contrasts(rows, complete_ids, low, high, *, effective_resistance):
    output = []
    for batch_id in complete_ids:
        low_row = rows[(batch_id, low)]
        high_row = rows[(batch_id, high)]
        low_raw = _anchor_metric(low_row, "raw_relative_l2_error")
        high_raw = _anchor_metric(high_row, "raw_relative_l2_error")
        endpoints = {
            "efficacy_log_high_over_low": _mean_extended(
                [_extended_log_ratio(h, l) for h, l in zip(high_raw, low_raw)]
            ),
            "ni_primary_log_low_over_high": _mean_extended(
                [_extended_log_ratio(l, h) for h, l in zip(high_raw, low_raw)]
            ),
            "ni_maximum_h_absolute_error_harm": float(np.mean(
                np.asarray(_anchor_metric(low_row, "maximum_h_absolute_error"))
                - np.asarray(_anchor_metric(high_row, "maximum_h_absolute_error"))
            )),
            "ni_rank_inversion_absolute_harm": float(np.mean(
                np.asarray(_anchor_metric(low_row, "rank_inversion_fraction"))
                - np.asarray(_anchor_metric(high_row, "rank_inversion_fraction"))
            )),
            "ni_source_diagonal_relative_error_harm": float(np.mean(
                np.asarray(_anchor_metric(low_row, "source_diagonal_relative_error"))
                - np.asarray(_anchor_metric(high_row, "source_diagonal_relative_error"))
            )),
            "ni_top8_overlap_loss": float(np.mean(
                np.asarray(_anchor_metric(high_row, "top8_overlap"))
                - np.asarray(_anchor_metric(low_row, "top8_overlap"))
            )),
            "ni_boundary_harmonic_absolute_harm": (
                _decode_hex(low_row["boundary_harmonic_max_hex"], "low boundary")
                - _decode_hex(high_row["boundary_harmonic_max_hex"], "high boundary")
            ),
        }
        if effective_resistance:
            endpoints["ni_effective_resistance_relative_error_harm"] = float(np.mean(
                np.asarray(_anchor_metric(low_row, "effective_resistance_relative_error"))
                - np.asarray(_anchor_metric(high_row, "effective_resistance_relative_error"))
            ))
        output.append({"batch_id": batch_id, "endpoints": endpoints})
    return output


def _bootstrap_statistics(batch_contrasts, statuses, schedule):
    complete_by_index = [bool(row["complete"]) for row in sorted(statuses, key=lambda row: row["batch_index"])]
    contrast_by_id = {row["batch_id"]: row["endpoints"] for row in batch_contrasts}
    complete_ids = [f"audit-{index:02d}" for index, complete in enumerate(complete_by_index) if complete]
    endpoint_names = sorted(next(iter(contrast_by_id.values()))) if contrast_by_id else []
    records = []
    samples = {name: [] for name in endpoint_names}
    for schedule_row in schedule:
        masked = [
            count if complete_by_index[index] else 0
            for index, count in enumerate(schedule_row["multiplicities"])
        ]
        retained_mass = sum(masked)
        if retained_mass == 0:
            raise HopAuditError("bootstrap replicate has zero retained multiplicity")
        weights = [masked[int(batch_id.rsplit("-", 1)[1])] for batch_id in complete_ids]
        encoded = {}
        for name in endpoint_names:
            value = _mean_extended(
                [contrast_by_id[batch_id][name] for batch_id in complete_ids], weights
            )
            samples[name].append(value)
            encoded[name] = _encode_extended(value)
        records.append(
            {
                "endpoints": encoded,
                "replicate_index": schedule_row["replicate_index"],
                "retained_multiplicity": retained_mass,
            }
        )
    intervals = {
        name: {
            "lower_0.05": _observed_order(values, 0.05, upper=False),
            "point": _mean_extended(
                [contrast_by_id[batch_id][name] for batch_id in complete_ids]
            ),
            "upper_0.95": _observed_order(values, 0.95, upper=True),
        }
        for name, values in samples.items()
    }
    return records, intervals


def _derive_decision(work_order, fidelity_rows, statuses, schedule, contract, peak_rss):
    complete_ids = [
        row["batch_id"]
        for row in sorted(statuses, key=lambda row: row["batch_index"])
        if row["complete"]
    ]
    index = _validate_evidence_against_work_order(
        work_order, fidelity_rows, statuses, contract
    )
    expected_rows = len(complete_ids) * len(work_order["candidate_roles"])
    if len(index) != expected_rows:
        raise HopAuditError("complete-batch fidelity evidence is not rectangular")
    statistics = contract["statistics"]
    summaries = {}
    checks = {}
    for role in work_order["candidate_roles"]:
        if complete_ids:
            summary = _absolute_summary(
                index,
                role,
                complete_ids,
                effective_resistance=contract["effective_resistance"],
            )
            role_checks, passed = _absolute_checks(summary, statistics["absolute_adequacy"])
            summaries[role] = summary
            checks[role] = {"checks": role_checks, "passed": passed}
    base = {
        "absolute_adequacy": hop_lock._hexify_scientific(checks),
        "audit_batch_count": EXPECTED_AUDIT_BATCHES,
        "all_24_batch_estimand_preserved": len(complete_ids) == EXPECTED_AUDIT_BATCHES,
        "complete_batch_count": len(complete_ids),
        "complete_batch_ids": complete_ids,
        "confirmatory_claim_authorized": False,
        "conditional_complete_case_inference": (
            MINIMUM_COMPLETE_BATCHES <= len(complete_ids) < EXPECTED_AUDIT_BATCHES
        ),
        "convergence_claim_authorized": False,
        "descriptive_results_authorized": bool(complete_ids),
        "larger_domain_efficacy_claim_authorized": False,
        "endpoint_summaries": hop_lock._hexify_scientific(summaries),
        "minimum_complete_batch_count": MINIMUM_COMPLETE_BATCHES,
        "missingness_correction_claimed": False,
        "resource_ceiling_passed": peak_rss <= contract["peak_rss_ceiling_bytes"],
    }
    if any(row.get("failure_kind") == "deterministic_safety_failure" for row in statuses):
        return {
            **base,
            "decision": "safety_or_resource_blocked",
            "reason": "one_or_more_batches_failed_a_deterministic_safety_gate",
        }, [], {}
    if peak_rss > contract["peak_rss_ceiling_bytes"]:
        return {**base, "decision": "safety_or_resource_blocked", "reason": "peak_rss_exceeded"}, [], {}
    if len(complete_ids) < MINIMUM_COMPLETE_BATCHES:
        return {**base, "decision": "descriptive_incomplete", "reason": "fewer_than_18_complete_batches"}, [], {}
    reference_checks, reference_adequate = _reference_checks(
        summaries["S_1024"], statistics["reference_adequacy"]
    )
    base = {
        **base,
        "reference_adequacy": {
            "checks": reference_checks,
            "passed": reference_adequate,
            "reference_role": "R_top",
            "support_role": "S_1024",
        },
    }
    if not reference_adequate:
        return {
            **base,
            "decision": "reference_inadequate",
            "reason": "untouched_audit_reference_adequacy_failed",
        }, [], {}
    mode = work_order["lock_mode"]
    if mode == "right_censored_diagnostics":
        return {**base, "decision": "right_censored_diagnostics", "reason": "lock_froze_diagnostic_only_mode"}, [], {}
    decision_roles = work_order["decision_roles"]
    low = context_low = decision_roles[0]
    if mode == "absolute_only":
        passed = checks[low]["passed"]
        return {
            **base,
            "absolute_endpoint_adequate": passed,
            "decision": "absolute_only",
            "reason": "absolute_only_lock_mode",
        }, [], {}
    high = decision_roles[1]
    contrasts = _batch_contrasts(
        index,
        complete_ids,
        context_low,
        high,
        effective_resistance=contract["effective_resistance"],
    )
    bootstrap_rows, intervals = _bootstrap_statistics(contrasts, statuses, schedule)
    efficacy = intervals["efficacy_log_high_over_low"]["upper_0.95"] < math.log(0.9)
    margins = statistics["noninferiority_intersection_margins"]
    endpoint_margins = {
        "ni_primary_log_low_over_high": math.log(1.10),
        "ni_maximum_h_absolute_error_harm": margins["maximum_h_absolute_error_harm"],
        "ni_rank_inversion_absolute_harm": margins["rank_inversion_absolute_harm"],
        "ni_source_diagonal_relative_error_harm": margins["source_diagonal_relative_error_harm"],
        "ni_top8_overlap_loss": margins["top8_overlap_loss"],
        "ni_boundary_harmonic_absolute_harm": margins["boundary_harmonic_absolute_harm"],
    }
    if contract["effective_resistance"]:
        endpoint_margins["ni_effective_resistance_relative_error_harm"] = margins[
            "effective_resistance_relative_error_harm"
        ]
    ni_checks = {
        name: intervals[name]["upper_0.95"] < margin
        for name, margin in endpoint_margins.items()
    }
    noninferior = all(ni_checks.values())
    low_adequate = checks[low]["passed"]
    node_reduction = all(
        index[(batch_id, low)]["candidate_nodes"] < index[(batch_id, high)]["candidate_nodes"]
        for batch_id in complete_ids
    )
    convergence_candidate = low_adequate and noninferior and node_reduction
    efficacy_candidate = efficacy and node_reduction
    if efficacy_candidate and convergence_candidate:
        result = "inconclusive_frontier"
        reason = "larger_domain_efficacy_and_smaller_domain_convergence_rules_conflict"
    elif efficacy_candidate:
        result = "larger_endpoint_efficacious"
        reason = "efficacy_upper_bound_below_log_0.9"
    elif not efficacy and convergence_candidate:
        result = "low_endpoint_converged"
        reason = "absolute_adequacy_and_noninferiority_with_realized_node_reduction"
    elif efficacy:
        result = "inconclusive_frontier"
        reason = "efficacy_passed_without_strict_realized_node_reduction"
    else:
        result = "inconclusive_frontier"
        reason = "frozen_efficacy_and_convergence_rules_did_not_resolve"
    confirmatory = result in {
        "larger_endpoint_efficacious",
        "low_endpoint_converged",
    }
    decision = {
        **base,
        "bootstrap_intervals": {
            name: {key: _encode_extended(value) for key, value in values.items()}
            for name, values in sorted(intervals.items())
        },
        "confirmatory_claim_authorized": confirmatory,
        "convergence_claim_authorized": result == "low_endpoint_converged",
        "decision": result,
        "efficacy_passed": efficacy,
        "noninferiority_intersection_passed": noninferior,
        "noninferiority_endpoint_checks": ni_checks,
        "larger_domain_efficacy_claim_authorized": (
            result == "larger_endpoint_efficacious"
        ),
        "reason": reason,
        "realized_node_reduction_passed": node_reduction,
    }
    return decision, bootstrap_rows, intervals


def _derive_payloads(context, runtime_blas):
    if runtime_blas != context["lock_manifest"]["scientific_core"]["actual_blas_identity"]:
        raise HopAuditError("audit BLAS identity differs from the frozen lock")
    work_order, schedule = _derive_audit_work_order(context)
    started = time.perf_counter()
    fidelity_rows, statuses, timings, evaluator_calls = _run_audit_batches(
        work_order, context["contract"]
    )
    peak_rss = _peak_rss_bytes()
    decision, bootstrap_rows, _intervals = _derive_decision(
        work_order,
        fidelity_rows,
        statuses,
        schedule,
        context["contract"],
        peak_rss,
    )
    scientific_payloads = {
        "audit_work_order.json": _canonical_json(work_order),
        "audit_fidelity.jsonl": _jsonl_bytes(fidelity_rows),
        "batch_status.jsonl": _jsonl_bytes(statuses),
        "bootstrap_statistics.jsonl": _jsonl_bytes(bootstrap_rows),
        "decision.json": _canonical_json(decision),
    }
    serialized_peak_rss = _peak_rss_bytes()
    if (
        peak_rss <= context["contract"]["peak_rss_ceiling_bytes"]
        < serialized_peak_rss
    ):
        # Bootstrap/statistics construction is inside the frozen resource gate.
        # A threshold crossing therefore changes the decision and scientific
        # payloads once, after which ru_maxrss cannot return below the ceiling.
        peak_rss = serialized_peak_rss
        decision, bootstrap_rows, _intervals = _derive_decision(
            work_order,
            fidelity_rows,
            statuses,
            schedule,
            context["contract"],
            peak_rss,
        )
        scientific_payloads["bootstrap_statistics.jsonl"] = _jsonl_bytes(
            bootstrap_rows
        )
        scientific_payloads["decision.json"] = _canonical_json(decision)
        serialized_peak_rss = _peak_rss_bytes()
    peak_rss = max(peak_rss, serialized_peak_rss)
    execution = {
        "actual_blas_identity": runtime_blas,
        "audit_batch_evaluator_calls": evaluator_calls,
        "batch_phase_timings": timings,
        "observed_peak_rss_bytes": peak_rss,
        "peak_rss_scope": context["contract"]["peak_rss_scope"],
        "peak_rss_ceiling_bytes": context["contract"]["peak_rss_ceiling_bytes"],
        "timing_and_rss_outside_scientific_fingerprint": True,
        "total_seconds": time.perf_counter() - started,
    }
    payloads = {
        **scientific_payloads,
        "execution.json": _canonical_json(execution),
    }
    return payloads, decision


def _build_manifest(payloads, context, decision, runtime_blas):
    if set(payloads) != set(ARTIFACT_NAMES):
        raise HopAuditError("audit payload inventory mismatch")
    if any(len(data) > MAXIMUM_AUDIT_ARTIFACT_BYTES[name] for name, data in payloads.items()):
        raise HopAuditError("generated audit artifact exceeds its size ceiling")
    scientific_records = {
        name: _content_record(payloads[name]) for name in SCIENTIFIC_ARTIFACT_NAMES
    }
    observational_records = {
        name: _content_record(payloads[name]) for name in OBSERVATIONAL_ARTIFACT_NAMES
    }
    core = {
        "actual_blas_identity": runtime_blas,
        "algorithm": ALGORITHM,
        "confirmatory_claim_authorized": decision["confirmatory_claim_authorized"],
        "convergence_claim_authorized": decision["convergence_claim_authorized"],
        "decision": decision["decision"],
        "implementation_records": _implementation_records(),
        "lock_fingerprint": context["lock_manifest"]["lock_fingerprint"],
        "lock_manifest_record": context["lock_manifest_record"],
        "larger_domain_efficacy_claim_authorized": decision[
            "larger_domain_efficacy_claim_authorized"
        ],
        "numeric_backend_contract": context["contract"]["numeric"],
        "plan_fingerprint": context["plan_context"]["manifest"]["plan_fingerprint"],
        "repository_commit": _git_commit(),
        "schema": SCHEMA,
        "scientific_artifact_records": scientific_records,
    }
    audit_fingerprint = hashlib.sha256(_canonical_json(core)).hexdigest()
    manifest = {
        "accepted": True,
        "algorithm": ALGORITHM,
        "audit_fingerprint": audit_fingerprint,
        "confirmatory_claim_authorized": decision["confirmatory_claim_authorized"],
        "convergence_claim_authorized": decision["convergence_claim_authorized"],
        "decision": decision["decision"],
        "larger_domain_efficacy_claim_authorized": decision[
            "larger_domain_efficacy_claim_authorized"
        ],
        "manifest_integrity_contract": "sha256-canonical-manifest-without-manifest_integrity_seal-v1",
        "marker_record": _content_record(MARKER_BYTES),
        "observational_artifact_records": observational_records,
        "schema": SCHEMA,
        "scientific_core": core,
    }
    manifest["manifest_integrity_seal"] = hashlib.sha256(
        _canonical_json(_manifest_without_seal(manifest))
    ).hexdigest()
    return manifest


def _read_audit_payloads(binding, manifest):
    records = {}
    records.update(manifest["scientific_core"]["scientific_artifact_records"])
    records.update(manifest["observational_artifact_records"])
    payloads = {}
    for name in ARTIFACT_NAMES:
        try:
            payloads[name] = hop_plan._read_bound_file(
                binding,
                name,
                records[name],
                f"audit {name}",
                maximum_size=MAXIMUM_AUDIT_ARTIFACT_BYTES[name],
            )
        except (KeyError, hop_plan.HopPlanError) as exc:
            raise HopAuditError(f"audit artifact {name} failed verification") from exc
    return payloads


def _validate_manifest(manifest):
    required = {
        "accepted", "algorithm", "audit_fingerprint", "confirmatory_claim_authorized",
        "convergence_claim_authorized", "decision",
        "larger_domain_efficacy_claim_authorized", "manifest_integrity_contract",
        "manifest_integrity_seal", "marker_record", "observational_artifact_records",
        "schema", "scientific_core",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise HopAuditError("audit manifest shape mismatch")
    if manifest["schema"] != SCHEMA or manifest["algorithm"] != ALGORITHM:
        raise HopAuditError("audit manifest schema mismatch")
    if manifest["marker_record"] != _content_record(MARKER_BYTES):
        raise HopAuditError("audit local-only marker record changed")
    if manifest["accepted"] is not True:
        raise HopAuditError("installed audit must be a completed transaction")
    if manifest["manifest_integrity_contract"] != "sha256-canonical-manifest-without-manifest_integrity_seal-v1":
        raise HopAuditError("audit manifest integrity contract changed")
    expected_seal = hashlib.sha256(_canonical_json(_manifest_without_seal(manifest))).hexdigest()
    if manifest["manifest_integrity_seal"] != expected_seal:
        raise HopAuditError("audit manifest seal mismatch")
    core = manifest["scientific_core"]
    expected_core_keys = {
        "actual_blas_identity",
        "algorithm",
        "confirmatory_claim_authorized",
        "convergence_claim_authorized",
        "decision",
        "implementation_records",
        "larger_domain_efficacy_claim_authorized",
        "lock_fingerprint",
        "lock_manifest_record",
        "numeric_backend_contract",
        "plan_fingerprint",
        "repository_commit",
        "schema",
        "scientific_artifact_records",
    }
    if not isinstance(core, dict) or set(core) != expected_core_keys:
        raise HopAuditError("audit scientific-core shape mismatch")
    if core["schema"] != SCHEMA or core["algorithm"] != ALGORITHM:
        raise HopAuditError("audit scientific-core schema mismatch")
    if (
        not isinstance(core["scientific_artifact_records"], dict)
        or not isinstance(manifest["observational_artifact_records"], dict)
        or set(core["scientific_artifact_records"]) != set(SCIENTIFIC_ARTIFACT_NAMES)
        or set(manifest["observational_artifact_records"])
        != set(OBSERVATIONAL_ARTIFACT_NAMES)
        or any(
            not hop_lock._content_record_is_valid(record)
            for record in (
                list(core["scientific_artifact_records"].values())
                + list(manifest["observational_artifact_records"].values())
            )
        )
    ):
        raise HopAuditError("audit artifact records are malformed")
    if manifest["audit_fingerprint"] != hashlib.sha256(_canonical_json(core)).hexdigest():
        raise HopAuditError("audit fingerprint mismatch")
    if (
        manifest["confirmatory_claim_authorized"] != core.get("confirmatory_claim_authorized")
        or manifest["convergence_claim_authorized"] != core.get("convergence_claim_authorized")
        or manifest["larger_domain_efficacy_claim_authorized"]
        != core.get("larger_domain_efficacy_claim_authorized")
        or manifest["decision"] != core.get("decision")
    ):
        raise HopAuditError("audit authorization fields disagree")
    expected_larger = manifest["decision"] == "larger_endpoint_efficacious"
    expected_convergence = manifest["decision"] == "low_endpoint_converged"
    if (
        manifest["larger_domain_efficacy_claim_authorized"] is not expected_larger
        or manifest["convergence_claim_authorized"] is not expected_convergence
        or manifest["confirmatory_claim_authorized"]
        is not (expected_larger or expected_convergence)
    ):
        raise HopAuditError("audit decision and authorization are inconsistent")


def _audit_manifest_from_binding(binding):
    try:
        data = hop_plan._read_bound_file(
            binding,
            MANIFEST_NAME,
            None,
            "audit manifest",
            maximum_size=MAXIMUM_AUDIT_MANIFEST_BYTES,
        )
    except hop_plan.HopPlanError as exc:
        raise HopAuditError("audit manifest is unavailable") from exc
    manifest = _strict_json_bytes(data, "audit manifest")
    _validate_manifest(manifest)
    return manifest, data


def _validate_status_control_flow(statuses):
    safety_positions = [
        index
        for index, row in enumerate(statuses)
        if row.get("failure_kind") == "deterministic_safety_failure"
    ]
    unattempted_positions = [
        index
        for index, row in enumerate(statuses)
        if row.get("failure_kind") == "not_attempted_after_global_block"
    ]
    if len(safety_positions) > 1:
        raise HopAuditError("audit ledger contains multiple global safety failures")
    if safety_positions:
        expected_tail = list(range(safety_positions[0] + 1, len(statuses)))
        if unattempted_positions != expected_tail:
            raise HopAuditError("audit global safety failure has an invalid tail")
    elif unattempted_positions:
        raise HopAuditError("audit has an unattempted tail without a safety failure")


def _verify_payload_derivations(payloads, context, manifest):
    expected_work_order, schedule = _derive_audit_work_order(context)
    work_order = _strict_json_bytes(payloads["audit_work_order.json"], "audit work order")
    if work_order != expected_work_order:
        raise HopAuditError("audit work order differs from the frozen projection")
    fidelity = _strict_jsonl_bytes(payloads["audit_fidelity.jsonl"], "audit fidelity")
    statuses = _strict_jsonl_bytes(payloads["batch_status.jsonl"], "batch status")
    bootstrap = _strict_jsonl_bytes(payloads["bootstrap_statistics.jsonl"], "bootstrap statistics")
    decision = _strict_json_bytes(payloads["decision.json"], "audit decision")
    execution = _strict_json_bytes(payloads["execution.json"], "audit execution")
    if len(statuses) != EXPECTED_AUDIT_BATCHES:
        raise HopAuditError("audit status ledger must contain 24 batches")
    batch_by_id = {row["batch_id"]: row for row in work_order["batches"]}
    adjacency = _adjacency_from_work_order(work_order)
    for index, row in enumerate(statuses):
        if (
            row.get("batch_index") != index
            or row.get("batch_id") != f"audit-{index:02d}"
            or not isinstance(row.get("complete"), bool)
            or set(row) != {
                "batch_id",
                "batch_index",
                "complete",
                "factor_provenance",
                "failure_class",
                "failure_details",
                "failure_kind",
                "roles_completed",
            }
        ):
            raise HopAuditError("audit status ledger identity mismatch")
        expected_roles = work_order["candidate_roles"] if row["complete"] else []
        if row["roles_completed"] != expected_roles:
            raise HopAuditError("audit status role completion is inconsistent")
        if row["complete"]:
            expected_factor, _owners = _expected_factor_provenance(
                batch_by_id[row["batch_id"]],
                work_order["candidate_roles"],
                adjacency,
                work_order["plan_fingerprint"],
            )
            if (
                row["failure_class"] is not None
                or row["failure_details"] is not None
                or row["failure_kind"] is not None
                or row["factor_provenance"] != expected_factor
            ):
                raise HopAuditError("complete audit batch carries a failure")
        elif (
            row["failure_kind"] not in {
                "registered_coverage_failure",
                "deterministic_safety_failure",
                "not_attempted_after_global_block",
            }
        ):
            raise HopAuditError("incomplete audit batch has an unregistered failure")
        elif row["failure_kind"] == "not_attempted_after_global_block":
            if (
                row["failure_class"] is not None
                or row["failure_details"] is not None
                or row["factor_provenance"] is not None
            ):
                raise HopAuditError("unattempted batch carries a failure class")
        elif not isinstance(row["failure_class"], str) or not row["failure_class"]:
            raise HopAuditError("failed audit batch lacks its failure class")
        elif row["failure_kind"] == "deterministic_safety_failure":
            if (
                row["failure_class"] not in {"FloatingPointError", "LinAlgError"}
                or row["failure_details"] is not None
                or row["factor_provenance"] is not None
            ):
                raise HopAuditError("safety-failed batch carries survivor provenance")
        else:
            if row["failure_class"] != ProtectedSetCoverageError.__name__:
                raise HopAuditError("coverage exclusion has the wrong failure class")
            batch = batch_by_id[row["batch_id"]]
            protected = set(batch["protected_nodes"])
            reference_nodes = set(
                hop_lock._domain_from_role(batch, "R_top", adjacency).nodes
            )
            missing_reference = sorted(
                protected.difference(reference_nodes), key=hop_plan._typed_id_key
            )
            missing_candidate = []
            if not missing_reference:
                for role in work_order["candidate_roles"]:
                    candidate_nodes = set(
                        hop_lock._domain_from_role(batch, role, adjacency).nodes
                    )
                    missing_candidate = sorted(
                        protected.difference(candidate_nodes),
                        key=hop_plan._typed_id_key,
                    )
                    if missing_candidate:
                        break
            expected_details = {
                "missing_candidate_nodes": missing_candidate,
                "missing_reference_nodes": missing_reference,
            }
            if (
                not (missing_candidate or missing_reference)
                or row["failure_details"] != expected_details
                or row["factor_provenance"] is not None
            ):
                raise HopAuditError("coverage exclusion lacks a rederived certificate")
    _validate_status_control_flow(statuses)
    expected_execution_keys = {
        "actual_blas_identity",
        "audit_batch_evaluator_calls",
        "batch_phase_timings",
        "observed_peak_rss_bytes",
        "peak_rss_ceiling_bytes",
        "peak_rss_scope",
        "timing_and_rss_outside_scientific_fingerprint",
        "total_seconds",
    }
    if not isinstance(execution, dict) or set(execution) != expected_execution_keys:
        raise HopAuditError("audit execution provenance shape mismatch")
    timing_rows = execution["batch_phase_timings"]
    if not isinstance(timing_rows, list) or len(timing_rows) != EXPECTED_AUDIT_BATCHES:
        raise HopAuditError("audit timing ledger must contain 24 batches")
    status_by_id = {row["batch_id"]: row for row in statuses}
    for index, timing in enumerate(timing_rows):
        batch_id = f"audit-{index:02d}"
        if not isinstance(timing, dict):
            raise HopAuditError("audit timing row is malformed")
        try:
            elapsed = float(timing.get("fidelity_seconds", math.nan))
        except (TypeError, ValueError) as exc:
            raise HopAuditError("audit timing row is malformed") from exc
        if (
            timing.get("batch_id") != batch_id
            or not math.isfinite(elapsed)
            or elapsed < 0.0
            or not isinstance(timing.get("role_timings"), list)
        ):
            raise HopAuditError("audit timing row is malformed")
        status = status_by_id[batch_id]
        if (
            status["failure_kind"] == "not_attempted_after_global_block"
            and elapsed != 0.0
        ):
            raise HopAuditError("unattempted audit batch has nonzero elapsed time")
        if (
            elapsed > context["contract"]["per_batch_elapsed_ceiling_seconds"]
            and status["failure_kind"] != "deterministic_safety_failure"
        ):
            raise HopAuditError("attempted audit batch exceeded its frozen ceiling")
        expected_roles = work_order["candidate_roles"] if status["complete"] else []
        if [row.get("role") for row in timing["role_timings"]] != expected_roles:
            raise HopAuditError("audit endpoint timing roles are incomplete")
        for role_timing in timing["role_timings"]:
            if set(role_timing) != {
                "candidate_build_seconds",
                "candidate_selection_seconds",
                "reference_build_seconds",
                "reference_selection_seconds",
                "role",
                "solve_seconds",
            }:
                raise HopAuditError("audit endpoint timing shape mismatch")
            for name, value in role_timing.items():
                if name != "role" and (
                    not math.isfinite(float(value)) or float(value) < 0.0
                ):
                    raise HopAuditError("audit endpoint timing is invalid")
    if (
        not isinstance(execution["observed_peak_rss_bytes"], int)
        or isinstance(execution["observed_peak_rss_bytes"], bool)
        or execution["observed_peak_rss_bytes"] < 0
        or execution["peak_rss_ceiling_bytes"] != context["contract"]["peak_rss_ceiling_bytes"]
        or execution["peak_rss_scope"] != context["contract"]["peak_rss_scope"]
        or execution["timing_and_rss_outside_scientific_fingerprint"] is not True
        or not math.isfinite(float(execution["total_seconds"]))
        or float(execution["total_seconds"]) < 0.0
    ):
        raise HopAuditError("audit resource provenance is malformed")
    expected_decision, expected_bootstrap, _ = _derive_decision(
        work_order,
        fidelity,
        statuses,
        schedule,
        context["contract"],
        execution["observed_peak_rss_bytes"],
    )
    if decision != expected_decision or bootstrap != expected_bootstrap:
        raise HopAuditError("audit statistics or decision failed rederivation")
    for name in (
        "confirmatory_claim_authorized",
        "convergence_claim_authorized",
        "larger_domain_efficacy_claim_authorized",
    ):
        if manifest.get(name) != decision.get(name) or manifest["scientific_core"].get(name) != decision.get(name):
            raise HopAuditError("audit authorization does not match the rederived decision")
    if (
        execution.get("actual_blas_identity") != manifest["scientific_core"]["actual_blas_identity"]
        or execution.get("audit_batch_evaluator_calls")
        != sum(
            row["failure_kind"] != "not_attempted_after_global_block"
            for row in statuses
        )
        or manifest["decision"] != decision["decision"]
    ):
        raise HopAuditError("audit execution or authorization provenance disagrees")


def _assert_audit_path_and_overlap(
    audit_dir,
    *,
    lock_dir,
    plan_dir,
    receipt_dir,
    attempt_a_dir,
    attempt_b_dir,
    source_spec,
    relation_policy,
    input_ceiling,
):
    try:
        resolved = Path(audit_dir).resolve(strict=True)
        hop_plan._assert_no_output_input_overlap(
            resolved,
            (lock_dir, plan_dir, receipt_dir, attempt_a_dir, attempt_b_dir),
            (source_spec, relation_policy),
            source_spec,
            input_ceiling,
        )
        return resolved
    except (OSError, hop_plan.HopPlanError) as exc:
        raise HopAuditError("audit path is not admissible") from exc


def _cleanup_audit_staging(parent_binding, staging_binding, staging_leaf):
    """Remove only this runner's known, still-bound output on failure."""

    try:
        hop_plan._assert_bound_directory(parent_binding, "audit output parent")
        hop_plan._assert_bound_directory(staging_binding, "audit staging")
        entries = set(
            hop_plan._bounded_directory_names(
                staging_binding, len(ALL_AUDIT_FILES), "audit staging"
            )
        )
        if not entries.issubset(ALL_AUDIT_FILES):
            raise HopAuditError("audit staging inventory changed")
        for name in sorted(entries):
            observed = os.stat(
                name, dir_fd=staging_binding.fd, follow_symlinks=False
            )
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise HopAuditError("audit staging artifact changed")
            os.unlink(name, dir_fd=staging_binding.fd)
        os.fsync(staging_binding.fd)
        hop_plan._assert_bound_directory(staging_binding, "audit staging")
        hop_plan._assert_bound_directory(parent_binding, "audit output parent")
        os.rmdir(staging_leaf, dir_fd=parent_binding.fd)
        os.fsync(parent_binding.fd)
        hop_plan._assert_bound_directory(parent_binding, "audit output parent")
    except (OSError, hop_plan.HopPlanError) as exc:
        raise HopAuditError("audit staging cleanup failed") from exc


def verify_audit(
    audit_dir,
    *,
    lock_dir,
    plan_dir,
    receipt_dir,
    attempt_a_dir,
    attempt_b_dir,
    source_spec,
    relation_policy,
):
    context = _capture_verified_context(
        lock_dir=lock_dir,
        plan_dir=plan_dir,
        receipt_dir=receipt_dir,
        attempt_a_dir=attempt_a_dir,
        attempt_b_dir=attempt_b_dir,
        source_spec=source_spec,
        relation_policy=relation_policy,
    )
    resolved = _assert_audit_path_and_overlap(
        audit_dir,
        lock_dir=lock_dir,
        plan_dir=plan_dir,
        receipt_dir=receipt_dir,
        attempt_a_dir=attempt_a_dir,
        attempt_b_dir=attempt_b_dir,
        source_spec=source_spec,
        relation_policy=relation_policy,
        input_ceiling=context["plan_context"]["manifest"]["fingerprint_core"]["resource_contract"]["planner_input_ceiling_bytes"],
    )
    try:
        binding = hop_plan._bind_directory(resolved, "HOP audit")
    except hop_plan.HopPlanError as exc:
        raise HopAuditError("HOP audit is unavailable") from exc
    try:
        if stat.S_IMODE(os.fstat(binding.fd).st_mode) != 0o700:
            raise HopAuditError("HOP audit directory mode mismatch")
        try:
            names = hop_plan._bounded_directory_names(binding, len(ALL_AUDIT_FILES), "HOP audit")
        except hop_plan.HopPlanError as exc:
            raise HopAuditError("HOP audit inventory is unavailable") from exc
        if set(names) != ALL_AUDIT_FILES or len(names) != len(ALL_AUDIT_FILES):
            raise HopAuditError("HOP audit inventory mismatch")
        manifest, manifest_data = _audit_manifest_from_binding(binding)
        try:
            marker_data = hop_plan._read_bound_file(
                binding,
                MARKER_NAME,
                manifest["marker_record"],
                "audit local-only marker",
                maximum_size=len(MARKER_BYTES),
            )
        except hop_plan.HopPlanError as exc:
            raise HopAuditError("audit local-only marker failed verification") from exc
        if marker_data != MARKER_BYTES:
            raise HopAuditError("audit local-only marker content changed")
        core = manifest["scientific_core"]
        if (
            core.get("repository_commit") != _git_commit()
            or core.get("implementation_records") != _implementation_records()
            or core.get("lock_fingerprint") != context["lock_manifest"]["lock_fingerprint"]
            or core.get("lock_manifest_record") != context["lock_manifest_record"]
            or core.get("plan_fingerprint")
            != context["plan_context"]["manifest"]["plan_fingerprint"]
            or core.get("actual_blas_identity")
            != context["lock_manifest"]["scientific_core"]["actual_blas_identity"]
            or core.get("numeric_backend_contract") != context["contract"]["numeric"]
        ):
            raise HopAuditError("audit implementation binding changed")
        payloads = _read_audit_payloads(binding, manifest)
        _verify_payload_derivations(payloads, context, manifest)
        if _canonical_json(manifest) != manifest_data:
            raise HopAuditError("audit manifest changed during verification")
        try:
            hop_plan._assert_private_directory_files(binding, "HOP audit")
        except hop_plan.HopPlanError as exc:
            raise HopAuditError("HOP audit privacy envelope failed") from exc
        final_context = _capture_verified_context(
            lock_dir=lock_dir,
            plan_dir=plan_dir,
            receipt_dir=receipt_dir,
            attempt_a_dir=attempt_a_dir,
            attempt_b_dir=attempt_b_dir,
            source_spec=source_spec,
            relation_policy=relation_policy,
        )
        if (
            final_context["lock_manifest"] != context["lock_manifest"]
            or final_context["lock_manifest_record"] != context["lock_manifest_record"]
            or final_context["plan_context"]["captured_records"] != context["plan_context"]["captured_records"]
        ):
            raise HopAuditError("verified inputs changed during audit verification")
        return manifest
    finally:
        binding.close()


def prepare_audit(
    *,
    audit_dir,
    local_root,
    lock_dir,
    plan_dir,
    receipt_dir,
    attempt_a_dir,
    attempt_b_dir,
    source_spec,
    relation_policy,
):
    context = _capture_verified_context(
        lock_dir=lock_dir,
        plan_dir=plan_dir,
        receipt_dir=receipt_dir,
        attempt_a_dir=attempt_a_dir,
        attempt_b_dir=attempt_b_dir,
        source_spec=source_spec,
        relation_policy=relation_policy,
    )
    try:
        target, parent_binding = hop_plan._validate_output_paths(
            audit_dir,
            local_root,
            (lock_dir, plan_dir, receipt_dir, attempt_a_dir, attempt_b_dir),
            (source_spec, relation_policy),
            source_spec,
            context["plan_context"]["manifest"]["fingerprint_core"]["resource_contract"]["planner_input_ceiling_bytes"],
        )
    except hop_plan.HopPlanError as exc:
        raise HopAuditError("audit output path is not admissible") from exc
    staging_binding = None
    staging_leaf = None
    committed = False
    try:
        with hop_lock._single_blas_thread() as runtime_blas:
            payloads, decision = _derive_payloads(context, runtime_blas)
            manifest = _build_manifest(payloads, context, decision, runtime_blas)
        final_context = _capture_verified_context(
            lock_dir=lock_dir,
            plan_dir=plan_dir,
            receipt_dir=receipt_dir,
            attempt_a_dir=attempt_a_dir,
            attempt_b_dir=attempt_b_dir,
            source_spec=source_spec,
            relation_policy=relation_policy,
        )
        if (
            final_context["lock_manifest"] != context["lock_manifest"]
            or final_context["lock_manifest_record"] != context["lock_manifest_record"]
            or final_context["plan_context"]["captured_records"] != context["plan_context"]["captured_records"]
        ):
            raise HopAuditError("verified inputs changed during audit")
        try:
            staging_leaf, staging_binding = hop_plan._create_bound_staging(parent_binding, target.name)
            for name in ARTIFACT_NAMES:
                hop_plan._write_bound_bytes(staging_binding, name, payloads[name])
            hop_plan._write_bound_bytes(staging_binding, MARKER_NAME, MARKER_BYTES)
            hop_plan._write_bound_bytes(staging_binding, MANIFEST_NAME, _canonical_json(manifest))
            os.fsync(staging_binding.fd)
        except hop_plan.HopPlanError as exc:
            raise HopAuditError("audit staging transaction failed") from exc
        staged = verify_audit(
            staging_binding.path,
            lock_dir=lock_dir,
            plan_dir=plan_dir,
            receipt_dir=receipt_dir,
            attempt_a_dir=attempt_a_dir,
            attempt_b_dir=attempt_b_dir,
            source_spec=source_spec,
            relation_policy=relation_policy,
        )
        if staged != manifest:
            raise HopAuditError("staged audit changed")
        try:
            hop_plan._assert_bound_directory(parent_binding, "audit output parent")
            hop_plan._assert_bound_directory(staging_binding, "audit staging")
            hop_plan._rename_directory_noreplace(parent_binding, staging_leaf, target.name)
            staging_binding.path = target
            staging_leaf = target.name
            hop_plan._assert_bound_directory(staging_binding, "installed HOP audit")
            os.fsync(parent_binding.fd)
        except hop_plan.HopPlanError as exc:
            raise HopAuditError("audit installation failed") from exc
        installed = verify_audit(
            target,
            lock_dir=lock_dir,
            plan_dir=plan_dir,
            receipt_dir=receipt_dir,
            attempt_a_dir=attempt_a_dir,
            attempt_b_dir=attempt_b_dir,
            source_spec=source_spec,
            relation_policy=relation_policy,
        )
        if installed != manifest:
            raise HopAuditError("installed audit changed")
        os.fsync(parent_binding.fd)
        committed = True
        return manifest
    finally:
        try:
            if not committed and staging_binding is not None:
                _cleanup_audit_staging(
                    parent_binding, staging_binding, staging_leaf
                )
        finally:
            if staging_binding is not None:
                staging_binding.close()
            parent_binding.close()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--audit-dir", required=True)
        command.add_argument("--lock-dir", required=True)
        command.add_argument("--plan-dir", required=True)
        command.add_argument("--receipt-dir", required=True)
        command.add_argument("--attempt-a-dir", required=True)
        command.add_argument("--attempt-b-dir", required=True)
        command.add_argument("--source-spec", required=True)
        command.add_argument("--relation-policy", required=True)
        if name == "prepare":
            command.add_argument("--local-root", required=True)
            command.add_argument("--local-only", action="store_true", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    common = {
        "audit_dir": args.audit_dir,
        "lock_dir": args.lock_dir,
        "plan_dir": args.plan_dir,
        "receipt_dir": args.receipt_dir,
        "attempt_a_dir": args.attempt_a_dir,
        "attempt_b_dir": args.attempt_b_dir,
        "source_spec": args.source_spec,
        "relation_policy": args.relation_policy,
    }
    try:
        if args.command == "prepare":
            manifest = prepare_audit(local_root=args.local_root, **common)
        else:
            manifest = verify_audit(**common)
    except (HopAuditError, hop_lock.CalibrationLockError) as exc:
        print(f"HOP audit failed closed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "audit_fingerprint": manifest["audit_fingerprint"],
                "confirmatory_claim_authorized": manifest["confirmatory_claim_authorized"],
                "convergence_claim_authorized": manifest["convergence_claim_authorized"],
                "decision": manifest["decision"],
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["confirmatory_claim_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
