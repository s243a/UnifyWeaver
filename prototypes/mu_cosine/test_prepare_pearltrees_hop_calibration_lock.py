#!/usr/bin/env python3
"""Focused contract tests for the calibration-only Pearltrees HOP lock."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import math
from types import SimpleNamespace

import numpy as np
import pytest

import prepare_pearltrees_hop_calibration_lock as lock
import prepare_pearltrees_hop_plan as plan


def _adequacy_summary(
    *,
    boundary: float = 0.01,
    h_error: float = 0.001,
    inversion: float = 0.01,
    raw_error: float = 0.005,
    overlap: float = 0.99,
) -> dict[str, float]:
    return {
        "boundary_harmonic_q90": boundary,
        "maximum_h_absolute_error_q90": h_error,
        "rank_inversion_fraction_q90": inversion,
        "raw_relative_l2_error_q90": raw_error,
        "source_diagonal_relative_error_q90": 0.001,
        "top8_overlap_q10": overlap,
    }


def _selection_contract() -> dict:
    return {
        "effective_resistance": False,
        "statistics": plan._statistical_contract("omitted"),
    }


def _endpoint_rows(*, identical_roles: tuple[str, str] | None = None) -> list[dict]:
    rows = []
    for batch_index in range(lock.EXPECTED_CALIBRATION_BATCHES):
        batch_id = f"calibration-{batch_index:02d}"
        hashes = {role: f"{batch_id}:{role}" for role in lock.ROLE_ORDER}
        if identical_roles is not None:
            left, right = identical_roles
            hashes[right] = hashes[left]
        for role in lock.ROLE_ORDER:
            rows.append(
                {
                    "batch_id": batch_id,
                    "node_content_sha256": hashes[role],
                    "realized_nodes": 1,
                    "role": role,
                }
            )
    return rows


def _select(monkeypatch, summaries, *, identical_roles=None):
    monkeypatch.setattr(
        lock,
        "_metric_aggregate",
        lambda _rows, role, *, effective_resistance: summaries[role],
    )
    monkeypatch.setattr(
        lock,
        "_endpoint_inventory",
        lambda _work_order, _adjacency: _endpoint_rows(
            identical_roles=identical_roles
        ),
    )
    return lock._select_lock_mode(
        [],
        {},
        {},
        _selection_contract(),
        alpha_top=0.125,
    )


def test_selection_freezes_finite_contrast_at_smallest_adequate_budget(
    monkeypatch,
) -> None:
    summaries = {
        role: _adequacy_summary() for role in lock.CANDIDATE_ROLES
    }
    selected = _select(monkeypatch, summaries)
    assert selected["lock_mode"] == "finite_contrast"
    assert selected["k_low"] == "S_256"
    assert selected["k_high"] == "S_512"
    assert selected["audit_solve_authorized"] is True
    assert selected["efficacy_or_resource_claim_authorized"] is True


def test_selection_freezes_absolute_only_at_1024_endpoint(monkeypatch) -> None:
    summaries = {
        "S_256": _adequacy_summary(raw_error=0.06),
        "S_512": _adequacy_summary(raw_error=0.06),
        "S_1024": _adequacy_summary(),
    }
    selected = _select(monkeypatch, summaries)
    assert selected["lock_mode"] == "absolute_only"
    assert selected["k_low"] == "S_1024"
    assert selected["k_high"] == "R_top"
    assert selected["audit_solve_authorized"] is True
    assert selected["efficacy_or_resource_claim_authorized"] is False


def test_selection_freezes_right_censored_diagnostics_when_no_candidate_passes(
    monkeypatch,
) -> None:
    # Boundary harmonic is an absolute-candidate gate, but is deliberately not
    # part of the S1024-vs-Rtop reference-convergence gate.
    summaries = {
        role: _adequacy_summary(boundary=0.2)
        for role in lock.CANDIDATE_ROLES
    }
    selected = _select(monkeypatch, summaries)
    assert selected["lock_mode"] == "right_censored_diagnostics"
    assert selected["k_low"] is None
    assert selected["k_high"] is None
    assert selected["audit_solve_authorized"] is True
    assert selected["efficacy_or_resource_claim_authorized"] is False


def test_selection_blocks_before_audit_when_reference_is_inadequate(
    monkeypatch,
) -> None:
    summaries = {
        role: _adequacy_summary() for role in lock.CANDIDATE_ROLES
    }
    summaries["S_1024"] = _adequacy_summary(raw_error=0.02)
    selected = _select(monkeypatch, summaries)
    assert selected["lock_mode"] == "blocked"
    assert selected["reason"] == "calibration_reference_inadequate"
    assert selected["audit_solve_authorized"] is False
    assert selected["frozen_audit_roles"] == []


def test_node_identical_exhaustion_blocks_phase_one_pending_gauge_amendment(
    monkeypatch,
) -> None:
    summaries = {
        role: _adequacy_summary() for role in lock.CANDIDATE_ROLES
    }
    selected = _select(
        monkeypatch,
        summaries,
        identical_roles=("S_256", "S_512"),
    )
    assert selected["lock_mode"] == "blocked"
    assert selected["k_low"] is None
    assert selected["k_high"] is None
    assert selected["endpoint_node_identical_across_all_batches"] is True
    assert selected["efficacy_or_resource_claim_authorized"] is False
    assert selected["audit_solve_authorized"] is False


def _calibration_work_order() -> dict:
    batches = []
    for batch_index in range(lock.EXPECTED_CALIBRATION_BATCHES):
        anchors = [f"a{4 * batch_index + offset:02d}" for offset in range(4)]
        batches.append(
            {
                "anchors_by_quartile": [
                    {"node_id": anchor, "quartile_id": quartile}
                    for anchor, quartile in zip(anchors, lock.EXPECTED_QUARTILES)
                ],
                "batch_id": f"calibration-{batch_index:02d}",
                "shells": [
                    {"anchor_node_id": anchor, "shell_nodes": [anchor]}
                    for anchor in anchors
                ],
            }
        )
    return {"batches": batches}


def _calibration_contract() -> dict:
    return {
        "cholesky_atol": 1e-12,
        "cholesky_rtol": 1e-11,
        "m_matrix_tolerance": 1e-12,
        "maximum_principle_tolerance": 1e-10,
        "minimum_rcond": 1e-12,
        "per_batch_elapsed_ceiling_seconds": 1000,
        "solve_residual_tolerance": 1e-10,
    }


def test_calibration_global_max_uses_all_32_unique_anchors_and_is_order_invariant(
    monkeypatch,
) -> None:
    work_order = _calibration_work_order()

    def domain_from_role(batch, _role, _adjacency):
        return SimpleNamespace(
            anchors=tuple(
                row["node_id"] for row in batch["anchors_by_quartile"]
            )
        )

    def calibrate(_reference, *, anchors, **_kwargs):
        records = tuple(
            SimpleNamespace(
                added_leakage_conductance=(int(anchor[1:]) + 1) / 100.0
            )
            for anchor in anchors
        )
        return SimpleNamespace(
            eigendecomposition_count=1,
            numerical_minimum_added_leakage=0.0,
            per_anchor=records,
            study_added_leakage_conductance=max(
                row.added_leakage_conductance for row in records
            ),
            total_evaluations=len(records),
        )

    monkeypatch.setattr(lock, "_domain_from_role", domain_from_role)
    monkeypatch.setattr(lock, "_zero_alpha_precheck", lambda *_a, **_k: {"status": "pass"})
    monkeypatch.setattr(lock.local, "calibrate_uniform_leakage_per_anchor", calibrate)
    contract = _calibration_contract()

    batches, alpha = lock._run_calibration_pass(work_order, {}, contract)
    reversed_order = {"batches": list(reversed(work_order["batches"]))}
    reversed_batches, reversed_alpha = lock._run_calibration_pass(
        reversed_order, {}, contract
    )

    assert len(batches) == len(reversed_batches) == 8
    assert sum(len(row["per_anchor"]) for row in batches) == 32
    assert alpha == reversed_alpha == pytest.approx(0.32)


def test_calibration_rejects_survivor_only_maximum_from_31_anchors(
    monkeypatch,
) -> None:
    work_order = _calibration_work_order()

    def domain_from_role(batch, _role, _adjacency):
        return SimpleNamespace(
            anchors=tuple(
                row["node_id"] for row in batch["anchors_by_quartile"]
            )
        )

    call_count = 0

    def calibrate(_reference, *, anchors, **_kwargs):
        nonlocal call_count
        call_count += 1
        records = tuple(
            SimpleNamespace(added_leakage_conductance=0.1)
            for anchor in (anchors[:-1] if call_count == 1 else anchors)
        )
        return SimpleNamespace(
            eigendecomposition_count=1,
            numerical_minimum_added_leakage=0.0,
            per_anchor=records,
            study_added_leakage_conductance=0.1,
            total_evaluations=len(records),
        )

    monkeypatch.setattr(lock, "_domain_from_role", domain_from_role)
    monkeypatch.setattr(lock, "_zero_alpha_precheck", lambda *_a, **_k: {"status": "pass"})
    monkeypatch.setattr(lock.local, "calibrate_uniform_leakage_per_anchor", calibrate)
    with pytest.raises(lock.CalibrationLockError, match="anchor count"):
        lock._run_calibration_pass(
            work_order,
            {},
            _calibration_contract(),
        )


def test_calibration_blocks_conditioning_derived_leakage(monkeypatch) -> None:
    work_order = _calibration_work_order()

    def domain_from_role(batch, _role, _adjacency):
        return SimpleNamespace(
            anchors=tuple(
                row["node_id"] for row in batch["anchors_by_quartile"]
            )
        )

    def calibrate(_reference, *, anchors, **_kwargs):
        return SimpleNamespace(
            eigendecomposition_count=1,
            numerical_minimum_added_leakage=1e-12,
            per_anchor=tuple(
                SimpleNamespace(added_leakage_conductance=0.1)
                for _anchor in anchors
            ),
            study_added_leakage_conductance=0.1,
            total_evaluations=len(anchors),
        )

    monkeypatch.setattr(lock, "_domain_from_role", domain_from_role)
    monkeypatch.setattr(
        lock, "_zero_alpha_precheck", lambda *_a, **_k: {"status": "pass"}
    )
    monkeypatch.setattr(lock.local, "calibrate_uniform_leakage_per_anchor", calibrate)

    with pytest.raises(np.linalg.LinAlgError, match="conditioning-derived leakage"):
        lock._run_calibration_pass(work_order, {}, _calibration_contract())


def _projection_context() -> dict:
    adjacency = {}
    batches = []
    domains = []
    boundaries = []
    shells = []
    for batch_index in range(lock.EXPECTED_CALIBRATION_BATCHES):
        batch_id = f"calibration-{batch_index:02d}"
        anchors = [f"pt:{4 * batch_index + offset + 1}" for offset in range(4)]
        adjacency.update({anchor: () for anchor in anchors})
        batches.append(
            {
                "anchors_by_quartile": [
                    {"node_id": anchor, "quartile_id": quartile}
                    for anchor, quartile in zip(anchors, lock.EXPECTED_QUARTILES)
                ],
                "batch_id": batch_id,
                "batch_index": batch_index,
                "protected_nodes": anchors,
                "split": "calibration",
            }
        )
        node_rows = [
            {"hop_distance": 0, "node_id": anchor} for anchor in anchors
        ]
        for role in lock.ROLE_ORDER:
            domains.append(
                {
                    "batch_id": batch_id,
                    "nodes": node_rows,
                    "requested_nodes": 4,
                    "role": role,
                    "truncated_final_shell_nodes": 0,
                }
            )
            boundaries.append({"batch_id": batch_id, "role": role})
        for anchor in anchors:
            shells.append(
                {
                    "anchor_node_id": anchor,
                    "batch_id": batch_id,
                    "hop_radius": 3,
                    "reasons": [],
                    "shell_nodes": [anchor],
                    "strictly_interior_pass": True,
                    "target_attenuation": "exp(-1)",
                }
            )
    batches.append(
        {
            "anchors_by_quartile": [],
            "batch_id": "audit-00",
            "batch_index": 0,
            "protected_nodes": [],
            "split": "audit",
        }
    )
    return {
        "adjacency": adjacency,
        "captured_records": {"plan_manifest": {"sha256": "0" * 64, "size_bytes": 1}},
        "manifest": {"plan_fingerprint": "1" * 64},
        "plan_artifacts": {
            "batches.jsonl": lock._jsonl_bytes(batches),
            "boundaries.jsonl": lock._jsonl_bytes(boundaries),
            "calibration_shells.jsonl": lock._jsonl_bytes(shells),
            "domains.jsonl": lock._jsonl_bytes(domains),
        },
    }


def _plan_rows(context: dict, artifact: str) -> list[dict]:
    return lock._strict_jsonl_bytes(
        context["plan_artifacts"][artifact], artifact[: -len(".jsonl")]
    )


def _set_plan_rows(context: dict, artifact: str, rows: list[dict]) -> None:
    context["plan_artifacts"][artifact] = lock._jsonl_bytes(rows)


def _derive_projection_work_order(monkeypatch, context: dict) -> dict:
    monkeypatch.setattr(
        lock.hop_plan,
        "_boundary_record",
        lambda batch_id, role, _requested, _nodes, _adjacency: {
            "batch_id": batch_id,
            "role": role,
        },
    )
    return lock._derive_calibration_work_order(context)


def test_work_order_projects_out_audit_rows_and_rejects_unknown_split(
    monkeypatch,
) -> None:
    context = _projection_context()
    work_order = _derive_projection_work_order(monkeypatch, context)
    assert len(work_order["batches"]) == 8
    assert {row["split"] for row in work_order["batches"]} == {"calibration"}
    assert all(not row["batch_id"].startswith("audit-") for row in work_order["batches"])
    assert work_order["contains_audit_metrics_or_responses"] is False

    rows = lock._strict_jsonl_bytes(
        context["plan_artifacts"]["batches.jsonl"], "batches"
    )
    rows.append(
        {
            "anchors_by_quartile": [],
            "batch_id": "unknown-00",
            "batch_index": 0,
            "protected_nodes": [],
            "split": "holdout",
        }
    )
    context["plan_artifacts"]["batches.jsonl"] = lock._jsonl_bytes(rows)
    with pytest.raises(lock.CalibrationLockError, match="unknown split"):
        lock._derive_calibration_work_order(context)


def test_work_order_rejects_wrong_batch_and_anchor_counts(monkeypatch) -> None:
    context = _projection_context()
    batches = _plan_rows(context, "batches.jsonl")
    batches = [row for row in batches if row["batch_id"] != "calibration-07"]
    _set_plan_rows(context, "batches.jsonl", batches)
    with pytest.raises(lock.CalibrationLockError, match="eight batches"):
        _derive_projection_work_order(monkeypatch, context)

    context = _projection_context()
    batches = _plan_rows(context, "batches.jsonl")
    batches[0]["anchors_by_quartile"].pop()
    _set_plan_rows(context, "batches.jsonl", batches)
    with pytest.raises(lock.CalibrationLockError, match="four anchors"):
        _derive_projection_work_order(monkeypatch, context)


def test_work_order_rejects_duplicate_and_missing_anchors(monkeypatch) -> None:
    context = _projection_context()
    batches = _plan_rows(context, "batches.jsonl")
    batches[0]["anchors_by_quartile"][1]["node_id"] = batches[0][
        "anchors_by_quartile"
    ][0]["node_id"]
    _set_plan_rows(context, "batches.jsonl", batches)
    with pytest.raises(lock.CalibrationLockError, match="quartile-balanced"):
        _derive_projection_work_order(monkeypatch, context)

    context = _projection_context()
    batches = _plan_rows(context, "batches.jsonl")
    batches[0]["anchors_by_quartile"][0]["node_id"] = "pt:missing"
    _set_plan_rows(context, "batches.jsonl", batches)
    with pytest.raises(lock.CalibrationLockError, match="absent from adjacency"):
        _derive_projection_work_order(monkeypatch, context)

    context = _projection_context()
    batches = _plan_rows(context, "batches.jsonl")
    batches[1]["anchors_by_quartile"][0]["node_id"] = batches[0][
        "anchors_by_quartile"
    ][0]["node_id"]
    _set_plan_rows(context, "batches.jsonl", batches)
    with pytest.raises(lock.CalibrationLockError, match="globally unique"):
        _derive_projection_work_order(monkeypatch, context)


def test_work_order_rejects_unbalanced_quartiles(monkeypatch) -> None:
    context = _projection_context()
    batches = _plan_rows(context, "batches.jsonl")
    batches[0]["anchors_by_quartile"][-1]["quartile_id"] = "q3"
    _set_plan_rows(context, "batches.jsonl", batches)
    with pytest.raises(lock.CalibrationLockError, match="quartile-balanced"):
        _derive_projection_work_order(monkeypatch, context)


def test_work_order_rejects_non_nested_and_missing_roles(monkeypatch) -> None:
    context = _projection_context()
    domains = _plan_rows(context, "domains.jsonl")
    s512 = next(
        row
        for row in domains
        if row["batch_id"] == "calibration-00" and row["role"] == "S_512"
    )
    s512["nodes"][0], s512["nodes"][1] = s512["nodes"][1], s512["nodes"][0]
    _set_plan_rows(context, "domains.jsonl", domains)
    with pytest.raises(lock.CalibrationLockError, match="not nested"):
        _derive_projection_work_order(monkeypatch, context)

    context = _projection_context()
    domains = _plan_rows(context, "domains.jsonl")
    domains = [
        row
        for row in domains
        if not (
            row["batch_id"] == "calibration-00" and row["role"] == "S_512"
        )
    ]
    _set_plan_rows(context, "domains.jsonl", domains)
    with pytest.raises(lock.CalibrationLockError, match="role is missing"):
        _derive_projection_work_order(monkeypatch, context)


def test_role_lookup_rejects_unknown_roles() -> None:
    batch = {"roles": [{"role": "S_256"}]}
    with pytest.raises(lock.CalibrationLockError, match="ambiguous"):
        lock._role_item(batch, "UNKNOWN")


def _two_node_domain(*, grounded: bool):
    exterior = ("outside",) if grounded else ()
    return lock.local.LocalDiffusionDomain(
        nodes=("a", "b"),
        anchors=("a",),
        hop_distance=np.asarray([0, 1], dtype=np.int64),
        neighbors=(("b",), ("a",) + exterior),
        maximum_nodes=2,
        complete_distance_shell=True,
        truncated_tie_count=0,
    )


def test_zero_alpha_precheck_accepts_structurally_grounded_reference() -> None:
    result = lock._zero_alpha_precheck(
        _two_node_domain(grounded=True),
        cholesky_atol=1e-12,
        cholesky_rtol=1e-11,
        m_matrix_tolerance=1e-12,
        maximum_principle_tolerance=1e-10,
        minimum_rcond=1e-12,
        solve_tolerance=1e-10,
    )
    assert result["status"] == "pass_without_added_leakage"
    assert float.fromhex(result["reciprocal_condition"]) > 0.0
    assert float.fromhex(result["solve_relative_residual"]) <= 1e-10


def test_zero_alpha_precheck_rejects_ungrounded_singular_reference() -> None:
    with pytest.raises(np.linalg.LinAlgError):
        lock._zero_alpha_precheck(
            _two_node_domain(grounded=False),
            cholesky_atol=1e-12,
            cholesky_rtol=1e-11,
            m_matrix_tolerance=1e-12,
            maximum_principle_tolerance=1e-10,
            minimum_rcond=1e-12,
            solve_tolerance=1e-10,
        )


def _valid_manifest(monkeypatch) -> dict:
    monkeypatch.setattr(lock, "_implementation_records", lambda: {})
    monkeypatch.setattr(lock, "_git_commit", lambda: "a" * 40)
    payloads = {name: b"" for name in lock.ARTIFACT_NAMES}
    context = {
        "captured_records": {
            "plan_manifest": {"sha256": "b" * 64, "size_bytes": 10}
        },
        "manifest": {"plan_fingerprint": "c" * 64},
    }
    selection = {
        "alpha_top": 0.125,
        "audit_solve_authorized": True,
        "lock_mode": "finite_contrast",
        "reason": "test_finite_contrast",
    }
    return lock._build_lock_manifest(
        payloads,
        context=context,
        contract={"effective_resistance": False, "numeric": {}},
        runtime_blas=[
            {
                "observed_num_threads": 1,
                "user_api": "blas",
                "version": "test",
            }
        ],
        selection=selection,
        calibration_completed=True,
    )


def _reseal(manifest: dict, *, scientific_core_changed: bool) -> None:
    if scientific_core_changed:
        manifest["lock_fingerprint"] = hashlib.sha256(
            lock._canonical_json(manifest["scientific_core"])
        ).hexdigest()
    manifest["manifest_integrity_seal"] = hashlib.sha256(
        lock._canonical_json(lock._manifest_without_seal(manifest))
    ).hexdigest()


def test_manifest_rejects_tampered_scientific_core_even_with_outer_reseal(
    monkeypatch,
) -> None:
    manifest = _valid_manifest(monkeypatch)
    lock._validate_manifest_invariants(manifest)
    tampered = deepcopy(manifest)
    tampered["scientific_core"]["plan_fingerprint"] = "d" * 64
    _reseal(tampered, scientific_core_changed=False)
    with pytest.raises(lock.CalibrationLockError, match="scientific fingerprint"):
        lock._validate_manifest_invariants(tampered)


def test_manifest_cannot_authorize_audit_in_blocked_mode_even_when_rehashed(
    monkeypatch,
) -> None:
    manifest = _valid_manifest(monkeypatch)
    invalid = deepcopy(manifest)
    invalid["lock_mode"] = "blocked"
    invalid["scientific_core"]["lock_mode"] = "blocked"
    # Deliberately retain accepted/audit authorization as true, then recompute
    # every public hash. The logical authorization invariant must still reject.
    _reseal(invalid, scientific_core_changed=True)
    with pytest.raises(lock.CalibrationLockError, match="authorization"):
        lock._validate_manifest_invariants(invalid)


def test_manifest_cannot_authorize_audit_with_incomplete_calibration(
    monkeypatch,
) -> None:
    manifest = _valid_manifest(monkeypatch)
    invalid = deepcopy(manifest)
    invalid["calibration_completed"] = False
    invalid["scientific_core"]["calibration_completed"] = False
    invalid["scientific_core"]["alpha_top_hex"] = None
    _reseal(invalid, scientific_core_changed=True)
    with pytest.raises(lock.CalibrationLockError, match="incomplete calibration"):
        lock._validate_manifest_invariants(invalid)


def test_manifest_cannot_authorize_audit_without_finite_frozen_alpha(
    monkeypatch,
) -> None:
    manifest = _valid_manifest(monkeypatch)
    invalid = deepcopy(manifest)
    invalid["scientific_core"]["alpha_top_hex"] = None
    _reseal(invalid, scientific_core_changed=True)
    with pytest.raises(lock.CalibrationLockError, match="freeze alpha"):
        lock._validate_manifest_invariants(invalid)


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("nonfinite numerical diagnostic"),
        lock.CalibrationLockError("nonfinite aggregate"),
    ],
)
def test_numerical_contract_errors_install_blocked_scientific_payloads(
    monkeypatch, failure,
) -> None:
    contract = {
        "peak_rss_ceiling_bytes": 10_000,
    }
    monkeypatch.setattr(lock, "_validated_bound_contract", lambda _manifest: contract)
    monkeypatch.setattr(
        lock,
        "_derive_calibration_work_order",
        lambda _context: {"batches": [], "schema": "test"},
    )
    monkeypatch.setattr(lock, "_adjacency_from_work_order", lambda _order: {})
    monkeypatch.setattr(
        lock,
        "_run_calibration_pass",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(lock, "_peak_rss_bytes", lambda: 1)
    payloads, selection, observed_contract, completed = lock._derive_lock_payloads(
        {"manifest": {}},
        [{"observed_num_threads": 1, "user_api": "blas"}],
    )
    assert observed_contract is contract
    assert completed is False
    assert selection["lock_mode"] == "blocked"
    assert selection["audit_solve_authorized"] is False
    assert payloads["anchor_calibrations.jsonl"] == b""
    assert payloads["batch_calibration.jsonl"] == b""
    assert payloads["calibration_fidelity.jsonl"] == b""


def _evidence_contract() -> dict:
    return {
        "calibration": {
            "bisection_relative_tolerance_hex": float(1e-8).hex(),
            "maximum_function_evaluations_per_anchor": 80,
        },
        "effective_resistance": False,
        "maximum_principle_tolerance": 1e-10,
        "minimum_rcond": 1e-12,
        "m_matrix_tolerance": 1e-12,
        "peak_rss_ceiling_bytes": 10_000,
        "solve_residual_tolerance": 1e-10,
        "statistics": plan._statistical_contract("omitted"),
    }


def _screening_record(anchor: str, *, attenuation: float) -> dict:
    return {
        "anchor": anchor,
        "attenuation_threshold": math.exp(-1.0).hex(),
        "distance_metric": "unweighted_hop",
        "maximum_observed_radius": float(3.0).hex(),
        "radius_lower": float(2.0).hex(),
        "radius_upper": float(3.0).hex(),
        "right_censored": False,
        "shell_attenuation": attenuation.hex(),
    }


def _zero_alpha_evidence() -> dict:
    return {
        "boundary_harmonic_max": float(0.0).hex(),
        "cholesky_reconstruction_relative_error": float(0.0).hex(),
        "maximum_kirchhoff_relative_error": float(0.0).hex(),
        "maximum_precision_eigenvalue": float(2.0).hex(),
        "maximum_principle_violation": float(0.0).hex(),
        "minimum_precision_eigenvalue": float(1.0).hex(),
        "reciprocal_condition": float(0.5).hex(),
        "solve_relative_residual": float(0.0).hex(),
        "status": "pass_without_added_leakage",
    }


def _valid_calibration_evidence():
    work_order = _calibration_work_order()
    added = 0.1
    target = math.exp(-1.0)
    lower_attenuation = target + 1e-10
    upper_attenuation = target - 1e-10
    lower_added = added * (1.0 - 0.5e-8)
    anchors = []
    batches = []
    fidelity = []
    for batch in work_order["batches"]:
        for anchor in batch["anchors_by_quartile"]:
            node_id = anchor["node_id"]
            anchors.append(
                {
                    "added_leakage_conductance": added.hex(),
                    "anchor_node_id": node_id,
                    "batch_id": batch["batch_id"],
                    "iterations": 2,
                    "minimality_certificate": {
                        "attenuation_at_lower": lower_attenuation.hex(),
                        "attenuation_at_upper": upper_attenuation.hex(),
                        "bracket_seed_radius": 3,
                        "initial_lower_passed": False,
                        "lower_added_leakage_conductance": lower_added.hex(),
                        "relative_tolerance": float(1e-8).hex(),
                        "target_attenuation": target.hex(),
                        "upper_added_leakage_conductance": added.hex(),
                    },
                    "numerical_minimum_added_leakage": float(0.0).hex(),
                    "quartile_id": anchor["quartile_id"],
                    "screening_at_batch_leakage": _screening_record(
                        node_id, attenuation=upper_attenuation
                    ),
                    "screening_at_global_leakage": _screening_record(
                        node_id, attenuation=upper_attenuation
                    ),
                    "screening_at_selected_leakage": _screening_record(
                        node_id, attenuation=upper_attenuation
                    ),
                    "shell_nodes": [node_id],
                }
            )
        batches.append(
            {
                "batch_id": batch["batch_id"],
                "batch_maximum_added_leakage": added.hex(),
                "eigendecomposition_count": 1,
                "factor_provenance": {
                    "batch_id": batch["batch_id"],
                    "candidate_build_count": 3,
                    "candidate_factorization_count": 3,
                    "candidate_model_index": [0, 1, 2],
                    "candidate_reference_reuse_count": 0,
                    "candidate_requested_result_count": 3,
                    "candidate_unique_model_count": 3,
                    "reference_build_count": 1,
                    "reference_factorization_count": 1,
                },
                "global_alpha_top": added.hex(),
                "numerical_minimum_added_leakage": float(0.0).hex(),
                "total_attenuation_evaluations": 8,
                "zero_alpha_precheck": _zero_alpha_evidence(),
            }
        )
        reference_safety = {"batch": batch["batch_id"], "kind": "reference"}
        for model_index, role in enumerate(lock.CANDIDATE_ROLES):
            fidelity.append(
                {
                    "batch_id": batch["batch_id"],
                    "candidate_model_index": model_index,
                    "candidate_safety": {
                        "batch": batch["batch_id"],
                        "model_index": model_index,
                    },
                    "reference_safety": reference_safety,
                    "role": role,
                }
            )
    return work_order, anchors, batches, fidelity, added


def test_calibration_evidence_rederives_anchor_batch_and_global_maxima() -> None:
    work_order, anchors, batches, fidelity, alpha_top = (
        _valid_calibration_evidence()
    )
    lock._validate_calibration_evidence(
        anchors,
        batches,
        fidelity,
        work_order,
        alpha_top,
        _evidence_contract(),
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (
            lambda anchors, batches, fidelity: batches[0].__setitem__(
                "batch_maximum_added_leakage", float(0.09).hex()
            ),
            "batch calibration evidence mismatch",
        ),
        (
            lambda anchors, batches, fidelity: anchors[0][
                "minimality_certificate"
            ].__setitem__("attenuation_at_lower", math.exp(-1.0).hex()),
            "minimality lower endpoint does not miss",
        ),
        (
            lambda anchors, batches, fidelity: batches[0][
                "factor_provenance"
            ].__setitem__("candidate_factorization_count", 2),
            "factor-count provenance mismatch",
        ),
        (
            lambda anchors, batches, fidelity: fidelity[1].__setitem__(
                "reference_safety", {"tampered": True}
            ),
            "reference safety diagnostics disagree",
        ),
    ],
)
def test_calibration_evidence_rejects_resealed_internal_inconsistency(
    tamper, message
) -> None:
    work_order, anchors, batches, fidelity, alpha_top = (
        _valid_calibration_evidence()
    )
    tamper(anchors, batches, fidelity)
    with pytest.raises(lock.CalibrationLockError, match=message):
        lock._validate_calibration_evidence(
            anchors,
            batches,
            fidelity,
            work_order,
            alpha_top,
            _evidence_contract(),
        )


def _path_adjacency(nodes: list[str]) -> dict[str, tuple[str, ...]]:
    adjacency = {}
    for index, node in enumerate(nodes):
        neighbors = []
        if index:
            neighbors.append(nodes[index - 1])
        if index + 1 < len(nodes):
            neighbors.append(nodes[index + 1])
        adjacency[node] = tuple(neighbors)
    return adjacency


def _generated_fidelity_fixture():
    anchors = [f"pt:{index}" for index in range(1, 5)]
    extra = [f"pt:{index}" for index in range(5, 15)]
    graph_nodes = anchors + extra
    adjacency = _path_adjacency(graph_nodes)
    protected = graph_nodes[:12]
    role_nodes = {
        "S_256": graph_nodes[:12],
        # This nominally larger endpoint is node-identical and must reuse the
        # S_256 model/safety diagnostics while retaining its own fingerprint.
        "S_512": graph_nodes[:12],
        "S_1024": graph_nodes[:13],
        "R_top": graph_nodes,
    }
    requested = {"S_256": 256, "S_512": 512, "S_1024": 1024, "R_top": 4096}
    roles = []
    for role in lock.ROLE_ORDER:
        nodes = role_nodes[role]
        roles.append(
            {
                "boundary": {"batch_id": "calibration-00", "role": role},
                "domain": {
                    "batch_id": "calibration-00",
                    "nodes": [
                        {
                            "hop_distance": (
                                0 if node in anchors else int(node.split(":")[1]) - 4
                            ),
                            "node_id": node,
                        }
                        for node in nodes
                    ],
                    "requested_nodes": requested[role],
                    "role": role,
                    "truncated_final_shell_nodes": 0,
                },
                "role": role,
            }
        )
    batch = {
        "anchors_by_quartile": [
            {"node_id": anchor, "quartile_id": quartile}
            for anchor, quartile in zip(anchors, lock.EXPECTED_QUARTILES)
        ],
        "batch_id": "calibration-00",
        "batch_index": 0,
        "protected_nodes": protected,
        "roles": roles,
        "shells": [],
        "split": "calibration",
    }
    work_order = {
        "batches": [batch],
        "incident_adjacency": [
            {"neighbors": list(adjacency[node]), "node_id": node}
            for node in graph_nodes
        ],
        "plan_fingerprint": "f" * 64,
    }
    alpha_top = 0.1
    candidates = tuple(
        lock._selection_from_role(
            batch, role, adjacency, work_order["plan_fingerprint"]
        )
        for role in lock.CANDIDATE_ROLES
    )
    reference = lock._selection_from_role(
        batch, "R_top", adjacency, work_order["plan_fingerprint"]
    )
    evaluated = lock.evaluate_nested_bounded_domain_fidelity(
        candidates,
        reference,
        protected_nodes=tuple(protected),
        intrinsic_leakage_conductance=alpha_top,
        rank_top_k=8,
        minimum_reciprocal_condition=_evidence_contract()["minimum_rcond"],
        include_effective_resistance=False,
    )
    assert evaluated.candidate_model_index[0] == evaluated.candidate_model_index[1]
    rows = []
    for result_index, (role, result) in enumerate(
        zip(lock.CANDIDATE_ROLES, evaluated)
    ):
        model_index = evaluated.candidate_model_index[result_index]
        candidate_safety = (
            evaluated.reference_model_diagnostics
            if model_index == -1
            else evaluated.candidate_model_diagnostics[model_index]
        )
        rows.append(
            {
                "batch_id": batch["batch_id"],
                "candidate_model_index": model_index,
                "candidate_safety": lock._hexify_scientific(
                    candidate_safety.as_dict()
                ),
                "reference_safety": lock._hexify_scientific(
                    evaluated.reference_model_diagnostics.as_dict()
                ),
                "result": lock._hexify_scientific(
                    lock._strip_timing_fields(result.as_dict())
                ),
                "role": role,
            }
        )
    return rows, work_order, adjacency, alpha_top


def test_fidelity_decoder_accepts_only_frozen_metric_domain_and_safety() -> None:
    rows, work_order, adjacency, alpha_top = _generated_fidelity_fixture()
    decoded = lock._decoded_fidelity_rows(
        rows,
        _evidence_contract(),
        work_order,
        adjacency,
        alpha_top,
    )
    assert len(decoded) == len(lock.CANDIDATE_ROLES)
    assert decoded[0]["result"].per_anchor_top_k_overlap == pytest.approx(
        tuple(
            float.fromhex(value)
            for value in rows[0]["result"]["per_anchor_top_k_overlap"]
        )
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (
            lambda row: row["result"]["per_anchor_raw_relative_l2_error"].__setitem__(
                0, float(-0.01).hex()
            ),
            "invalid domain",
        ),
        (
            lambda row: row["result"]["per_anchor_top_k_overlap"].__setitem__(
                0, float(1.01).hex()
            ),
            "outside \\[0, 1\\]",
        ),
        (
            lambda row: row["candidate_safety"].__setitem__(
                "maximum_positive_off_diagonal", float(1e-4).hex()
            ),
            "stored numerical gate failed",
        ),
        (
            lambda row: row["result"].__setitem__(
                "per_anchor_effective_resistance_relative_error",
                [float(0.01).hex()] * 4,
            ),
            "omitted effective-resistance arm",
        ),
    ],
)
def test_fidelity_decoder_fails_closed_on_metric_or_safety_tampering(
    tamper, message
) -> None:
    rows, work_order, adjacency, alpha_top = _generated_fidelity_fixture()
    tamper(rows[0])
    with pytest.raises(lock.CalibrationLockError, match=message):
        lock._decoded_fidelity_rows(
            rows,
            _evidence_contract(),
            work_order,
            adjacency,
            alpha_top,
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (
            lambda rows: rows[0]["result"].__setitem__(
                "candidate_selection_fingerprint", "0" * 64
            ),
            "provenance disagrees with the work order",
        ),
        (
            lambda rows: rows[0]["result"].__setitem__(
                "reference_selection_fingerprint", "0" * 64
            ),
            "provenance disagrees with the work order",
        ),
        (
            lambda rows: rows[0]["result"].__setitem__(
                "source_nodes", list(reversed(rows[0]["result"]["source_nodes"]))
            ),
            "provenance disagrees with the work order",
        ),
        (
            lambda rows: rows[0]["result"].__setitem__(
                "alpha_fingerprint", "0" * 64
            ),
            "provenance disagrees with the work order",
        ),
        (
            lambda rows: rows[1]["candidate_safety"].update(
                {
                    "selection_fingerprint": rows[1]["result"][
                        "candidate_selection_fingerprint"
                    ],
                    "strategy": rows[1]["result"]["candidate_strategy"],
                }
            ),
            "candidate safety identity disagrees with the frozen selection",
        ),
        (
            lambda rows: rows[0]["result"].__setitem__(
                "raw_relative_l2_error_90th_percentile", float(0.123).hex()
            ),
            "order-statistic summary disagrees",
        ),
    ],
)
def test_fidelity_decoder_binds_domain_alpha_reuse_and_order_statistics(
    tamper, message
) -> None:
    rows, work_order, adjacency, alpha_top = _generated_fidelity_fixture()
    tamper(rows)
    with pytest.raises(lock.CalibrationLockError, match=message):
        lock._decoded_fidelity_rows(
            rows,
            _evidence_contract(),
            work_order,
            adjacency,
            alpha_top,
        )


def _incomplete_payload_fixture(
    *, observed_rss: int, ceiling: int, resource_blocked: bool
):
    reason = (
        "calibration_resource_ceiling_exceeded"
        if resource_blocked
        else "calibration_numerical_contract_failed"
    )
    work_order = {"batches": [], "schema": "test-work-order"}
    runtime = [
        {
            "observed_num_threads": 1,
            "user_api": "blas",
            "version": "test",
        }
    ]
    selection = {
        "alpha_top": None,
        "audit_solve_authorized": False,
        "confirmatory_claim_authorized": False,
        "efficacy_or_resource_claim_authorized": False,
        "frozen_audit_roles": [],
        "k_high": None,
        "k_low": None,
        "lock_mode": "blocked",
        "reason": reason,
    }
    payloads = {
        "anchor_calibrations.jsonl": b"",
        "batch_calibration.jsonl": b"",
        "calibration_fidelity.jsonl": b"",
        "calibration_work_order.json": lock._canonical_json(work_order),
        "execution.json": lock._canonical_json(
            {
                "actual_blas_identity": runtime,
                "audit_metrics_materialized": False,
                "audit_solves_executed": 0,
                "batch_phase_timings": [],
                "calibration_only_work_order": True,
                "calibration_seconds": None,
                "fidelity_seconds": None,
                "observed_peak_rss_bytes": observed_rss,
                "peak_rss_ceiling_bytes": ceiling,
                "timing_and_rss_outside_scientific_fingerprint": True,
                "total_seconds": 0.0,
            }
        ),
        "selection.json": lock._canonical_json(selection),
    }
    manifest = {
        "audit_solve_authorized": False,
        "calibration_completed": False,
        "lock_mode": "blocked",
        "reason": reason,
        "scientific_core": {
            "actual_blas_identity": runtime,
            "alpha_top_hex": None,
        },
    }
    contract = {
        "peak_rss_ceiling_bytes": ceiling,
        "per_batch_elapsed_ceiling_seconds": 10.0,
    }
    return payloads, manifest, work_order, contract


def _completed_execution_fixture():
    runtime = [
        {
            "observed_num_threads": 1,
            "user_api": "blas",
            "version": "test",
        }
    ]
    work_order = {
        "batches": [
            {"batch_id": f"calibration-{index:02d}"}
            for index in range(lock.EXPECTED_CALIBRATION_BATCHES)
        ]
    }
    execution = {
        "actual_blas_identity": runtime,
        "audit_metrics_materialized": False,
        "audit_solves_executed": 0,
        "batch_phase_timings": [
            {
                "batch_id": batch["batch_id"],
                "calibration_seconds": 1.0,
                "fidelity_seconds": 1.0,
            }
            for batch in work_order["batches"]
        ],
        "calibration_only_work_order": True,
        "calibration_seconds": 8.0,
        "fidelity_seconds": 8.0,
        "observed_peak_rss_bytes": 9_000,
        "peak_rss_ceiling_bytes": 10_000,
        "timing_and_rss_outside_scientific_fingerprint": True,
        "total_seconds": 16.0,
    }
    core = {"actual_blas_identity": deepcopy(runtime)}
    contract = {
        "peak_rss_ceiling_bytes": 10_000,
        "per_batch_elapsed_ceiling_seconds": 10.0,
    }
    return execution, core, contract, work_order


def test_completed_execution_rejects_batch_elapsed_ceiling_violation() -> None:
    execution, core, contract, work_order = _completed_execution_fixture()
    execution["batch_phase_timings"][3]["fidelity_seconds"] = 10.1
    with pytest.raises(
        lock.CalibrationLockError,
        match="batch elapsed time exceeds its frozen ceiling",
    ):
        lock._validate_execution_provenance(
            execution, core, contract, True, work_order
        )


def test_execution_runtime_identity_must_match_scientific_core() -> None:
    execution, core, contract, work_order = _completed_execution_fixture()
    execution["actual_blas_identity"][0]["version"] = "tampered"
    with pytest.raises(
        lock.CalibrationLockError,
        match="execution provenance violates audit isolation",
    ):
        lock._validate_execution_provenance(
            execution, core, contract, True, work_order
        )


@pytest.mark.parametrize(
    ("observed_rss", "ceiling", "resource_blocked"),
    [(9_999, 10_000, False), (10_001, 10_000, True)],
)
def test_execution_rss_and_resource_decision_agree(
    observed_rss, ceiling, resource_blocked
) -> None:
    payloads, manifest, work_order, contract = _incomplete_payload_fixture(
        observed_rss=observed_rss,
        ceiling=ceiling,
        resource_blocked=resource_blocked,
    )
    lock._validate_lock_payload_structure(
        payloads, manifest, work_order, contract
    )


@pytest.mark.parametrize(
    ("observed_rss", "resource_blocked"),
    [(10_001, False), (9_999, True)],
)
def test_execution_rss_cannot_disagree_with_resource_decision(
    observed_rss, resource_blocked
) -> None:
    payloads, manifest, work_order, contract = _incomplete_payload_fixture(
        observed_rss=observed_rss,
        ceiling=10_000,
        resource_blocked=resource_blocked,
    )
    with pytest.raises(
        lock.CalibrationLockError,
        match="resource evidence disagrees with the lock decision",
    ):
        lock._validate_lock_payload_structure(
            payloads, manifest, work_order, contract
        )


def test_execution_rss_must_use_the_frozen_ceiling() -> None:
    payloads, manifest, work_order, contract = _incomplete_payload_fixture(
        observed_rss=9_999,
        ceiling=10_000,
        resource_blocked=False,
    )
    contract["peak_rss_ceiling_bytes"] = 9_000
    with pytest.raises(lock.CalibrationLockError, match="resource provenance"):
        lock._validate_lock_payload_structure(
            payloads, manifest, work_order, contract
        )


def test_lock_reader_rejects_oversized_content_record_before_payload_read(
    monkeypatch,
) -> None:
    record = {"sha256": "a" * 64, "size_bytes": 0}
    scientific = {
        name: dict(record) for name in lock.SCIENTIFIC_ARTIFACT_NAMES
    }
    oversized_name = "anchor_calibrations.jsonl"
    scientific[oversized_name]["size_bytes"] = (
        lock.MAXIMUM_LOCK_ARTIFACT_BYTES[oversized_name] + 1
    )
    manifest = {
        "marker_record": dict(record),
        "observational_artifact_records": {
            name: dict(record) for name in lock.OBSERVATIONAL_ARTIFACT_NAMES
        },
        "scientific_core": {"scientific_artifact_records": scientific},
    }

    def unexpected_read(*_args, **_kwargs):
        pytest.fail("oversized content record reached bulk payload reading")

    monkeypatch.setattr(lock, "_read_bound_bytes", unexpected_read)
    with pytest.raises(
        lock.CalibrationLockError,
        match="scientific artifact exceeds its frozen read ceiling",
    ):
        lock._read_lock_payloads(object(), manifest)


def test_manifest_rejects_non_single_thread_blas_identity_even_when_resealed(
    monkeypatch,
) -> None:
    manifest = _valid_manifest(monkeypatch)
    invalid = deepcopy(manifest)
    invalid["scientific_core"]["actual_blas_identity"][0][
        "observed_num_threads"
    ] = 2
    _reseal(invalid, scientific_core_changed=True)
    with pytest.raises(
        lock.CalibrationLockError,
        match="BLAS thread identity is invalid",
    ):
        lock._validate_manifest_invariants(invalid)


def test_runtime_blas_gate_rejects_observed_two_thread_backend(
    monkeypatch,
) -> None:
    import threadpoolctl

    @contextmanager
    def requested_one_thread(*, limits, user_api):
        assert limits == 1
        assert user_api == "blas"
        yield

    monkeypatch.setattr(threadpoolctl, "threadpool_limits", requested_one_thread)
    monkeypatch.setattr(
        threadpoolctl,
        "threadpool_info",
        lambda: [
            {
                "architecture": "test",
                "internal_api": "test",
                "num_threads": 2,
                "prefix": "test",
                "threading_layer": "test",
                "user_api": "blas",
                "version": "test",
            }
        ],
    )
    with pytest.raises(
        lock.CalibrationLockError,
        match="BLAS thread contract is not one",
    ):
        with lock._single_blas_thread():
            pytest.fail("two-thread backend passed the runtime gate")


def test_blocked_lock_prepare_install_verify_round_trip_and_commit_binding(
    monkeypatch, tmp_path
) -> None:
    payloads, _fixture_manifest, work_order, contract = (
        _incomplete_payload_fixture(
            observed_rss=9_999,
            ceiling=10_000,
            resource_blocked=False,
        )
    )
    contract = {
        **contract,
        "effective_resistance": False,
        "numeric": {},
    }
    context = {
        "captured_records": {
            "plan_manifest": {"sha256": "b" * 64, "size_bytes": 10}
        },
        "manifest": {
            "fingerprint_core": {
                "resource_contract": {"planner_input_ceiling_bytes": 1_024}
            },
            "plan_fingerprint": "c" * 64,
        },
    }
    selection = lock._strict_json_bytes(payloads["selection.json"], "selection")
    runtime = lock._strict_json_bytes(payloads["execution.json"], "execution")[
        "actual_blas_identity"
    ]

    @contextmanager
    def one_thread_runtime():
        yield deepcopy(runtime)

    monkeypatch.setattr(
        lock,
        "_capture_verified_context",
        lambda *_args, **_kwargs: deepcopy(context),
    )
    monkeypatch.setattr(
        lock,
        "_validated_bound_contract",
        lambda _manifest: deepcopy(contract),
    )
    monkeypatch.setattr(
        lock,
        "_derive_calibration_work_order",
        lambda _context: deepcopy(work_order),
    )
    monkeypatch.setattr(
        lock,
        "_derive_lock_payloads",
        lambda _context, _runtime: (
            deepcopy(payloads),
            deepcopy(selection),
            deepcopy(contract),
            False,
        ),
    )
    monkeypatch.setattr(lock, "_implementation_records", lambda: {})
    monkeypatch.setattr(lock, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(lock, "_single_blas_thread", one_thread_runtime)
    monkeypatch.setattr(
        lock.hop_plan,
        "_assert_no_output_input_overlap",
        lambda *_args, **_kwargs: None,
    )

    lock_dir = tmp_path / "calibration-lock"
    common = {
        "attempt_a_dir": tmp_path / "attempt-a",
        "attempt_b_dir": tmp_path / "attempt-b",
        "lock_dir": lock_dir,
        "plan_dir": tmp_path / "plan",
        "receipt_dir": tmp_path / "receipt",
        "relation_policy": tmp_path / "policy.json",
        "source_spec": tmp_path / "sources.json",
    }
    prepared, exit_code = lock.prepare_lock(local_root=tmp_path, **common)
    assert exit_code == 2
    assert prepared["lock_mode"] == "blocked"
    assert prepared["audit_solve_authorized"] is False
    assert lock.verify_lock(**common) == prepared

    monkeypatch.setattr(lock, "_git_commit", lambda: "d" * 40)
    with pytest.raises(
        lock.CalibrationLockError,
        match="implementation binding changed",
    ):
        lock.verify_lock(**common)
