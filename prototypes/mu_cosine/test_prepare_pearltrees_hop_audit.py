#!/usr/bin/env python3
"""Focused statistical-contract tests for the untouched HOP audit."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import math
from types import SimpleNamespace

import pytest

import prepare_pearltrees_hop_audit as audit
import prepare_pearltrees_hop_plan as plan


def _statuses(
    complete_count: int,
    *,
    deterministic_failure: bool = False,
) -> list[dict]:
    rows = []
    for batch_index in range(audit.EXPECTED_AUDIT_BATCHES):
        complete = batch_index < complete_count
        failure_kind = None if complete else "registered_coverage_failure"
        if deterministic_failure and batch_index == complete_count:
            failure_kind = "deterministic_safety_failure"
        rows.append(
            {
                "batch_id": f"audit-{batch_index:02d}",
                "batch_index": batch_index,
                "complete": complete,
                "failure_class": None if complete else "SyntheticFailure",
                "failure_kind": failure_kind,
                "roles_completed": (
                    ["S_256", "S_512", "S_1024"] if complete else []
                ),
            }
        )
    return rows


def _finite_work_order() -> dict:
    return {
        "candidate_roles": ["S_256", "S_512", "S_1024"],
        "decision_roles": ["S_256", "S_512"],
        "lock_mode": "finite_contrast",
    }


def _contract() -> dict:
    return {
        "effective_resistance": False,
        "peak_rss_ceiling_bytes": 10**12,
        "statistics": plan._statistical_contract("omitted"),
    }


def _synthetic_plan_material() -> tuple[
    dict, dict[str, bytes], dict, bytes, bytes, bytes
]:
    """Build a small but complete plan projection without running a graph study."""

    adjacency: dict[str, tuple[str, ...]] = {}
    batches = []
    domains = []
    boundaries = []
    calibration_shells = []
    audit_shells = []
    role_budgets = {
        "S_256": 256,
        "S_512": 512,
        "S_1024": 1024,
        "R_top": 4096,
    }
    next_node = 1
    for split, batch_count in (
        ("calibration", 8),
        ("audit", audit.EXPECTED_AUDIT_BATCHES),
    ):
        for batch_index in range(batch_count):
            batch_id = f"{split}-{batch_index:02d}"
            anchors = [f"pt:{next_node + offset}" for offset in range(4)]
            next_node += 4
            for anchor in anchors:
                adjacency[anchor] = ()
            # Keep one real graph node outside every frozen domain so the
            # projection preserves a nontrivial protected-set boundary.
            omitted_protected = f"pt:{next_node}"
            next_node += 1
            adjacency[omitted_protected] = ()
            anchors_by_quartile = [
                {"node_id": anchor, "quartile_id": quartile}
                for anchor, quartile in zip(anchors, audit.EXPECTED_QUARTILES)
            ]
            batches.append(
                {
                    "anchors_by_quartile": anchors_by_quartile,
                    "batch_id": batch_id,
                    "batch_index": batch_index,
                    "protected_nodes": [*anchors, omitted_protected],
                    "split": split,
                }
            )
            node_rows = [
                {"hop_distance": 0, "node_id": anchor} for anchor in anchors
            ]
            for role in audit.ROLE_ORDER:
                domain = {
                    "batch_id": batch_id,
                    "nodes": copy.deepcopy(node_rows),
                    "requested_nodes": role_budgets[role],
                    "role": role,
                    "truncated_final_shell_nodes": 0,
                }
                domains.append(domain)
                boundaries.append(
                    plan._boundary_record(
                        batch_id,
                        role,
                        role_budgets[role],
                        anchors,
                        adjacency,
                    )
                )
            shell_target = (
                calibration_shells if split == "calibration" else audit_shells
            )
            for anchor in anchors:
                shell_target.append(
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

    bootstrap = [
        {
            "multiplicities": [1] * audit.EXPECTED_AUDIT_BATCHES,
            "replicate_index": replicate,
        }
        for replicate in range(audit.EXPECTED_BOOTSTRAP_REPLICATES)
    ]
    artifacts = {
        "audit_shells.jsonl": audit._jsonl_bytes(audit_shells),
        "batches.jsonl": audit._jsonl_bytes(batches),
        "bootstrap_multiplicities.jsonl": audit._jsonl_bytes(bootstrap),
        "boundaries.jsonl": audit._jsonl_bytes(boundaries),
        "calibration_shells.jsonl": audit._jsonl_bytes(calibration_shells),
        "domains.jsonl": audit._jsonl_bytes(domains),
    }
    attempt_manifest = audit._canonical_json({"schema": "synthetic-attempt-v1"})
    adjacency_data = audit._jsonl_bytes(
        [
            {"neighbors": list(adjacency[node]), "node_id": node}
            for node in sorted(adjacency, key=plan._typed_id_key)
        ]
    )
    eligibility_data = b""
    manifest = {
        "accepted": True,
        "audit_solve_authorized": False,
        "calibration_solve_authorized": True,
        "diffusion_or_fidelity_metrics_computed": False,
        "fingerprint_core": {
            "artifact_records": {
                name: audit._content_record(data) for name, data in artifacts.items()
            },
            "input_bindings": {
                "canonical_attempt_a_manifest": audit._content_record(
                    attempt_manifest
                ),
                "planning_graph_artifacts": {
                    "adjacency.jsonl": audit._content_record(adjacency_data),
                    "anchor_eligibility.jsonl": audit._content_record(
                        eligibility_data
                    ),
                },
            },
            "resource_contract": {"planner_input_ceiling_bytes": 10**8},
        },
        "plan_fingerprint": "1" * 64,
    }
    return (
        adjacency,
        artifacts,
        manifest,
        attempt_manifest,
        adjacency_data,
        eligibility_data,
    )


def _write_private(path, data: bytes) -> None:
    path.write_bytes(data)
    path.chmod(0o600)


def _capture_synthetic_plan_context(tmp_path, monkeypatch) -> dict:
    (
        adjacency,
        artifacts,
        manifest,
        attempt_manifest,
        adjacency_data,
        eligibility_data,
    ) = _synthetic_plan_material()
    plan_dir = tmp_path / "plan"
    attempt_dir = tmp_path / "attempt-a"
    plan_dir.mkdir(mode=0o700)
    attempt_dir.mkdir(mode=0o700)
    plan_dir.chmod(0o700)
    attempt_dir.chmod(0o700)
    _write_private(plan_dir / plan.MANIFEST_NAME, audit._canonical_json(manifest))
    for name, data in artifacts.items():
        _write_private(plan_dir / name, data)
    _write_private(attempt_dir / "manifest.json", attempt_manifest)
    _write_private(attempt_dir / "adjacency.jsonl", adjacency_data)
    _write_private(attempt_dir / "anchor_eligibility.jsonl", eligibility_data)

    monkeypatch.setattr(plan, "verify_plan", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        plan,
        "_load_graph",
        lambda *_args, **_kwargs: ({}, adjacency, ()),
    )
    context = audit.hop_lock._capture_verified_context(
        plan_dir,
        receipt_dir=tmp_path / "receipt",
        attempt_a_dir=attempt_dir,
        attempt_b_dir=tmp_path / "attempt-b",
        source_spec=tmp_path / "source-spec.json",
        relation_policy=tmp_path / "relation-policy.json",
    )
    return context


def _audit_context(plan_context: dict) -> dict:
    runtime = [
        {
            "observed_num_threads": 1,
            "user_api": "blas",
            "version": "synthetic",
        }
    ]
    return {
        "contract": {
            "effective_resistance": False,
            "minimum_rcond": 1e-12,
            "numeric": {},
            "peak_rss_ceiling_bytes": 10**9,
            "peak_rss_scope": "synthetic-audit-transaction",
            "per_batch_elapsed_ceiling_seconds": 10.0,
            "statistics": plan._statistical_contract("omitted"),
        },
        "lock_execution": {"actual_blas_identity": runtime},
        "lock_manifest": {
            "lock_fingerprint": "2" * 64,
            "scientific_core": {
                "actual_blas_identity": runtime,
                "alpha_top_hex": float(0.1).hex(),
            },
        },
        "lock_manifest_record": {"sha256": "3" * 64, "size_bytes": 1},
        "plan_context": plan_context,
        "selection": {
            "frozen_audit_roles": ["S_256", "S_512"],
            "k_high": "S_512",
            "k_low": "S_256",
            "lock_mode": "finite_contrast",
            "required_audit_model_roles": list(audit.ROLE_ORDER),
        },
    }


def test_verified_plan_capture_feeds_complete_audit_projection(
    monkeypatch, tmp_path
) -> None:
    plan_context = _capture_synthetic_plan_context(tmp_path, monkeypatch)

    assert set(plan_context["plan_artifacts"]) == {
        "audit_shells.jsonl",
        "batches.jsonl",
        "bootstrap_multiplicities.jsonl",
        "boundaries.jsonl",
        "calibration_shells.jsonl",
        "domains.jsonl",
    }
    for name in ("audit_shells.jsonl", "bootstrap_multiplicities.jsonl"):
        assert plan_context["captured_records"]["plan_artifacts"][name] == (
            plan_context["manifest"]["fingerprint_core"]["artifact_records"][name]
        )

    work_order, schedule = audit._derive_audit_work_order(
        _audit_context(plan_context)
    )
    assert work_order["batch_count"] == audit.EXPECTED_AUDIT_BATCHES
    assert work_order["audit_anchor_count"] == audit.EXPECTED_AUDIT_ANCHORS
    assert {batch["split"] for batch in work_order["batches"]} == {"audit"}
    assert len(schedule) == audit.EXPECTED_BOOTSTRAP_REPLICATES
    assert work_order["contains_calibration_metrics_or_responses"] is False


def test_audit_projection_rejects_calibration_anchor_overlap(
    monkeypatch, tmp_path
) -> None:
    plan_context = _capture_synthetic_plan_context(tmp_path, monkeypatch)
    context = _audit_context(plan_context)
    batches = audit._strict_jsonl_bytes(
        plan_context["plan_artifacts"]["batches.jsonl"], "batches"
    )
    calibration = next(row for row in batches if row["split"] == "calibration")
    first_audit = next(row for row in batches if row["split"] == "audit")
    calibration["anchors_by_quartile"][0]["node_id"] = first_audit[
        "anchors_by_quartile"
    ][0]["node_id"]
    plan_context["plan_artifacts"]["batches.jsonl"] = audit._jsonl_bytes(batches)

    with pytest.raises(audit.HopAuditError, match="anchor identities overlap"):
        audit._derive_audit_work_order(context)


def test_audit_capture_rejects_lock_without_audit_authorization(
    monkeypatch, tmp_path
) -> None:
    lock_manifest = {
        "accepted": True,
        "audit_solve_authorized": False,
        "audit_solves_executed": 0,
        "calibration_completed": True,
        "confirmatory_claim_authorized": False,
        "lock_mode": "finite_contrast",
    }
    monkeypatch.setattr(
        audit.hop_lock,
        "verify_lock",
        lambda *_args, **_kwargs: lock_manifest,
    )

    def unexpected_plan_capture(*_args, **_kwargs):
        pytest.fail("an unauthorized lock reached plan capture")

    monkeypatch.setattr(
        audit.hop_lock,
        "_capture_verified_context",
        unexpected_plan_capture,
    )
    with pytest.raises(
        audit.HopAuditError,
        match="does not authorize an audit solve",
    ):
        audit._capture_verified_context(
            lock_dir=tmp_path / "lock",
            plan_dir=tmp_path / "plan",
            receipt_dir=tmp_path / "receipt",
            attempt_a_dir=tmp_path / "attempt-a",
            attempt_b_dir=tmp_path / "attempt-b",
            source_spec=tmp_path / "sources.json",
            relation_policy=tmp_path / "policy.json",
        )


def test_audit_prepare_verify_replay_and_tamper_rejection(
    monkeypatch, tmp_path
) -> None:
    plan_context = _capture_synthetic_plan_context(tmp_path, monkeypatch)
    context = _audit_context(plan_context)
    runtime = copy.deepcopy(
        context["lock_manifest"]["scientific_core"]["actual_blas_identity"]
    )

    @contextmanager
    def one_thread_runtime():
        yield copy.deepcopy(runtime)

    def deterministic_safety_failure(*_args, **_kwargs):
        raise audit.np.linalg.LinAlgError("synthetic deterministic safety stop")

    monkeypatch.setattr(
        audit,
        "_capture_verified_context",
        lambda **_kwargs: copy.deepcopy(context),
    )
    monkeypatch.setattr(audit, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(audit, "_implementation_records", lambda: {})
    monkeypatch.setattr(audit.hop_lock, "_single_blas_thread", one_thread_runtime)
    monkeypatch.setattr(audit, "_peak_rss_bytes", lambda: 100)
    monkeypatch.setattr(audit.time, "perf_counter", lambda: 10.0)
    monkeypatch.setattr(
        audit,
        "evaluate_nested_bounded_domain_fidelity",
        deterministic_safety_failure,
    )
    monkeypatch.setattr(
        audit.hop_plan,
        "_assert_no_output_input_overlap",
        lambda *_args, **_kwargs: None,
    )

    common = {
        "attempt_a_dir": tmp_path / "attempt-a-input",
        "attempt_b_dir": tmp_path / "attempt-b-input",
        "lock_dir": tmp_path / "lock-input",
        "plan_dir": tmp_path / "plan-input",
        "receipt_dir": tmp_path / "receipt-input",
        "relation_policy": tmp_path / "policy.json",
        "source_spec": tmp_path / "sources.json",
    }
    first_dir = tmp_path / "audit-first"
    first = audit.prepare_audit(
        audit_dir=first_dir,
        local_root=tmp_path,
        **common,
    )
    assert first["decision"] == "safety_or_resource_blocked"
    assert first["confirmatory_claim_authorized"] is False
    assert audit.verify_audit(audit_dir=first_dir, **common) == first

    second_dir = tmp_path / "audit-second"
    second = audit.prepare_audit(
        audit_dir=second_dir,
        local_root=tmp_path,
        **common,
    )
    assert second["audit_fingerprint"] == first["audit_fingerprint"]
    for name in audit.SCIENTIFIC_ARTIFACT_NAMES:
        assert (second_dir / name).read_bytes() == (first_dir / name).read_bytes()

    monkeypatch.setattr(audit, "_git_commit", lambda: "b" * 40)
    with pytest.raises(audit.HopAuditError, match="implementation binding changed"):
        audit.verify_audit(audit_dir=first_dir, **common)
    monkeypatch.setattr(audit, "_git_commit", lambda: "a" * 40)

    decision_path = first_dir / "decision.json"
    decision_path.write_bytes(decision_path.read_bytes() + b"\n")
    with pytest.raises(audit.HopAuditError, match="artifact decision.json"):
        audit.verify_audit(audit_dir=first_dir, **common)


def _adequacy_summary() -> dict[str, float]:
    return {
        "boundary_harmonic_q90": 0.01,
        "maximum_h_absolute_error_q90": 0.001,
        "rank_inversion_fraction_q90": 0.01,
        "raw_relative_l2_error_q90": 0.005,
        "source_diagonal_relative_error_q90": 0.001,
        "top8_overlap_q10": 0.99,
    }


def _encoded_fidelity_result() -> dict:
    graph = {
        node: tuple(
            neighbor
            for neighbor in (node - 1, node + 1)
            if 0 <= neighbor < 8
        )
        for node in range(8)
    }
    candidate = audit.fidelity_module.select_hop_budget_domain(
        (0,), graph, maximum_nodes=5
    )
    reference = audit.fidelity_module.select_hop_budget_domain(
        (0,), graph, maximum_nodes=8
    )
    result = audit.fidelity_module.evaluate_bounded_domain_fidelity(
        candidate,
        reference,
        protected_nodes=tuple(range(5)),
        intrinsic_leakage_conductance=0.2,
    )
    return audit.hop_lock._hexify_scientific(
        audit.hop_lock._strip_timing_fields(result.as_dict())
    )


def _decision_intervals(
    *,
    efficacy_upper: float,
    primary_upper: float = 0.0,
    maximum_h_upper: float = 0.0,
) -> dict[str, dict[str, float]]:
    upper = {
        "efficacy_log_high_over_low": efficacy_upper,
        "ni_boundary_harmonic_absolute_harm": 0.0,
        "ni_maximum_h_absolute_error_harm": maximum_h_upper,
        "ni_primary_log_low_over_high": primary_upper,
        "ni_rank_inversion_absolute_harm": 0.0,
        "ni_source_diagonal_relative_error_harm": 0.0,
        "ni_top8_overlap_loss": 0.0,
    }
    return {
        name: {"lower_0.05": value, "point": value, "upper_0.95": value}
        for name, value in upper.items()
    }


def _patch_decision_inputs(monkeypatch, statuses, intervals) -> None:
    complete_ids = [row["batch_id"] for row in statuses if row["complete"]]
    node_counts = {"S_256": 10, "S_512": 20, "S_1024": 30}
    index = {
        (batch_id, role): {"candidate_nodes": node_counts[role]}
        for batch_id in complete_ids
        for role in _finite_work_order()["candidate_roles"]
    }
    monkeypatch.setattr(
        audit,
        "_validate_evidence_against_work_order",
        lambda *_args, **_kwargs: index,
    )
    monkeypatch.setattr(
        audit,
        "_absolute_summary",
        lambda *_args, **_kwargs: _adequacy_summary(),
    )
    monkeypatch.setattr(
        audit,
        "_absolute_checks",
        lambda *_args, **_kwargs: ({"frozen_absolute_gates": True}, True),
    )
    monkeypatch.setattr(
        audit,
        "_reference_checks",
        lambda *_args, **_kwargs: ({"frozen_reference_gates": True}, True),
    )
    monkeypatch.setattr(audit, "_batch_contrasts", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        audit,
        "_bootstrap_statistics",
        lambda *_args, **_kwargs: ([], intervals),
    )


def _derive_patched_decision(monkeypatch, statuses, intervals) -> dict:
    _patch_decision_inputs(monkeypatch, statuses, intervals)
    decision, bootstrap, returned_intervals = audit._derive_decision(
        _finite_work_order(),
        [],
        statuses,
        [],
        _contract(),
        peak_rss=1,
    )
    assert bootstrap == []
    if decision["decision"] not in {
        "descriptive_incomplete",
        "safety_or_resource_blocked",
    }:
        assert returned_intervals == intervals
    return decision


def test_extended_log_ratio_freezes_all_zero_cases() -> None:
    assert audit._extended_log_ratio(0.0, 0.0) == 0.0
    assert audit._extended_log_ratio(0.0, 2.0) == -math.inf
    assert audit._extended_log_ratio(2.0, 0.0) == math.inf
    assert audit._extended_log_ratio(2.0, 4.0) == pytest.approx(math.log(0.5))
    assert audit._decode_extended(
        audit._encode_extended(-math.inf), "negative infinity"
    ) == -math.inf
    assert audit._decode_extended(
        audit._encode_extended(math.inf), "positive infinity"
    ) == math.inf
    with pytest.raises(audit.HopAuditError):
        audit._extended_log_ratio(-1.0, 1.0)
    with pytest.raises(audit.HopAuditError):
        audit._extended_log_ratio(math.inf, 1.0)


def test_observed_order_statistics_do_not_interpolate() -> None:
    values = [0.0, 10.0]
    assert audit._observed_order(values, 0.90, upper=True) == 10.0
    assert audit._observed_order(values, 0.10, upper=False) == 0.0
    assert audit._observed_order([3.0, 1.0, 2.0], 0.50, upper=True) == 2.0
    with pytest.raises(audit.HopAuditError):
        audit._observed_order([], 0.90, upper=True)
    with pytest.raises(audit.HopAuditError):
        audit._observed_order([0.0, math.nan], 0.90, upper=True)


def test_full_fidelity_replay_rejects_resealed_out_of_range_metrics() -> None:
    contract = {
        "maximum_principle_tolerance": 1e-10,
        "minimum_rcond": 1e-12,
    }
    encoded = _encoded_fidelity_result()
    replayed = audit._decode_full_fidelity_result(encoded, contract, "test")
    assert replayed.candidate_nodes == 5

    negative = copy.deepcopy(encoded)
    negative["per_anchor_raw_relative_l2_error"][0] = float(-0.1).hex()
    with pytest.raises(audit.HopAuditError, match="invariant failed"):
        audit._decode_full_fidelity_result(negative, contract, "test")

    above_one = copy.deepcopy(encoded)
    above_one["per_anchor_top_k_overlap"][0] = float(1.1).hex()
    with pytest.raises(audit.HopAuditError, match="invariant failed"):
        audit._decode_full_fidelity_result(above_one, contract, "test")


def test_unattempted_tail_requires_exactly_one_preceding_safety_failure() -> None:
    valid = [
        {"failure_kind": None},
        {"failure_kind": "registered_coverage_failure"},
        {"failure_kind": "deterministic_safety_failure"},
        {"failure_kind": "not_attempted_after_global_block"},
        {"failure_kind": "not_attempted_after_global_block"},
    ]
    audit._validate_status_control_flow(valid)

    missing_cause = copy.deepcopy(valid)
    missing_cause[2]["failure_kind"] = "registered_coverage_failure"
    with pytest.raises(audit.HopAuditError, match="without a safety failure"):
        audit._validate_status_control_flow(missing_cause)

    broken_tail = copy.deepcopy(valid)
    broken_tail[3]["failure_kind"] = None
    with pytest.raises(audit.HopAuditError, match="invalid tail"):
        audit._validate_status_control_flow(broken_tail)

    repeated = copy.deepcopy(valid)
    repeated[1]["failure_kind"] = "deterministic_safety_failure"
    with pytest.raises(audit.HopAuditError, match="multiple global safety"):
        audit._validate_status_control_flow(repeated)


def test_masked_bootstrap_uses_frozen_replicate_index_and_one_retained_mass() -> None:
    statuses = _statuses(18)
    contrasts = [
        {
            "batch_id": f"audit-{index:02d}",
            "endpoints": {"a": float(index), "b": float(2 * index)},
        }
        for index in range(18)
    ]
    # Incomplete batches carry large frozen counts.  They must contribute
    # neither endpoint weight nor retained mass after applying the fixed mask.
    multiplicities = [0] * 24
    multiplicities[0] = 2
    multiplicities[1] = 1
    multiplicities[2] = 1
    multiplicities[3] = 1
    multiplicities[18:] = [4, 3, 3, 3, 3, 3]
    schedule = [{"replicate_index": 713, "multiplicities": multiplicities}]

    rows, intervals = audit._bootstrap_statistics(contrasts, statuses, schedule)

    assert rows == [
        {
            "endpoints": {
                "a": audit._encode_extended(6.0 / 5.0),
                "b": audit._encode_extended(12.0 / 5.0),
            },
            "replicate_index": 713,
            "retained_multiplicity": 5,
        }
    ]
    assert audit._decode_extended(rows[0]["endpoints"]["b"], "b") == pytest.approx(
        2.0 * audit._decode_extended(rows[0]["endpoints"]["a"], "a")
    )
    assert intervals["a"]["upper_0.95"] == pytest.approx(6.0 / 5.0)


def test_masked_bootstrap_fails_closed_on_zero_retained_mass() -> None:
    statuses = _statuses(18)
    contrasts = [
        {"batch_id": f"audit-{index:02d}", "endpoints": {"x": 0.0}}
        for index in range(18)
    ]
    multiplicities = [0] * 24
    multiplicities[23] = 24
    with pytest.raises(
        audit.HopAuditError,
        match="zero retained multiplicity",
    ):
        audit._bootstrap_statistics(
            contrasts,
            statuses,
            [{"replicate_index": 8, "multiplicities": multiplicities}],
        )


def test_absolute_and_reference_threshold_equality_passes() -> None:
    statistics = plan._statistical_contract("enabled")
    absolute_limits = statistics["absolute_adequacy"]
    summary = {
        "boundary_harmonic_q90": absolute_limits["boundary_harmonic_q90_max"],
        "effective_resistance_relative_error_q90": absolute_limits[
            "effective_resistance_relative_error_q90_max"
        ],
        "maximum_h_absolute_error_q90": absolute_limits[
            "maximum_h_absolute_error_q90_max"
        ],
        "rank_inversion_fraction_q90": absolute_limits[
            "rank_inversion_fraction_q90_max"
        ],
        "raw_relative_l2_error_q90": absolute_limits[
            "raw_relative_l2_error_q90_max"
        ],
        "source_diagonal_relative_error_q90": 1.0,
        "top8_overlap_q10": absolute_limits["top8_overlap_q10_min"],
    }
    checks, passed = audit._absolute_checks(summary, absolute_limits)
    assert passed is True
    assert all(checks.values())

    reference_limits = statistics["reference_adequacy"]
    reference_summary = {
        "maximum_h_absolute_error_q90": reference_limits[
            "maximum_h_absolute_error_q90_max"
        ],
        "raw_relative_l2_error_q90": reference_limits[
            "raw_relative_l2_error_q90_max"
        ],
        "top8_overlap_q10": reference_limits["top8_overlap_q10_min"],
    }
    reference_checks, reference_passed = audit._reference_checks(
        reference_summary, reference_limits
    )
    assert reference_passed is True
    assert all(reference_checks.values())


def test_strict_efficacy_threshold_equality_does_not_count_as_efficacy(
    monkeypatch,
) -> None:
    decision = _derive_patched_decision(
        monkeypatch,
        _statuses(18),
        _decision_intervals(efficacy_upper=math.log(0.9)),
    )
    assert decision["efficacy_passed"] is False
    assert decision["decision"] == "low_endpoint_converged"


@pytest.mark.parametrize(
    ("interval_overrides", "failed_endpoint"),
    [
        (
            {"primary_upper": math.log(1.10)},
            "ni_primary_log_low_over_high",
        ),
        (
            {"maximum_h_upper": 0.01},
            "ni_maximum_h_absolute_error_harm",
        ),
    ],
)
def test_noninferiority_iut_uses_strict_thresholds(
    monkeypatch,
    interval_overrides,
    failed_endpoint,
) -> None:
    decision = _derive_patched_decision(
        monkeypatch,
        _statuses(18),
        _decision_intervals(efficacy_upper=0.0, **interval_overrides),
    )
    assert decision["noninferiority_endpoint_checks"][failed_endpoint] is False
    assert decision["noninferiority_intersection_passed"] is False
    assert decision["decision"] == "inconclusive_frontier"
    assert decision["confirmatory_claim_authorized"] is False


def test_decision_authorizes_larger_domain_efficacy_only_when_iut_fails(
    monkeypatch,
) -> None:
    decision = _derive_patched_decision(
        monkeypatch,
        _statuses(18),
        _decision_intervals(efficacy_upper=-0.2, primary_upper=0.2),
    )
    assert decision["decision"] == "larger_endpoint_efficacious"
    assert decision["larger_domain_efficacy_claim_authorized"] is True
    assert decision["convergence_claim_authorized"] is False
    assert decision["confirmatory_claim_authorized"] is True


def test_decision_authorizes_low_convergence(monkeypatch) -> None:
    decision = _derive_patched_decision(
        monkeypatch,
        _statuses(18),
        _decision_intervals(efficacy_upper=0.0),
    )
    assert decision["decision"] == "low_endpoint_converged"
    assert decision["efficacy_passed"] is False
    assert decision["noninferiority_intersection_passed"] is True
    assert decision["realized_node_reduction_passed"] is True
    assert decision["convergence_claim_authorized"] is True
    assert decision["confirmatory_claim_authorized"] is True


def test_decision_marks_simultaneous_efficacy_and_iut_as_conflict(
    monkeypatch,
) -> None:
    decision = _derive_patched_decision(
        monkeypatch,
        _statuses(18),
        _decision_intervals(efficacy_upper=-0.2),
    )
    assert decision["decision"] == "inconclusive_frontier"
    assert decision["reason"] == (
        "larger_domain_efficacy_and_smaller_domain_convergence_rules_conflict"
    )
    assert decision["efficacy_passed"] is True
    assert decision["noninferiority_intersection_passed"] is True
    assert decision["confirmatory_claim_authorized"] is False
    assert decision["convergence_claim_authorized"] is False


def test_fewer_than_18_complete_batches_is_descriptive_only(monkeypatch) -> None:
    decision = _derive_patched_decision(
        monkeypatch,
        _statuses(17),
        _decision_intervals(efficacy_upper=-1.0),
    )
    assert decision["decision"] == "descriptive_incomplete"
    assert decision["complete_batch_count"] == 17
    assert decision["descriptive_results_authorized"] is True
    assert decision["confirmatory_claim_authorized"] is False


def test_deterministic_safety_failure_blocks_confirmatory_use(monkeypatch) -> None:
    decision = _derive_patched_decision(
        monkeypatch,
        _statuses(23, deterministic_failure=True),
        _decision_intervals(efficacy_upper=-1.0),
    )
    assert decision["decision"] == "safety_or_resource_blocked"
    assert decision["confirmatory_claim_authorized"] is False
    assert decision["convergence_claim_authorized"] is False


def test_metric_evidence_maps_quartiles_by_source_id_not_position() -> None:
    batch = {
        "batch_id": "audit-00",
        "anchors_by_quartile": [
            {"node_id": "pt:30", "quartile_id": "q1"},
            {"node_id": "pt:10", "quartile_id": "q2"},
            {"node_id": "pt:40", "quartile_id": "q3"},
            {"node_id": "pt:20", "quartile_id": "q4"},
        ],
    }
    result = SimpleNamespace(
        as_dict=lambda: {"source_nodes": ["pt:10", "pt:20", "pt:30", "pt:40"]},
        candidate_boundary_harmonic_max=0.01,
        candidate_nodes=10,
        effective_resistance_evaluated=False,
        per_anchor_candidate_cut_current_fraction=[0.1, 0.2, 0.3, 0.4],
        per_anchor_maximum_h_absolute_error=[1.0, 2.0, 3.0, 4.0],
        per_anchor_rank_inversion_fraction=[0.0, 0.0, 0.0, 0.0],
        per_anchor_raw_relative_l2_error=[10.0, 20.0, 30.0, 40.0],
        per_anchor_reference_cut_current_fraction=[0.01, 0.02, 0.03, 0.04],
        per_anchor_source_diagonal_relative_error=[0.0, 0.0, 0.0, 0.0],
        per_anchor_top_k_overlap=[1.0, 1.0, 1.0, 1.0],
        protected_candidate_fraction=1.0,
        protected_reference_fraction=1.0,
        reference_nodes=20,
        source_nodes=("pt:10", "pt:20", "pt:30", "pt:40"),
    )
    safety = SimpleNamespace(as_dict=lambda: {"checks_passed": True})

    evidence = audit._metric_evidence(
        batch,
        "S_256",
        result,
        safety,
        safety,
        [
            {"anchor": "pt:20"},
            {"anchor": "pt:40"},
            {"anchor": "pt:10"},
            {"anchor": "pt:30"},
        ],
    )

    assert [row["quartile_id"] for row in evidence["anchors"]] == [
        "q1",
        "q2",
        "q3",
        "q4",
    ]
    assert [
        audit._decode_hex(row["metrics"]["raw_relative_l2_error"], "raw")
        for row in evidence["anchors"]
    ] == [30.0, 10.0, 40.0, 20.0]
    assert [row["anchor"] for row in evidence["screening"]] == [
        "pt:30",
        "pt:10",
        "pt:40",
        "pt:20",
    ]
