#!/usr/bin/env python3
"""Contract tests for the outcome-blind Pearltrees no-solve HOP plan."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import stat

import pytest

import declare_pearltrees_diffusion_sources as declaration
import prepare_pearltrees_diffusion_consensus as consensus
import prepare_pearltrees_hop_plan as plan


SNAPSHOT_RESOURCE_CEILING = 200_000_000
PLANNER_INPUT_CEILING = 200_000_000
PLANNER_TOUCH_CEILING = 200_000_000
STUDY_RSS_CEILING = 200_000_000
PHYSICAL_RELATIONS = {
    "alias": True,
    "collection": True,
    "cross_link": False,
    "path": True,
    "ref": True,
    "shortcut": True,
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _chain_rdf(node_count: int, *, private_isolated: bool = False) -> bytes:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:pt="https://www.pearltrees.com/rdf/0.1/#">',
    ]
    final_node = node_count + (1 if private_isolated else 0)
    for node in range(1, final_node + 1):
        privacy = 1 if private_isolated and node == final_node else 0
        lines.extend(
            [
                f'  <pt:Tree rdf:about="https://www.pearltrees.com/synthetic/id{node}">',
                f"    <pt:treeId>{node}</pt:treeId>",
                f"    <pt:title>Synthetic title {node}</pt:title>",
                f"    <pt:privacy>{privacy}</pt:privacy>",
                "  </pt:Tree>",
            ]
        )
    for child in range(2, node_count + 1):
        parent = child - 1
        lines.extend(
            [
                "  <pt:RefPearl>",
                "    <pt:parentTree "
                f'rdf:resource="https://www.pearltrees.com/synthetic/id{parent}" />',
                "    <pt:seeAlso "
                f'rdf:resource="https://www.pearltrees.com/synthetic/id{child}" />',
                f"    <pt:title>Synthetic title {child}</pt:title>",
                "    <pt:privacy>0</pt:privacy>",
                "  </pt:RefPearl>",
            ]
        )
    lines.append("</rdf:RDF>")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _make_case(
    root: Path,
    *,
    node_count: int,
    minimum_anchors: int,
    private_isolated: bool = False,
) -> dict[str, Path]:
    inputs = root / "inputs"
    local = root / "local"
    inputs.mkdir(parents=True)
    local.mkdir()
    rdf = inputs / "private-structural-source.rdf"
    rdf.write_bytes(_chain_rdf(node_count, private_isolated=private_isolated))
    policy = inputs / "policy.json"
    _write_json(
        policy,
        {
            "physical_edges": PHYSICAL_RELATIONS,
            "schema": "pearltrees-diffusion-relation-policy-v1",
        },
    )
    declaration_dir = local / "declaration"
    spec = declaration.build_source_spec(
        snapshot_label="private-synthetic-hop-plan",
        rdf=(("private-source-id", "synthetic", str(rdf)),),
    )
    declaration.install_source_spec(
        spec, output_dir=declaration_dir, local_root=local
    )
    case = {
        "attempt_a": local / "attempt-a",
        "attempt_b": local / "attempt-b",
        "local": local,
        "policy": policy,
        "receipt": local / "consensus",
        "source_spec": declaration_dir / declaration.SPEC_FILENAME,
    }
    receipt, exit_code = consensus.prepare_consensus(
        case["source_spec"],
        case["policy"],
        case["attempt_a"],
        case["attempt_b"],
        case["receipt"],
        case["local"],
        minimum_anchors=minimum_anchors,
        resource_ceiling_bytes=SNAPSHOT_RESOURCE_CEILING,
    )
    assert exit_code == 0
    assert receipt["accepted"] is True
    return case


def _prepare_plan(case: dict[str, Path], name: str, **overrides):
    values = {
        "planner_input_ceiling_bytes": PLANNER_INPUT_CEILING,
        "planner_edge_touch_ceiling": PLANNER_TOUCH_CEILING,
        "study_peak_rss_ceiling_bytes": STUDY_RSS_CEILING,
        "effective_resistance_arm": "enabled",
    }
    values.update(overrides)
    output = case["local"] / name
    manifest, exit_code = plan.prepare_plan(
        receipt_dir=case["receipt"],
        attempt_a_dir=case["attempt_a"],
        attempt_b_dir=case["attempt_b"],
        source_spec=case["source_spec"],
        relation_policy=case["policy"],
        plan_dir=output,
        local_root=case["local"],
        **values,
    )
    return output, manifest, exit_code


def _verify(case: dict[str, Path], output: Path):
    return plan.verify_plan(
        output,
        receipt_dir=case["receipt"],
        attempt_a_dir=case["attempt_a"],
        attempt_b_dir=case["attempt_b"],
        source_spec=case["source_spec"],
        relation_policy=case["policy"],
    )


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def ready_case(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return _make_case(
        tmp_path_factory.mktemp("pearltrees-hop-ready"),
        node_count=1100,
        minimum_anchors=128,
        private_isolated=True,
    )


def test_numeric_typed_id_order_and_hash_preimage_are_frozen() -> None:
    assert plan._typed_id_key("pt:2") < plan._typed_id_key("pt:10")
    assert plan._typed_id_key("pt:10") < plan._typed_id_key("zz:1")
    expected = plan.hashlib.sha256(
        b'[3882001,"select","pt:2"]\n'
    ).hexdigest()
    assert plan._selection_key("select", "pt:2")[0] == expected
    with pytest.raises(plan.HopPlanError):
        plan._typed_id_key("pt:02")


def test_quartile_remainder_allocation_is_frozen() -> None:
    assert plan._quartile_slices(130) == (
        (0, 33),
        (33, 66),
        (66, 98),
        (98, 130),
    )


def test_cycle_convergence_is_not_a_cut() -> None:
    adjacency = {
        "pt:1": ("pt:2", "pt:3"),
        "pt:2": ("pt:1", "pt:4"),
        "pt:3": ("pt:1", "pt:4"),
        "pt:4": ("pt:2", "pt:3", "pt:5"),
        "pt:5": ("pt:4",),
    }
    order, distances, truncated = plan._hop_order(("pt:1",), adjacency)
    assert order == ("pt:1", "pt:2", "pt:3", "pt:4", "pt:5")
    assert distances["pt:4"] == 2
    assert truncated == 0
    boundary = plan._boundary_record(
        "audit-001", "S_4", 4, order[:4], adjacency
    )
    assert boundary["cut_edges"] == [["pt:4", "pt:5"]]
    assert boundary["beta"] == [{"cut_conductance": 1, "node_id": "pt:4"}]


def test_reference_truncation_counts_the_omitted_final_shell() -> None:
    leaf_count = 4100
    center = "pt:1"
    leaves = tuple(f"pt:{index}" for index in range(2, leaf_count + 2))
    adjacency = {center: leaves}
    adjacency.update({leaf: (center,) for leaf in leaves})
    anchors = leaves[:4]
    batch = {
        "anchors_by_quartile": [
            {"node_id": anchor, "quartile_id": f"q{index + 1}"}
            for index, anchor in enumerate(anchors)
        ],
        "batch_id": "audit-001",
        "split": "audit",
    }
    internal = {
        anchor: {
            "pair_distances": {other: (0 if other == anchor else 2) for other in anchors},
            "protected": (anchor,),
            "shell": (),
        }
        for anchor in anchors
    }
    _batches, domains, _boundaries, _shells, _maximum, blocks = plan._freeze_domains(
        (batch,), internal, adjacency, plan._TouchBudget(100_000)
    )
    assert blocks == []
    reference = next(row for row in domains if row["role"] == "R_top")
    assert reference["realized_nodes"] == 4096
    assert reference["truncated_final_shell_nodes"] == 5


def test_planner_source_has_a_static_no_solve_import_and_call_boundary() -> None:
    source = Path(plan.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_modules = {"numpy", "scipy", "torch", "unifyweaver.graph"}
    forbidden_calls = {"solve", "cholesky", "eig", "eigh", "eigvalsh", "qr", "inv"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                not any(alias.name == name or alias.name.startswith(name + ".") for name in forbidden_modules)
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not any(
                node.module == name or node.module.startswith(name + ".")
                for name in forbidden_modules
            )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_calls


def test_private_bundle_fifo_fails_without_a_blocking_open(tmp_path: Path) -> None:
    bundle = tmp_path / "private-bundle"
    bundle.mkdir(mode=0o700)
    fifo = bundle / "corrupt-leaf"
    os.mkfifo(fifo, mode=0o600)
    binding = plan._bind_directory(bundle, "synthetic private bundle")
    try:
        with pytest.raises(plan.HopPlanError, match="artifact contract mismatch"):
            plan._assert_private_directory_files(
                binding, "synthetic private bundle"
            )
    finally:
        binding.close()


def test_private_bundle_inventory_scan_is_bounded(tmp_path: Path) -> None:
    bundle = tmp_path / "oversized-private-bundle"
    bundle.mkdir(mode=0o700)
    for index in range(4):
        leaf = bundle / f"leaf-{index}"
        leaf.write_bytes(b"")
        os.chmod(leaf, 0o600)
    binding = plan._bind_directory(bundle, "synthetic private bundle")
    try:
        with pytest.raises(plan.HopPlanError, match="inventory exceeds its bound"):
            plan._bounded_directory_names(
                binding, 3, "synthetic private bundle"
            )
    finally:
        binding.close()


def test_ready_plan_is_balanced_nested_local_only_and_byte_deterministic(
    ready_case: dict[str, Path],
) -> None:
    first, manifest, exit_code = _prepare_plan(ready_case, "plan-one")
    second, second_manifest, second_exit = _prepare_plan(ready_case, "plan-two")

    assert exit_code == second_exit == 0
    assert manifest == second_manifest
    assert manifest["accepted"] is True
    assert manifest["reason"] == "hop_plan_frozen"
    assert manifest["solves_executed"] == 0
    assert manifest["structural_metrics_computed"] is True
    assert manifest["diffusion_or_fidelity_metrics_computed"] is False
    assert manifest["calibration_solve_authorized"] is True
    assert manifest["audit_solve_authorized"] is False
    assert manifest["aggregate"] == {
        "audit_batches": 24,
        "calibration_batches": 8,
        "eligible_anchor_count": 1100,
        "publishable": False,
        "selected_anchor_count": 128,
    }
    assert _verify(ready_case, first) == manifest
    assert {item.name for item in first.iterdir()} == plan.ALL_PLAN_FILES
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(item.stat().st_mode) == 0o600 for item in first.iterdir())
    assert all(
        (first / name).read_bytes() == (second / name).read_bytes()
        for name in plan.ALL_PLAN_FILES
    )

    quartiles = _jsonl(first / "quartiles.jsonl")
    assert [row["member_count"] for row in quartiles] == [275, 275, 275, 275]
    anchors = _jsonl(first / "selected_anchors.jsonl")
    assert len(anchors) == len({row["node_id"] for row in anchors}) == 128
    assert sum(row["split"] == "calibration" for row in anchors) == 32
    assert sum(row["split"] == "audit" for row in anchors) == 96
    batches = _jsonl(first / "batches.jsonl")
    assert len(batches) == 32
    assert all(
        [row["quartile_id"] for row in batch["anchors_by_quartile"]]
        == ["q1", "q2", "q3", "q4"]
        for batch in batches
    )
    domains = _jsonl(first / "domains.jsonl")
    by_batch = {}
    for row in domains:
        by_batch.setdefault(row["batch_id"], {})[row["role"]] = row
    assert len(by_batch) == 32
    for roles in by_batch.values():
        s256 = [row["node_id"] for row in roles["S_256"]["nodes"]]
        s512 = [row["node_id"] for row in roles["S_512"]["nodes"]]
        s1024 = [row["node_id"] for row in roles["S_1024"]["nodes"]]
        reference = [row["node_id"] for row in roles["R_top"]["nodes"]]
        assert s256 == s512[:256]
        assert s512 == s1024[:512]
        assert s1024 == reference[:1024]
        assert [len(s256), len(s512), len(s1024), len(reference)] == [
            256,
            512,
            1024,
            1100,
        ]
    boundaries = _jsonl(first / "boundaries.jsonl")
    assert all(row["closure_policy"] == "exact_dirichlet_no_closure" for row in boundaries)
    assert all(row["cut_mass"] == row["cut_edge_count"] for row in boundaries)
    shells = _jsonl(first / "calibration_shells.jsonl")
    assert len(shells) == 32
    assert all(row["strictly_interior_pass"] is True for row in shells)
    bootstrap = _jsonl(first / "bootstrap_multiplicities.jsonl")
    assert len(bootstrap) == 9999
    assert bootstrap[0] == {
        "multiplicities": [
            4,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            4,
            0,
            2,
            1,
            1,
            1,
            0,
            1,
        ],
        "replicate_index": 0,
    }
    assert all(sum(row["multiplicities"]) == 24 for row in bootstrap)
    statistical = manifest["fingerprint_core"]["statistical_contract"]
    assert statistical["minimum_complete_audit_batches"] == 18
    assert statistical["reference_adequacy"] == {
        "maximum_h_absolute_error_q90_max": 0.005,
        "raw_relative_l2_error_q90_max": 0.01,
        "top8_overlap_q10_min": 0.98,
    }

    eligibility = _jsonl(ready_case["attempt_a"] / "anchor_eligibility.jsonl")
    private = next(row for row in eligibility if row["node_id"] == "pt:1101")
    assert private == {
        "eligible": False,
        "node_id": "pt:1101",
        "reason": "direct_private",
    }
    adjacency_ids = {
        row["node_id"] for row in _jsonl(ready_case["attempt_a"] / "adjacency.jsonl")
    }
    assert "pt:1101" not in adjacency_ids

    manifest_text = (first / "manifest.json").read_text(encoding="utf-8")
    assert "Synthetic title" not in manifest_text
    assert "private-source-id" not in manifest_text
    assert str(ready_case["local"].parent) not in manifest_text
    assert '"K_low"' not in manifest_text and '"K_high"' not in manifest_text


def test_small_valid_snapshot_installs_an_immutable_coverage_block(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path / "small", node_count=20, minimum_anchors=1)
    output, manifest, exit_code = _prepare_plan(case, "blocked-plan")

    assert exit_code == 2
    assert manifest["accepted"] is False
    assert manifest["reason"] == "quartile_coverage_inadequate"
    assert manifest["block_reasons"] == ["quartile_coverage_inadequate"]
    assert manifest["calibration_solve_authorized"] is False
    assert manifest["audit_solve_authorized"] is False
    assert _verify(case, output) == manifest
    assert (output / "batches.jsonl").read_bytes() == b""


def test_tiny_study_ceiling_records_resource_block(
    ready_case: dict[str, Path],
) -> None:
    output, manifest, exit_code = _prepare_plan(
        ready_case,
        "resource-block",
        study_peak_rss_ceiling_bytes=1,
    )

    assert exit_code == 2
    assert manifest["accepted"] is False
    assert "study_resource_inadequate" in manifest["block_reasons"]
    assert _verify(ready_case, output) == manifest


def test_plan_verifier_rejects_changed_node_artifact(
    ready_case: dict[str, Path], tmp_path: Path
) -> None:
    source = ready_case["local"] / "plan-one"
    changed = tmp_path / "changed-plan"
    shutil.copytree(source, changed)
    path = changed / "batches.jsonl"
    path.write_bytes(path.read_bytes() + b'{}\n')
    os.chmod(path, 0o600)

    with pytest.raises(plan.HopPlanError):
        _verify(ready_case, changed)


def test_full_chain_rejects_changed_attempt_a_leaf_before_plan_install(
    ready_case: dict[str, Path], tmp_path: Path
) -> None:
    changed_attempt = tmp_path / "attempt-a"
    shutil.copytree(ready_case["attempt_a"], changed_attempt)
    adjacency = changed_attempt / "adjacency.jsonl"
    adjacency.write_bytes(adjacency.read_bytes() + b'{}\n')
    os.chmod(adjacency, 0o600)

    with pytest.raises(plan.HopPlanError):
        plan.prepare_plan(
            receipt_dir=ready_case["receipt"],
            attempt_a_dir=changed_attempt,
            attempt_b_dir=ready_case["attempt_b"],
            source_spec=ready_case["source_spec"],
            relation_policy=ready_case["policy"],
            plan_dir=ready_case["local"] / "must-not-install",
            local_root=ready_case["local"],
            planner_input_ceiling_bytes=PLANNER_INPUT_CEILING,
            planner_edge_touch_ceiling=PLANNER_TOUCH_CEILING,
            study_peak_rss_ceiling_bytes=STUDY_RSS_CEILING,
            effective_resistance_arm="enabled",
        )
    assert not (ready_case["local"] / "must-not-install").exists()


def test_plan_target_cannot_nest_inside_the_source_declaration(
    ready_case: dict[str, Path],
) -> None:
    target = ready_case["source_spec"].parent / "must-not-install"
    with pytest.raises(plan.HopPlanError):
        plan.prepare_plan(
            receipt_dir=ready_case["receipt"],
            attempt_a_dir=ready_case["attempt_a"],
            attempt_b_dir=ready_case["attempt_b"],
            source_spec=ready_case["source_spec"],
            relation_policy=ready_case["policy"],
            plan_dir=target,
            local_root=ready_case["local"],
            planner_input_ceiling_bytes=PLANNER_INPUT_CEILING,
            planner_edge_touch_ceiling=PLANNER_TOUCH_CEILING,
            study_peak_rss_ceiling_bytes=STUDY_RSS_CEILING,
            effective_resistance_arm="enabled",
        )
    assert not target.exists()
    declaration.verify_installed_source_spec(ready_case["source_spec"])


def test_source_declaration_hardlink_fails_before_plan_install(
    ready_case: dict[str, Path],
) -> None:
    marker = ready_case["source_spec"].parent / declaration.LOCAL_ONLY_MARKER
    extra_link = ready_case["local"] / "temporary-declaration-marker-hardlink"
    target = ready_case["local"] / "hardlink-must-not-install"
    os.link(marker, extra_link)
    try:
        with pytest.raises(plan.HopPlanError):
            plan.prepare_plan(
                receipt_dir=ready_case["receipt"],
                attempt_a_dir=ready_case["attempt_a"],
                attempt_b_dir=ready_case["attempt_b"],
                source_spec=ready_case["source_spec"],
                relation_policy=ready_case["policy"],
                plan_dir=target,
                local_root=ready_case["local"],
                planner_input_ceiling_bytes=PLANNER_INPUT_CEILING,
                planner_edge_touch_ceiling=PLANNER_TOUCH_CEILING,
                study_peak_rss_ceiling_bytes=STUDY_RSS_CEILING,
                effective_resistance_arm="enabled",
            )
        assert not target.exists()
    finally:
        extra_link.unlink()


def test_unused_attempt_leaf_hardlink_is_rejected(
    ready_case: dict[str, Path],
) -> None:
    source = ready_case["attempt_a"] / "nodes.jsonl"
    extra_link = ready_case["local"] / "temporary-unused-attempt-hardlink"
    os.link(source, extra_link)
    try:
        with pytest.raises(plan.HopPlanError, match="artifact contract mismatch"):
            _verify(ready_case, ready_case["local"] / "plan-one")
    finally:
        extra_link.unlink()


def test_plan_target_cannot_nest_inside_authoritative_api_directory(
    ready_case: dict[str, Path],
) -> None:
    api_dir = ready_case["local"] / "authoritative-api-json"
    api_dir.mkdir()
    _write_json(api_dir / "page.json", {"synthetic": True})
    alternate_dir = ready_case["local"] / "api-source-declaration"
    alternate = declaration.build_source_spec(
        snapshot_label="private-api-overlap-regression",
        api_json_dir=(("api-source", str(api_dir)),),
    )
    declaration.install_source_spec(
        alternate, output_dir=alternate_dir, local_root=ready_case["local"]
    )
    target = api_dir / "must-not-install"
    with pytest.raises(plan.HopPlanError, match="overlaps a verified input"):
        plan.prepare_plan(
            receipt_dir=ready_case["receipt"],
            attempt_a_dir=ready_case["attempt_a"],
            attempt_b_dir=ready_case["attempt_b"],
            source_spec=alternate_dir / declaration.SPEC_FILENAME,
            relation_policy=ready_case["policy"],
            plan_dir=target,
            local_root=ready_case["local"],
            planner_input_ceiling_bytes=PLANNER_INPUT_CEILING,
            planner_edge_touch_ceiling=PLANNER_TOUCH_CEILING,
            study_peak_rss_ceiling_bytes=STUDY_RSS_CEILING,
            effective_resistance_arm="enabled",
        )
    assert not target.exists()


def test_cli_failure_is_generic_and_does_not_echo_private_paths(
    ready_case: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    private_missing = ready_case["local"] / "private-missing-plan"
    exit_code = plan.main(
        [
            "verify",
            "--receipt-dir",
            str(ready_case["receipt"]),
            "--attempt-a-dir",
            str(ready_case["attempt_a"]),
            "--attempt-b-dir",
            str(ready_case["attempt_b"]),
            "--source-spec",
            str(ready_case["source_spec"]),
            "--relation-policy",
            str(ready_case["policy"]),
            "--plan-dir",
            str(private_missing),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert json.loads(captured.err) == {"error": "HOP plan failed closed"}
    assert str(private_missing) not in captured.err
