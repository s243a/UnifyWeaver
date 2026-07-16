#!/usr/bin/env python3
"""Tests for the portable topology-only source-dependence runner."""

import json

import pytest

import run_repeated_judge_source_dependence as runner


def graph(label):
    node = f"node-{label}"
    return {"parents": {node: set()}, "children": {node: set()}}


def record(character):
    return {"size_bytes": 1, "sha256": character * 64}


def graph_inputs():
    return {
        "exploratory": {
            "corpus_identity": "fixture-a",
            "artifacts": {"graph": record("a")},
        },
        "fresh": {
            "corpus_identity": "fixture-b",
            "artifacts": {"graph": record("b")},
        },
    }


def fake_audit(passing):
    def audit(parents, _children, region_count, registered_sizes, *, rho_grid):
        corpus = "exploratory" if "node-exploratory" in parents else "fresh"
        passed = bool(passing.get((corpus, region_count), False))
        return {
            "target_region_count": region_count,
            "rho_grid": list(rho_grid),
            "registered_size_results": {
                str(size): {
                    "components_per_corpus": size,
                    "gates": {"all_topology_gates_pass": passed},
                }
                for size in registered_sizes
            },
            "gates": {"all_registered_sizes_pass": passed},
        }

    return audit


def build(monkeypatch, passing, **kwargs):
    monkeypatch.setattr(runner, "audit_source_dependence", fake_audit(passing))
    return runner.build_source_dependence_payload(
        {
            "exploratory": graph("exploratory"),
            "fresh": graph("fresh"),
        },
        graph_inputs(),
        implementation={"fixture.py": record("c")},
        **kwargs,
    )


def test_same_region_count_must_pass_both_corpora_and_no_count_is_selected(monkeypatch):
    payload = build(
        monkeypatch,
        {
            ("exploratory", 64): True,
            ("fresh", 96): True,
        },
        region_count_grid=(64, 96),
        registered_sizes=(160,),
    )
    assert payload["decision"]["structural_bridge_defined"] is False
    assert payload["decision"]["jointly_passing_region_counts"] == []
    assert payload["decision"]["region_count_selected"] is None
    assert payload["joint_region_count_grid"]["64"]["passing_corpora"] == [
        "exploratory"
    ]
    assert payload["joint_region_count_grid"]["96"]["passing_corpora"] == [
        "fresh"
    ]


def test_joint_structural_pass_unlocks_nothing(monkeypatch):
    payload = build(
        monkeypatch,
        {
            ("exploratory", 64): True,
            ("fresh", 64): True,
            ("exploratory", 96): True,
            ("fresh", 96): True,
        },
        region_count_grid=(64, 96),
        registered_sizes=(160,),
    )
    assert payload["decision"]["structural_bridge_defined"] is True
    assert payload["decision"]["jointly_passing_region_counts"] == [64, 96]
    assert payload["configuration"]["region_count_selection_performed"] is False
    assert payload["configuration"]["all_jointly_passing_region_counts_continue"] is True
    assert payload["decision"]["downstream_pipeline_must_stop"] is True
    assert all(value is False for value in payload["authorization"].values())
    assert payload["authorization"]["historical_inventory_unlocked"] is False


def test_default_scientific_grid_is_passed_to_exactly_both_corpora(monkeypatch):
    calls = []

    def audit(parents, _children, region_count, registered_sizes, *, rho_grid):
        corpus = "exploratory" if "node-exploratory" in parents else "fresh"
        calls.append((corpus, region_count, tuple(registered_sizes), tuple(rho_grid)))
        return {"gates": {"all_registered_sizes_pass": True}}

    monkeypatch.setattr(runner, "audit_source_dependence", audit)
    payload = runner.build_source_dependence_payload(
        {
            "exploratory": graph("exploratory"),
            "fresh": graph("fresh"),
        },
        graph_inputs(),
        implementation={"fixture.py": record("c")},
    )
    assert set(calls) == {
        (corpus, region_count, (160, 320, 512, 800), runner.FROZEN_RHO_GRID)
        for corpus in ("exploratory", "fresh")
        for region_count in (64, 96, 128)
    }
    assert payload["configuration"]["region_count_grid"] == [64, 96, 128]
    assert payload["configuration"]["registered_components_per_corpus"] == [
        160, 320, 512, 800
    ]
    assert payload["configuration"]["rho_grid"] == [0.0, 0.025, 0.05, 0.1, 0.2]


def test_payload_is_deterministic_portable_and_atomic(monkeypatch, tmp_path):
    passing = {("exploratory", 8): True, ("fresh", 8): True}
    first = build(
        monkeypatch,
        passing,
        region_count_grid=(8,),
        registered_sizes=(20,),
        rho_grid=(0.0, 0.2),
    )
    second = build(
        monkeypatch,
        passing,
        region_count_grid=(8,),
        registered_sizes=(20,),
        rho_grid=(0.0, 0.2),
    )
    assert first == second
    data = runner._json_bytes(first)
    assert str(tmp_path).encode() not in data
    output = tmp_path / "nested" / "audit.json"
    assert runner._atomic_write(output, first) == runner._atomic_write(output, second)
    assert output.read_bytes() == data
    assert not output.with_name("audit.json.tmp").exists()
    loaded = json.loads(data)
    assert loaded["inputs"]["outcomes_consumed"] is False
    assert loaded["inputs"]["historical_inventory_consumed"] is False
    assert loaded["inputs"]["nomic_cache_consumed"] is False
    assert loaded["inputs"]["candidate_pool_consumed"] is False
    assert loaded["inputs"]["judge_responses_consumed"] is False


def test_summary_projection_hashes_bulky_exact_arrays(monkeypatch):
    passing = {("exploratory", 8): True, ("fresh", 8): True}
    payload = build(
        monkeypatch,
        passing,
        region_count_grid=(8,),
        registered_sizes=(20,),
        rho_grid=(0.0, 0.2),
    )
    for corpus in ("exploratory", "fresh"):
        audit = payload["results"][corpus]["8"]
        audit["exposure"] = {"matrix": [[1.0, 0.2], [0.2, 1.0]]}
        row = audit["registered_size_results"]["20"]
        row["capacities_by_region"] = {"r0": 10, "r1": 10}
        row["allocation"] = {
            "counts_by_region": {"r0": 10, "r1": 10},
            "assignment_region_ids": ["r0"] * 10 + ["r1"] * 10,
            "used_region_count": 2,
        }
    projected = runner._summary_projection(payload)
    assert projected["artifact_projection"] == "tracked-summary-v1"
    assert projected["full_payload_record"] == runner.content_record(
        runner._json_bytes(payload)
    )
    for corpus in ("exploratory", "fresh"):
        audit = projected["results"][corpus]["8"]
        assert "matrix" not in audit["exposure"]
        assert audit["exposure"]["matrix_shape"] == [2, 2]
        row = audit["registered_size_results"]["20"]
        assert "capacities_by_region" not in row
        assert "counts_by_region" not in row["allocation"]
        assert "assignment_region_ids" not in row["allocation"]
        assert all(
            record["size_bytes"] > 0 and len(record["sha256"]) == 64
            for record in (
                audit["exposure"]["matrix_record"],
                row["capacities_by_region_record"],
                row["allocation"]["counts_by_region_record"],
                row["allocation"]["assignment_region_ids_record"],
            )
        )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"region_count_grid": ()}, "positive"),
        ({"region_count_grid": (8, 8)}, "unique"),
        ({"region_count_grid": (16, 8)}, "increasing"),
        ({"registered_sizes": (True,)}, "positive"),
        ({"registered_sizes": (20, 10)}, "increasing"),
        ({"rho_grid": ()}, "nonempty"),
        ({"rho_grid": (0.0, 0.0)}, "unique"),
        ({"rho_grid": (0.0, 0.2, 0.1)}, "increasing"),
        ({"rho_grid": (0.1, 0.2)}, "start at zero"),
        ({"rho_grid": (0.0, True)}, "finite numbers"),
        ({"rho_grid": (0.0, 1.0)}, r"\[0,1\)"),
        ({"rho_grid": (0.0, float("nan"))}, "finite numbers"),
    ],
)
def test_invalid_grids_fail_closed(monkeypatch, kwargs, message):
    with pytest.raises(ValueError, match=message):
        build(monkeypatch, {}, **kwargs)


@pytest.mark.parametrize(
    "graphs,inputs,message",
    [
        ({"exploratory": graph("exploratory")}, graph_inputs(), "exactly"),
        (
            {"exploratory": graph("exploratory"), "fresh": graph("fresh")},
            {"exploratory": graph_inputs()["exploratory"]},
            "align exactly",
        ),
    ],
)
def test_both_corpora_are_required(monkeypatch, graphs, inputs, message):
    monkeypatch.setattr(runner, "audit_source_dependence", fake_audit({}))
    with pytest.raises(ValueError, match=message):
        runner.build_source_dependence_payload(
            graphs,
            inputs,
            implementation={"fixture.py": record("c")},
        )


def test_malformed_module_gate_fails_closed(monkeypatch):
    monkeypatch.setattr(
        runner,
        "audit_source_dependence",
        lambda *_args, **_kwargs: {"gates": {"all_registered_sizes_pass": 1}},
    )
    with pytest.raises(RuntimeError, match="must be boolean"):
        runner.build_source_dependence_payload(
            {
                "exploratory": graph("exploratory"),
                "fresh": graph("fresh"),
            },
            graph_inputs(),
            region_count_grid=(8,),
            registered_sizes=(20,),
            implementation={"fixture.py": record("c")},
        )


def test_default_and_audit_only_exit_behavior(monkeypatch, tmp_path):
    implementation = {"source.py": record("a")}
    payload = {
        "decision": {
            "structural_bridge_defined": True,
            "jointly_passing_region_counts": [64],
        }
    }
    monkeypatch.setattr(runner, "_implementation_records", lambda: implementation)
    monkeypatch.setattr(
        runner,
        "load_frozen_graphs",
        lambda *_args, **_kwargs: ({}, {}, {}),
    )
    monkeypatch.setattr(
        runner,
        "build_source_dependence_payload",
        lambda *_args, **_kwargs: payload,
    )
    blocked = tmp_path / "blocked.json"
    assert runner.main(["--artifact-repo", "unused", "--out", str(blocked)]) == 2
    report = tmp_path / "report.json"
    assert runner.main(
        ["--artifact-repo", "unused", "--out", str(report), "--audit-only"]
    ) == 0
    assert blocked.read_bytes() == report.read_bytes()


def test_direct_payload_fails_if_scientific_records_change(monkeypatch):
    records = iter(
        [
            {"source.py": record("a")},
            {"source.py": record("b")},
        ]
    )
    monkeypatch.setattr(runner, "_implementation_records", lambda: next(records))
    monkeypatch.setattr(runner, "audit_source_dependence", fake_audit({}))
    with pytest.raises(RuntimeError, match="provenance changed"):
        runner.build_source_dependence_payload(
            {
                "exploratory": graph("exploratory"),
                "fresh": graph("fresh"),
            },
            graph_inputs(),
            region_count_grid=(8,),
            registered_sizes=(20,),
        )


def test_main_rechecks_scientific_records_before_writing(monkeypatch, tmp_path):
    records = iter(
        [
            {"source.py": record("a")},
            {"source.py": record("b")},
        ]
    )
    monkeypatch.setattr(runner, "_implementation_records", lambda: next(records))
    monkeypatch.setattr(
        runner,
        "load_frozen_graphs",
        lambda *_args, **_kwargs: ({}, {}, {}),
    )
    monkeypatch.setattr(
        runner,
        "build_source_dependence_payload",
        lambda *_args, **_kwargs: {
            "decision": {
                "structural_bridge_defined": False,
                "jointly_passing_region_counts": [],
            }
        },
    )
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(RuntimeError, match="provenance changed"):
        runner.main(["--artifact-repo", "unused", "--out", str(output)])
    assert not output.exists()


def test_scientific_identity_includes_protocol_and_graph_dependencies():
    records = runner._implementation_records()
    for name in (
        "DESIGN_repeated_judge_source_dependence.md",
        "PREREG_graph_geometry_repeated_judge.md",
        "graph_geometry.py",
        "repeated_judge_source_regions.py",
        "repeated_judge_source_dependence.py",
        "run_repeated_judge_candidate_capacity.py",
    ):
        assert name in records


def test_portable_blas_identity_excludes_machine_paths():
    portable = runner._portable_blas_runtime([{
        "user_api": "blas",
        "internal_api": "openblas",
        "prefix": "libopenblas",
        "version": "1.2.3",
        "threading_layer": "pthreads",
        "architecture": "x86_64",
        "filepath": "/machine/specific/libopenblas.so",
        "num_threads": 99,
    }])
    assert portable == [{
        "user_api": "blas",
        "internal_api": "openblas",
        "prefix": "libopenblas",
        "version": "1.2.3",
        "threading_layer": "pthreads",
        "architecture": "x86_64",
    }]


def test_main_rechecks_external_graph_records_before_writing(monkeypatch, tmp_path):
    implementation = {"source.py": record("a")}
    graph_records = iter([
        ({}, {"exploratory": record("a"), "fresh": record("b")}, {}),
        ({}, {"exploratory": record("a"), "fresh": record("c")}, {}),
    ])
    monkeypatch.setattr(runner, "_implementation_records", lambda: implementation)
    monkeypatch.setattr(
        runner, "load_frozen_graphs", lambda *_args, **_kwargs: next(graph_records)
    )
    monkeypatch.setattr(
        runner,
        "build_source_dependence_payload",
        lambda *_args, **_kwargs: {
            "decision": {
                "structural_bridge_defined": False,
                "jointly_passing_region_counts": [],
            }
        },
    )
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(RuntimeError, match="graph content provenance changed"):
        runner.main(["--artifact-repo", "unused", "--out", str(output)])
    assert not output.exists()
