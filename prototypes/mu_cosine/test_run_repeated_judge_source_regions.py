#!/usr/bin/env python3
"""Tests for the portable source-region audit runner."""

import json

import pytest

import run_repeated_judge_source_regions as runner


def line_graph(size):
    nodes = [f"n{index:04d}" for index in range(size)]
    parents = {node: set() for node in nodes}
    children = {node: set() for node in nodes}
    for child, parent in zip(nodes[1:], nodes[:-1]):
        parents[child].add(parent)
        children[parent].add(child)
    return {"parents": parents, "children": children}


def star_graph(size):
    nodes = [f"n{index:04d}" for index in range(size)]
    parents = {node: set() for node in nodes}
    children = {node: set() for node in nodes}
    for child in nodes[1:]:
        parents[child].add(nodes[0])
        children[nodes[0]].add(child)
    return {"parents": parents, "children": children}


def record(character):
    return {"size_bytes": 1, "sha256": character * 64}


def graph_inputs():
    return {
        "exploratory": {"corpus_identity": "fixture-a", "artifacts": {"graph": record("a")}},
        "fresh": {"corpus_identity": "fixture-b", "artifacts": {"graph": record("b")}},
    }


def build(graph_a, graph_b, **kwargs):
    return runner.build_source_region_payload(
        {"exploratory": graph_a, "fresh": graph_b},
        graph_inputs(),
        implementation={"fixture.py": record("c")},
        **kwargs,
    )


def test_coarsest_jointly_passing_grid_value_is_selected():
    graph = line_graph(400)
    payload = build(
        graph,
        graph,
        region_count_grid=(10, 20),
        registered_sizes=(10,),
        min_effective_regions=10,
    )
    assert payload["decision"]["source_region_topology_gate_passed"] is True
    assert payload["decision"]["selected_region_count"] == 10
    assert payload["authorization"]["historical_inventory_unlocked"] is True
    assert payload["authorization"]["candidate_enumeration_unlocked"] is False
    assert payload["decision"]["candidate_builder_must_stop"] is True
    assert payload["configuration"]["cumulative_walk_weights"] == [
        1.0, 0.5, 0.25, 0.125
    ]
    assert payload["configuration"]["cumulative_walk_support_radius_hops"] == 3
    assert all(
        value is False
        for name, value in payload["authorization"].items()
        if name != "historical_inventory_unlocked"
    )


def test_one_corpus_failure_blocks_every_authorization():
    payload = build(
        line_graph(400),
        star_graph(400),
        region_count_grid=(10,),
        registered_sizes=(10,),
        min_effective_regions=10,
    )
    assert payload["decision"]["source_region_topology_gate_passed"] is False
    assert payload["decision"]["selected_region_count"] is None
    assert all(value is False for value in payload["authorization"].values())
    assert payload["joint_region_count_grid"]["10"]["passing_corpora"] == [
        "exploratory"
    ]


def test_payload_is_deterministic_portable_and_atomic(tmp_path):
    graph = line_graph(200)
    first = build(
        graph,
        graph,
        region_count_grid=(10,),
        registered_sizes=(10,),
        min_effective_regions=10,
    )
    second = build(
        graph,
        graph,
        region_count_grid=(10,),
        registered_sizes=(10,),
        min_effective_regions=10,
    )
    assert first == second
    data = runner._json_bytes(first)
    assert str(tmp_path).encode() not in data
    output = tmp_path / "nested" / "audit.json"
    assert runner._atomic_write(output, first) == runner._atomic_write(output, second)
    assert output.read_bytes() == data
    assert not output.with_name("audit.json.tmp").exists()
    assert json.loads(data)["inputs"]["outcomes_consumed"] is False


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"region_count_grid": ()}, "positive"),
        ({"region_count_grid": (10, 10)}, "unique"),
        ({"region_count_grid": (20, 10)}, "increasing"),
        ({"registered_sizes": (True,)}, "positive"),
    ],
)
def test_invalid_grids_fail_closed(kwargs, message):
    graph = line_graph(100)
    with pytest.raises(ValueError, match=message):
        build(graph, graph, **kwargs)


def test_blocked_exit_is_two_and_audit_only_is_zero(monkeypatch, tmp_path):
    graphs = {"exploratory": star_graph(100), "fresh": star_graph(100)}
    monkeypatch.setattr(
        runner,
        "load_frozen_graphs",
        lambda *_args, **_kwargs: (graphs, graph_inputs(), {}),
    )
    monkeypatch.setattr(runner, "DEFAULT_REGION_COUNT_GRID", (10,))
    # main() uses build defaults bound at definition time, so substitute a small
    # wrapper rather than changing the scientific default constants.
    original = runner.build_source_region_payload
    monkeypatch.setattr(
        runner,
        "build_source_region_payload",
        lambda loaded, inputs, **kwargs: original(
            loaded,
            inputs,
            region_count_grid=(10,),
            registered_sizes=(10,),
            min_effective_regions=10,
            **kwargs,
        ),
    )
    blocked = tmp_path / "blocked.json"
    assert runner.main(["--artifact-repo", "unused", "--out", str(blocked)]) == 2
    report = tmp_path / "report.json"
    assert runner.main(
        ["--artifact-repo", "unused", "--out", str(report), "--audit-only"]
    ) == 0
    assert blocked.read_bytes() == report.read_bytes()


def test_direct_payload_fails_if_implementation_changes_during_audit(monkeypatch):
    graph = line_graph(200)
    records = iter(
        [
            {"source.py": record("a")},
            {"source.py": record("b")},
        ]
    )
    monkeypatch.setattr(runner, "_implementation_records", lambda: next(records))
    with pytest.raises(RuntimeError, match="implementation provenance changed"):
        runner.build_source_region_payload(
            {"exploratory": graph, "fresh": graph},
            graph_inputs(),
            region_count_grid=(10,),
            registered_sizes=(10,),
            min_effective_regions=10,
        )


def test_main_rechecks_implementation_before_writing(monkeypatch, tmp_path):
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
        "build_source_region_payload",
        lambda *_args, **_kwargs: {
            "decision": {
                "source_region_topology_gate_passed": False,
                "selected_region_count": None,
            }
        },
    )
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(RuntimeError, match="implementation provenance changed"):
        runner.main(["--artifact-repo", "unused", "--out", str(output)])
    assert not output.exists()


def test_implementation_identity_includes_walk_feature_semantics():
    assert "graph_geometry.py" in runner._implementation_records()
