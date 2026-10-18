#!/usr/bin/env python3
"""Tests for the no-spend repeated-judge structural capacity preflight."""

import json

import pytest

from independent_embedding_cache import MODEL_SPECS
from repeated_judge_campaign import FROZEN_NOMIC_PREFIX
from repeated_judge_candidate_capacity import (
    CandidateCapacityError,
    ENDPOINTS_PER_COMPONENT,
    audit_graph_capacity,
    connected_component_sizes,
    optimistic_capacity_bound,
    source_component_cap,
)
import run_repeated_judge_candidate_capacity as runner


def graph_from_component_sizes(sizes, *, reverse=False):
    parents = {}
    children = {}
    for component, size in enumerate(sizes):
        nodes = [f"c{component}-n{value}" for value in range(size)]
        for node in nodes:
            parents[node] = set()
            children[node] = set()
        for child, parent in zip(nodes[1:], nodes[:-1]):
            parents[child].add(parent)
            children[parent].add(child)
    if reverse:
        parents = dict(reversed(list(parents.items())))
        children = dict(reversed(list(children.items())))
    return parents, children


def fake_record(token):
    return {"size_bytes": len(token), "sha256": token * 64}


def fake_inputs():
    return {
        "exploratory": {
            "corpus_identity": "fixture-a",
            "format": "fixture",
            "artifacts": {"graph": fake_record("a")},
        },
        "fresh": {
            "corpus_identity": "fixture-b",
            "format": "fixture",
            "artifacts": {"graph": fake_record("b")},
        },
    }


def test_one_connected_component_cannot_satisfy_a_ten_percent_cap():
    bound = optimistic_capacity_bound((75_901,), 160)
    assert bound["source_component_cap"] == 16
    assert bound["optimistic_capacity_upper_bound"] == 16
    assert bound["necessary_capacity_gate_passes"] is False


def test_exact_exploratory_upper_bounds_and_g512_floor():
    sizes = (
        83_194, 767, 77, 19, 5, 4, 4, 4, 4, 4,
        3, 3, 3, 3, 3, 3, 3,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        1,
    )
    assert sum(sizes) == 84_136
    conservative = {
        value: optimistic_capacity_bound(
            sizes, value, endpoints_per_component=1
        )["optimistic_capacity_upper_bound"]
        for value in (160, 320, 512, 800)
    }
    sharper = {
        value: optimistic_capacity_bound(
            sizes, value, endpoints_per_component=ENDPOINTS_PER_COMPONENT
        )[
            "optimistic_capacity_upper_bound"
        ]
        for value in (160, 320, 512, 800)
    }
    assert conservative == {160: 143, 320: 194, 512: 251, 800: 335}
    assert sharper == {160: 58, 320: 93, 512: 131, 800: 189}
    assert source_component_cap(512) == 51
    assert ENDPOINTS_PER_COMPONENT == 4


def test_component_sizes_and_audit_are_input_order_invariant():
    first = graph_from_component_sizes((9, 5, 1))
    second = graph_from_component_sizes((9, 5, 1), reverse=True)
    assert connected_component_sizes(*first) == (9, 5, 1)
    assert connected_component_sizes(*second) == (9, 5, 1)
    assert audit_graph_capacity(*first, (16,)) == audit_graph_capacity(
        *second, (16,)
    )


def test_removing_nodes_from_frozen_source_components_cannot_increase_bound():
    before = optimistic_capacity_bound((40, 20, 8), 32)
    after = optimistic_capacity_bound((36, 16, 4), 32)
    assert after["optimistic_capacity_upper_bound"] <= before[
        "optimistic_capacity_upper_bound"
    ]


@pytest.mark.parametrize(
    "parents,children,message",
    [
        ({}, {}, "at least one node"),
        ({"a": "b"}, {}, "not text"),
        ({"": set()}, {}, "non-empty"),
        ({"a": [["unhashable"]]}, {}, "hashable"),
    ],
)
def test_malformed_graphs_fail_closed(parents, children, message):
    with pytest.raises(CandidateCapacityError, match=message):
        connected_component_sizes(parents, children)


def test_inconsistent_parent_child_maps_fail_closed():
    with pytest.raises(CandidateCapacityError, match="maps disagree"):
        connected_component_sizes({"child": {"parent"}}, {"parent": set()})


def test_invalid_capacity_arguments_fail_closed():
    with pytest.raises(CandidateCapacityError, match="integer"):
        optimistic_capacity_bound((4,), True)
    with pytest.raises(CandidateCapacityError, match="positive integers"):
        optimistic_capacity_bound((4, 0), 10)
    with pytest.raises(CandidateCapacityError, match=r"\(0,1\]"):
        optimistic_capacity_bound((4,), 10, cap_fraction=0.0)


@pytest.mark.parametrize("registered_sizes", [(0,), (True,), (1.5,), ("16",)])
def test_invalid_registered_sizes_fail_closed(registered_sizes):
    graph = graph_from_component_sizes((8,))
    graphs = {
        corpus: {"parents": graph[0], "children": graph[1]}
        for corpus in ("exploratory", "fresh")
    }
    with pytest.raises(ValueError, match="positive non-bool integers"):
        runner.build_capacity_payload(
            graphs,
            fake_inputs(),
            registered_sizes=registered_sizes,
            implementation={"runner.py": fake_record("c")},
        )


def test_payload_is_deterministic_path_free_and_fail_closed(tmp_path):
    exploratory = graph_from_component_sizes((12, 4))
    fresh = graph_from_component_sizes((16,))
    graphs = {
        "exploratory": {"parents": exploratory[0], "children": exploratory[1]},
        "fresh": {"parents": fresh[0], "children": fresh[1]},
    }
    implementation = {"runner.py": fake_record("c")}
    first = runner.build_capacity_payload(
        graphs,
        fake_inputs(),
        registered_sizes=(16,),
        implementation=implementation,
    )
    second = runner.build_capacity_payload(
        graphs,
        fake_inputs(),
        registered_sizes=(16,),
        implementation=implementation,
    )
    assert first == second
    assert first["decision"]["candidate_builder_must_stop"] is True
    assert all(value is False for value in first["authorization"].values())
    assert first["inputs"]["outcomes_consumed"] is False
    assert first["inputs"]["historical_inventory_consumed"] is False
    assert first["inputs"]["nomic_cache_consumed"] is False
    serialized = runner._json_bytes(first)
    assert str(tmp_path).encode() not in serialized
    output = tmp_path / "nested" / "capacity.json"
    record_a = runner._atomic_write(output, first)
    record_b = runner._atomic_write(output, second)
    assert record_a == record_b
    assert output.read_bytes() == serialized
    assert not (output.parent / "capacity.json.tmp").exists()
    assert json.loads(serialized)["schema_version"] == 1


def test_passing_payload_unlocks_only_no_spend_enumeration():
    graph = graph_from_component_sizes((1,) * 16)
    graphs = {
        corpus: {"parents": graph[0], "children": graph[1]}
        for corpus in ("exploratory", "fresh")
    }
    payload = runner.build_capacity_payload(
        graphs,
        fake_inputs(),
        registered_sizes=(16,),
        implementation={"runner.py": fake_record("c")},
    )
    assert payload["decision"]["capacity_gate_passed"] is True
    assert payload["decision"]["candidate_builder_must_stop"] is False
    assert payload["authorization"]["candidate_enumeration_unlocked"] is True
    assert all(
        value is False
        for name, value in payload["authorization"].items()
        if name != "candidate_enumeration_unlocked"
    )
    assert "PASSED" in payload["status"]
    assert "does not establish actual packability" in payload["decision"]["reason"]


def test_mixed_grid_reports_at_least_one_failure_without_overclaiming():
    graph = graph_from_component_sizes((1,))
    graphs = {
        corpus: {"parents": graph[0], "children": graph[1]}
        for corpus in ("exploratory", "fresh")
    }
    payload = runner.build_capacity_payload(
        graphs,
        fake_inputs(),
        registered_sizes=(1, 16),
        implementation={"runner.py": fake_record("c")},
    )
    assert payload["joint_grid"]["1"]["joint_necessary_capacity_gate_passes"] is True
    assert payload["joint_grid"]["16"]["joint_necessary_capacity_gate_passes"] is False
    assert "at least one registered size" in payload["decision"]["reason"]
    assert "every registered size" not in payload["decision"]["reason"]


def test_blocked_audit_returns_two_by_default(monkeypatch, tmp_path):
    graph = graph_from_component_sizes((16,))
    graphs = {
        corpus: {"parents": graph[0], "children": graph[1]}
        for corpus in ("exploratory", "fresh")
    }
    monkeypatch.setattr(
        runner,
        "load_frozen_graphs",
        lambda *_args, **_kwargs: (graphs, fake_inputs(), {}),
    )
    output = tmp_path / "capacity.json"
    status = runner.main(
        [
            "--artifact-repo", "unused",
            "--out", str(output),
        ]
    )
    assert status == 2
    assert json.loads(output.read_text())["decision"]["candidate_builder_must_stop"] is True


def test_audit_only_explicitly_returns_zero_for_blocked_audit(monkeypatch, tmp_path):
    graph = graph_from_component_sizes((16,))
    graphs = {
        corpus: {"parents": graph[0], "children": graph[1]}
        for corpus in ("exploratory", "fresh")
    }
    monkeypatch.setattr(
        runner,
        "load_frozen_graphs",
        lambda *_args, **_kwargs: (graphs, fake_inputs(), {}),
    )
    blocked_output = tmp_path / "blocked.json"
    blocked_status = runner.main(
        [
            "--artifact-repo", "unused",
            "--out", str(blocked_output),
        ]
    )
    output = tmp_path / "audit-only.json"
    status = runner.main(
        [
            "--artifact-repo", "unused",
            "--out", str(output),
            "--audit-only",
        ]
    )
    assert blocked_status == 2
    assert status == 0
    assert output.read_bytes() == blocked_output.read_bytes()
    payload = json.loads(output.read_text())
    assert payload["decision"]["candidate_builder_must_stop"] is True
    assert all(value is False for value in payload["authorization"].values())


def test_passing_audit_returns_zero_by_default(monkeypatch, tmp_path):
    graph = graph_from_component_sizes((1,) * 800)
    graphs = {
        corpus: {"parents": graph[0], "children": graph[1]}
        for corpus in ("exploratory", "fresh")
    }
    monkeypatch.setattr(
        runner,
        "load_frozen_graphs",
        lambda *_args, **_kwargs: (graphs, fake_inputs(), {}),
    )
    output = tmp_path / "capacity.json"
    status = runner.main(
        [
            "--artifact-repo", "unused",
            "--out", str(output),
        ]
    )
    assert status == 0
    assert json.loads(output.read_text())["decision"]["capacity_gate_passed"] is True


def test_audit_only_does_not_swallow_load_errors(monkeypatch, tmp_path):
    def fail_load(*_args, **_kwargs):
        raise RuntimeError("fixture load failure")

    monkeypatch.setattr(runner, "load_frozen_graphs", fail_load)
    output = tmp_path / "capacity.json"
    with pytest.raises(RuntimeError, match="fixture load failure"):
        runner.main(
            [
                "--artifact-repo", "unused",
                "--out", str(output),
                "--audit-only",
            ]
        )
    assert not output.exists()


def test_audit_only_does_not_swallow_write_errors(monkeypatch, tmp_path):
    graph = graph_from_component_sizes((16,))
    graphs = {
        corpus: {"parents": graph[0], "children": graph[1]}
        for corpus in ("exploratory", "fresh")
    }
    monkeypatch.setattr(
        runner,
        "load_frozen_graphs",
        lambda *_args, **_kwargs: (graphs, fake_inputs(), {}),
    )

    def fail_write(*_args, **_kwargs):
        raise OSError("fixture write failure")

    monkeypatch.setattr(runner, "_atomic_write", fail_write)
    with pytest.raises(OSError, match="fixture write failure"):
        runner.main(
            [
                "--artifact-repo", "unused",
                "--out", str(tmp_path / "capacity.json"),
                "--audit-only",
            ]
        )


def test_graph_input_sets_must_match_exactly():
    graph = graph_from_component_sizes((8,))
    graphs = {"exploratory": {"parents": graph[0], "children": graph[1]}}
    with pytest.raises(ValueError, match="exactly exploratory and fresh"):
        runner.build_capacity_payload(graphs, {"exploratory": {}})


def test_frozen_nomic_prefix_matches_exact_cache_input_bytes():
    assert FROZEN_NOMIC_PREFIX == MODEL_SPECS["nomic"].task_prefix
    assert FROZEN_NOMIC_PREFIX == "clustering: "
