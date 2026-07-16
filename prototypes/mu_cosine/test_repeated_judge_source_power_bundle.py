#!/usr/bin/env python3
"""Tests for exact Stage-A source-design extraction and loading."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import repeated_judge_source_power_bundle as bundle
import repeated_judge_source_power as science


HERE = Path(__file__).resolve().parent
TRACKED_BUNDLE = HERE / "repro" / "repeated_judge_source_power" / "source_design.json"
TRACKED_SUMMARY = (
    HERE / "repro" / "repeated_judge_source_dependence" / "summary.json"
)


def _parent_fixture():
    configuration = {
        "required_corpora": list(bundle.EXPECTED_CORPORA),
        "region_count_grid": list(bundle.EXPECTED_REGION_COUNTS),
        "registered_components_per_corpus": list(bundle.EXPECTED_COMPONENT_COUNTS),
        "rho_grid": [0.0, 0.025, 0.05, 0.1, 0.2],
        "cumulative_walk_weights": [1.0, 0.5, 0.25, 0.125],
        "source_region_cap_fraction": 0.1,
        "endpoints_charged_per_campaign_component": 4,
    }
    inputs = {
        "graph_bundle": bundle.content_record(b"graph bundle"),
        "graphs": {
            corpus: {
                "corpus_identity": f"fixture-{corpus}",
                "artifacts": {"graph": bundle.content_record(corpus.encode("utf-8"))},
            }
            for corpus in bundle.EXPECTED_CORPORA
        },
        "outcomes_consumed": False,
        "historical_inventory_consumed": False,
        "nomic_cache_consumed": False,
        "candidate_pool_consumed": False,
        "judge_responses_consumed": False,
    }
    results = {}
    for corpus in bundle.EXPECTED_CORPORA:
        results[corpus] = {}
        for region_count in bundle.EXPECTED_REGION_COUNTS:
            region_ids = [
                f"{corpus}-K{region_count}-region-{index:03d}"
                for index in range(region_count)
            ]
            matrix = [
                [1.0 if row == column else 0.0 for column in range(region_count)]
                for row in range(region_count)
            ]
            sizes = {}
            for components in bundle.EXPECTED_COMPONENT_COUNTS:
                assignment = [
                    region_ids[index % region_count] for index in range(components)
                ]
                counts = {region_id: 0 for region_id in region_ids}
                for region_id in assignment:
                    counts[region_id] += 1
                capacities = {
                    region_id: max(counts[region_id], 1) for region_id in region_ids
                }
                sizes[str(components)] = {
                    "components_per_corpus": components,
                    "capacities_by_region": capacities,
                    "allocation": {
                        "assignment_region_ids": assignment,
                        "counts_by_region": counts,
                        "used_region_count": sum(value > 0 for value in counts.values()),
                        "quadratic_exposure": float(
                            sum(value * value for value in counts.values())
                        ),
                    },
                }
            results[corpus][str(region_count)] = {
                "target_region_count": region_count,
                "actual_region_count": region_count,
                "partition_assignment_record": bundle.content_record(
                    f"{corpus}:{region_count}:partition".encode("utf-8")
                ),
                "exposure": {"region_ids": region_ids, "matrix": matrix},
                "registered_size_results": sizes,
            }
    return {
        "schema_version": 1,
        "algorithm": bundle.PARENT_ALGORITHM,
        "configuration": configuration,
        "inputs": inputs,
        "results": results,
    }


def _summary_for(full, *, full_bytes=None):
    if full_bytes is None:
        full_bytes = bundle.canonical_json_bytes(full)
    summary = copy.deepcopy(full)
    summary["artifact_projection"] = "tracked-summary-v1"
    summary["full_payload_record"] = bundle.content_record(full_bytes)
    for corpus in bundle.EXPECTED_CORPORA:
        for region_count in bundle.EXPECTED_REGION_COUNTS:
            audit = summary["results"][corpus][str(region_count)]
            matrix = audit["exposure"].pop("matrix")
            audit["exposure"]["matrix_shape"] = [region_count, region_count]
            audit["exposure"]["matrix_record"] = bundle.canonical_value_record(matrix)
            for components in bundle.EXPECTED_COMPONENT_COUNTS:
                row = audit["registered_size_results"][str(components)]
                capacities = row.pop("capacities_by_region")
                row["capacities_by_region_record"] = bundle.canonical_value_record(
                    capacities
                )
                allocation = row["allocation"]
                assignments = allocation.pop("assignment_region_ids")
                counts = allocation.pop("counts_by_region")
                allocation["assignment_region_ids_record"] = (
                    bundle.canonical_value_record(assignments)
                )
                allocation["counts_by_region_record"] = bundle.canonical_value_record(
                    counts
                )
    return summary


def _fixture_bytes(full=None):
    full = _parent_fixture() if full is None else full
    full_bytes = bundle.canonical_json_bytes(full)
    summary = _summary_for(full, full_bytes=full_bytes)
    return full_bytes, bundle.canonical_json_bytes(summary)


@pytest.fixture(scope="module")
def valid_fixture():
    full_bytes, summary_bytes = _fixture_bytes()
    extracted = bundle.build_source_design_bundle(full_bytes, summary_bytes)
    return full_bytes, summary_bytes, extracted


def test_tracked_bundle_is_exact_compact_path_free_and_runner_loadable():
    loaded = bundle.load_source_design_bundle(TRACKED_BUNDLE, TRACKED_SUMMARY)
    assert bundle.source_design_bundle_identity(loaded) == {
        "size_bytes": 969914,
        "sha256": "da7c2ec6d003150aeb0465eb099508aea9918b495ff00ae25ea3f6e44cfe5fb9",
    }
    assert loaded["parent"]["full_payload_record"] == {
        "size_bytes": 2767735,
        "sha256": "bf9a09c35e54bd36c2e7efea19c432ccf1e9105ff67c4154cfc1c6e744a843b2",
    }
    assert sum(len(value) for value in loaded["designs"].values()) == 6
    assert sum(
        len(design["allocations"])
        for corpus in loaded["designs"].values()
        for design in corpus.values()
    ) == 24
    assert TRACKED_BUNDLE.stat().st_size < 1_000_000
    serialized = TRACKED_BUNDLE.read_text(encoding="utf-8")
    assert "/tmp/" not in serialized
    assert "/home/" not in serialized
    assert all(value is False for value in loaded["authorization"].values())


def test_tracked_bundle_builds_all_24_scientific_designs_and_source_splits():
    loaded = bundle.load_source_design_bundle(TRACKED_BUNDLE, TRACKED_SUMMARY)
    assert tuple(loaded["configuration"]["source_eta_grid"]) == (
        bundle.EXPECTED_SOURCE_ETA_GRID
    )
    assert bundle.EXPECTED_SOURCE_ETA_GRID == science.SOURCE_ETA_GRID
    checked = 0
    for corpus in bundle.EXPECTED_CORPORA:
        for region_count in bundle.EXPECTED_REGION_COUNTS:
            row = loaded["designs"][corpus][str(region_count)]
            region_ids = tuple(row["region_ids"])
            for components in bundle.EXPECTED_COMPONENT_COUNTS:
                allocation = row["allocations"][str(components)]
                assignments = tuple(
                    region_ids[index]
                    for index in allocation["assignment_region_indices"]
                )
                design = science.build_source_design(
                    region_ids, row["exposure_matrix"], assignments
                )
                splits = science.source_atomic_component_splits(design)
                diagnostics = science.source_split_diagnostics(design, splits)
                assert diagnostics["gates"]["all_source_split_gates_pass"]
                checked += 1
    assert checked == 24


def test_extraction_contains_exact_six_matrices_and_24_reconstructable_allocations(
    valid_fixture,
):
    _full_bytes, _summary_bytes, extracted = valid_fixture
    designs = extracted["designs"]
    assert set(designs) == set(bundle.EXPECTED_CORPORA)
    for corpus in bundle.EXPECTED_CORPORA:
        for region_count in bundle.EXPECTED_REGION_COUNTS:
            design = designs[corpus][str(region_count)]
            assert len(design["region_ids"]) == region_count
            assert len(design["exposure_matrix"]) == region_count
            assert set(design["allocations"]) == {
                str(value) for value in bundle.EXPECTED_COMPONENT_COUNTS
            }
            for components in bundle.EXPECTED_COMPONENT_COUNTS:
                allocation = design["allocations"][str(components)]
                assert len(allocation["assignment_region_indices"]) == components
                assert sum(allocation["counts_by_region"]) == components
                assert all(
                    count <= capacity
                    for count, capacity in zip(
                        allocation["counts_by_region"],
                        allocation["capacities_by_region"],
                    )
                )


def test_parent_byte_mutation_and_projected_full_payload_fail_closed(valid_fixture):
    full_bytes, summary_bytes, _extracted = valid_fixture
    mutated = full_bytes[:-2] + b',"mutation":true}\n'
    with pytest.raises(bundle.SourcePowerBundleError, match="parent record"):
        bundle.build_source_design_bundle(mutated, summary_bytes)
    with pytest.raises(bundle.SourcePowerBundleError, match="unprojected"):
        bundle.build_source_design_bundle(summary_bytes, summary_bytes)


def test_parent_legacy_source_eta_grid_must_match_scientific_grid():
    full = _parent_fixture()
    full["configuration"]["rho_grid"][-1] = 0.15
    full_bytes, summary_bytes = _fixture_bytes(full)
    with pytest.raises(bundle.SourcePowerBundleError, match="legacy source eta grid"):
        bundle.build_source_design_bundle(full_bytes, summary_bytes)


def test_omitted_array_record_mutation_fails_closed():
    full = _parent_fixture()
    full_bytes = bundle.canonical_json_bytes(full)
    summary = _summary_for(full, full_bytes=full_bytes)
    summary["results"]["fresh"]["64"]["exposure"]["matrix_record"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(bundle.SourcePowerBundleError, match="content record mismatch"):
        bundle.build_source_design_bundle(
            full_bytes, bundle.canonical_json_bytes(summary)
        )


def test_asymmetric_and_non_psd_exposure_matrices_fail_even_with_matching_records():
    asymmetric = _parent_fixture()
    asymmetric["results"]["exploratory"]["64"]["exposure"]["matrix"][0][1] = 0.2
    full_bytes, summary_bytes = _fixture_bytes(asymmetric)
    with pytest.raises(bundle.SourcePowerBundleError, match="exactly symmetric"):
        bundle.build_source_design_bundle(full_bytes, summary_bytes)

    non_psd = _parent_fixture()
    matrix = non_psd["results"]["fresh"]["64"]["exposure"]["matrix"]
    matrix[0][1] = matrix[1][0] = 0.9
    matrix[0][2] = matrix[2][0] = 0.9
    full_bytes, summary_bytes = _fixture_bytes(non_psd)
    with pytest.raises(bundle.SourcePowerBundleError, match="positive semidefinite"):
        bundle.build_source_design_bundle(full_bytes, summary_bytes)


def test_assignment_count_and_capacity_mismatches_fail_closed():
    mismatch = _parent_fixture()
    allocation = mismatch["results"]["exploratory"]["96"][
        "registered_size_results"
    ]["160"]["allocation"]
    allocation["assignment_region_ids"][0] = allocation["assignment_region_ids"][1]
    full_bytes, summary_bytes = _fixture_bytes(mismatch)
    with pytest.raises(bundle.SourcePowerBundleError, match="assignment/count mismatch"):
        bundle.build_source_design_bundle(full_bytes, summary_bytes)

    over_capacity = _parent_fixture()
    row = over_capacity["results"]["fresh"]["128"]["registered_size_results"][
        "320"
    ]
    first = row["allocation"]["assignment_region_ids"][0]
    row["capacities_by_region"][first] = 0
    full_bytes, summary_bytes = _fixture_bytes(over_capacity)
    with pytest.raises(bundle.SourcePowerBundleError, match="exceeds a region capacity"):
        bundle.build_source_design_bundle(full_bytes, summary_bytes)


def test_loader_rejects_canonical_bundle_mutation_and_projection(valid_fixture, tmp_path):
    _full_bytes, summary_bytes, extracted = valid_fixture
    summary_path = tmp_path / "summary.json"
    summary_path.write_bytes(summary_bytes)

    mutated = copy.deepcopy(extracted)
    mutated["designs"]["fresh"]["64"]["exposure_matrix"][0][1] = 0.1
    path = tmp_path / "mutated.json"
    path.write_bytes(bundle.canonical_json_bytes(mutated))
    with pytest.raises(bundle.SourcePowerBundleError, match="symmetric|record"):
        bundle.load_source_design_bundle(path, summary_path)

    projected = copy.deepcopy(extracted)
    projected["artifact_projection"] = "tracked-summary-v1"
    path.write_bytes(bundle.canonical_json_bytes(projected))
    with pytest.raises(bundle.SourcePowerBundleError, match="projection"):
        bundle.load_source_design_bundle(path, summary_path)


def test_loader_requires_canonical_bytes_and_writer_is_deterministic_atomic(
    valid_fixture, tmp_path
):
    _full_bytes, summary_bytes, extracted = valid_fixture
    summary_path = tmp_path / "summary.json"
    summary_path.write_bytes(summary_bytes)
    noncanonical = tmp_path / "pretty.json"
    noncanonical.write_text(json.dumps(extracted, indent=2), encoding="utf-8")
    with pytest.raises(bundle.SourcePowerBundleError, match="not canonical"):
        bundle.load_source_design_bundle(noncanonical, summary_path)

    first = tmp_path / "one" / "bundle.json"
    second = tmp_path / "two" / "bundle.json"
    first_record = bundle.write_source_design_bundle(first, extracted)
    second_record = bundle.write_source_design_bundle(second, extracted)
    assert first_record == second_record == bundle.source_design_bundle_identity(extracted)
    assert first.read_bytes() == second.read_bytes()
    assert bundle.load_source_design_bundle(first, summary_path) == extracted
    assert not list(tmp_path.rglob("*.tmp"))


def test_duplicate_keys_and_nonfinite_json_are_rejected():
    with pytest.raises(bundle.SourcePowerBundleError, match="duplicate JSON key"):
        bundle._loads(b'{"a":1,"a":2}', "fixture")
    with pytest.raises(bundle.SourcePowerBundleError, match="non-finite"):
        bundle._loads(b'{"a":NaN}', "fixture")
