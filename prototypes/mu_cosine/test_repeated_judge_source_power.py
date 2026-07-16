"""Tests for the no-spend source-dependent repeated-judge Stage-A primitives."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import repeated_judge_power as base
import repeated_judge_source_power as source


def make_design(component_count=60, region_count=20):
    region_ids = tuple(f"region-{index:03d}" for index in range(region_count))
    # Strictly positive definite, nonnegative, PSD, and unit diagonal.
    exposure = 0.82 * np.eye(region_count) + 0.18 * np.ones(
        (region_count, region_count)
    )
    assignments = tuple(
        region_ids[(7 * index + index // region_count) % region_count]
        for index in range(component_count)
    )
    return source.build_source_design(region_ids, exposure, assignments)


def paired_designs(component_count=160, region_count=64):
    design = make_design(component_count, region_count)
    return {"exploratory": design, "fresh": design}


def test_source_design_factor_matches_dense_component_path():
    design = make_design(12, 4)
    for eta in source.SOURCE_ETA_GRID:
        factor = source.source_component_factor(design, eta)
        dense = (
            (1.0 - eta) * np.eye(design.component_count)
            + eta * design.incidence @ design.exposure @ design.incidence.T
        )
        np.testing.assert_allclose(factor @ factor.T, dense, rtol=0.0, atol=2e-15)
        np.testing.assert_allclose(
            source.source_component_correlation(design, eta),
            dense,
            rtol=0.0,
            atol=2e-15,
        )
        np.testing.assert_allclose(np.diag(dense), 1.0, rtol=0.0, atol=2e-15)

    assert design.exposure.flags.writeable is False
    assert design.exposure_factor.flags.writeable is False
    assert design.component_region_index.flags.writeable is False
    assert design.incidence.flags.writeable is False
    assert sum(design.counts_by_region.values()) == design.component_count


def test_source_design_rejects_reordered_or_repaired_inputs():
    with pytest.raises(ValueError, match="canonical order"):
        source.build_source_design(("z", "a"), np.eye(2), ("z", "a"))
    with pytest.raises(ValueError, match="positive definite"):
        source.build_source_design(
            ("a", "b"), np.ones((2, 2)), ("a", "b")
        )
    with pytest.raises(ValueError, match="canonical source-region"):
        source.build_source_design(("a",), np.eye(1), ("missing",))
    design = make_design()
    with pytest.raises(ValueError, match="must belong"):
        source.source_component_factor(design, 0.075)


def test_exact_persistent_draw_uses_source_then_item_channel_factors():
    # The geometry only needs the same component count as the source design.
    geometry = base.build_campaign_geometry(12, seed=2)
    small_design = make_design(12, 6)
    scenario = base.SCENARIO_BY_NAME["cumulative_rho_0.10"]
    eta = 0.20
    independent_white = np.arange(12 * 12, dtype=float).reshape(12, 12) / 100.0
    region_white = np.arange(6 * 12, dtype=float).reshape(6, 12) / 80.0

    class SequenceRng:
        def __init__(self):
            self.values = [independent_white, region_white]

        def standard_normal(self, shape):
            value = self.values.pop(0)
            assert value.shape == shape
            return value.copy()

    observed = source.draw_source_persistent(
        geometry, scenario, small_design, eta, SequenceRng()
    ).reshape(12, 12)
    item = base.rho_matched_correlation(
        base.scenario_item_kernel(geometry, scenario), scenario.truth_rho
    )
    within = np.linalg.cholesky(
        np.kron(item, base.PERSISTENT_CHANNEL_COVARIANCE)
    )
    expected = (
        np.sqrt(1.0 - eta) * (independent_white @ within.T)
        + np.sqrt(eta)
        * (
            small_design.exposure_factor @ (region_white @ within.T)
        )[small_design.component_region_index]
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2e-15)


def test_block_null_retains_the_exact_source_covariance_algebra():
    design = make_design(8, 4)
    eta = 0.20
    scenario = base.SCENARIO_BY_NAME["block_null"]
    geometry = base.build_campaign_geometry(8 + 4, seed=4)
    # Use a matching design for the geometry after satisfying the base minimum G.
    design = make_design(12, 4)
    source_covariance = source.source_component_correlation(design, eta)
    item = base.rho_matched_correlation(
        base.scenario_item_kernel(geometry, scenario), scenario.truth_rho
    )
    np.testing.assert_allclose(item, np.eye(3), rtol=0.0, atol=0.0)
    covariance = np.kron(
        source_covariance,
        np.kron(item, base.PERSISTENT_CHANNEL_COVARIANCE),
    )
    same_region = np.flatnonzero(
        design.component_region_index == design.component_region_index[0]
    )[1]
    different_region = np.flatnonzero(
        design.component_region_index != design.component_region_index[0]
    )[0]
    dimension = base.ROWS_PER_COMPONENT * base.CHANNELS
    block_same = covariance[
        0:dimension,
        same_region * dimension:(same_region + 1) * dimension,
    ]
    block_different = covariance[
        0:dimension,
        different_region * dimension:(different_region + 1) * dimension,
    ]
    expected_within = np.kron(np.eye(3), base.PERSISTENT_CHANNEL_COVARIANCE)
    np.testing.assert_allclose(block_same, eta * expected_within)
    expected_cross = eta * design.exposure[
        design.component_region_index[0],
        design.component_region_index[different_region],
    ] * expected_within
    np.testing.assert_allclose(block_different, expected_cross)


def test_source_atomic_splits_are_deterministic_balanced_and_prompt_mixed():
    design = make_design(173, 64)
    first = source.source_atomic_component_splits(design, seed=51)
    # Fold and prompt incidence are entirely stable; the seed is recorded but
    # cannot perturb the frozen count/region-ID algorithms.
    second = source.source_atomic_component_splits(design, seed=999)
    np.testing.assert_array_equal(first.outer_label, second.outer_label)
    np.testing.assert_array_equal(first.inner_label, second.inner_label)
    np.testing.assert_array_equal(
        first.prompt_block_index, second.prompt_block_index
    )
    assert [block.tolist() for block in first.prompt_blocks] == [
        block.tolist() for block in second.prompt_blocks
    ]
    assert sorted(np.concatenate(first.prompt_blocks).tolist()) == list(range(173))
    assert max(map(len, first.prompt_blocks)) <= base.MAX_PROMPT_ROWS

    assignments = np.asarray(design.assignment_region_ids, dtype=object)
    for region in design.region_ids:
        positions = np.flatnonzero(assignments == region)
        if len(positions):
            assert len(set(first.outer_label[positions])) == 1
            assert len(set(first.inner_label[positions])) == 1
    for block in first.prompt_blocks:
        assert len(set(first.outer_label[block])) == 1
        assert len(set(first.inner_label[block])) == 1
    # Round-robin construction must actually mix sources where a split cell
    # contains more than one active source.
    assert any(
        len({design.assignment_region_ids[index] for index in block}) > 1
        for block in first.prompt_blocks
        if len(block) > 1
    )
    outer_sizes = [len(fold.held) for fold in first.outer]
    largest_group = max(design.counts_by_region.values())
    assert max(outer_sizes) - min(outer_sizes) <= largest_group
    for fold in first.outer:
        assert not set(fold.train) & set(fold.held)
        for fit, held in fold.inner:
            assert set(fit) | set(held) == set(fold.train)
            assert not set(fit) & set(held)
    diagnostics = source.source_split_diagnostics(design, first)
    assert all(diagnostics["gates"].values())
    assert len(diagnostics["outer_local_inner_assignment"]) == 5
    manual_max_share = max(
        max(
            sum(design.assignment_region_ids[index] == region for index in block)
            / len(block)
            for region in {design.assignment_region_ids[index] for index in block}
        )
        for block in first.prompt_blocks
    )
    assert diagnostics["prompt_by_source"]["maximum_source_share"] == manual_max_share
    prompt_source = diagnostics["prompt_by_source"]
    incidence = np.asarray(prompt_source["incidence_counts"], dtype=int)
    assert incidence.shape == (
        prompt_source["prompt_blocks"], prompt_source["active_source_regions"]
    )
    np.testing.assert_array_equal(
        incidence.sum(axis=1), np.asarray(list(map(len, first.prompt_blocks)))
    )
    assert incidence.sum() == design.component_count
    assert [
        row["prompt_block"]
        for row in prompt_source["analysis_signature_by_prompt_block"]
    ] == list(range(len(first.prompt_blocks)))
    assert all(
        len(set(first.outer_label[block])) == 1
        and row["outer_fold"] == int(first.outer_label[block[0]])
        and len(set(first.inner_label[block])) == 1
        and row["global_inner_fold"] == int(first.inner_label[block[0]])
        for row, block in zip(
            prompt_source["analysis_signature_by_prompt_block"],
            first.prompt_blocks,
        )
    )


def test_prompt_packer_uses_frozen_least_source_total_id_key():
    design = source.build_source_design(
        ("a", "b", "c"),
        0.8 * np.eye(3) + 0.2 * np.ones((3, 3)),
        ("b", "a", "b", "a", "a", "c"),
    )
    blocks = source._least_source_prompt_blocks(
        np.arange(6), design, max_prompt_rows=3
    )
    # Source groups are processed by (-count, stable ID), then component ID.
    # This deliberately differs from processing all component IDs globally.
    assert [block.tolist() for block in blocks] == [[1, 4, 2], [3, 0, 5]]


def test_mean_only_source_smooth_term_is_deterministic_and_omitted():
    design = make_design(60, 20)
    first = source.source_smooth_omitted_mean(design, seed=55)
    second = source.source_smooth_omitted_mean(design, seed=55)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (60, 3, 4)
    assert np.max(np.abs(first)) > 0.0
    assert np.allclose(first.mean(axis=0), 0.0, atol=1e-15)
    assert np.sqrt(np.mean(first[:, 0, 0] ** 2)) == pytest.approx(0.07)
    # Components sharing a source receive the same omitted source signal.
    for region in design.region_ids:
        positions = np.flatnonzero(
            np.asarray(design.assignment_region_ids, dtype=object) == region
        )
        if len(positions) > 1:
            np.testing.assert_array_equal(
                first[positions],
                np.repeat(first[positions[0]][None], len(positions), axis=0),
            )
    assert not np.array_equal(
        first, source.source_smooth_omitted_mean(design, seed=56)
    )


def test_repeated_field_is_deterministic_and_missingness_is_request_level():
    design = make_design(160, 64)
    geometry, splits = source.build_source_campaign_geometry(design, seed=61)
    scenario = base.SCENARIO_BY_NAME["block_null"]
    first = source.draw_source_repeated_field(
        geometry, scenario, design, 0.20, 3, 62, missing_rate=0.12
    )
    second = source.draw_source_repeated_field(
        geometry, scenario, design, 0.20, 3, 62, missing_rate=0.12
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (160, 3, 3, 4)
    for block in splits.prompt_blocks:
        for row in range(3):
            for family, start in enumerate((0, 2)):
                schedules = np.isfinite(first[block, row, :, start])
                assert np.all(schedules == schedules[0])
                assert np.all(np.sum(schedules, axis=1) >= 2)
                assert np.array_equal(
                    schedules,
                    np.isfinite(first[block, row, :, start + 1]),
                )
    without_wave = source.draw_source_repeated_field(
        geometry,
        scenario,
        design,
        0.20,
        3,
        62,
        missing_rate=0.12,
        include_wave_effect=False,
    )
    # Independent nuisance RNG namespaces keep missingness and every other
    # draw fixed when the wave term is ablated.
    np.testing.assert_array_equal(np.isfinite(first), np.isfinite(without_wave))
    difference = first - without_wave
    for repeat in range(3):
        for channel in range(4):
            finite = np.isfinite(difference[:, :, repeat, channel])
            values = difference[:, :, repeat, channel][finite]
            assert np.ptp(values) <= 2e-15

    # The five generator-eta worlds are common-random-number sensitivities:
    # eta changes only the persistent mixture.  Call/request/wave draws and
    # request-level missingness remain bit-identical within one corpus world.
    eta_zero = source.draw_source_repeated_field(
        geometry, scenario, design, 0.0, 3, 63, missing_rate=0.12
    )
    eta_high = source.draw_source_repeated_field(
        geometry, scenario, design, 0.20, 3, 63, missing_rate=0.12
    )
    np.testing.assert_array_equal(np.isfinite(eta_zero), np.isfinite(eta_high))
    persistent_delta = eta_high - eta_zero
    assert np.nanmax(np.abs(persistent_delta)) > 0.0
    for component in range(design.component_count):
        for row in range(base.ROWS_PER_COMPONENT):
            for channel in range(base.CHANNELS):
                values = persistent_delta[component, row, :, channel]
                values = values[np.isfinite(values)]
                if len(values) > 1:
                    assert np.ptp(values) <= 3e-15


def test_component_marginal_quasi_nll_matches_direct_dense_formula():
    rng = np.random.default_rng(70)
    count = 7
    observed = np.ones((count, 3, 3, 2), dtype=bool)
    nuisance = SimpleNamespace(
        persistent_covariance=base.PERSISTENT_CHANNEL_COVARIANCE.copy(),
        request_covariance=base.REQUEST_CHANNEL_COVARIANCE.copy(),
        repeat_covariance=(
            base.CALL_CHANNEL_COVARIANCE + base.REQUEST_CHANNEL_COVARIANCE
        ),
        evaluate_repeat_counts=np.sum(observed, axis=2),
        evaluate_observed_calls=observed,
    )
    residuals = rng.normal(size=(count, 3, 4))
    kernel = base.gamma_item_kernel(0.5)
    rho = 0.10
    covariance = base.candidate_covariance(nuisance, kernel, rho)
    observed_nll = source.component_marginal_quasi_nll(
        residuals, nuisance, kernel, rho
    )
    flat = residuals.reshape(count, -1)
    expected = []
    for vector, matrix in zip(flat, covariance):
        expected.append(0.5 * (
            vector @ np.linalg.solve(matrix, vector)
            + np.linalg.slogdet(matrix)[1]
            + len(vector) * np.log(2.0 * np.pi)
        ) / len(vector))
    np.testing.assert_allclose(observed_nll, expected, rtol=0.0, atol=2e-14)


def test_prompt_source_standard_errors_match_dense_covariance_exactly():
    design = make_design(24, 8)
    rng = np.random.default_rng(80)
    values = rng.normal(size=(24, 2))
    prompt_ids = np.repeat(np.arange(6), 4)
    centered = values - values.mean(axis=0)
    for eta in source.SOURCE_ETA_GRID:
        covariance = source.prompt_source_multiplier_covariance(
            design, prompt_ids, eta
        )
        # Both source and prompt fields have unit diagonal; no intersection is
        # subtracted, hence the deliberately conservative diagonal is two.
        np.testing.assert_allclose(np.diag(covariance), 2.0)
        dense_variance = np.einsum(
            "ge,gh,he->e", centered, covariance, centered
        ) / len(values) ** 2
        observed = source.graph_aware_influence_standard_errors(
            values, prompt_ids, design, eta
        )
        np.testing.assert_allclose(
            observed * observed, dense_variance, rtol=0.0, atol=2e-16
        )


def test_graph_aware_bounds_use_one_max_t_and_worst_source_eta_deterministically():
    first_design = make_design(30, 10)
    second_design = make_design(30, 10)
    rng = np.random.default_rng(90)
    values = (rng.normal(size=(30, 2)), rng.normal(size=(30, 2)))
    prompts = (np.repeat(np.arange(10), 3), np.repeat(np.arange(6), 5))
    first = source.graph_aware_prompt_source_lower_bounds(
        values,
        prompts,
        (first_design, second_design),
        draws=199,
        seed=91,
    )
    second = source.graph_aware_prompt_source_lower_bounds(
        values,
        prompts,
        (first_design, second_design),
        draws=199,
        seed=91,
    )
    assert first.inference_source_eta_grid == source.SOURCE_ETA_GRID
    assert first.critical_value == second.critical_value
    np.testing.assert_array_equal(
        first.lower_bounds_by_source_eta, second.lower_bounds_by_source_eta
    )
    np.testing.assert_allclose(
        first.lower_bounds_by_source_eta,
        first.point_estimates[:, None, :]
        - first.critical_value * first.standard_errors_by_source_eta,
    )
    np.testing.assert_array_equal(
        first.worst_source_eta_lower_bounds,
        np.min(first.lower_bounds_by_source_eta, axis=1),
    )
    np.testing.assert_array_equal(
        first.worst_source_eta_indices,
        np.argmin(first.lower_bounds_by_source_eta, axis=1),
    )
    for corpus_index, design in enumerate((first_design, second_design)):
        for eta_index, eta in enumerate(source.SOURCE_ETA_GRID):
            np.testing.assert_allclose(
                first.analytic_standard_errors_by_source_eta[
                    corpus_index, eta_index
                ],
                source.graph_aware_influence_standard_errors(
                    values[corpus_index], prompts[corpus_index], design, eta
                ),
                rtol=0.0,
                atol=3e-16,
            )
    assert first.order_statistic_rank_one_based == int(
        np.ceil(0.95 * (199 + 1))
    )

    prepared = {
        corpus: source.prepare_graph_aware_source_corpus_multiplier(
            values[corpus_index],
            prompts[corpus_index],
            design,
            corpus_name=corpus,
            draws=199,
            multiplier_seed=91,
        )
        for corpus_index, (corpus, design) in enumerate(zip(
            source.REQUIRED_SOURCE_CORPORA,
            (first_design, second_design),
        ))
    }
    prepared_bounds = (
        source.combine_prepared_graph_aware_source_corpus_multipliers(
            prepared
        )
    )
    assert prepared_bounds.inference_source_eta_grid == (
        first.inference_source_eta_grid
    )
    assert prepared_bounds.critical_value == first.critical_value
    assert (
        prepared_bounds.order_statistic_rank_one_based
        == first.order_statistic_rank_one_based
    )
    assert prepared_bounds.prompt_block_counts == first.prompt_block_counts
    assert (
        prepared_bounds.active_source_region_counts
        == first.active_source_region_counts
    )
    for field in (
        "point_estimates",
        "standard_errors_by_source_eta",
        "analytic_standard_errors_by_source_eta",
        "lower_bounds_by_source_eta",
        "worst_source_eta_lower_bounds",
        "worst_source_eta_indices",
    ):
        np.testing.assert_array_equal(
            getattr(prepared_bounds, field), getattr(first, field)
        )

    # Reconstruct every frozen multiplier field independently: source fields
    # are Gaussian, prompt fields are Rademacher, and the one critical value is
    # the exact finite upper order statistic over corpus x eta x endpoint.
    deviations = np.empty((199, 2, len(source.SOURCE_ETA_GRID), 2))
    for corpus_index, (value, prompt_ids, design) in enumerate(zip(
        values, prompts, (first_design, second_design)
    )):
        centered = value - value.mean(axis=0)
        prompt_index, prompt_count = source._prompt_positions(prompt_ids)
        prompt_sums = np.zeros((prompt_count, 2))
        np.add.at(prompt_sums, prompt_index, centered)
        epsilon = np.random.default_rng(
            base.derive_seed(91, "source-iid", corpus_index)
        ).standard_normal((199, design.component_count))
        region_white = np.random.default_rng(
            base.derive_seed(91, "source-region", corpus_index)
        ).standard_normal((199, design.region_count))
        prompt_white = np.random.default_rng(
            base.derive_seed(91, "prompt", corpus_index)
        ).choice((-1.0, 1.0), size=(199, prompt_count))
        region = region_white @ design.exposure_factor.T
        prompt_deviation = prompt_white @ prompt_sums / design.component_count
        for eta_index, eta in enumerate(source.SOURCE_ETA_GRID):
            multiplier = (
                np.sqrt(1.0 - eta) * epsilon
                + np.sqrt(eta) * region[:, design.component_region_index]
            )
            deviations[:, corpus_index, eta_index] = (
                multiplier @ centered / design.component_count
                + prompt_deviation
            )
    sample_se = np.std(deviations, axis=0, ddof=1)
    np.testing.assert_array_equal(
        first.standard_errors_by_source_eta, sample_se
    )
    max_t = np.max(-deviations / sample_se[None], axis=(1, 2, 3))
    expected_critical = np.sort(max_t)[
        first.order_statistic_rank_one_based - 1
    ]
    assert first.critical_value == expected_critical
    with pytest.raises(source.MultiplierNotIdentifiedError, match="zero or nonfinite"):
        source.graph_aware_prompt_source_lower_bounds(
            (np.tile([0.1, 0.2], (30, 1)), np.tile([0.3, 0.4], (30, 1))),
            prompts,
            (first_design, second_design),
            draws=19,
            seed=92,
        )


def test_prepared_nonidentified_sentinel_matches_fail_closed_power_records():
    designs = {
        "exploratory": make_design(30, 10),
        "fresh": make_design(30, 10),
    }
    prompt_ids = np.repeat(np.arange(10), 3)
    scenario = base.SCENARIO_BY_NAME["block_null"]

    def record(corpus, endpoint_gains):
        endpoint_gains = np.asarray(endpoint_gains, dtype=float)
        return source.SourceCorpusPowerReplicate(
            corpus_name=corpus,
            scenario=scenario.name,
            generator_source_eta=0.0,
            selected=(base.BLOCK_CANDIDATE,) * base.OUTER_FOLDS,
            selector_rejected=False,
            maximum_inner_gain=0.0,
            endpoint_component_gains=endpoint_gains,
            prompt_block_ids=prompt_ids.copy(),
            topology_component_advantage=None,
            call_loading=0.0,
            persistent_loading=0.0,
            request_loading=0.0,
        )

    zero = np.zeros((30, len(base.PRIMARY_ENDPOINTS)))
    positive = np.random.default_rng(930).normal(
        size=(30, len(base.PRIMARY_ENDPOINTS))
    )
    exploratory_zero = record("exploratory", zero)
    fresh_zero = record("fresh", zero)
    fresh_positive = record("fresh", positive)
    multiplier_seed = 931

    def prepared(records):
        return {
            corpus: source.prepare_graph_aware_source_corpus_multiplier(
                item.endpoint_component_gains,
                item.prompt_block_ids,
                designs[corpus],
                corpus_name=corpus,
                draws=39,
                multiplier_seed=multiplier_seed,
            )
            for corpus, item in records.items()
        }

    def assert_same_record(left, right):
        for field in left.__dataclass_fields__:
            left_value = getattr(left, field)
            right_value = getattr(right, field)
            if isinstance(left_value, np.ndarray):
                np.testing.assert_array_equal(left_value, right_value)
            else:
                assert left_value == right_value

    for records in (
        {"exploratory": exploratory_zero, "fresh": fresh_positive},
        {"exploratory": exploratory_zero, "fresh": fresh_zero},
    ):
        components = prepared(records)
        assert components["exploratory"].inference_identified is False
        if records["fresh"] is fresh_positive:
            assert components["fresh"].inference_identified is True
        else:
            assert components["fresh"].inference_identified is False
        with pytest.raises(
            source.MultiplierNotIdentifiedError, match="zero or nonfinite"
        ):
            source.combine_prepared_graph_aware_source_corpus_multipliers(
                components
            )
        ordinary = source.combine_source_power_corpus_replicates(
            records,
            designs,
            scenario,
            multiplier_seed=multiplier_seed,
            multiplier_draws=39,
        )
        cached = source.combine_source_power_corpus_replicates(
            records,
            designs,
            scenario,
            multiplier_seed=multiplier_seed,
            multiplier_draws=39,
            prepared_multiplier_components_by_corpus=components,
        )
        assert ordinary.inference_identified is False
        assert ordinary.promoted is False
        assert_same_record(ordinary, cached)


def test_tiny_source_null_and_power_paths_are_deterministic_and_aggregate():
    designs = paired_designs(160, 64)
    common = dict(
        source_eta_by_corpus={"exploratory": 0.20, "fresh": 0.20},
        repeats=3,
        gammas=(0.0, 1.0),
        rhos=(0.0, 0.10),
        mean_ridges=(0.0, 0.10),
        missing_rate=0.0,
    )
    first_null = source.source_null_maximum(designs, seed=101, **common)
    second_null = source.source_null_maximum(designs, seed=101, **common)
    assert first_null == second_null
    scenario = base.SCENARIO_BY_NAME["cumulative_rho_0.10"]
    first = source.run_source_power_replicate(
        designs,
        scenario,
        seed=102,
        null_threshold=first_null,
        multiplier_draws=39,
        **common,
    )
    second = source.run_source_power_replicate(
        designs,
        scenario,
        seed=102,
        null_threshold=first_null,
        multiplier_draws=39,
        **common,
    )
    assert first.selected == second.selected
    assert first.promoted == second.promoted
    np.testing.assert_array_equal(
        first.endpoint_component_gains, second.endpoint_component_gains
    )
    np.testing.assert_array_equal(
        first.endpoint_lower_bounds_by_source_eta,
        second.endpoint_lower_bounds_by_source_eta,
    )
    assert first.endpoint_component_gains.shape == (2, 160, 2)
    assert first.endpoint_lower_bounds_by_source_eta.shape == (
        2, len(source.SOURCE_ETA_GRID), 2
    )
    assert first.endpoint_worst_source_eta_lower_bounds.shape == (2, 2)
    aggregate = source.aggregate_source_power_records((first, second))
    assert aggregate["scenario"] == scenario.name
    assert aggregate["generator_source_eta_by_corpus"] == {
        "exploratory": 0.20,
        "fresh": 0.20,
    }
    assert aggregate["replicates"] == 2
    assert aggregate["inference_source_eta_grid"] == list(
        source.SOURCE_ETA_GRID
    )
    assert set(aggregate["endpoint_mean_gain_per_scalar"]) == {
        "exploratory", "fresh"
    }

    nonidentified = source.run_source_power_replicate(
        designs,
        base.SCENARIO_BY_NAME["block_null"],
        source_eta_by_corpus={"exploratory": 0.20, "fresh": 0.20},
        repeats=3,
        seed=103,
        null_threshold=0.0,
        gammas=(0.0,),
        rhos=(0.0,),
        mean_ridges=(0.0, 0.10),
        missing_rate=0.0,
        multiplier_draws=19,
    )
    assert nonidentified.inference_identified is False
    assert nonidentified.promoted is False
    np.testing.assert_array_equal(
        nonidentified.endpoint_worst_source_eta_lower_bounds, 0.0
    )
    nonidentified_aggregate = source.aggregate_source_power_records(
        (nonidentified,)
    )
    assert nonidentified_aggregate["inference_nonidentified_replicates"] == 1
    assert nonidentified_aggregate["all_replicates_inference_identified"] is False


def test_asymmetric_source_eta_pair_reuses_only_the_unchanged_corpus_world():
    designs = paired_designs(160, 64)
    splits = {
        corpus: source.source_atomic_component_splits(design)
        for corpus, design in designs.items()
    }
    scenario = base.SCENARIO_BY_NAME["cumulative_rho_0.10"]
    common = dict(
        repeats=3,
        null_threshold=0.0,
        gammas=(0.0, 1.0),
        rhos=(0.0, 0.10),
        mean_ridges=(0.0, 0.10),
        shrinkage=0.05,
        missing_rate=0.0,
        max_prompt_rows=base.MAX_PROMPT_ROWS,
    )
    seed = 1701
    exploratory = source.run_source_corpus_power_replicate(
        designs["exploratory"],
        scenario,
        0.0,
        corpus_name="exploratory",
        seed=base.derive_seed(seed, "exploratory"),
        splits=splits["exploratory"],
        **common,
    )
    fresh = source.run_source_corpus_power_replicate(
        designs["fresh"],
        scenario,
        0.20,
        corpus_name="fresh",
        seed=base.derive_seed(seed, "fresh"),
        splits=splits["fresh"],
        **common,
    )
    combined = source.combine_source_power_corpus_replicates(
        {"exploratory": exploratory, "fresh": fresh},
        designs,
        scenario,
        multiplier_seed=base.derive_seed(
            seed, "joint-prompt-source-multiplier"
        ),
        multiplier_draws=39,
    )
    multiplier_seed = base.derive_seed(
        seed, "joint-prompt-source-multiplier"
    )
    prepared = {
        corpus: source.prepare_graph_aware_source_corpus_multiplier(
            record.endpoint_component_gains,
            record.prompt_block_ids,
            designs[corpus],
            corpus_name=corpus,
            draws=39,
            multiplier_seed=multiplier_seed,
        )
        for corpus, record in (
            ("exploratory", exploratory),
            ("fresh", fresh),
        )
    }
    prepared_combined = source.combine_source_power_corpus_replicates(
        {"exploratory": exploratory, "fresh": fresh},
        designs,
        scenario,
        multiplier_seed=multiplier_seed,
        multiplier_draws=39,
        prepared_multiplier_components_by_corpus=prepared,
    )
    assert (
        prepared_combined.multiplier_critical_value
        == combined.multiplier_critical_value
    )
    np.testing.assert_array_equal(
        prepared_combined.endpoint_lower_bounds_by_source_eta,
        combined.endpoint_lower_bounds_by_source_eta,
    )
    np.testing.assert_array_equal(
        prepared_combined.endpoint_worst_source_eta_lower_bounds,
        combined.endpoint_worst_source_eta_lower_bounds,
    )
    stale_fresh = source.prepare_graph_aware_source_corpus_multiplier(
        fresh.endpoint_component_gains.copy(),
        fresh.prompt_block_ids,
        designs["fresh"],
        corpus_name="fresh",
        draws=39,
        multiplier_seed=multiplier_seed,
    )
    with pytest.raises(ValueError, match="record/design/inference identity"):
        source.combine_source_power_corpus_replicates(
            {"exploratory": exploratory, "fresh": fresh},
            designs,
            scenario,
            multiplier_seed=multiplier_seed,
            multiplier_draws=39,
            prepared_multiplier_components_by_corpus={
                "exploratory": prepared["exploratory"],
                "fresh": stale_fresh,
            },
        )
    reference_bounds = source.graph_aware_prompt_source_lower_bounds(
        (
            exploratory.endpoint_component_gains,
            fresh.endpoint_component_gains,
        ),
        (exploratory.prompt_block_ids, fresh.prompt_block_ids),
        (designs["exploratory"], designs["fresh"]),
        draws=39,
        seed=base.derive_seed(seed, "joint-prompt-source-multiplier"),
    )
    # Pair combination recomputes one joint 2-corpus x 5-eta x 2-endpoint
    # max-t.  It must not splice together already-final corpuswise bounds.
    assert combined.multiplier_critical_value == reference_bounds.critical_value
    np.testing.assert_array_equal(
        combined.endpoint_lower_bounds_by_source_eta,
        reference_bounds.lower_bounds_by_source_eta,
    )
    direct = source.run_source_power_replicate(
        designs,
        scenario,
        source_eta_by_corpus={"exploratory": 0.0, "fresh": 0.20},
        seed=seed,
        multiplier_draws=39,
        splits_by_corpus=splits,
        **common,
    )
    assert direct.generator_source_eta_by_corpus == (0.0, 0.20)
    assert direct.selected == combined.selected
    np.testing.assert_array_equal(
        direct.endpoint_component_gains, combined.endpoint_component_gains
    )
    np.testing.assert_array_equal(
        direct.endpoint_lower_bounds_by_source_eta,
        combined.endpoint_lower_bounds_by_source_eta,
    )

    changed = source.run_source_power_replicate(
        designs,
        scenario,
        source_eta_by_corpus={"exploratory": 0.10, "fresh": 0.20},
        seed=seed,
        multiplier_draws=39,
        splits_by_corpus=splits,
        **common,
    )
    # Corpus-specific seed derivation does not include the other corpus's eta:
    # the unchanged fresh world is bit-identical, while the exploratory world
    # changes under its different persistent-source mixture.
    np.testing.assert_array_equal(
        direct.endpoint_component_gains[1],
        changed.endpoint_component_gains[1],
    )
    assert direct.selected[1] == changed.selected[1]
    assert not np.array_equal(
        direct.endpoint_component_gains[0],
        changed.endpoint_component_gains[0],
    )

    null_common = dict(
        repeats=3,
        gammas=(0.0, 1.0),
        rhos=(0.0, 0.10),
        mean_ridges=(0.0, 0.10),
        missing_rate=0.0,
    )
    joint_null = source.source_null_maximum(
        designs,
        source_eta_by_corpus={"exploratory": 0.0, "fresh": 0.20},
        seed=seed,
        splits_by_corpus=splits,
        **null_common,
    )
    per_corpus_null = {
        corpus: source.source_corpus_null_maximum(
            designs[corpus],
            corpus_name=corpus,
            source_eta=eta,
            seed=base.derive_seed(seed, corpus),
            splits=splits[corpus],
            **null_common,
        )
        for corpus, eta in (("exploratory", 0.0), ("fresh", 0.20))
    }
    assert joint_null == source.combine_source_corpus_null_maxima(
        per_corpus_null
    )


def test_high_level_paths_fail_closed_on_mismatched_designs_and_nonnull():
    designs = paired_designs(160, 64)
    with pytest.raises(ValueError, match="source_eta_by_corpus"):
        source.source_null_maximum(
            designs,
            source_eta_by_corpus={"exploratory": 0.20},
            repeats=3,
            seed=1,
            gammas=(0.0,),
            rhos=(0.0,),
            mean_ridges=(0.0,),
            missing_rate=0.0,
        )
    with pytest.raises(ValueError, match="exactly exploratory and fresh"):
        source.source_null_maximum(
            {"exploratory": designs["exploratory"]},
            source_eta_by_corpus={"exploratory": 0.20, "fresh": 0.20},
            repeats=3,
            seed=1,
            gammas=(0.0,),
            rhos=(0.0,),
            mean_ridges=(0.0,),
            missing_rate=0.0,
        )
    with pytest.raises(ValueError, match="zero item coupling"):
        source.source_null_maximum(
            designs,
            base.SCENARIO_BY_NAME["cumulative_rho_0.10"],
            source_eta_by_corpus={"exploratory": 0.20, "fresh": 0.20},
            repeats=3,
            seed=1,
            gammas=(0.0,),
            rhos=(0.0,),
            mean_ridges=(0.0,),
            missing_rate=0.0,
        )
