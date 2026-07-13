import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repeated_judge_power import (
    BLOCK_CANDIDATE,
    CALL_CHANNEL_COVARIANCE,
    DEFAULT_GAMMAS,
    DEFAULT_MEAN_RIDGES,
    DEFAULT_RHOS,
    MAX_PROMPT_ROWS,
    REQUEST_CHANNEL_COVARIANCE,
    Candidate,
    CandidateSummary,
    InnerSearch,
    SCENARIOS,
    SCENARIO_BY_NAME,
    build_campaign_geometry,
    calibrate_synthetic_selector_null,
    candidate_covariance,
    candidate_grid,
    component_gaussian_nll,
    component_splits,
    deranged_item_kernel,
    draw_latent_states,
    draw_repeated_field,
    explicit_item_kernels,
    finite_null_maximum_threshold,
    fit_repeat_nuisance,
    gamma_item_kernel,
    inner_candidate_search,
    maximum_off_diagonal,
    posterior_state_nll,
    prompt_block_candidate_covariance,
    prompt_block_gaussian_nll,
    prompt_block_multiplier_simultaneous_lower_bounds,
    prompt_block_posterior_state_nll,
    rho_matched_correlation,
    run_power_replicate,
    select_strictly_calibrated,
)


def geometry_and_splits(component_count=48, seed=1):
    splits = component_splits(component_count, seed=seed)
    geometry = build_campaign_geometry(
        component_count, seed=seed + 1, prompt_blocks=splits.prompt_blocks
    )
    return geometry, splits


def test_frozen_grids_scenarios_and_psd_geometry_are_exact():
    assert DEFAULT_GAMMAS == (0.0, 0.25, 0.50, 0.75, 1.0)
    assert DEFAULT_RHOS == (0.0, 0.025, 0.05, 0.10, 0.20)
    assert DEFAULT_MEAN_RIDGES == (0.0, 0.01, 0.10, 1.0, 10.0)
    expected_scenarios = ["block_null", "mean_only"] + [
        f"{family}_rho_{rho}"
        for family in ("cumulative", "nomic", "mixture", "deranged")
        for rho in ("0.04", "0.10", "0.20")
    ]
    assert [scenario.name for scenario in SCENARIOS] == expected_scenarios
    grid = candidate_grid()
    assert len(grid) == 1 + len(DEFAULT_GAMMAS) * (len(DEFAULT_RHOS) - 1)
    assert grid[0] == BLOCK_CANDIDATE

    kernels = explicit_item_kernels()
    for name in ("cumulative", "nomic", "deranged_cumulative"):
        kernel = kernels[name]
        assert np.allclose(kernel, kernel.T)
        assert np.allclose(np.diag(kernel), 1.0)
        assert np.linalg.eigvalsh(kernel)[0] >= -1e-12
    assert not np.allclose(kernels["cumulative"], kernels["nomic"])
    assert np.allclose(
        np.linalg.eigvalsh(kernels["cumulative"]),
        np.linalg.eigvalsh(kernels["deranged_cumulative"]),
    )


def test_rho_matched_path_has_requested_maximum_and_is_spd():
    for gamma in DEFAULT_GAMMAS:
        kernel = gamma_item_kernel(gamma)
        for rho in DEFAULT_RHOS:
            correlation = rho_matched_correlation(kernel, rho)
            assert np.allclose(np.diag(correlation), 1.0)
            assert np.linalg.eigvalsh(correlation)[0] > 0.0
            assert np.isclose(maximum_off_diagonal(correlation), rho)
    with pytest.raises(ValueError, match="0.95"):
        rho_matched_correlation(np.full((3, 3), 1.0), 0.96)


def test_prompt_blocks_are_bounded_stable_and_never_cross_split_signatures():
    _geometry, splits = geometry_and_splits(160, seed=123)
    assert len(splits.outer) == 5
    assert np.ptp([len(fold.held) for fold in splits.outer]) <= 1
    assert sorted(np.concatenate([fold.held for fold in splits.outer]).tolist()) == list(range(160))
    assert max(map(len, splits.prompt_blocks)) <= MAX_PROMPT_ROWS
    for block in splits.prompt_blocks:
        assert len(set(splits.outer_label[block])) == 1
        assert len(set(splits.inner_label[block])) == 1
    for fold in splits.outer:
        assert not set(fold.train) & set(fold.held)
        train_blocks = set(splits.prompt_block_index[fold.train])
        held_blocks = set(splits.prompt_block_index[fold.held])
        assert not train_blocks & held_blocks
        # With five equal G=160 outer folds and one stable global three-way
        # inner label, a <=1 leave-one-outer margin is combinatorially
        # impossible for every outer fold; the deterministic optimum is 2.
        assert np.ptp([len(held) for _fit, held in fold.inner]) <= 2
        for fit, held in fold.inner:
            assert not set(fit) & set(held)
            assert set(fit) | set(held) == set(fold.train)
            assert not (
                set(splits.prompt_block_index[fit])
                & set(splits.prompt_block_index[held])
            )
    changed = component_splits(160, seed=124)
    assert not np.array_equal(splits.prompt_block_index, changed.prompt_block_index)


def test_repeat_generator_has_request_level_missingness_and_is_deterministic():
    geometry, _splits = geometry_and_splits(64, seed=14)
    scenario = SCENARIO_BY_NAME["cumulative_rho_0.20"]
    first = draw_repeated_field(geometry, scenario, repeats=4, seed=702)
    second = draw_repeated_field(geometry, scenario, repeats=4, seed=702)
    np.testing.assert_allclose(first, second, equal_nan=True)
    for block in geometry.prompt_blocks:
        for row in range(3):
            for repeat in range(4):
                for start in (0, 2):
                    observed = np.isfinite(first[block, row, repeat, start])
                    assert np.all(observed == observed[0])
        counts = np.stack([
            np.sum(np.isfinite(first[block, ..., start]), axis=2)
            for start in (0, 2)
        ], axis=-1)
        assert np.min(counts) >= 2


def test_request_and_row_call_moments_are_separately_refit():
    component_count = 600
    geometry = build_campaign_geometry(component_count, seed=91)
    field = draw_repeated_field(
        geometry,
        SCENARIO_BY_NAME["block_null"],
        repeats=4,
        seed=92,
        missing_rate=0.0,
        include_wave_effect=False,
    )
    nuisance = fit_repeat_nuisance(
        field, geometry, np.arange(component_count), np.arange(20)
    )
    assert np.allclose(
        nuisance.request_covariance,
        REQUEST_CHANNEL_COVARIANCE,
        atol=0.035,
    )
    assert np.allclose(nuisance.call_covariance, CALL_CHANNEL_COVARIANCE, atol=0.055)
    assert np.linalg.eigvalsh(nuisance.request_covariance)[0] > 0.0
    assert np.linalg.eigvalsh(nuisance.call_covariance)[0] > 0.0


def test_nuisance_refit_uses_only_training_prompt_blocks():
    geometry, splits = geometry_and_splits(80, seed=20)
    field = draw_repeated_field(
        geometry, SCENARIO_BY_NAME["cumulative_rho_0.10"], repeats=4, seed=22
    )
    fold = splits.outer[0]
    fitted = fit_repeat_nuisance(field, geometry, fold.train, fold.held)
    changed = field.copy()
    changed[fold.held] += 100.0
    refitted = fit_repeat_nuisance(changed, geometry, fold.train, fold.held)
    for name in (
        "coefficients",
        "wave_effects",
        "call_covariance",
        "request_covariance",
        "persistent_covariance",
    ):
        assert np.array_equal(getattr(fitted, name), getattr(refitted, name))
    assert fitted.selected_mean_ridge in DEFAULT_MEAN_RIDGES
    assert not np.array_equal(
        fitted.evaluate_centered_means, refitted.evaluate_centered_means
    )


def test_candidate_marginal_and_prompt_schedule_covariances_are_both_present():
    geometry, splits = geometry_and_splits(80, seed=30)
    field = draw_repeated_field(
        geometry, SCENARIO_BY_NAME["block_null"], repeats=3, seed=31
    )
    fold = splits.outer[0]
    nuisance = fit_repeat_nuisance(field, geometry, fold.train, fold.held)
    covariance = candidate_covariance(
        nuisance, gamma_item_kernel(0.5, geometry.kernels), 0.10
    )
    assert covariance.shape == (len(fold.held), 12, 12)
    row, family = 0, 0
    start = row * 4 + family * 2
    count = nuisance.evaluate_repeat_counts[0, row, family]
    expected = (
        nuisance.persistent_covariance[:2, :2]
        + nuisance.repeat_covariance[:2, :2] / count
    )
    assert np.allclose(covariance[0, start:start + 2, start:start + 2], expected)

    held_blocks = geometry.prompt_block_index[fold.held]
    positions = np.flatnonzero(held_blocks == held_blocks[0])
    assert len(positions) >= 2
    joint = prompt_block_candidate_covariance(
        nuisance, gamma_item_kernel(0.5, geometry.kernels), 0.0, positions
    )
    overlap = np.sum(
        nuisance.evaluate_observed_calls[positions[0], row, :, family]
        & nuisance.evaluate_observed_calls[positions[1], row, :, family]
    )
    coefficient = overlap / (
        nuisance.evaluate_repeat_counts[positions[0], row, family]
        * nuisance.evaluate_repeat_counts[positions[1], row, family]
    )
    cross = joint[start:start + 2, 12 + start:12 + start + 2]
    assert np.allclose(cross, nuisance.request_covariance[:2, :2] * coefficient)
    assert np.max(np.abs(cross)) > 0.0


def test_component_nll_matches_direct_dense_formula():
    rng = np.random.default_rng(33)
    residuals = rng.standard_normal((4, 3, 4))
    base = rng.standard_normal((12, 12))
    covariance = base @ base.T + np.eye(12)
    observed = component_gaussian_nll(residuals, covariance)
    vector = residuals[0].reshape(-1)
    sign, logdet = np.linalg.slogdet(covariance)
    expected = 0.5 * (
        vector @ np.linalg.solve(covariance, vector)
        + logdet
        + len(vector) * np.log(2.0 * np.pi)
    ) / len(vector)
    assert sign > 0
    assert np.isclose(observed[0], expected)


def test_primary_posterior_endpoint_is_not_a_duplicate_residual_score():
    geometry, splits = geometry_and_splits(64, seed=40)
    field = draw_repeated_field(
        geometry, SCENARIO_BY_NAME["cumulative_rho_0.10"], repeats=3, seed=41
    )
    fold = splits.outer[0]
    nuisance = fit_repeat_nuisance(field, geometry, fold.train, fold.held)
    blocks = geometry.prompt_block_index[fold.held]
    kernel = gamma_item_kernel(1.0, geometry.kernels)
    residual_nll = prompt_block_gaussian_nll(
        nuisance.evaluate_centered_means, nuisance, kernel, 0.10, blocks
    )
    states = draw_latent_states(len(fold.held), seed=42)
    posterior_nll = prompt_block_posterior_state_nll(
        nuisance.evaluate_centered_means, nuisance, kernel, 0.10, states, blocks
    )
    changed_states = states + 0.5
    changed_posterior = prompt_block_posterior_state_nll(
        nuisance.evaluate_centered_means,
        nuisance,
        kernel,
        0.10,
        changed_states,
        blocks,
    )
    assert np.isfinite(residual_nll).all()
    assert np.isfinite(posterior_nll).all()
    assert not np.allclose(residual_nll, posterior_nll)
    assert not np.allclose(posterior_nll, changed_posterior)


def test_inner_search_uses_complete_grid_three_folds_and_frozen_tie_break():
    geometry, splits = geometry_and_splits(48, seed=50)
    field = draw_repeated_field(
        geometry, SCENARIO_BY_NAME["cumulative_rho_0.10"], repeats=3, seed=51
    )
    search = inner_candidate_search(field, geometry, splits.outer[0])
    assert len(search.summaries) == 21
    assert all(len(row.fold_gains) == 3 for row in search.summaries)
    # Exact tie order: lower rho, then gamma nearer .5, then lower gamma.
    summaries = [
        CandidateSummary(Candidate(0.0, 0.10), 1.0, 2, (1.0, 1.0, -1.0)),
        CandidateSummary(Candidate(0.5, 0.10), 1.0, 2, (1.0, 1.0, -1.0)),
        CandidateSummary(Candidate(0.5, 0.05), 1.0, 2, (1.0, 1.0, -1.0)),
    ]
    chosen = min(summaries, key=lambda row: (
        -row.macro_gain,
        row.candidate.rho,
        abs(row.candidate.gamma - 0.5),
        row.candidate.gamma,
    ))
    assert chosen.candidate == Candidate(0.5, 0.05)


def test_prompt_block_multiplier_uses_clusters_and_constant_bounds_are_exact():
    values = np.tile([0.1, 0.2], (20, 1))
    block_ids = np.repeat(np.arange(4), 5)
    lower, critical, clusters = prompt_block_multiplier_simultaneous_lower_bounds(
        values, block_ids, draws=99, seed=60
    )
    assert clusters == 4
    assert np.allclose(lower, [0.1, 0.2])
    assert np.isfinite(critical)
    with pytest.raises(ValueError, match="two independent prompt blocks"):
        prompt_block_multiplier_simultaneous_lower_bounds(
            values, np.zeros(20, dtype=int), draws=9
        )


def test_finite_null_threshold_and_strict_rejection_boundary():
    threshold, rank = finite_null_maximum_threshold(np.arange(20.0), confidence=0.95)
    assert rank == 20
    assert threshold == 19.0
    candidate = Candidate(1.0, 0.1)
    row = CandidateSummary(candidate, 19.0, 3, (19.0, 19.0, 19.0))
    search = InnerSearch(candidate, (row,), 19.0)
    selected, rejected = select_strictly_calibrated(search, threshold)
    assert selected == BLOCK_CANDIDATE
    assert not rejected
    stronger = InnerSearch(candidate, (row,), np.nextafter(19.0, np.inf))
    selected, rejected = select_strictly_calibrated(stronger, threshold)
    assert selected == candidate
    assert rejected


def test_tiny_null_and_power_runs_are_deterministic_and_regenerate_designs():
    null_first = calibrate_synthetic_selector_null(
        32,
        repeats=3,
        draws=2,
        seed=70,
        gammas=(0.0, 1.0),
        rhos=(0.0, 0.10),
        mean_ridges=(0.0, 0.1),
        missing_rate=0.0,
    )
    null_second = calibrate_synthetic_selector_null(
        32,
        repeats=3,
        draws=2,
        seed=70,
        gammas=(0.0, 1.0),
        rhos=(0.0, 0.10),
        mean_ridges=(0.0, 0.1),
        missing_rate=0.0,
    )
    np.testing.assert_array_equal(null_first[0], null_second[0])
    assert null_first[1:] == null_second[1:]
    record = run_power_replicate(
        32,
        SCENARIO_BY_NAME["cumulative_rho_0.10"],
        repeats=3,
        seed=71,
        null_threshold=null_first[1],
        gammas=(0.0, 1.0),
        rhos=(0.0, 0.10),
        mean_ridges=(0.0, 0.1),
        multiplier_draws=19,
        missing_rate=0.0,
    )
    repeated = run_power_replicate(
        32,
        SCENARIO_BY_NAME["cumulative_rho_0.10"],
        repeats=3,
        seed=71,
        null_threshold=null_first[1],
        gammas=(0.0, 1.0),
        rhos=(0.0, 0.10),
        mean_ridges=(0.0, 0.1),
        multiplier_draws=19,
        missing_rate=0.0,
    )
    np.testing.assert_array_equal(record.endpoint_component_gains, repeated.endpoint_component_gains)
    assert record.selected == repeated.selected
    assert record.endpoint_component_gains.shape == (2, 32, 2)
    assert len(record.inference_prompt_blocks) == 2
    assert max(record.inference_prompt_blocks) < 32
