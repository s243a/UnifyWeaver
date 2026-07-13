import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repeated_judge_power as power
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


def shared_varying_schedule(component_count):
    shared = np.zeros((3, 5, 2), dtype=bool)
    shared[0, :, 0] = True
    shared[0, :4, 1] = True
    shared[1, (0, 2, 4), 0] = True
    shared[1, :, 1] = True
    shared[2, :4, 0] = True
    shared[2, :2, 1] = True
    return np.repeat(shared[None], component_count, axis=0)


def score_only_nuisance(observed):
    counts = np.sum(observed, axis=2)
    return SimpleNamespace(
        persistent_covariance=power.PERSISTENT_CHANNEL_COVARIANCE.copy(),
        call_covariance=CALL_CHANNEL_COVARIANCE.copy(),
        request_covariance=REQUEST_CHANNEL_COVARIANCE.copy(),
        repeat_covariance=(
            CALL_CHANNEL_COVARIANCE + REQUEST_CHANNEL_COVARIANCE
        ),
        evaluate_repeat_counts=counts,
        evaluate_observed_calls=observed,
    )


def dense_prompt_score_reference(residuals, nuisance, kernel, rho, states):
    positions = np.arange(len(residuals))
    covariance = prompt_block_candidate_covariance(
        nuisance, kernel, rho, positions
    )
    residual_vector = residuals.reshape(-1)
    residual_score = 0.5 * (
        residual_vector @ np.linalg.solve(covariance, residual_vector)
        + np.linalg.slogdet(covariance)[1]
        + len(residual_vector) * np.log(2.0 * np.pi)
    ) / len(residual_vector)

    component_design = np.kron(np.eye(3), power.MEASUREMENT_DESIGN)
    component_prior = np.kron(
        np.eye(3), power.PRIOR_STATE_COVARIANCE
    )
    design = np.kron(np.eye(len(residuals)), component_design)
    prior = np.kron(np.eye(len(residuals)), component_prior)
    truth = states.reshape(-1)
    observed = design @ truth + residual_vector
    solved_design = np.linalg.solve(covariance, design)
    precision = np.linalg.solve(prior, np.eye(len(prior))) + design.T @ solved_design
    posterior_covariance = np.linalg.solve(precision, np.eye(len(precision)))
    posterior_mean = posterior_covariance @ (
        design.T @ np.linalg.solve(covariance, observed)
    )
    delta = truth - posterior_mean
    posterior_score = 0.5 * (
        delta @ precision @ delta
        + np.linalg.slogdet(posterior_covariance)[1]
        + len(delta) * np.log(2.0 * np.pi)
    ) / len(delta)
    return residual_score, posterior_score


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


@pytest.mark.parametrize("component_count", [1, 2, 10])
def test_compound_symmetric_fast_scores_match_dense_for_shared_varying_counts(
    component_count,
    monkeypatch,
):
    observed = shared_varying_schedule(component_count)
    nuisance = score_only_nuisance(observed)
    rng = np.random.default_rng(3100 + component_count)
    residuals = rng.standard_normal((component_count, 3, 4))
    states = rng.standard_normal((component_count, 3, 2))
    blocks = np.zeros(component_count, dtype=int)
    kernel = gamma_item_kernel(0.75)
    rho = 0.10
    dense_residual, dense_posterior = dense_prompt_score_reference(
        residuals, nuisance, kernel, rho, states
    )

    def forbidden_dense_fallback(*_args, **_kwargs):
        raise AssertionError("exchangeable schedule used the dense fallback")

    monkeypatch.setattr(
        power, "_dense_prompt_block_gaussian_score", forbidden_dense_fallback
    )
    monkeypatch.setattr(
        power,
        "_dense_prompt_block_posterior_state_score",
        forbidden_dense_fallback,
    )
    fast_residual = prompt_block_gaussian_nll(
        residuals, nuisance, kernel, rho, blocks
    )
    fast_posterior = prompt_block_posterior_state_nll(
        residuals, nuisance, kernel, rho, states, blocks
    )
    np.testing.assert_allclose(
        fast_residual,
        np.full(component_count, dense_residual),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        fast_posterior,
        np.full(component_count, dense_posterior),
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize("schedule_case", ["unequal_counts", "different_schedule"])
def test_nonexchangeable_schedules_use_dense_fallback_and_remain_exact(
    schedule_case,
    monkeypatch,
):
    observed = shared_varying_schedule(3)
    if schedule_case == "unequal_counts":
        observed[1, 0, 4, 0] = False
    else:
        # Preserve the count while changing which repeat was recorded.  With
        # three components this creates nonconstant pairwise overlaps.
        observed[1, 0, 0, 1] = False
        observed[1, 0, 4, 1] = True
    nuisance = score_only_nuisance(observed)
    rng = np.random.default_rng(3200 + int(schedule_case == "different_schedule"))
    residuals = rng.standard_normal((3, 3, 4))
    states = rng.standard_normal((3, 3, 2))
    blocks = np.zeros(3, dtype=int)
    kernel = gamma_item_kernel(0.25)
    rho = 0.05
    dense_residual, dense_posterior = dense_prompt_score_reference(
        residuals, nuisance, kernel, rho, states
    )
    assert power._exchangeable_prompt_schedule(nuisance, np.arange(3)) is None

    fallback_calls = {"residual": 0, "posterior": 0}
    original_residual = power._dense_prompt_block_gaussian_score
    original_posterior = power._dense_prompt_block_posterior_state_score

    def residual_spy(*args, **kwargs):
        fallback_calls["residual"] += 1
        return original_residual(*args, **kwargs)

    def posterior_spy(*args, **kwargs):
        fallback_calls["posterior"] += 1
        return original_posterior(*args, **kwargs)

    monkeypatch.setattr(power, "_dense_prompt_block_gaussian_score", residual_spy)
    monkeypatch.setattr(
        power, "_dense_prompt_block_posterior_state_score", posterior_spy
    )
    actual_residual = prompt_block_gaussian_nll(
        residuals, nuisance, kernel, rho, blocks
    )
    actual_posterior = prompt_block_posterior_state_nll(
        residuals, nuisance, kernel, rho, states, blocks
    )
    assert fallback_calls == {"residual": 1, "posterior": 1}
    np.testing.assert_allclose(
        actual_residual, np.full(3, dense_residual), rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        actual_posterior, np.full(3, dense_posterior), rtol=2e-12, atol=2e-12
    )


def test_exchangeable_mode_factorizations_are_cached_by_exact_signature(monkeypatch):
    observed = shared_varying_schedule(4)
    nuisance = score_only_nuisance(observed)
    rng = np.random.default_rng(3300)
    residuals = rng.standard_normal((4, 3, 4))
    states = rng.standard_normal((4, 3, 2))
    blocks = np.repeat(np.arange(2), 2)
    kernel = gamma_item_kernel(0.5)
    calls = {"residual": 0, "posterior": 0}
    original_residual = power._factor_compound_symmetric_gaussian_modes
    original_posterior = power._factor_compound_symmetric_posterior_modes

    def residual_spy(*args, **kwargs):
        calls["residual"] += 1
        return original_residual(*args, **kwargs)

    def posterior_spy(*args, **kwargs):
        calls["posterior"] += 1
        return original_posterior(*args, **kwargs)

    monkeypatch.setattr(
        power, "_factor_compound_symmetric_gaussian_modes", residual_spy
    )
    monkeypatch.setattr(
        power, "_factor_compound_symmetric_posterior_modes", posterior_spy
    )
    prompt_block_gaussian_nll(residuals, nuisance, kernel, 0.10, blocks)
    prompt_block_posterior_state_nll(
        residuals, nuisance, kernel, 0.10, states, blocks
    )
    assert calls == {"residual": 1, "posterior": 1}


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


def test_inner_search_recomputes_dense_before_a_last_bit_zero_decision(monkeypatch):
    geometry, splits = geometry_and_splits(36, seed=61)
    field = draw_repeated_field(
        geometry, SCENARIO_BY_NAME["block_null"], repeats=3, seed=62
    )
    tolerance = power.DENSE_DECISION_RECHECK_ATOL
    dense_calls = 0

    def fast_score(residuals, _nuisance, _kernel, rho, _blocks):
        value = -0.5 * tolerance if rho > 0.0 else 0.0
        return np.full(len(residuals), value)

    def dense_score(residuals, _nuisance, _kernel, rho, _blocks):
        nonlocal dense_calls
        dense_calls += 1
        value = 0.5 * tolerance if rho > 0.0 else 0.0
        return np.full(len(residuals), value)

    monkeypatch.setattr(power, "prompt_block_gaussian_nll", fast_score)
    monkeypatch.setattr(power, "_dense_prompt_block_gaussian_nll", dense_score)
    search = inner_candidate_search(
        field,
        geometry,
        splits.outer[0],
        gammas=(0.5,),
        rhos=(0.0, 0.10),
    )
    candidate_summary = next(
        row for row in search.summaries if not row.candidate.is_block
    )
    assert dense_calls == 6  # block and candidate in each of three folds
    assert candidate_summary.fold_gains == pytest.approx(
        (-0.5 * tolerance,) * 3
    )
    assert candidate_summary.positive_folds == 0
    assert search.selected == BLOCK_CANDIDATE


def test_dense_zero_guard_is_a_trigger_not_a_changed_threshold(monkeypatch):
    geometry, splits = geometry_and_splits(36, seed=63)
    field = draw_repeated_field(
        geometry, SCENARIO_BY_NAME["block_null"], repeats=3, seed=64
    )
    gain = 2.0 * power.DENSE_DECISION_RECHECK_ATOL

    def fast_score(residuals, _nuisance, _kernel, rho, _blocks):
        return np.full(len(residuals), -gain if rho > 0.0 else 0.0)

    def forbidden_dense_score(*_args, **_kwargs):
        raise AssertionError("a gain outside the numerical guard was recomputed")

    monkeypatch.setattr(power, "prompt_block_gaussian_nll", fast_score)
    monkeypatch.setattr(
        power, "_dense_prompt_block_gaussian_nll", forbidden_dense_score
    )
    search = inner_candidate_search(
        field,
        geometry,
        splits.outer[0],
        gammas=(0.5,),
        rhos=(0.0, 0.10),
    )
    assert search.selected == Candidate(0.5, 0.10)
    assert search.maximum_eligible_gain == pytest.approx(gain)


@pytest.mark.parametrize(
    ("seed", "scenario_name"),
    [
        (71, "block_null"),
        (72, "cumulative_rho_0.10"),
        (73, "mixture_rho_0.20"),
    ],
)
def test_inner_search_selection_matches_forced_dense_scoring(
    seed,
    scenario_name,
    monkeypatch,
):
    geometry, splits = geometry_and_splits(36, seed=seed)
    field = draw_repeated_field(
        geometry,
        SCENARIO_BY_NAME[scenario_name],
        repeats=4,
        seed=seed + 200,
        missing_rate=0.15,
    )
    fast = inner_candidate_search(field, geometry, splits.outer[0])
    with monkeypatch.context() as context:
        context.setattr(
            power,
            "_exchangeable_prompt_schedule",
            lambda _nuisance, _positions: None,
        )
        dense = inner_candidate_search(field, geometry, splits.outer[0])

    assert fast.selected == dense.selected
    assert fast.maximum_eligible_gain == pytest.approx(
        dense.maximum_eligible_gain, abs=1e-12
    )
    for fast_summary, dense_summary in zip(fast.summaries, dense.summaries):
        assert fast_summary.candidate == dense_summary.candidate
        assert fast_summary.positive_folds == dense_summary.positive_folds
        assert fast_summary.macro_gain == pytest.approx(
            dense_summary.macro_gain, abs=1e-12
        )
        np.testing.assert_allclose(
            fast_summary.fold_gains,
            dense_summary.fold_gains,
            rtol=0.0,
            atol=1e-12,
        )


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
