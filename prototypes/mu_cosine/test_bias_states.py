"""Synthetic tests for fit_bias_states.py (spec: DESIGN_bias_state_augmentation.md §5.1).

Everything here is deterministic and self-contained: a tiny synthetic graph exercises each bin
(h1..h5, sib, cous, rand) plus the explicit `missing` state, and known injected offsets validate
recovery, shrinkage, the fail-closed information floor, and the affine-first calibration order.
"""
import numpy as np
import pytest

from fit_bias_states import (
    BINS,
    fit_bias_states,
    fit_channel_offsets,
    pair_distance_features,
    soft_bin_weights,
    stratum_sign_table,
    tune_bandwidth,
)

BIN_INDEX = {b: i for i, b in enumerate(BINS)}


def synthetic_graph_pairs(n_per_bin=40):
    """A graph with one exemplar relation per stratum, replicated n_per_bin times per stratum.

    Replicas share no nodes across strata; every non-missing node has a parent chain so it is
    'known'.  Returns (parents, pairs, tags) with tags in BINS order (missing included).
    """
    parents = {}
    pairs, tags = [], []
    for rep in range(n_per_bin):
        for h in range(1, 6):
            chain = [f"h{h}_r{rep}_n{k}" for k in range(h + 1)]
            for lo, hi in zip(chain[:-1], chain[1:]):
                parents[lo] = [hi]
            pairs.append((chain[0], chain[-1]))
            tags.append(f"h{h}")
        p = f"sib_r{rep}_p"
        a, b = f"sib_r{rep}_a", f"sib_r{rep}_b"
        parents[a] = [p]
        parents[b] = [p]
        parents[p] = [f"sib_r{rep}_top"]
        pairs.append((a, b))
        tags.append("sib")
        g = f"cous_r{rep}_g"
        pa, pb = f"cous_r{rep}_pa", f"cous_r{rep}_pb"
        ca, cb = f"cous_r{rep}_ca", f"cous_r{rep}_cb"
        parents[pa] = [g]
        parents[pb] = [g]
        parents[ca] = [pa]
        parents[cb] = [pb]
        pairs.append((ca, cb))
        tags.append("cous")
        ra, rb = f"rand_r{rep}_a", f"rand_r{rep}_b"
        parents[ra] = [f"rand_r{rep}_atop"]
        parents[rb] = [f"rand_r{rep}_btop"]
        pairs.append((ra, rb))
        tags.append("rand")
        pairs.append((f"ghost_r{rep}_a", f"ghost_r{rep}_b"))
        tags.append("missing")
    return parents, pairs, tags


def test_features_and_hard_weights_hit_expected_bins():
    parents, pairs, tags = synthetic_graph_pairs(n_per_bin=3)
    feats = pair_distance_features(parents, pairs)
    W = soft_bin_weights(feats, tau=1e-6)
    assert np.allclose(W.sum(axis=1), 1.0)
    for row, tag in enumerate(tags):
        assert W[row].argmax() == BIN_INDEX[tag], f"row {row} ({tag}) landed in {BINS[W[row].argmax()]}"
        assert W[row].max() > 0.999  # tau→0 is the hard-switching special case


def test_missing_state_never_maps_to_rand():
    parents, pairs, tags = synthetic_graph_pairs(n_per_bin=2)
    feats = pair_distance_features(parents, pairs)
    missing_rows = [i for i, t in enumerate(tags) if t == "missing"]
    for tau in (1e-6, 0.5, 1.5):
        W = soft_bin_weights(feats, tau)
        assert np.allclose(W[missing_rows, BIN_INDEX["missing"]], 1.0)
        assert np.allclose(W[missing_rows, BIN_INDEX["rand"]], 0.0)
        # and no non-missing row leaks into the missing state
        other = [i for i, t in enumerate(tags) if t != "missing"]
        assert np.allclose(W[other, BIN_INDEX["missing"]], 0.0)


def test_soft_weights_are_deterministic_and_share_across_neighbors():
    parents, pairs, _ = synthetic_graph_pairs(n_per_bin=2)
    feats = pair_distance_features(parents, pairs)
    W1 = soft_bin_weights(feats, tau=0.75)
    W2 = soft_bin_weights(feats, tau=0.75)
    assert np.array_equal(W1, W2)
    h2 = feats[:, 0] > 0.5
    h2 &= feats[:, 1] == 2.0
    row = np.where(h2)[0][0]
    assert W1[row, BIN_INDEX["h2"]] > W1[row, BIN_INDEX["h1"]] > 0.0
    assert W1[row, BIN_INDEX["h3"]] > 0.0  # neighbors borrow strength through the design


def test_offset_recovery_signs_and_magnitudes():
    parents, pairs, tags = synthetic_graph_pairs(n_per_bin=60)
    feats = pair_distance_features(parents, pairs)
    true = dict(h1=0.09, h2=0.06, h3=0.04, h4=0.03, h5=0.02, sib=-0.06, cous=-0.05, rand=-0.01,
                missing=0.0)
    rng = np.random.default_rng(7)
    resid = np.array([true[t] for t in tags]) + rng.normal(0.0, 0.05, len(tags))
    train = np.arange(len(pairs))
    states = fit_bias_states(feats, train, {("luna", "D"): resid}, prior_sd=0.10, verbose=False)
    # With a smoothing bandwidth the coefficients are kernel weights, not bin means; the deployable
    # quantity is the IMPLIED correction W·b on each stratum's rows — assert on that.
    corr = states.corrections(("luna", "D"))
    tags_arr = np.asarray(tags)
    for b, v in true.items():
        implied = float(corr[tags_arr == b].mean())
        if abs(v) >= 0.02:
            assert np.sign(implied) == np.sign(v), f"{b}: implied {implied} vs {v}"
            assert abs(implied - v) < 0.03, f"{b}: implied {implied} vs {v}"
    # debias removes the structure it fitted
    corrected = states.debias(("luna", "D"), resid)
    for b in ("h1", "sib"):
        m = np.asarray(tags) == b
        assert abs(corrected[m].mean()) < abs(resid[m].mean())


def test_fail_closed_zero_support_state_returns_prior():
    parents, pairs, tags = synthetic_graph_pairs(n_per_bin=30)
    keep = [i for i, t in enumerate(tags) if t not in ("h4", "h5", "missing")]
    feats = pair_distance_features(parents, [pairs[i] for i in keep])
    rng = np.random.default_rng(1)
    resid = rng.normal(0.05, 0.05, len(keep))
    W = soft_bin_weights(feats, tau=1e-6)  # hard bins: h4/h5/missing get exactly zero support
    fit = fit_channel_offsets(W, resid, prior_sd=0.10, name="test")
    for b in ("h4", "h5", "missing"):
        k = BIN_INDEX[b]
        assert fit.support[k] == 0.0
        assert fit.fallback[k], f"{b} must fall back to its prior"
        assert fit.offsets[k] == 0.0
        assert fit.info_ratio[k] < 0.10
    assert fit.rank < len(BINS)  # the printed rank exposes the unsupported states
    assert not fit.fallback[BIN_INDEX["h1"]]


def test_shrinkage_pulls_thin_bins_toward_zero():
    parents, pairs, tags = synthetic_graph_pairs(n_per_bin=60)
    feats = pair_distance_features(parents, pairs)
    W = soft_bin_weights(feats, tau=1e-6)
    rng = np.random.default_rng(3)
    resid = rng.normal(0.0, 0.05, len(tags))
    thin = np.asarray(tags) == "h5"
    resid[thin.nonzero()[0][:2]] += 0.30  # 2 loud rows in an otherwise-zero bin
    keep = ~thin
    keep[thin.nonzero()[0][:2]] = True
    fit_thin = fit_channel_offsets(W[keep], resid[keep], prior_sd=0.10, name="thin")
    unshrunk = resid[thin.nonzero()[0][:2]].mean()
    k = BIN_INDEX["h5"]
    # 2 rows against lam = var(resid)/prior_sd² keeps h5 above the 0.10 info floor, so the
    # assertion must actually run — a fallback here would make the test vacuous
    assert not fit_thin.fallback[k]
    assert abs(fit_thin.offsets[k]) < abs(unshrunk)  # partial pooling shrinks 2-row evidence


def test_affine_first_slope_error_is_not_expressed_as_offsets():
    """The #3648 tilt lesson: a slope error must be absorbed by the affine, not the offsets."""
    from run_product_kalman_realdata import affine_calibrate

    parents, pairs, tags = synthetic_graph_pairs(n_per_bin=60)
    feats = pair_distance_features(parents, pairs)
    rng = np.random.default_rng(11)
    y = np.clip(rng.uniform(0.1, 0.9, len(tags)), 0, 1)
    raw = 0.7 * y + 0.2 + rng.normal(0, 0.02, len(tags))  # pure affine tilt, no bin structure
    train = np.arange(len(pairs))
    cal = affine_calibrate(raw[train], y[train], raw)
    states = fit_bias_states(feats, train, {("luna", "D"): cal - y}, prior_sd=0.10, verbose=False)
    # nothing left for the bins once the affine is retained: the implied per-row correction is noise
    assert np.abs(states.corrections(("luna", "D"))).max() < 0.015


def test_bandwidth_tuning_is_deterministic_and_train_only():
    parents, pairs, tags = synthetic_graph_pairs(n_per_bin=40)
    feats = pair_distance_features(parents, pairs)
    rng = np.random.default_rng(5)
    resid = {("luna", "D"): np.array([0.08 if t == "h1" else -0.04 for t in tags])
             + rng.normal(0, 0.03, len(tags))}
    tau1, table1 = tune_bandwidth(feats, resid, prior_sd=0.10)
    tau2, table2 = tune_bandwidth(feats, resid, prior_sd=0.10)
    assert tau1 == tau2 and table1 == table2


def test_stratum_sign_table_matches_injected_signs():
    parents, pairs, tags = synthetic_graph_pairs(n_per_bin=50)
    feats = pair_distance_features(parents, pairs)
    rng = np.random.default_rng(9)
    y = rng.uniform(0.2, 0.8, len(tags))
    bias = np.array([{"h1": 0.09, "sib": -0.06}.get(t, 0.0) for t in tags])
    raw = y + bias + rng.normal(0, 0.02, len(tags))
    from run_product_kalman_realdata import affine_calibrate

    train = np.arange(len(pairs))
    cal = affine_calibrate(raw[train], y[train], raw)
    states = fit_bias_states(feats, train, {("luna", "D"): cal - y}, prior_sd=0.10, verbose=False)
    table = stratum_sign_table(np.asarray(tags), y, raw, cal, states.W,
                               states.fits[("luna", "D")].offsets)
    by = {s: (measured, implied, ok) for s, _, measured, implied, ok in table}
    assert by["h1"][2] and by["h1"][0] > 0
    assert by["sib"][2] and by["sib"][0] < 0


def test_boolean_train_mask_equals_integer_indices():
    parents, pairs, tags = synthetic_graph_pairs(n_per_bin=30)
    feats = pair_distance_features(parents, pairs)
    rng = np.random.default_rng(13)
    resid = {("luna", "D"): rng.normal(0.03, 0.05, len(tags))}
    mask = np.zeros(len(pairs), dtype=bool)
    mask[::2] = True
    st_mask = fit_bias_states(feats, mask, resid, prior_sd=0.10, verbose=False)
    st_idx = fit_bias_states(feats, np.flatnonzero(mask), resid, prior_sd=0.10, verbose=False)
    assert np.array_equal(st_mask.fits[("luna", "D")].offsets, st_idx.fits[("luna", "D")].offsets)


def test_offsets_are_the_coherent_joint_posterior():
    """The prior is the fail-closed mechanism: offsets = the FULL joint ridge posterior mean
    (no column surgery — external review 2026-07-17), with zero-support states exactly at the
    prior mean and the full posterior covariance exposed for downstream propagation."""
    parents, pairs, tags = synthetic_graph_pairs(n_per_bin=40)
    keep = [i for i, t in enumerate(tags) if t != "missing"]
    feats = pair_distance_features(parents, [pairs[i] for i in keep])
    rng = np.random.default_rng(17)
    resid = np.array([0.08 if tags[i].startswith("h") else -0.04 for i in keep])
    resid = resid + rng.normal(0, 0.04, len(keep))
    W = soft_bin_weights(feats, tau=0.75)  # smooth: hop columns overlap
    fit = fit_channel_offsets(W, resid, prior_sd=0.10, info_floor=0.10, name="posterior")
    lam = fit.noise_var / fit.prior_sd**2
    A = W.T @ W + lam * np.eye(len(BINS))
    assert np.allclose(fit.offsets, np.linalg.solve(A, W.T @ resid))
    assert np.allclose(fit.posterior_cov, fit.noise_var * np.linalg.inv(A))
    k = BIN_INDEX["missing"]  # zero support here → posterior exactly equals the prior
    assert fit.offsets[k] == 0.0
    assert np.isclose(fit.posterior_var[k], fit.prior_sd**2)
    assert fit.fallback[k]


def test_correction_var_matches_quadratic_form():
    parents, pairs, tags = synthetic_graph_pairs(n_per_bin=20)
    feats = pair_distance_features(parents, pairs)
    rng = np.random.default_rng(23)
    resid = {("luna", "D"): rng.normal(0.02, 0.05, len(tags))}
    train = np.array([j for j, t in enumerate(tags) if t != "missing"])  # missing UNSEEN in train
    states = fit_bias_states(feats, train, resid, prior_sd=0.10, verbose=False)
    cv = states.correction_var(("luna", "D"))
    P = states.fits[("luna", "D")].posterior_cov
    i = 3
    assert np.isclose(cv[i], states.W[i] @ P @ states.W[i])
    # an unseen missing-state row carries the FULL prior variance — it is not "known-unbiased"
    missing_rows = [j for j, t in enumerate(tags) if t == "missing"]
    assert np.allclose(cv[missing_rows], 0.10**2)


def test_input_validation():
    with pytest.raises(ValueError):
        fit_channel_offsets(np.ones((4, 3)), np.zeros(4))
    with pytest.raises(ValueError):
        fit_channel_offsets(np.ones((4, len(BINS))), np.zeros(5))
    with pytest.raises(ValueError):
        fit_channel_offsets(np.ones((4, len(BINS))), np.full(4, np.nan))
    with pytest.raises(ValueError):
        fit_channel_offsets(np.ones((4, len(BINS))), np.zeros(4), prior_sd=0.0)
