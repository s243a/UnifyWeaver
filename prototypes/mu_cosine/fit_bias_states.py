#!/usr/bin/env python3
"""Batch hierarchical bias fit: per-(judge, distance-bin, channel) offset states.

DESIGN_bias_state_augmentation.md §5.1 (build-plan item 1).  Bias states are residual OFFSETS fitted
ON TOP of the retained global per-channel affine calibration (`affine_calibrate`, train-split-only) —
never in place of it: offsets cannot express a slope error, and dropping the affine would reinvite the
tilt that flipped the #3648 headline.

Gauge: the OPERATING judge (gpt-5.5-low) has its bias pinned to 0 — here structurally, because the
5.5 labels ARE the regression target, so a 5.5 "channel" would have identically-zero residuals.  Every
estimate below is bias RELATIVE TO gpt-5.5-low (fidelity to the operating judge, never "semantic
accuracy"; the absolute frame needs the human-verified gold subset).

Soft distance assignment is a deterministic, OUTCOME-BLIND kernel basis over graph features only
(never labels): bin centers = the campaign strata classes h1..h5 / sib / cous / rand, one shared
bandwidth in center-spacing units, tuned on train rows only.  Rows with NO usable distance signal get
their own explicit `missing` basis state — never silently mapped to `rand` ("no signal" and "measured
unrelated" are different facts).  Overlapping columns make the bias ESTIMATES share information across
neighboring bins (kernel smoothing through the design); they are not correlated measurement noise.

The fit is one small ridge regression per (judge, channel): the weak zero prior b ~ N(0, prior_sd²)
is the pseudo-measurement form of hierarchical partial pooling, and with identity transition and no
process noise this batch fit equals the sequential filter's posterior (design §1, "Transition").

Diagnostics printed per fit, FAIL CLOSED: unregularized design rank (kernel columns overlap, so Σw is
support mass, NOT effective sample size — rank is the honest check), per-state conditional posterior
variance (as an information ratio against the prior), and the design condition number.  The closing
mechanism is the prior itself: the applied offsets are the coherent joint ridge posterior, so a state
with no information keeps EXACTLY its prior (mean 0, variance prior_sd²) and weakly-informed states
are shrunk in proportion; states below the information floor are flagged in the diagnostics.  The
full posterior covariance is exposed (`posterior_cov`, `BiasStates.correction_var`) for downstream
uncertainty propagation (the square-root/QR lane).

Importable API (target-factory use):

    feats  = pair_distance_features(parents, pairs)               # outcome-blind, once per corpus
    states = fit_bias_states(feats, train_idx, residuals={("luna", "D"): resid, ...})
    luna_D_debiased = states.debias(("luna", "D"), luna_D_affine_calibrated)

CLI (standalone fit on the dual-judge campaign; prints the per-bin posteriors and the sign check
against the measured stratum-bias table):

    python3 fit_bias_states.py [--prior-sd 0.10] [--info-floor 0.10] [--seed 0]
"""
import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BINS = ("h1", "h2", "h3", "h4", "h5", "sib", "cous", "rand", "missing")
HOP_SLICE = slice(0, 5)          # h1..h5: ancestor pairs, center = hop distance 1..5
LAT_SLICE = slice(5, 8)          # sib/cous/rand: non-ancestor pairs, centers on the d_sym coordinate
MISSING_COL = 8
DEFAULT_TAUS = (1e-6, 0.25, 0.5, 0.75, 1.0, 1.5)
LOG2PI = float(np.log(2 * np.pi))


def pair_distance_features(parents, pairs, hmax=6, cap=13, in_graph=None):
    """Outcome-blind per-pair distance features: [is_anc, d_up, d_sym, known] as a float array.

    is_anc: either endpoint is an ancestor of the other within hmax; d_up = that hop distance (nan
    otherwise).  d_sym = min common-ancestor hop sum (self counts at 0), capped at `cap`; nan when no
    common ancestor is found.  known = both endpoints are in the graph — a pair with known=0 and no
    measured relation has NO usable distance signal (the `missing` state); a pair with known=1 and no
    common ancestor within the horizon was MEASURED unrelated (the `rand` neighborhood).
    Mirrors run_sym_channel_fusion.sym_graph_features (same ancestors() walk, same cap/hmax defaults)
    but keeps the raw distances that the kernel basis needs.
    """
    from sample_channel_campaign import ancestors

    if in_graph is None:
        in_graph = {n for n, ps in parents.items() if ps} | {p for ps in parents.values() for p in ps}
    feats = np.zeros((len(pairs), 4))
    anc_cache = {}

    def anc(n):
        if n not in anc_cache:
            a = ancestors(parents, n, hmax)
            a[n] = 0
            anc_cache[n] = a
        return anc_cache[n]

    for i, (x, y) in enumerate(pairs):
        ax, ay = anc(x), anc(y)
        d_up = ax.get(y, ay.get(x))
        is_anc = d_up is not None and d_up > 0
        common = set(ax) & set(ay)
        d_sym = min((ax[c] + ay[c] for c in common), default=None)
        known = x in in_graph and y in in_graph
        feats[i] = (
            float(is_anc),
            float(d_up) if is_anc else np.nan,
            min(float(d_sym), float(cap)) if d_sym is not None else (float(cap) if known else np.nan),
            float(known),
        )
    return feats


def _lateral_coord(d_sym, cap):
    """Map d_sym onto a unit-spaced lateral coordinate: sib(d_sym=2)→1, cous(4)→2, rand(cap)→3."""
    d = np.minimum(d_sym, cap)
    return np.where(d <= 4.0, d / 2.0, 2.0 + (d - 4.0) / (cap - 4.0))


def soft_bin_weights(feats, tau, cap=13):
    """Deterministic soft assignment over BINS from distance features alone (never outcomes).

    Ancestor rows spread over h1..h5 by a Gaussian kernel in hop units — the hop coordinate is
    clamped at 5, so h5 explicitly means "5 OR MORE hops" (ancestors() walks to hmax=6; without the
    clamp a 6-hop row would be asymmetrically down-weighted instead of landing on the deepest bin).
    Non-ancestor rows spread over sib/cous/rand by the same kernel on the unit-spaced lateral
    coordinate; rows without usable signal are one-hot `missing`.  Weights are normalized within the
    row's family, so tau→0 recovers hard nearest-center switching (the bandwidth→0 special case).
    """
    tau = max(float(tau), 1e-9)
    W = np.zeros((len(feats), len(BINS)))
    is_anc = feats[:, 0] > 0.5
    no_signal = ~is_anc & ~np.isfinite(feats[:, 2])

    def kernel_rows(rows, coords, centers, cols):
        if len(rows):
            logw = -0.5 * ((coords[:, None] - centers[None, :]) / tau) ** 2
            logw -= logw.max(axis=1, keepdims=True)
            w = np.exp(logw)
            W[rows, cols] = w / w.sum(axis=1, keepdims=True)

    hop_rows = np.where(is_anc)[0]
    kernel_rows(hop_rows, np.minimum(feats[hop_rows, 1], 5.0), np.arange(1.0, 6.0), HOP_SLICE)
    lat_rows = np.where(~is_anc & ~no_signal)[0]
    kernel_rows(lat_rows, _lateral_coord(feats[lat_rows, 2], cap), np.array([1.0, 2.0, 3.0]), LAT_SLICE)
    W[no_signal, MISSING_COL] = 1.0
    return W


@dataclass
class ChannelBiasFit:
    """Shrunk per-bin offsets for one (judge, channel), with fail-closed diagnostics.

    `offsets` is the coherent joint ridge posterior mean; `fallback` flags prior-dominated states
    (info below floor — their offsets are ≈0 by shrinkage, exactly 0 at zero support).
    `posterior_cov` is the full state covariance for downstream uncertainty propagation
    (per-row correction variance = w·P_b·wᵀ; see BiasStates.correction_var).
    """

    name: str
    offsets: np.ndarray            # joint ridge posterior mean per state
    posterior_cov: np.ndarray      # full posterior covariance of the states (len(BINS)²)
    posterior_var: np.ndarray      # its diagonal, kept for cheap access
    info_ratio: np.ndarray         # 1 - posterior_var/prior_var (0 = prior only, 1 = data-determined)
    fallback: np.ndarray           # bool: prior-dominated (info below floor); flagged, not zeroed
    support: np.ndarray            # Σw per state (support mass, NOT effective sample size)
    rank: int                      # unregularized design rank of WᵀW
    cond: float                    # condition number of the supported unregularized design
    noise_var: float
    prior_sd: float
    n_train: int

    def diagnostics_lines(self):
        lines = [
            f"[{self.name}] n_train={self.n_train} rank={self.rank}/{int((self.support > 1e-9).sum())} "
            f"supported states, cond={self.cond:.1f}, noise_var={self.noise_var:.5f}, "
            f"prior_sd={self.prior_sd}, prior-dominated={int(self.fallback.sum())}"
        ]
        if self.rank < int((self.support > 1e-9).sum()):
            lines.append("    WARNING: unregularized design is rank-deficient over its supported "
                         "states — bin estimates are identified only through the prior")
        for k, b in enumerate(BINS):
            flag = " → prior-dominated (info below floor)" if self.fallback[k] else ""
            lines.append(
                f"    {b:8s} offset={self.offsets[k]:+.4f} "
                f"post_var={self.posterior_var[k]:.2e} info={self.info_ratio[k]:.2f} "
                f"support={self.support[k]:.1f}{flag}"
            )
        return lines


def fit_channel_offsets(W, resid, prior_sd=0.10, noise_var=None, info_floor=0.10, name=""):
    """One ridge fit: resid ≈ W·b with the weak zero prior b ~ N(0, prior_sd²) per state.

    `noise_var` defaults to var(resid) — an upper bound on the row noise (it still contains the bin
    structure), so the implied shrinkage is conservative.

    Fail-closed semantics (external review, 2026-07-17): the PRIOR ITSELF is the fallback mechanism.
    The returned offsets are the coherent joint ridge posterior mean — a state with zero support has
    posterior exactly equal to its prior (offset exactly 0), and a weakly-informed state is shrunk
    toward 0 in proportion to its missing information.  States below `info_floor` are FLAGGED
    (`fallback`) so callers and diagnostics can see which estimates are prior-dominated; they are
    NOT surgically zeroed or dropped — a point mass at 0 is a claim the model does not make, and
    removing columns would break the batch-fit ≡ sequential-posterior equivalence the design relies
    on.  The full posterior covariance is returned for downstream uncertainty propagation.
    """
    W = np.asarray(W, dtype=float)
    r = np.asarray(resid, dtype=float)
    if W.ndim != 2 or W.shape[1] != len(BINS):
        raise ValueError(f"W must be (n, {len(BINS)})")
    if r.shape != (W.shape[0],):
        raise ValueError("resid must align with W rows")
    if not np.isfinite(W).all() or not np.isfinite(r).all():
        raise ValueError("W and resid must be finite")
    if prior_sd <= 0:
        raise ValueError("prior_sd must be positive")

    if noise_var is None:
        noise_var = float(max(r.var(), 1e-8))
    lam = noise_var / (prior_sd**2)
    G = W.T @ W
    A = G + lam * np.eye(len(BINS))
    b = np.linalg.solve(A, W.T @ r)
    posterior_cov = noise_var * np.linalg.inv(A)
    posterior_var = np.diag(posterior_cov).copy()
    info_ratio = 1.0 - posterior_var / (prior_sd**2)
    fallback = info_ratio < info_floor

    support = W.sum(axis=0)
    supported = support > 1e-9
    rank = int(np.linalg.matrix_rank(G))
    if supported.any():
        eig = np.linalg.eigvalsh(G[np.ix_(supported, supported)])
        cond = float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf")
    else:
        cond = float("nan")
    return ChannelBiasFit(
        name=name,
        offsets=b,
        posterior_cov=posterior_cov,
        posterior_var=posterior_var,
        info_ratio=info_ratio,
        fallback=fallback,
        support=support,
        rank=rank,
        cond=cond,
        noise_var=noise_var,
        prior_sd=float(prior_sd),
        n_train=W.shape[0],
    )


def cv_fold_ids(pairs, folds):
    """Deterministic ENDPOINT-DISJOINT fold assignment for pair rows (no RNG).

    Each node is hashed (crc32) into one of `folds` groups; a pair participates in CV only when
    BOTH endpoints fall in the same group (fold = that group), else it gets -1 (excluded from CV
    scoring but still used in the final full-train fit).  This is the inner-CV analog of the outer
    node-disjoint split: same-node residuals are correlated, and letting a shared endpoint straddle
    fit/validation folds would bias the bandwidth toward overfit values.
    """
    import zlib

    def h(node):
        return zlib.crc32(repr(node).encode()) % folds

    return np.array([h(x) if h(x) == h(y) else -1 for x, y in pairs], dtype=int)


def tune_bandwidth(feats_train, residuals_train, taus=DEFAULT_TAUS, prior_sd=0.10, folds=5, cap=13,
                   info_floor=0.10, pairs=None):
    """Pick the shared kernel bandwidth by deterministic K-fold CV on TRAIN rows only.

    The row→bin map stays a function of graph features alone for every candidate tau; train labels
    enter only through the CV score of the ridge fit, so the assignment remains outcome-blind.
    Deterministic, no RNG: when `pairs` (the rows' endpoint tuples) is given, folds are
    endpoint-disjoint via cv_fold_ids (cross-fold pairs are excluded from scoring); else folds fall
    back to row index modulo `folds`.  CV uses the same shrinkage AND the same information floor as
    the deployed fit, so the selected tau optimizes the estimator that is actually applied.
    (Known, accepted residue: the per-channel affine calibration behind `residuals_train` is fit on
    the full outer-train set, not per inner fold — a 2-parameter calibrator on ~350 rows; the outer
    held-node evaluation is unaffected.)
    """
    n = len(feats_train)
    if pairs is not None:
        if len(pairs) != n:
            raise ValueError("pairs must align with feats_train rows")
        fold_id = cv_fold_ids(pairs, folds)
    else:
        fold_id = np.arange(n) % folds
    best_tau, best_score, table = None, None, []
    for tau in taus:
        W = soft_bin_weights(feats_train, tau, cap=cap)
        score = 0.0
        for r in residuals_train.values():
            r = np.asarray(r, dtype=float)
            for f in range(folds):
                fit_rows, val_rows = (fold_id != f) & (fold_id >= 0), fold_id == f
                if not val_rows.any() or not fit_rows.any():
                    continue
                cf = fit_channel_offsets(W[fit_rows], r[fit_rows], prior_sd=prior_sd,
                                         info_floor=info_floor)
                pred = W[val_rows] @ cf.offsets
                score += float(((r[val_rows] - pred) ** 2).sum())
        table.append((float(tau), score))
        if best_score is None or score < best_score:
            best_tau, best_score = float(tau), score
    return best_tau, table


@dataclass
class BiasStates:
    """Fitted bias states for a set of (judge, channel) keys over one corpus' pairs."""

    tau: float
    W: np.ndarray                                  # soft bin weights for ALL rows (train + held)
    fits: dict = field(default_factory=dict)       # (judge, channel) -> ChannelBiasFit
    cv_table: list = field(default_factory=list)

    def corrections(self, key):
        return self.W @ self.fits[key].offsets

    def correction_var(self, key):
        """Per-row variance of the applied correction, w·P_b·wᵀ (marginal, no cross-row terms).

        Downstream fusion that wants to price bias-estimation uncertainty should ADD this to the
        channel's measurement variance (a `missing` row then carries the full prior variance
        rather than being treated as known-unbiased).  The step-1 ladder does not yet consume it —
        cross-row covariance and the square-root/QR propagation are the follow-up lanes.
        """
        P = self.fits[key].posterior_cov
        return np.einsum("ij,jk,ik->i", self.W, P, self.W)

    def debias(self, key, values):
        """Apply the shrunk offsets on top of the (already affine-calibrated) channel readings."""
        return np.asarray(values, dtype=float) - self.corrections(key)

    def diagnostics_lines(self):
        lines = [f"bias states: tau={self.tau} (train-tuned), bins={','.join(BINS)}"]
        for fit in self.fits.values():
            lines.extend(fit.diagnostics_lines())
        return lines


def fit_bias_states(feats, train_idx, residuals, prior_sd=0.10, info_floor=0.10,
                    taus=DEFAULT_TAUS, cap=13, cv_pairs=None, verbose=True):
    """Fit shrunk per-bin offsets for every (judge, channel) residual series.

    `feats` covers ALL rows (pair_distance_features); `residuals` maps (judge, channel) to
    affine-calibrated-residual arrays over all rows.  Fitting (bandwidth + offsets) uses `train_idx`
    rows only (integer indices or a boolean mask); the returned object applies corrections to any
    row through the outcome-blind W.  `cv_pairs` (optional, aligned with ALL rows) gives each row's
    endpoint tuple for endpoint-disjoint CV folds — see tune_bandwidth.
    """
    feats = np.asarray(feats, dtype=float)
    train_idx = np.asarray(train_idx)
    if train_idx.dtype == bool:
        if len(train_idx) != len(feats):
            raise ValueError("boolean train_idx mask must align with feats rows")
        train_idx = np.flatnonzero(train_idx)
    else:
        train_idx = train_idx.astype(int)
    res_train = {k: np.asarray(v, dtype=float)[train_idx] for k, v in residuals.items()}
    pairs_train = None
    if cv_pairs is not None:
        if len(cv_pairs) != len(feats):
            raise ValueError("cv_pairs must align with feats rows")
        pairs_train = [cv_pairs[i] for i in train_idx]
    tau, cv_table = tune_bandwidth(feats[train_idx], res_train, taus=taus, prior_sd=prior_sd,
                                   cap=cap, info_floor=info_floor, pairs=pairs_train)
    W = soft_bin_weights(feats, tau, cap=cap)
    states = BiasStates(tau=tau, W=W, cv_table=cv_table)
    for key, r in res_train.items():
        name = f"{key[0]}.{key[1]}" if isinstance(key, tuple) else str(key)
        states.fits[key] = fit_channel_offsets(
            W[train_idx], r, prior_sd=prior_sd, info_floor=info_floor, name=name
        )
    if verbose:
        for line in states.diagnostics_lines():
            print(line)
    return states


def stratum_sign_table(tags, y, raw, calibrated, W, offsets):
    """Measured raw per-stratum bias vs the model-implied bias (affine part + shrunk offsets).

    measured_s = mean_s(raw − y);  implied_s = mean_s(raw − calibrated) + mean_s(W)·b.
    The SIGN agreement of these two columns is the acceptance check against the
    REPORT_luna_campaign §1 stratum table (offsets alone are residuals on top of the affine and are
    not comparable to the raw table).
    """
    tags = np.asarray(tags)
    rows = []
    for s in sorted(set(tags.tolist())):
        m = tags == s
        measured = float(np.mean(raw[m] - y[m]))
        implied = float(np.mean(raw[m] - calibrated[m]) + W[m].mean(axis=0) @ offsets)
        rows.append((s, int(m.sum()), measured, implied, np.sign(measured) == np.sign(implied)))
    return rows


# --------------------------------------------------------------------------------------------------
# Standalone CLI: fit on the dual-judge campaign and print the acceptance sign check.
# --------------------------------------------------------------------------------------------------

def _require(path, what):
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {what}: {path}\n"
            "The dual-judge campaign artifacts live in /tmp/mu_data (see HANDOFF_PR3648.md §3 for "
            "md5 provenance). If /tmp was cleared, restore from backup or regenerate via "
            "sample_channel_campaign.py + score_with_codex.py (scoring costs money)."
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--luna-campaign", default="/tmp/mu_data/campaign_scored_luna.tsv")
    ap.add_argument("--seed", type=int, default=0, help="node-disjoint split seed for the train fit")
    ap.add_argument("--prior-sd", type=float, default=0.10)
    ap.add_argument("--info-floor", type=float, default=0.10)
    ap.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    a = ap.parse_args(argv)

    from eval_luna_transfer import load_luna
    from fine_tune_channel_heads import CAMPAIGN_SCORED, load_campaign_datasets
    from node_disjoint_eval import node_disjoint_pair_split
    from run_product_kalman_logit import dequant
    from run_product_kalman_realdata import DATASETS, affine_calibrate
    from run_sym_channel_fusion import calibrate_luna, sym_graph_features
    from sigma_hop_confirmatory import FeatureGraphConfig, load_feature_graph

    _require(CAMPAIGN_SCORED, "gpt-5.5 campaign labels")
    _require(a.luna_campaign, "luna campaign labels")
    lp, lD, lS = load_luna(a.luna_campaign)
    luna_by = {p: (lD[i], lS[i]) for i, p in enumerate(lp)}
    dss = load_campaign_datasets()

    for name, ds in dss.items():
        corpus = name.replace("-campaign", "")
        parents, children, _, _ = load_feature_graph(FeatureGraphConfig(**DATASETS[corpus]["graph"]))
        in_graph = set(parents) | {c for kids in children.values() for c in kids}
        keep = [i for i, pair in enumerate(ds["pairs"]) if pair in luna_by]
        pairs = [ds["pairs"][i] for i in keep]
        tags = [ds["tags"][i].replace("campaign_", "") for i in keep]
        y = dequant(np.column_stack([ds["D"][keep], ds["S"][keep]]))
        d = ds["d"][keep]
        luna = np.array([luna_by[pair] for pair in pairs])
        split = node_disjoint_pair_split(pairs, a.seed, strata=tags)
        tr = split.train

        print(f"\n=== {name}: {len(pairs)} dual-judge rows; train={len(tr)} "
              f"(node-disjoint seed {a.seed}) ===")
        graph_D = affine_calibrate(d[tr], y[tr, 0], d)
        F = sym_graph_features(parents, pairs)
        X = np.column_stack([F, np.ones(len(F))])
        beta, *_ = np.linalg.lstsq(X[tr], y[tr, 1], rcond=None)
        graph_S = X @ beta
        luna_cal = calibrate_luna(luna, y, tr)

        feats = pair_distance_features(parents, pairs, in_graph=in_graph)
        residuals = {
            ("luna", "D"): luna_cal[:, 0] - y[:, 0],
            ("luna", "S"): luna_cal[:, 1] - y[:, 1],
            ("graph", "D"): graph_D - y[:, 0],
            ("graph", "S"): graph_S - y[:, 1],
        }
        states = fit_bias_states(feats, tr, residuals, prior_sd=a.prior_sd,
                                 info_floor=a.info_floor, taus=a.taus,
                                 cv_pairs=pairs)

        print("\nsign check vs measured stratum bias (train rows; bias relative to gpt-5.5-low):")
        for (judge, ch), raw_col, cal_col, y_col in [
            (("luna", "D"), luna[:, 0], luna_cal[:, 0], y[:, 0]),
            (("luna", "S"), luna[:, 1], luna_cal[:, 1], y[:, 1]),
        ]:
            print(f"  {judge}.{ch}:  stratum   n     measured   implied   sign-match")
            table = stratum_sign_table(
                np.asarray(tags)[tr], y_col[tr], raw_col[tr], cal_col[tr],
                states.W[tr], states.fits[(judge, ch)].offsets,
            )
            for s, n_s, measured, implied, ok in table:
                print(f"            {s:8s} {n_s:4d}   {measured:+.4f}    {implied:+.4f}   "
                      f"{'OK' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
