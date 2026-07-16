#!/usr/bin/env python3
"""Matched-cost simulation (DESIGN_cheap_judge_pipeline §5.2): at equal scoring budget, do n pure-5.5
labels or the cheap-judge scheme (0.3n overlap + luna bulk with fused targets) train a better predictor?

Budget accounting (in 5.5-call units, price ratio k): arm A spends n on n 5.5-labeled pairs. Arm B spends
n_ov·(1+1/k) on a dual-scored overlap (n_ov = max(30, 0.3n): a 30-row floor for a stable block fit) and
the rest on luna-only bulk at 1/k per pair: n_bulk = k·(n − n_ov·(1+1/k)) = k·n − n_ov·(k+1). (Blocker 2:
the floor makes n_ov > 0.3n at n=80, so the bulk MUST be sized with n_ov, not 0.3n — the old
0.7kn−0.3n form overspent arm B at n=80.) Realized spend is asserted == n and printed per cell; cells that
would need more bulk rows than the pool holds are flagged TRUNC and excluded from matched-cost claims.
Fusion blocks (prior ⊕ graph_D ⊕ graph_S ⊕ luna, correlated) fit on the overlap; the bulk trains on fused
posteriors; the overlap trains on its 5.5 labels.

Downstream estimator: ridge regression from frozen e5 pair-features (p_x⊙q_y ++ |p_x−q_y|) — a fast proxy
for the head fine-tune; all arms use the SAME estimator so the comparison is about label quality×quantity,
not the estimator (caveat: absolute numbers are proxy-level; the transformer head sees more). Lambda is
selected on an order-invariant PAID-label split: calibration, covariance, pseudo-targets, ridge, and scaler use
paid inner-train only, candidates score only paid inner-valid true labels, then the selected pipeline is refit on
the full arm. The default evaluation partition is strict node-disjoint; crossing pairs are dropped. Eval: held
corr vs 5.5 labels (D and S), paired resamples per grid point.

The A+free control spends the same n paid calls as arm A, fits the prior⊕graph conditioner on those rows,
then fills the rest of the available training pool with zero-scoring-cost fused targets. It deliberately uses
the whole free pool (and is labelled accordingly), rather than pretending that free rows consume budget.

  python3 sim_matched_cost.py --k 2 4 8
"""
import argparse
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_luna_transfer import load_luna as load_scored_mu_tsv
from fine_tune_channel_heads import load_campaign_datasets, load_expanded
from fine_tune_fused_head import agnostic_readouts
from node_disjoint_eval import format_split_diagnostics, node_disjoint_pair_split
from product_kalman import fit_residual_covariance
from run_judge_channel import correlated_update_H
from run_product_kalman_logit import dequant
from run_product_kalman_realdata import DATASETS, affine_calibrate
from run_sym_channel_fusion import H4, calibrate_luna, sym_graph_features
from sigma_hop_confirmatory import FeatureGraphConfig, descendant_disjoint_split, load_feature_graph

ROOT = os.path.dirname(os.path.abspath(__file__))
LUNA_CAMPAIGN = "/tmp/mu_data/campaign_scored_luna.tsv"
H_FREE = np.array([[1.0, 0.0], [0.0, 1.0]])


def positive_integer(value):
    """Argparse/type helper for price ratios, which this row-count simulation requires to be integral."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if not np.isfinite(number) or number <= 0 or not number.is_integer():
        raise argparse.ArgumentTypeError(
            f"k must be a positive integer price ratio; got {value!r} (fractional rows cannot spend exactly)"
        )
    return int(number)


def stable_seed(seed, *parts):
    """Stable 64-bit seed, independent of Python hash randomization and grid traversal order."""
    h = hashlib.blake2b(digest_size=8, person=b"mu-cost")
    for value in (int(seed),) + parts:
        payload = repr(value).encode("utf-8")
        h.update(len(payload).to_bytes(8, "little"))
        h.update(payload)
    return int.from_bytes(h.digest(), "little")


def replicate_permutation(pool, seed, corpus, rep):
    """One reusable permutation per corpus/replicate, so n cells are nested and order-independent."""
    pool = np.asarray(pool)
    rng = np.random.default_rng(stable_seed(seed, corpus, int(rep), "sample"))
    return pool[rng.permutation(len(pool))]


def budget_plan(n, k, n_ov, avail):
    """Matched-cost bulk sizing (blocker 2), as a pure/testable unit. The overlap is dual-scored (5.5 +
    luna) at cost 1+1/k per row; the bulk is luna-only at 1/k. From the matched budget n:
        spend(overlap) = n_ov*(1+1/k);  n_bulk = k*(n - n_ov*(1+1/k)) = k*n - n_ov*(k+1).
    Realized spend equals n exactly when the cell is feasible and untruncated, and is < n otherwise.
    Returns (n_bulk_want, n_bulk_used, truncated, spend, feasible)."""
    try:
        k = positive_integer(k)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(str(exc)) from exc
    spend_ov = n_ov * (1.0 + 1.0 / k)
    feasible = spend_ov <= n + 1e-6                 # can we even afford the dual-scored overlap?
    n_bulk = k * n - n_ov * (k + 1)
    trunc = feasible and n_bulk > avail
    n_bulk_used = max(0, min(n_bulk, avail)) if feasible else 0
    spend = spend_ov + n_bulk_used / k
    return n_bulk, n_bulk_used, trunc, spend, feasible


def _canonical_ids(X, y, sample_ids):
    """Return stable row identities; callers should pass dataset indices when available."""
    if sample_ids is not None:
        ids = np.asarray(sample_ids)
        if len(ids) != len(X):
            raise ValueError("sample_ids must have one value per training row")
        labels = [repr(value.item() if hasattr(value, "item") else value) for value in ids]
        if len(set(labels)) != len(labels):
            raise ValueError("sample_ids must be unique")
        return ids
    # Content-derived fallback keeps the public helper permutation-invariant. Exact duplicate rows may
    # share an identity, but their contribution is identical because the target is included in the digest.
    ids = []
    for row, target in zip(np.asarray(X), np.asarray(y)):
        h = hashlib.blake2b(digest_size=16, person=b"mu-ridge-row")
        h.update(np.ascontiguousarray(row).tobytes())
        h.update(np.ascontiguousarray(np.asarray(target)).tobytes())
        ids.append(h.hexdigest())
    return np.asarray(ids)


def _inner_partition(X, y, sample_ids=None, split_seed=0, valid_frac=0.20):
    """Order-invariant inner split, canonically ordered to remove input-order floating-point drift."""
    n = len(X)
    if n < 3:
        raise ValueError("ridge lambda selection needs at least three training rows")
    if not 0.0 < valid_frac < 1.0:
        raise ValueError("valid_frac must lie strictly between zero and one")
    ids = _canonical_ids(X, y, sample_ids)
    keyed = []
    for i, value in enumerate(ids):
        value = value.item() if hasattr(value, "item") else value
        keyed.append((stable_seed(split_seed, "inner", value), repr(value), i))
    order = np.array([item[2] for item in sorted(keyed)], dtype=int)
    n_valid = min(n - 2, max(1, int(round(valid_frac * n))))
    return order[n_valid:], order[:n_valid], order


def _stable_order(sample_ids, seed, purpose):
    """Canonical row order keyed only by stable identity, never by a target value or input position."""
    keyed = []
    for i, value in enumerate(np.asarray(sample_ids)):
        value = value.item() if hasattr(value, "item") else value
        keyed.append((stable_seed(seed, purpose, value), repr(value), i))
    return np.array([item[2] for item in sorted(keyed)], dtype=int)


def _ridge_weights(X, y, lam):
    """Fit standardized ridge plus an unpenalized intercept, returning its complete transform."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    y_mu = float(y.mean())
    Z = (X - mu) / sd
    eye = np.eye(Z.shape[1])
    w = np.linalg.solve(Z.T @ Z + float(lam) * eye, Z.T @ (y - y_mu))
    return mu, sd, y_mu, w


def _select_ridge_lambda(X, y, Xv, yv, lams, train_ids, valid_ids, split_seed):
    """Fit/scaler on candidate-train only and score every lambda on paid true-label validation only."""
    train_order = _stable_order(train_ids, split_seed, "ridge-candidate-train")
    valid_order = _stable_order(valid_ids, split_seed, "ridge-candidate-valid")
    X = np.asarray(X, dtype=float)[train_order]
    y = np.asarray(y, dtype=float)[train_order]
    Xv = np.asarray(Xv, dtype=float)[valid_order]
    yv = np.asarray(yv, dtype=float)[valid_order]
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    y_mu = float(y.mean())
    Z = (X - mu) / sd
    Zv = (Xv - mu) / sd
    G = Z.T @ Z
    b = Z.T @ (y - y_mu)
    eye = np.eye(X.shape[1])
    best_score = -np.inf
    best_lam = None
    for lam in lams:
        w = np.linalg.solve(G + float(lam) * eye, b)
        pv = y_mu + Zv @ w
        score = np.corrcoef(pv, yv)[0, 1] if pv.std() > 1e-9 and yv.std() > 1e-9 else -np.inf
        # Fixed lambda order is the deterministic tie-breaker.
        if score > best_score:
            best_score, best_lam = float(score), float(lam)
    if best_lam is None:  # All validation correlations were undefined: choose the strongest regularizer.
        best_lam = float(max(lams))
    return best_lam, mu, sd


def nested_ridge_fit_predict(
    X, y, Xq, paid_idx, bulk_idx=(), *, pseudo_target_builder=None,
    lams=(3.0, 30.0, 300.0, 3000.0), sample_ids=None, split_seed=0, return_info=False,
):
    """Nested paid-label lambda selection followed by a full-budget refit.

    ``pseudo_target_builder(fit_paid_idx)`` must return one pseudo-target per row in ``X``. During lambda
    selection it is called with paid inner-train rows only; paid inner-valid labels are used exclusively to
    score candidate lambdas. After selection, the builder and ridge are refit using all paid rows. This keeps
    supervised calibration/covariance fitting out of the validation fold while retaining the whole paid
    budget in the final arm.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xq = np.asarray(Xq, dtype=float)
    paid_idx = np.asarray(paid_idx, dtype=int)
    bulk_idx = np.asarray(bulk_idx, dtype=int)
    lams = tuple(float(lam) for lam in lams)
    if X.ndim != 2 or y.shape != (len(X),):
        raise ValueError("X must be 2-D with one true target per row")
    if len(paid_idx) < 3:
        raise ValueError("nested ridge selection needs at least three paid rows")
    if paid_idx.ndim != 1 or bulk_idx.ndim != 1:
        raise ValueError("paid_idx and bulk_idx must be 1-D")
    all_idx = np.concatenate([paid_idx, bulk_idx])
    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError("paid and bulk rows must be unique and disjoint")
    if len(all_idx) and (all_idx.min() < 0 or all_idx.max() >= len(X)):
        raise ValueError("paid_idx and bulk_idx must index rows of X")
    if not lams or any(lam <= 0 for lam in lams):
        raise ValueError("lams must contain positive values")
    if len(bulk_idx) and pseudo_target_builder is None:
        raise ValueError("bulk rows require a pseudo_target_builder")

    ids = _canonical_ids(X, y, sample_ids)
    paid_inner, paid_valid, _ = _inner_partition(
        X[paid_idx], y[paid_idx], sample_ids=ids[paid_idx], split_seed=split_seed,
    )
    fit_paid_idx = paid_idx[paid_inner]
    valid_paid_idx = paid_idx[paid_valid]
    bulk_order = _stable_order(ids[bulk_idx], split_seed, "bulk") if len(bulk_idx) else np.array([], dtype=int)
    ordered_bulk_idx = bulk_idx[bulk_order]

    if len(ordered_bulk_idx):
        pseudo_inner = np.asarray(pseudo_target_builder(fit_paid_idx), dtype=float)
        if pseudo_inner.shape != (len(X),) or not np.isfinite(pseudo_inner).all():
            raise ValueError("pseudo_target_builder must return one finite target per row in X")
        candidate_idx = np.concatenate([fit_paid_idx, ordered_bulk_idx])
        candidate_y = np.concatenate([y[fit_paid_idx], pseudo_inner[ordered_bulk_idx]])
    else:
        candidate_idx = fit_paid_idx
        candidate_y = y[fit_paid_idx]

    best_lam, inner_mu, inner_sd = _select_ridge_lambda(
        X[candidate_idx], candidate_y, X[valid_paid_idx], y[valid_paid_idx], lams,
        ids[candidate_idx], ids[valid_paid_idx], split_seed,
    )

    paid_full_order = _stable_order(ids[paid_idx], split_seed, "paid-full")
    ordered_paid_idx = paid_idx[paid_full_order]
    if len(ordered_bulk_idx):
        pseudo_full = np.asarray(pseudo_target_builder(ordered_paid_idx), dtype=float)
        if pseudo_full.shape != (len(X),) or not np.isfinite(pseudo_full).all():
            raise ValueError("pseudo_target_builder must return one finite target per row in X")
        final_idx = np.concatenate([ordered_paid_idx, ordered_bulk_idx])
        final_y = np.concatenate([y[ordered_paid_idx], pseudo_full[ordered_bulk_idx]])
    else:
        final_idx = ordered_paid_idx
        final_y = y[ordered_paid_idx]
    final_order = _stable_order(ids[final_idx], split_seed, "ridge-final")
    mu, sd, y_mu, w = _ridge_weights(X[final_idx][final_order], final_y[final_order], best_lam)
    pred = y_mu + ((Xq - mu) / sd) @ w
    if return_info:
        return pred, {
            "lambda": best_lam,
            "inner_train": fit_paid_idx.copy(),
            "inner_valid": valid_paid_idx.copy(),
            "inner_mu": inner_mu.copy(),
            "inner_sd": inner_sd.copy(),
            "selection_target_fit_ids": np.asarray(ids[fit_paid_idx]).copy(),
            "paid_valid_ids": np.asarray(ids[valid_paid_idx]).copy(),
            "final_target_fit_ids": np.asarray(ids[ordered_paid_idx]).copy(),
        }
    return pred


def ridge_fit_predict(
    X, y, Xq, lams=(3.0, 30.0, 300.0, 3000.0), *, sample_ids=None, split_seed=0,
    return_info=False,
):
    """Paid-label-only convenience wrapper around :func:`nested_ridge_fit_predict`."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return nested_ridge_fit_predict(
        X, y, Xq, np.arange(len(X)), lams=lams, sample_ids=sample_ids,
        split_seed=split_seed, return_info=return_info,
    )


def paired_delta_summary(baseline, candidate, *, n_boot=5000, confidence=0.95, seed=0):
    """Paired candidate-minus-baseline replicate delta with a percentile bootstrap CI for its mean."""
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if baseline.shape != candidate.shape or baseline.ndim != 1 or len(baseline) == 0:
        raise ValueError("baseline and candidate must be nonempty paired 1-D arrays of equal length")
    if not (np.isfinite(baseline).all() and np.isfinite(candidate).all()):
        raise ValueError("paired observations must be finite")
    if int(n_boot) != n_boot or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    delta = candidate - baseline
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), size=(int(n_boot), len(delta)))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return {
        "mean": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "confidence": float(confidence),
        "n_pairs": int(len(delta)),
    }


def fused_targets(prior, meas, y, fit_idx, shrink=0.05, H=H4):
    """Blocks fit on fit_idx (the overlap); posteriors for ALL rows."""
    H = np.asarray(H, dtype=float)
    if H.shape != (meas.shape[1], prior.shape[1]):
        raise ValueError("H must map the latent target dimensions to the measurement dimensions")
    E = np.column_stack([y - prior, meas - y @ H.T])[fit_idx]
    cov = fit_residual_covariance(E, shrinkage=shrink)
    P0, C_pm, R0 = cov[:2, :2], cov[:2, 2:], cov[2:, 2:]
    post = np.zeros_like(prior)
    for i in range(len(prior)):
        xp, _ = correlated_update_H(prior[i], P0, meas[i], R0, C_pm, H)
        post[i] = np.clip(xp, 0.0, 1.0)
    return post


def main():
    ap = argparse.ArgumentParser()
    # Campaign-INDEPENDENT prior (blocker 1) — see run_sym_channel_fusion.py.
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_prod_namecond.pt"))
    ap.add_argument("--k", type=positive_integer, nargs="+", default=[2, 4, 8])
    ap.add_argument("--n", type=int, nargs="+", default=[80, 160, 320, 640])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; each corpus/replicate permutation is stable across --n list/order")
    ap.add_argument("--split", choices=("node-disjoint", "descendant-disjoint"), default="node-disjoint",
                    help="node-disjoint is the leakage-resistant primary split; descendant-disjoint reproduces PR #3648")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--held-node-frac", type=float, default=0.40)
    ap.add_argument("--split-candidates", type=int, default=64)
    ap.add_argument("--bootstrap-reps", type=int, default=5000,
                    help="paired-replicate bootstrap draws for candidate-minus-A confidence intervals")
    ap.add_argument("--confidence", type=float, default=0.95)
    a = ap.parse_args()
    if a.reps < 1:
        ap.error("--reps must be positive")
    if a.bootstrap_reps < 1:
        ap.error("--bootstrap-reps must be positive")
    if not 0.0 < a.confidence < 1.0:
        ap.error("--confidence must lie strictly between zero and one")

    ref, _ = load_expanded(a.ckpt, dev="cpu")
    ref.eval()
    lp, lD, lS = load_scored_mu_tsv(LUNA_CAMPAIGN)
    luna_by = {p: (lD[i], lS[i]) for i, p in enumerate(lp)}
    dss = load_campaign_datasets()

    for n_name, ds in dss.items():
        corpus = n_name.replace("-campaign", "")
        parents, _, _, _ = load_feature_graph(FeatureGraphConfig(**DATASETS[corpus]["graph"]))
        keep = [i for i, p in enumerate(ds["pairs"]) if p in luna_by]
        pairs = [ds["pairs"][i] for i in keep]
        tags = [ds["tags"][i] for i in keep]
        y = dequant(np.column_stack([ds["D"][keep], ds["S"][keep]]))
        ro = agnostic_readouts(ref, ds, "cpu")
        prior = np.column_stack([ro["prior_D"][keep], ro["prior_S"][keep]])
        d = ds["d"][keep]
        luna = np.array([luna_by[p] for p in pairs])
        F = sym_graph_features(parents, pairs)
        tok = ds["tok"]
        ii = [tok.idx[x] for x, _ in pairs]; jj = [tok.idx[yy] for _, yy in pairs]
        px = tok.p[ii].numpy(); qy = tok.q[jj].numpy()
        feat = np.column_stack([px * qy, np.abs(px - qy)])
        graph_design = np.column_stack([F, np.ones(len(F))])
        row_ids = np.arange(len(feat))

        if a.split == "node-disjoint":
            split = node_disjoint_pair_split(
                pairs, a.split_seed, held_node_fraction=a.held_node_frac,
                strata=tags, candidates=a.split_candidates, minimum_per_stratum=1,
            )
            tr, he = split.train, split.held
            split_note = "STRICT node-disjoint; crossing pairs dropped"
        else:
            tr, he = descendant_disjoint_split(pairs, a.split_seed, held_frac=0.30)
            tr, he = np.array(tr), np.array(he)
            split = None
            split_note = "EXPLORATORY descendant-disjoint; roots may overlap"
        print(f"\n=== {n_name}: {len(pairs)} rows (pool {len(tr)}, held {len(he)}; {split_note}) ===")
        if split is not None:
            for line in format_split_diagnostics(split).splitlines():
                print(f"  {line}")
        print("paired training-subsample replicates; bootstrap intervals condition on this fixed held split")
        print(f"{'n':>5s} {'arm':>20s}" + "".join(f" {c:>8s}" for c in ("D corr", "S corr")))
        rep_orders = [replicate_permutation(tr, a.seed, corpus, rep) for rep in range(a.reps)]
        for n in a.n:
            if n > len(tr):
                continue
            # Budget accounting (blocker 2). The overlap is dual-scored (5.5 + luna) at cost 1+1/k per row;
            # the bulk is luna-only at 1/k. The overlap size uses a 30-row FLOOR, so it can exceed 0.3n at
            # small n (n=80: n_ov=30 = 0.375n). Prior code sized the bulk with 0.3n instead of len(ov),
            # which OVERSPENT arm B whenever the floor bound. Correct, from the matched budget n:
            #   spend(overlap) = n_ov*(1+1/k);  n_bulk = k*(n - n_ov*(1+1/k)) = k*n - n_ov*(k+1).
            n_ov = max(30, int(0.3 * n))
            avail = len(tr) - n_ov                          # bulk rows available in the pool (const/rep)
            acct = {}                                       # k -> (n_bulk_want, n_bulk_used, trunc, spend, feasible)
            for k in a.k:
                acct[k] = budget_plan(n, k, n_ov, avail)
                if acct[k][4]:
                    assert acct[k][3] <= n + 1e-6, f"arm B overspends: n={n} k={k} spend={acct[k][3]:.2f} > {n}"
                    if not acct[k][2]:
                        assert abs(acct[k][3] - n) <= 1e-6, (
                            f"untruncated arm B must match budget exactly: n={n} k={k} "
                            f"spend={acct[k][3]:.6f}"
                        )
            print(f"  n={n}: overlap n_ov={n_ov}; " + "  ".join(
                (f"k={k:g}:INFEASIBLE" if not acct[k][4] else
                 f"k={k:g}:n_bulk={acct[k][0]}" + (f"*TRUNC->{acct[k][1]}" if acct[k][2] else "")
                 + f"(spend {acct[k][3]:.1f})") for k in a.k))
            res = {}
            for rep in range(a.reps):
                sel = rep_orders[rep]
                # arm A: n pure 5.5 labels
                A_idx = sel[:n]
                for ch, col in (("D", 0), ("S", 1)):
                    inner_seed = stable_seed(a.seed, corpus, n, rep, ch, "ridge-inner")
                    pred = nested_ridge_fit_predict(
                        feat, y[:, col], feat[he], A_idx, sample_ids=row_ids, split_seed=inner_seed,
                    )
                    res.setdefault(("A: 5.5 only", ch), []).append(np.corrcoef(pred, y[he, col])[0, 1])

                # Same paid spend as A, plus all remaining pool rows labelled by prior⊕free graph channels.
                free_bulk = sel[n:]
                free_post_cache = {}

                def build_free_post(fit_paid_idx):
                    key = tuple(sorted(np.asarray(fit_paid_idx, dtype=int).tolist()))
                    if key not in free_post_cache:
                        fit = np.asarray(key, dtype=int)
                        free_d = affine_calibrate(d[fit], y[fit, 0], d)
                        free_beta, *_ = np.linalg.lstsq(graph_design[fit], y[fit, 1], rcond=None)
                        free_meas = np.column_stack([free_d, graph_design @ free_beta])
                        free_post_cache[key] = fused_targets(prior, free_meas, y, fit, H=H_FREE)
                    return free_post_cache[key]

                for ch, col in (("D", 0), ("S", 1)):
                    inner_seed = stable_seed(a.seed, corpus, n, rep, ch, "ridge-inner")
                    pred = nested_ridge_fit_predict(
                        feat, y[:, col], feat[he], A_idx, free_bulk,
                        pseudo_target_builder=lambda fit, col=col: build_free_post(fit)[:, col],
                        sample_ids=row_ids, split_seed=inner_seed,
                    )
                    res.setdefault(("A+free: full pool", ch), []).append(
                        np.corrcoef(pred, y[he, col])[0, 1]
                    )

                # arm B per k: overlap n_ov (5.5 labels) + luna bulk with fused targets
                ov = sel[:n_ov]
                luna_post_cache = {}

                def build_luna_post(fit_paid_idx):
                    key = tuple(sorted(np.asarray(fit_paid_idx, dtype=int).tolist()))
                    if key not in luna_post_cache:
                        fit = np.asarray(key, dtype=int)
                        m_cal = affine_calibrate(d[fit], y[fit, 0], d)
                        beta, *_ = np.linalg.lstsq(graph_design[fit], y[fit, 1], rcond=None)
                        luna_c = calibrate_luna(luna, y, fit)         # bias first (blocker 3 / DESIGN §2)
                        meas = np.column_stack([
                            m_cal, graph_design @ beta, luna_c[:, 0], luna_c[:, 1],
                        ])
                        luna_post_cache[key] = fused_targets(prior, meas, y, fit)
                    return luna_post_cache[key]

                for k in a.k:
                    if not acct[k][4]:
                        continue
                    B_bulk = sel[n_ov:n_ov + acct[k][1]]
                    for ch, col in (("D", 0), ("S", 1)):
                        inner_seed = stable_seed(a.seed, corpus, n, rep, ch, "ridge-inner")
                        pred = nested_ridge_fit_predict(
                            feat, y[:, col], feat[he], ov, B_bulk,
                            pseudo_target_builder=lambda fit, col=col: build_luna_post(fit)[:, col],
                            sample_ids=row_ids, split_seed=inner_seed,
                        )
                        res.setdefault((f"B: scheme k={k:g}", ch), []).append(
                            np.corrcoef(pred, y[he, col])[0, 1])
            arms = ["A: 5.5 only", "A+free: full pool"] + [
                f"B: scheme k={k:g}" for k in a.k if (f"B: scheme k={k:g}", "D") in res
            ]
            baselines = {
                "A": {ch: np.asarray(res[("A: 5.5 only", ch)]) for ch in ("D", "S")},
                "A+free": {ch: np.asarray(res[("A+free: full pool", ch)]) for ch in ("D", "S")},
            }
            for arm in arms:
                cd = np.array(res[(arm, "D")]); cs = np.array(res[(arm, "S")])
                flag = ""
                if arm.startswith("B: scheme"):
                    kv = int(arm.split("k=")[1])
                    if acct[kv][2]:
                        flag = "  [TRUNC: pool-limited, excluded from matched-cost claim]"
                elif arm.startswith("A+free"):
                    flag = "  [same paid spend; zero-cost targets fill pool]"
                print(f"{n:>5d} {arm:>20s} {cd.mean():+8.3f} {cs.mean():+8.3f}"
                      f"   (replicate SD {cd.std():.3f}/{cs.std():.3f}){flag}")
                if arm != "A: 5.5 only":
                    comparisons = ["A"] + (["A+free"] if arm.startswith("B: scheme") else [])
                    for comparison in comparisons:
                        summaries = {}
                        for ch, vals in (("D", cd), ("S", cs)):
                            summaries[ch] = paired_delta_summary(
                                baselines[comparison][ch], vals,
                                n_boot=a.bootstrap_reps, confidence=a.confidence,
                                seed=stable_seed(
                                    a.seed, corpus, n, arm, comparison, ch, "paired-bootstrap"
                                ),
                            )
                        pct = 100.0 * a.confidence
                        print(" " * 28 + f"paired Δ vs {comparison}: " + "  ".join(
                            f"{ch} {summary['mean']:+.3f} "
                            f"({pct:g}% CI {summary['ci_low']:+.3f},{summary['ci_high']:+.3f})"
                            for ch, summary in summaries.items()
                        ))


if __name__ == "__main__":
    main()
