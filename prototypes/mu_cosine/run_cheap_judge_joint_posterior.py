#!/usr/bin/env python3
"""Same-split JointPosterior comparison for the cheap-judge pipeline.

The campaign cannot support an honest eight-way relation-class comparison: ``cur_rel`` is the
sampler's structural provenance and is ``subcategory`` for every scored row.  Instead, this script
declares a *decision-space bridge* from the operating judge's eight probabilities to three macro
decisions that the continuous D/S state can plausibly represent:

    directional = element_of + subcategory + subtopic + super_category
    symmetric   = see_also + assoc
    other       = unknown + none

The target is the argmax macro decision.  It is GPT-5.5 decision fidelity, not ground truth and not
eight-way relation recovery.  A train-only logistic ``p(decision | D,S)`` bridge maps the correlated
Gaussian posterior into that same decision space.  Its expectation under ``N(mu_post, P_post)`` is
computed with two-dimensional Gauss-Hermite quadrature, so the Gaussian covariance affects confidence.
The bridge applied to the held row's operating-judge D/S is reported as a finite-sample linear reference.
Its gap mixes representation loss with bridge misspecification and estimation error; it is not a Bayes ceiling.

Both learned methods use the identical train-calibrated source vector on a node-disjoint
combiner/calibration split (upstream checkpoint endpoint exposure is not claimed):

    [prior_D, prior_S, graph_D, graph_S, luna_D, luna_S]

JointPosterior learns ``p(decision | all sources)`` directly.  The dense Gaussian baseline first fuses
the sources into ``N(D,S)`` and then applies the declared decision bridge.  Equal and separability-
weighted factored products are controls, not recommended estimators.  Metrics are accuracy, log-loss,
ECE (10 equal-width confidence bins), and margin-gated AURC with an endpoint-connected-component
bootstrap.  The component bootstrap keeps every row sharing a node in the same resampled block.

Typical run (local scored artifacts and campaign-independent checkpoint required):

    python3 run_cheap_judge_joint_posterior.py --seed 0 --boot 500

The script intentionally does not modify the PR #3648 experiment or report files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_luna_transfer import load_luna as load_scored_mu_tsv
from emit_transitive_hops import hit_prob
from fine_tune_channel_heads import load_campaign_datasets, load_expanded
from fine_tune_fused_head import agnostic_readouts
from mu_posterior import JointPosterior, MuPosterior, _eval, aurc, margin_conf, pearson
from node_disjoint_eval import node_disjoint_pair_split
from product_kalman import fit_residual_covariance
from run_judge_channel import correlated_update_H
from run_product_kalman_logit import dequant
from run_product_kalman_realdata import DATASETS, affine_calibrate
from run_sym_channel_fusion import H4, calibrate_luna, sym_graph_features
from sigma_hop_confirmatory import FeatureGraphConfig, load_feature_graph

ROOT = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = "/tmp/mu_data/campaign_scored.tsv"
LUNA_CAMPAIGN = "/tmp/mu_data/campaign_scored_luna.tsv"
CLASSES = ("directional", "symmetric", "other")
GROUPS = {
    "directional": ("element_of", "subcategory", "subtopic", "super_category"),
    "symmetric": ("see_also", "assoc"),
    "other": ("unknown", "none"),
}
SOURCES = ("prior_D", "prior_S", "graph_D", "graph_S", "luna_D", "luna_S")
GAUSSIAN_RUNGS = {
    "gaussian/prior": (),
    "gaussian/+graph": (0, 1),
    "gaussian/+luna": (2, 3),
    "gaussian/all": (0, 1, 2, 3),
}
JOINT_RUNGS = {
    "joint/prior": (0, 1),
    "joint/+graph": (0, 1, 2, 3),
    "joint/+luna": (0, 1, 4, 5),
    "joint/all": (0, 1, 2, 3, 4, 5),
}


@dataclass
class SplitData:
    """Train-calibrated sources and dense Gaussian blocks for one split."""

    X: np.ndarray
    y_ds: np.ndarray
    prior: np.ndarray
    meas: np.ndarray
    P0: np.ndarray
    C_pm: np.ndarray
    R0: np.ndarray


def aggregate_decision_probabilities(row, col):
    """Aggregate one campaign TSV row into the declared three-way decision simplex."""
    p = np.array([sum(float(row[col[f"P[{rel}]"]]) for rel in GROUPS[group]) for group in CLASSES])
    z = float(p.sum())
    if not np.isfinite(z) or z <= 0:
        raise ValueError("judge relation probabilities do not have positive finite mass")
    return p / z


def macro_decision(probability, *, tie_atol=0.0):
    """Return fixed-order hard decision, top-two margin, and an explicit top-tie flag."""
    probability = np.asarray(probability, dtype=float)
    if probability.shape != (len(CLASSES),) or not np.isfinite(probability).all():
        raise ValueError(f"expected {len(CLASSES)} finite macro probabilities")
    order = np.argsort(-probability, kind="stable")
    top = probability[order[0]]
    tied = np.flatnonzero(np.isclose(probability, top, rtol=0.0, atol=tie_atol))
    # Stable CLASSES order resolves exact/near-exact ties; persist the tie flag so this is auditable.
    decision_index = int(tied[0]) if len(tied) else int(order[0])
    margin = float(top - probability[order[1]])
    return CLASSES[decision_index], margin, bool(len(tied) > 1)


def load_decision_targets(path=CAMPAIGN):
    """Return pair -> (normalised macro probabilities, hard macro decision).

    The hard decision is used because the repository JointPosterior trains on class labels. The runner
    persists row-level aggregated vectors, margins, tie flags, and split membership for auditing.
    Duplicate pairs must agree rather than being silently overwritten.
    """
    out = {}
    cur_rel = Counter()
    with open(path, encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip().split("\t")
        col = {name: i for i, name in enumerate(header)}
        required = {"node", "root", "cur_rel"} | {
            f"P[{rel}]" for rels in GROUPS.values() for rel in rels
        }
        missing = required - set(col)
        if missing:
            raise ValueError(f"campaign file lacks columns: {sorted(missing)}")
        for line in f:
            row = line.rstrip("\n").split("\t")
            if len(row) < len(header):
                continue
            pair = (row[col["node"]], row[col["root"]])
            prob = aggregate_decision_probabilities(row, col)
            decision, _margin, _tied = macro_decision(prob)
            value = (prob, decision)
            if pair in out and (out[pair][1] != value[1] or not np.allclose(out[pair][0], prob)):
                raise ValueError(f"conflicting duplicate decision target for {pair!r}")
            out[pair] = value
            cur_rel[row[col["cur_rel"]]] += 1
    return out, cur_rel


def pool_relation_values(mu, probability, temperature=0.10):
    """Return hard-max, conditional-probability, and temperature-softmax pooled values."""
    mu, probability = np.asarray(mu, float), np.asarray(probability, float)
    if mu.ndim != 1 or probability.shape != mu.shape or len(mu) == 0:
        raise ValueError("mu and probability must be same-length non-empty vectors")
    if temperature <= 0:
        raise ValueError("pooling temperature must be positive")
    p_total = probability.sum()
    soft_weight = np.exp((mu - mu.max()) / temperature)
    return (float(mu.max()),
            float(probability @ mu / p_total) if p_total > 0 else float(mu.mean()),
            float(soft_weight @ mu / soft_weight.sum()))


def load_within_judge_pooling_states(path=CAMPAIGN, temperature=0.10):
    """Alternative reductions of one judge's relation-specific mu values to D/S.

    These are within-judge relation pooling diagnostics. They are not expert selection: all terms come
    from one GPT-5.5 response. ``softmax`` is a temperature-softmax weighted mean (bounded by the input
    mu values), while ``prob_weighted`` uses the judge's conditional relation probabilities within each
    D/S group. The #3648 estimator remains on its preregistered hard maximum; alternatives are linear-reference
    diagnostics until all prior/measurement channels are retrained with the same pooling rule.
    """
    if temperature <= 0:
        raise ValueError("pooling temperature must be positive")
    out = {}
    pooled_groups = (GROUPS["directional"], GROUPS["symmetric"])
    with open(path, encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip().split("\t")
        col = {name: i for i, name in enumerate(header)}
        required = {"node", "root"} | {
            prefix + rel + "]" for rels in pooled_groups for rel in rels for prefix in ("P[", "mu[")
        }
        missing = required - set(col)
        if missing:
            raise ValueError(f"campaign file lacks pooling columns: {sorted(missing)}")
        for line in f:
            row = line.rstrip("\n").split("\t")
            if len(row) < len(header):
                continue
            pair = (row[col["node"]], row[col["root"]])
            hard, prob_weighted, softmax = [], [], []
            for relations in pooled_groups:
                mu = np.array([float(row[col[f"mu[{rel}]"]]) for rel in relations])
                probability = np.array([float(row[col[f"P[{rel}]"]]) for rel in relations])
                hp, pp, sp = pool_relation_values(mu, probability, temperature)
                hard.append(hp); prob_weighted.append(pp); softmax.append(sp)
            value = {
                "hard_max": np.asarray(hard),
                "prob_weighted": np.asarray(prob_weighted),
                f"softmax_t{temperature:g}": np.asarray(softmax),
            }
            if pair in out and any(not np.allclose(out[pair][key], val) for key, val in value.items()):
                raise ValueError(f"conflicting duplicate pooling state for {pair!r}")
            out[pair] = value
    return out


def endpoint_components(pairs):
    """Row-index blocks induced by connected components of the pair endpoint graph."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)
    by = {}
    for i, (a, _b) in enumerate(pairs):
        by.setdefault(find(a), []).append(i)
    return [np.asarray(ix, int) for ix in by.values()]


def component_bootstrap_aurc(proba, labels, pairs, B=500, seed=0):
    """Margin AURC and 95% CI from endpoint-connected-component resampling."""
    proba = np.asarray(proba, float)
    labels = np.asarray(labels, int)
    correct = (proba.argmax(1) == labels).astype(float)
    conf = margin_conf(proba)
    blocks = endpoint_components(pairs)
    point = aurc(conf, correct)
    if B <= 0:
        return point, float("nan"), float("nan"), len(blocks), max(map(len, blocks), default=0)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        chosen = rng.integers(0, len(blocks), len(blocks))
        ix = np.concatenate([blocks[j] for j in chosen])
        vals.append(aurc(conf[ix], correct[ix]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi), len(blocks), max(map(len, blocks), default=0)


def component_bootstrap_aurc_delta(proba_a, proba_b, labels, pairs, B=500, seed=0):
    """Paired component-bootstrap CI for AURC(A)-AURC(B); negative favours A."""
    labels = np.asarray(labels, int)
    ca = (np.asarray(proba_a).argmax(1) == labels).astype(float)
    cb = (np.asarray(proba_b).argmax(1) == labels).astype(float)
    ma, mb = margin_conf(proba_a), margin_conf(proba_b)
    blocks = endpoint_components(pairs)
    point = aurc(ma, ca) - aurc(mb, cb)
    if B <= 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        chosen = rng.integers(0, len(blocks), len(blocks))
        ix = np.concatenate([blocks[j] for j in chosen])
        vals.append(aurc(ma[ix], ca[ix]) - aurc(mb[ix], cb[ix]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def gaussian_bridge_proba(bridge, means, covariances, order=5):
    """E[p(class | Z)] for Z~N(mean,cov), by product Gauss-Hermite quadrature."""
    means = np.asarray(means, float)
    covariances = np.asarray(covariances, float)
    if means.ndim != 2 or means.shape[1] != 2 or covariances.shape != (len(means), 2, 2):
        raise ValueError("expected means [N,2] and covariances [N,2,2]")
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    z = np.array([(a, b) for a in nodes for b in nodes], float) * np.sqrt(2.0)
    w = np.array([wa * wb for wa in weights for wb in weights], float) / np.pi
    out = []
    for mu, cov in zip(means, covariances):
        eigval, eigvec = np.linalg.eigh(0.5 * (cov + cov.T))
        root = eigvec @ np.diag(np.sqrt(np.clip(eigval, 0.0, None)))
        points = mu + z @ root.T
        out.append(w @ bridge.proba(points))
    out = np.asarray(out)
    return out / out.sum(axis=1, keepdims=True)


def fit_factored(Xtr, labels_tr, Xhe, weights="equal", nbins=12):
    """Factored histogram product control fit strictly on training rows."""
    post = MuPosterior(nbins=nbins, lo=-0.10, hi=1.10, smoothing=1.0)
    for j, source in enumerate(SOURCES):
        post.fit_source(source, zip(labels_tr, Xtr[:, j]))
    if weights == "separability":
        for source in SOURCES:
            post.weights[source] = post.separability(source)[0]
    elif weights != "equal":
        raise ValueError("weights must be 'equal' or 'separability'")
    return np.array([
        [post.posterior(dict(zip(SOURCES, row))).get(cls, 1e-12) for cls in CLASSES]
        for row in Xhe
    ])


def calibrate_sources(prior, graph_raw, graph_features, luna, y_ds, train):
    """Fit every affine/graph calibration and residual block on ``train`` only."""
    graph_d = affine_calibrate(graph_raw[train], y_ds[train, 0], graph_raw)
    design = np.column_stack([graph_features, np.ones(len(graph_features))])
    beta, *_ = np.linalg.lstsq(design[train], y_ds[train, 1], rcond=None)
    graph_s = design @ beta
    luna_c = calibrate_luna(luna, y_ds, train)
    meas = np.column_stack([graph_d, graph_s, luna_c[:, 0], luna_c[:, 1]])
    X = np.column_stack([prior, meas])
    observed_state = y_ds[:, [0, 1, 0, 1]]
    errors = np.column_stack([y_ds - prior, meas - observed_state])[train]
    cov = fit_residual_covariance(errors, shrinkage=0.05)
    return SplitData(X=X, y_ds=y_ds, prior=prior, meas=meas,
                     P0=cov[:2, :2], C_pm=cov[:2, 2:], R0=cov[2:, 2:])


def gaussian_posterior(split, indices, selected):
    """Dense Gaussian posterior for a measurement subset (prior enters exactly once)."""
    means, covs = [], []
    selected = list(selected)
    for i in indices:
        if selected:
            xp, Pp = correlated_update_H(
                split.prior[i], split.P0, split.meas[i, selected],
                split.R0[np.ix_(selected, selected)], split.C_pm[:, selected], H4[selected],
            )
        else:
            xp, Pp = split.prior[i].copy(), split.P0.copy()
        means.append(xp)
        covs.append(Pp)
    return np.asarray(means), np.asarray(covs)


def metric_record(proba, label_names, pairs, boot, seed):
    ri = {name: i for i, name in enumerate(CLASSES)}
    labels = np.array([ri[x] for x in label_names], int)
    acc, logloss, ece = _eval(proba, label_names, ri, bins=10)
    a, lo, hi, blocks, largest = component_bootstrap_aurc(proba, labels, pairs, B=boot, seed=seed)
    return {
        "accuracy": acc,
        "log_loss": logloss,
        "ece_10_equal_width": ece,
        "aurc_margin": a,
        "aurc_ci95_component_boot": [lo, hi],
        "bootstrap_blocks": blocks,
        "largest_block_rows": largest,
    }


def print_source_diagnostics(Xtr, label_names):
    print("  source separability on TRAIN (SD of per-class means):")
    for j, source in enumerate(SOURCES):
        means = [np.mean(Xtr[np.array(label_names) == cls, j]) for cls in CLASSES]
        print(f"    {source:9s} {np.std(means):.4f}  means=" + "/".join(f"{v:.3f}" for v in means))
    print("  source correlation on TRAIN (Pearson; large |r| means redundancy):")
    print("               " + " ".join(f"{s[:8]:>8s}" for s in SOURCES))
    for i, source in enumerate(SOURCES):
        vals = " ".join(f"{pearson(Xtr[:, i], Xtr[:, j]):+8.2f}" for j in range(len(SOURCES)))
        print(f"    {source:9s} {vals}")


def run_corpus(name, ds, target_by_pair, pooling_by_pair, luna_by_pair, checkpoint, args):
    corpus = name.replace("-campaign", "")
    parents, _, _, _ = load_feature_graph(FeatureGraphConfig(**DATASETS[corpus]["graph"]))
    # Graph loaders expose parent sets. Both the walk DP's cycle guard and tokenizer ancestor order can
    # otherwise depend on Python's per-process hash seed. Canonicalise locally before either feature path.
    parents = {node: tuple(sorted(values)) for node, values in parents.items()}
    ds["tok"].parents = parents
    model, _ = checkpoint
    ro = agnostic_readouts(model, ds, "cpu")
    keep = [i for i, pair in enumerate(ds["pairs"])
            if pair in target_by_pair and pair in pooling_by_pair and pair in luna_by_pair]
    pairs = [ds["pairs"][i] for i in keep]
    if len(pairs) != len(set(pairs)):
        raise ValueError("comparison requires unique campaign pairs")
    y_ds = dequant(np.column_stack([ds["D"][keep], ds["S"][keep]]))
    prior = np.column_stack([ro["prior_D"][keep], ro["prior_S"][keep]])
    graph_raw = np.array([hit_prob(parents, x, y) for x, y in pairs])
    graph_features = sym_graph_features(parents, pairs)
    luna = np.asarray([luna_by_pair[pair] for pair in pairs])
    soft_target = np.asarray([target_by_pair[pair][0] for pair in pairs])
    hard_target = np.asarray([target_by_pair[pair][1] for pair in pairs])
    pooling_states = {
        key: dequant(np.asarray([pooling_by_pair[pair][key] for pair in pairs]))
        for key in next(iter(pooling_by_pair.values()))
    }
    if not np.allclose(y_ds, pooling_states["hard_max"]):
        raise AssertionError("campaign loader's D/S differs from audited hard-max relation pooling")

    split_spec = node_disjoint_pair_split(
        pairs,
        args.seed,
        held_node_fraction=args.held_frac,
        strata=hard_target,
        candidates=args.split_candidates,
        minimum_per_stratum=args.min_class,
    )
    train, held = split_spec.train, split_spec.held
    train_nodes = {node for i in train for node in pairs[i]}
    held_nodes = {node for i in held for node in pairs[i]}
    if train_nodes & held_nodes:
        raise AssertionError("audited node-disjoint split leaked an endpoint")
    train_counts, held_counts = Counter(hard_target[train]), Counter(hard_target[held])
    if min((train_counts[c] for c in CLASSES), default=0) < args.min_class:
        raise ValueError(f"{name}: seed {args.seed} has too few training rows per class: {train_counts}")
    if min((held_counts[c] for c in CLASSES), default=0) < args.min_class:
        raise ValueError(f"{name}: seed {args.seed} has too few held rows per class: {held_counts}")
    print(f"\n=== {name}: combiner/calibration node-disjoint seed={args.seed}; "
          f"candidate={split_spec.selected_candidate}/{split_spec.candidates}; "
          f"train={len(train)} held={len(held)} cross-dropped={len(split_spec.cross)} ===")
    print(f"  train classes {dict(train_counts)}; held classes {dict(held_counts)}")

    split = calibrate_sources(prior, graph_raw, graph_features, luna, y_ds, train)
    print_source_diagnostics(split.X[train], hard_target[train])
    bridge = JointPosterior(CLASSES, n_features=2, hidden=0, seed=1000 + args.seed).fit(
        y_ds[train], hard_target[train], epochs=args.epochs,
    )
    methods = {"bridge/D-S-hard-max-reference": bridge.proba(y_ds[held])}
    pooling_diagnostics = {}
    print("  within-judge pooling diagnostics (full matched corpus; one response; NOT expert selection):")
    for j, (pool_name, state) in enumerate(pooling_states.items()):
        delta = state - y_ds
        pooling_diagnostics[pool_name] = {
            "mean_delta_from_hard_max_D_S": delta.mean(axis=0).tolist(),
            "correlation_with_hard_max_D_S": [pearson(state[:, k], y_ds[:, k]) for k in range(2)],
        }
        print(f"    {pool_name:14s} mean delta D/S {delta[:, 0].mean():+.3f}/{delta[:, 1].mean():+.3f}; "
              f"corr {pearson(state[:, 0], y_ds[:, 0]):+.3f}/{pearson(state[:, 1], y_ds[:, 1]):+.3f}")
        if pool_name != "hard_max":
            # Linear-reference target-construction diagnostic. A production comparison must retrain every
            # prior and measurement channel under this same within-judge pooling definition.
            alt_bridge = JointPosterior(CLASSES, n_features=2, hidden=0,
                                        seed=1100 + args.seed + j).fit(
                state[train], hard_target[train], epochs=args.epochs,
            )
            methods[f"bridge/D-S-{pool_name}-reference"] = alt_bridge.proba(state[held])

    for method, selected in GAUSSIAN_RUNGS.items():
        means, covs = gaussian_posterior(split, held, selected)
        methods[method] = gaussian_bridge_proba(bridge, means, covs, order=args.quad_order)
    for method, selected in JOINT_RUNGS.items():
        cols = list(selected)
        joint = JointPosterior(CLASSES, n_features=len(cols), hidden=args.hidden,
                               seed=2000 + args.seed).fit(
            split.X[train][:, cols], hard_target[train], epochs=args.epochs,
        )
        methods[method] = joint.proba(split.X[held][:, cols])
    methods["factored/equal"] = fit_factored(
        split.X[train], hard_target[train], split.X[held], weights="equal", nbins=args.nbins,
    )
    methods["factored/separability"] = fit_factored(
        split.X[train], hard_target[train], split.X[held], weights="separability", nbins=args.nbins,
    )

    held_pairs = [pairs[i] for i in held]
    records = {}
    print("  metrics (ECE=10 equal-width confidence bins; AURC=margin gate, component bootstrap):")
    print(f"    {'method':26s} {'acc':>6s} {'logloss':>8s} {'ECE':>7s} {'AURC [95% CI]':>25s}")
    for j, (method, proba) in enumerate(methods.items()):
        rec = metric_record(proba, hard_target[held], held_pairs, args.boot, args.seed * 100 + j)
        records[method] = rec
        lo, hi = rec["aurc_ci95_component_boot"]
        print(f"    {method:26s} {rec['accuracy']:6.3f} {rec['log_loss']:8.3f} {rec['ece_10_equal_width']:7.3f} "
              f"{rec['aurc_margin']:7.3f} [{lo:.3f}, {hi:.3f}]")
    ri = {name: i for i, name in enumerate(CLASSES)}
    y_int = np.array([ri[x] for x in hard_target[held]])
    delta, dlo, dhi = component_bootstrap_aurc_delta(
        methods["joint/all"], methods["gaussian/all"], y_int, held_pairs,
        B=args.boot, seed=9000 + args.seed,
    )
    print(f"  paired AURC delta joint/all - gaussian/all: {delta:+.4f} [{dlo:+.4f}, {dhi:+.4f}] "
          "(negative favours JointPosterior)")
    source_deltas = {}
    for j, (family, source, with_method, without_method) in enumerate([
        ("gaussian", "graph", "gaussian/all", "gaussian/+luna"),
        ("gaussian", "luna", "gaussian/all", "gaussian/+graph"),
        ("joint", "graph", "joint/all", "joint/+luna"),
        ("joint", "luna", "joint/all", "joint/+graph"),
    ]):
        point, lo, hi = component_bootstrap_aurc_delta(
            methods[with_method], methods[without_method], y_int, held_pairs,
            B=args.boot, seed=10000 + args.seed * 10 + j,
        )
        source_deltas[f"{family}_add_{source}"] = [point, lo, hi]
        print(f"  paired AURC delta {family}/all - {without_method} (add {source}): "
              f"{point:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    train_set, held_set = set(map(int, train)), set(map(int, held))
    decision_rows = []
    for i, pair in enumerate(pairs):
        decision, margin, tied = macro_decision(soft_target[i])
        decision_rows.append({
            "pair": list(pair),
            "split": "train" if i in train_set else "held" if i in held_set else "cross-dropped",
            "probability": dict(zip(CLASSES, soft_target[i].tolist())),
            "hard_decision": decision,
            "top_two_margin": margin,
            "top_tie_fixed_class_order": tied,
        })
    return {
        "corpus": name,
        "seed": args.seed,
        "split": {
            "kind": "combiner-calibration-node-disjoint",
            "held_fraction_nodes": args.held_frac,
            "train_rows": len(train),
            "held_rows": len(held),
            "cross_rows_dropped": len(split_spec.cross),
            "candidates": split_spec.candidates,
            "selected_candidate": split_spec.selected_candidate,
            "retained_fraction": split_spec.retained_fraction,
            "train_classes": dict(train_counts),
            "held_classes": dict(held_counts),
        },
        "target": {
            "frame": "gpt-5.5-low macro-decision fidelity; not ground truth",
            "classes": {key: list(value) for key, value in GROUPS.items()},
            "mean_soft_probability_held": dict(zip(CLASSES, soft_target[held].mean(axis=0).tolist())),
            "decision_rows": decision_rows,
            "audit_counts": {
                "top_ties": int(sum(row["top_tie_fixed_class_order"] for row in decision_rows)),
                "margin_below_0.02": int(sum(row["top_two_margin"] < 0.02 for row in decision_rows)),
            },
            "within_judge_pooling": {
                "role": "same-response linear-reference diagnostic only; hard max remains the #3648 estimator",
                "not_expert_selection": True,
                "shift_correlation_scope": "full matched corpus, including train/held/cross-dropped rows",
                "diagnostics": pooling_diagnostics,
            },
        },
        "metrics": records,
        "paired_aurc_delta_joint_minus_gaussian": [delta, dlo, dhi],
        "paired_source_aurc_deltas": source_deltas,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_prod_namecond.pt"),
                    help="campaign-independent prior checkpoint")
    ap.add_argument("--campaign", default=CAMPAIGN)
    ap.add_argument("--luna", default=LUNA_CAMPAIGN)
    ap.add_argument("--seed", type=int, default=0, help="node partition and model-init seed")
    ap.add_argument("--held-frac", type=float, default=0.40, help="fraction of nodes assigned held")
    ap.add_argument("--split-candidates", type=int, default=64,
                    help="deterministic coverage-aware node partitions considered per seed")
    ap.add_argument("--min-class", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=0,
                    help="JointPosterior hidden units; 0 is the preregistered multinomial LR")
    ap.add_argument("--nbins", type=int, default=12, help="factored-control histogram bins")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--quad-order", type=int, default=5)
    ap.add_argument("--pool-temperature", type=float, default=0.10,
                    help="temperature for the within-judge softmax-pooling linear-reference diagnostic")
    ap.add_argument("--out", default=None, help="optional JSON result path")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)  # tiny heads are much faster and deterministic without a large BLAS pool
    target_by_pair, cur_rel = load_decision_targets(args.campaign)
    pooling_by_pair = load_within_judge_pooling_states(args.campaign, args.pool_temperature)
    if len(cur_rel) == 1:
        only, count = next(iter(cur_rel.items()))
        print(f"campaign cur_rel is degenerate: {only}={count}; using declared macro decision bridge")
    luna_pairs, luna_d, luna_s = load_scored_mu_tsv(args.luna)
    luna_by_pair = {pair: (luna_d[i], luna_s[i]) for i, pair in enumerate(luna_pairs)}
    model, cfg = load_expanded(args.ckpt, dev="cpu")
    model.eval()
    dss = load_campaign_datasets()
    results = [run_corpus(name, ds, target_by_pair, pooling_by_pair, luna_by_pair, (model, cfg), args)
               for name, ds in dss.items()]
    payload = {
        "protocol": "same split; combiner/calibration node-disjoint; train-only calibration; macro decision bridge",
        "checkpoint": os.path.abspath(args.ckpt),
        "results": results,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
