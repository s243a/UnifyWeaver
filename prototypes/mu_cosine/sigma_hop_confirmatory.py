#!/usr/bin/env python3
"""Run the pre-registered confirmatory Σ(hop) test.

This is the implementation companion to `PREREG_sigma_hop_confirmatory.md`. It keeps the preregistered decision
surface deliberately narrow: descendant-disjoint splits, constant-Σ baseline, smooth 6-parameter Σ(hop), and a
one-sided hop-shuffle permutation test on the averaged held-out NLL gain.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from contextlib import closing
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emit_direction_blend import parse_responses
from emit_transitive_hops import hit_prob
from eval_relatedness import build_model
from mu_attention import OPS, Tokenizer, load_dag
from sample_sigma_hop_fresh_corpus import FreshCorpusError, LmdbTitleGraph, load_lmdb_slice_maps


DIR = ["subcategory", "subtopic", "element_of", "super_category"]
SYM = ["see_also", "assoc"]
SCHEMA_VERSION = 1
EXPECTED_HOPS = {1, 2, 3, 4, 5}


class ConfirmatoryInputError(ValueError):
    pass


class OverlapError(ConfirmatoryInputError):
    pass


@dataclass(frozen=True)
class FeatureGraphConfig:
    graph: Optional[str]
    candidate_lmdb: Optional[str] = None
    lmdb_root: Optional[str] = None
    exploratory_graph: Optional[str] = None
    title_i2s_db: str = "title_i2s"
    title_s2i_db: str = "title_s2i"
    lmdb_no_lock: bool = False


@dataclass(frozen=True)
class ConfirmatoryData:
    pairs: tuple[tuple[str, str], ...]
    hop: np.ndarray
    D: np.ndarray
    S: np.ndarray
    X: np.ndarray


def sigma_of_hop(params, hop):
    """Pre-registered smooth covariance: log-linear sigmas and tanh correlation."""
    aD, bD, aS, bS, c, e = params
    return np.exp(aD + bD * hop), np.exp(aS + bS * hop), np.tanh(c + e * hop)


def biv_nll(rD, rS, sD, sS, rho):
    # Apply the same numerical boundary to sample-rho baselines and tanh-parametrized smooth rhos.
    rho = np.clip(rho, -0.98, 0.98)
    sD = np.maximum(sD, 1e-6)
    sS = np.maximum(sS, 1e-6)
    zD, zS = rD / sD, rS / sS
    q = (zD**2 - 2 * rho * zD * zS + zS**2) / (1 - rho**2)
    return np.log(2 * np.pi) + np.log(sD * sS) + 0.5 * np.log(1 - rho**2) + 0.5 * q


def fit_sigma_of_hop(rD, rS, hop):
    corr = np.corrcoef(rD, rS)[0, 1] if len(rD) > 1 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    p0 = [
        np.log(np.std(rD) + 1e-6),
        0.0,
        np.log(np.std(rS) + 1e-6),
        0.0,
        np.arctanh(np.clip(corr, -0.9, 0.9)),
        0.0,
    ]

    def objective(params):
        sD, sS, rho = sigma_of_hop(params, hop)
        return biv_nll(rD, rS, sD, sS, rho).mean()

    result = minimize(
        objective,
        p0,
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-7},
    )
    if not result.success:
        warnings.warn(f"Nelder-Mead did not converge while fitting Sigma(hop): {result.message}")
    return result.x


def descendant_disjoint_split(pairs, seed, held_frac=0.30):
    rng = np.random.default_rng(seed)
    descendants = sorted({x for x, _ in pairs})
    rng.shuffle(descendants)
    held_desc = set(descendants[: max(1, int(held_frac * len(descendants)))])
    train = np.array([i for i, (x, _) in enumerate(pairs) if x not in held_desc], dtype=int)
    held = np.array([i for i, (x, _) in enumerate(pairs) if x in held_desc], dtype=int)
    return train, held


def residuals_for_split(D, S, X, train, held):
    rank = np.linalg.matrix_rank(X[train])
    if rank < X.shape[1]:
        warnings.warn(
            f"rank-deficient residual mean design on split: rank {rank} < {X.shape[1]}; "
            "using numpy lstsq minimum-norm solution"
        )
    out = []
    for y in (D, S):
        beta, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        out.append((y[train] - X[train] @ beta, y[held] - X[held] @ beta))
    return out


def split_gain(data, train, held, hop_for_sigma):
    (rD_train, rD_held), (rS_train, rS_held) = residuals_for_split(data.D, data.S, data.X, train, held)
    sD = np.std(rD_train) + 1e-6
    sS = np.std(rS_train) + 1e-6
    rho = np.corrcoef(rD_train, rS_train)[0, 1]
    if not np.isfinite(rho):
        rho = 0.0

    const_nll = biv_nll(rD_held, rS_held, sD, sS, rho).mean()
    params = fit_sigma_of_hop(rD_train, rS_train, hop_for_sigma[train].astype(float))
    sDf, sSf, rhof = sigma_of_hop(params, hop_for_sigma[held].astype(float))
    smooth_nll = biv_nll(rD_held, rS_held, sDf, sSf, rhof).mean()
    return const_nll - smooth_nll, const_nll, smooth_nll


def valid_splits(data, seeds, held_frac=0.30, min_train=30, min_held=12):
    splits = []
    skipped = []
    for seed in seeds:
        train, held = descendant_disjoint_split(data.pairs, seed, held_frac=held_frac)
        if len(train) < min_train or len(held) < min_held:
            skipped.append({"seed": seed, "train": len(train), "held": len(held)})
            continue
        splits.append((seed, train, held))
    return splits, skipped


def mean_gain(data, splits, hop_for_sigma):
    gains = []
    const_nll = []
    smooth_nll = []
    held_sizes = []
    for _, train, held in splits:
        gain, c_nll, s_nll = split_gain(data, train, held, hop_for_sigma)
        gains.append(gain)
        const_nll.append(c_nll)
        smooth_nll.append(s_nll)
        held_sizes.append(len(held))
    return {
        "mean_gain": float(np.mean(gains)),
        "split_gains": [float(x) for x in gains],
        "constant_nll": float(np.mean(const_nll)),
        "sigma_hop_nll": float(np.mean(smooth_nll)),
        "mean_held_pairs": float(np.mean(held_sizes)),
    }


def permutation_test(data, splits, k=1000, seed=1, allow_small_k=False):
    if k < 1000 and not allow_small_k:
        raise ValueError("permutation_test requires k >= 1000 for confirmatory use")
    observed = mean_gain(data, splits, data.hop)
    rng = np.random.default_rng(seed)
    # Sequential RNG use is intentional: it gives a reproducible ordered stream of hop shuffles for the audit log.
    null = np.array(
        [mean_gain(data, splits, data.hop[rng.permutation(len(data.hop))])["mean_gain"] for _ in range(k)]
    )
    p = (1 + int(np.sum(null >= observed["mean_gain"]))) / (k + 1)
    confirmed = bool(observed["mean_gain"] > 0 and p < 0.01)
    return {
        **observed,
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "permutation_k": int(k),
        "permutation_p": float(p),
        "decision_inputs": {"mean_gain": float(observed["mean_gain"]), "permutation_p": float(p)},
        # The positive-gain check repeats the preregistered direction even though the p-value is upper-tail.
        "confirmed": confirmed,
        "decision": "confirmed" if confirmed else "not confirmed",
    }


def gmu(obj, rel):
    val = (obj.get(rel, {}) or {}).get("mu_fwd" if rel in DIR else "mu", 0)
    return float(val)


def load_scored_pairs(score_in, responses, prefix="transitive_h"):
    with open(score_in, encoding="utf-8") as fh:
        rows = [ln.rstrip("\n").split("\t") for ln in fh if not ln.startswith("#")]
    byid = parse_responses(responses)
    pairs, hop, D, S = [], [], [], []
    dropped_no_response = 0
    dropped_malformed = 0
    for idx, row in enumerate(rows):
        if len(row) < 5:
            dropped_malformed += 1
            continue
        if not row[4].startswith(prefix):
            continue
        if idx not in byid:
            dropped_no_response += 1
            continue
        raw_hop = row[4][len(prefix) :]
        try:
            h = int(raw_hop)
        except ValueError as exc:
            raise ConfirmatoryInputError(
                f"cannot parse hop count {raw_hop!r} from row {idx} in {score_in}"
            ) from exc
        pairs.append((row[0], row[1]))
        hop.append(h)
        D.append(max((gmu(byid[idx], rel) for rel in DIR), default=0.0))
        S.append(max((gmu(byid[idx], rel) for rel in SYM), default=0.0))
    if dropped_malformed:
        warnings.warn(f"{dropped_malformed} malformed score rows ignored in {score_in}")
    if dropped_no_response:
        warnings.warn(f"{dropped_no_response} {prefix} rows dropped: no LLM response found in {responses}")
    if not pairs:
        raise ConfirmatoryInputError(f"no scored {prefix} pairs with responses found in {score_in}")
    return tuple(pairs), np.array(hop), np.array(D), np.array(S)


def validate_hop_range(hop):
    got = {int(h) for h in set(hop)}
    if got != EXPECTED_HOPS:
        raise ConfirmatoryInputError(f"expected hop levels {sorted(EXPECTED_HOPS)}, got {sorted(got)}")


def load_exploratory_nodes(graph_path):
    nodes = set()
    with open(graph_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 2:
                nodes.add(cols[0])
                nodes.add(cols[1])
    return nodes


def assert_no_node_overlap(pairs, exploratory_graph):
    if not exploratory_graph:
        raise ConfirmatoryInputError("--exploratory-graph is required for the preregistered no-overlap check")
    exploratory_nodes = load_exploratory_nodes(exploratory_graph)
    confirmatory_nodes = {n for pair in pairs for n in pair}
    overlap = sorted(confirmatory_nodes & exploratory_nodes)
    if overlap:
        preview = ", ".join(overlap[:10])
        raise OverlapError(f"confirmatory pairs overlap exploratory graph nodes ({len(overlap)} total): {preview}")
    return overlap


def degree_from_maps(parents, children):
    nodes = set(parents) | set(children)
    for values in parents.values():
        nodes.update(values)
    for values in children.values():
        nodes.update(values)
    return {
        node: len(parents.get(node, ())) + len(children.get(node, ()))
        for node in nodes
    }


def load_feature_graph(config):
    """Return parents/children/degree maps for confirmatory feature construction.

    TSV keeps backward compatibility with the exploratory runner. LMDB reuses the sampler's retained-slice
    traversal so `hit_prob` is computed on the same no-overlap/admin-filtered graph that produced the fresh pairs.
    """
    if config.graph:
        parents, children, deg = load_dag(config.graph)
        return parents, children, deg, {}

    if not config.candidate_lmdb:
        raise ConfirmatoryInputError("one of --graph or --candidate-lmdb is required")
    if not config.lmdb_root:
        raise ConfirmatoryInputError("--candidate-lmdb requires --lmdb-root for confirmatory feature construction")
    if not config.exploratory_graph:
        raise ConfirmatoryInputError("--exploratory-graph is required for LMDB no-overlap graph filtering")

    exploratory_nodes = load_exploratory_nodes(config.exploratory_graph)
    try:
        with closing(LmdbTitleGraph(
            config.candidate_lmdb,
            config.title_i2s_db,
            config.title_s2i_db,
            lock=not config.lmdb_no_lock,
        )) as lmdb_graph:
            root_id = lmdb_graph.node_id(config.lmdb_root)
            root_title, slice_nodes, parents, children, stats = load_lmdb_slice_maps(
                lmdb_graph, root_id, exploratory_nodes
            )
    except FreshCorpusError as exc:
        raise ConfirmatoryInputError(str(exc)) from exc

    if not slice_nodes:
        raise ConfirmatoryInputError(f"LMDB root `{config.lmdb_root}` produced an empty retained feature graph")
    deg = degree_from_maps(parents, children)
    return parents, children, deg, {
        "feature_graph_source": "lmdb",
        "feature_graph_lmdb": config.candidate_lmdb,
        "feature_graph_root": root_title,
        "feature_graph_slice_nodes": len(slice_nodes),
        "feature_graph_lmdb_stats": dict(sorted(stats.items())),
    }


def load_confirmatory_labels(score_in, responses, prefix, exploratory_graph):
    pairs, hop, D, S = load_scored_pairs(score_in, responses, prefix=prefix)
    assert_no_node_overlap(pairs, exploratory_graph)
    validate_hop_range(hop)
    return pairs, hop, D, S


def load_e5_cache_and_filter(pairs, hop, D, S, e5_cache):
    # This project cache is a trusted local torch pickle produced by the scoring pipeline. Do not use untrusted caches.
    cache = torch.load(e5_cache, weights_only=False)
    idx = {name: i for i, name in enumerate(cache["names"])}
    keep = np.array([i for i, (x, y) in enumerate(pairs) if x in idx and y in idx], dtype=int)
    dropped = len(pairs) - len(keep)
    if dropped:
        warnings.warn(f"{dropped}/{len(pairs)} scored pairs dropped: endpoint missing from e5 cache {e5_cache}")
    if len(keep) == 0:
        raise ConfirmatoryInputError("no scored pairs survived the e5-cache endpoint filter")
    filtered_pairs = tuple(pairs[i] for i in keep)
    return cache, idx, filtered_pairs, hop[keep], D[keep], S[keep]


def build_confirmatory_features(pairs, cache, idx, feature_graph_config, model_path, device):
    parents, _, deg, _ = load_feature_graph(feature_graph_config)
    tokenizer = Tokenizer(cache["query"], cache["passage"], idx, parents, deg)
    model = build_model(model_path, device)

    def mu(op, pair_batch):
        rows = [(x, y, OPS[op]) for x, y in pair_batch]
        out = []
        for i in range(0, len(rows), 512):
            batch = tokenizer.build(rows[i : i + 512], train=False)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                out += model(**batch).cpu().tolist()
        return np.array(out)

    muD = np.maximum(mu("HIER", pairs), mu("HIER", [(y, x) for x, y in pairs]))
    muS = mu("SYM", pairs)
    gd = np.array([hit_prob(parents, x, y) for x, y in pairs])
    return np.column_stack([muD, muS, gd, np.ones(len(pairs))])


def build_confirmatory_data_from_labels(pairs, hop, D, S, e5_cache, feature_graph_config, model_path, device):
    cache, idx, pairs, hop, D, S = load_e5_cache_and_filter(pairs, hop, D, S, e5_cache)
    validate_hop_range(hop)
    X = build_confirmatory_features(pairs, cache, idx, feature_graph_config, model_path, device)
    return ConfirmatoryData(pairs=pairs, hop=hop, D=D, S=S, X=X)


def build_confirmatory_data(
    score_in, responses, e5_cache, feature_graph_config, model_path, device, prefix, exploratory_graph
):
    pairs, hop, D, S = load_confirmatory_labels(score_in, responses, prefix, exploratory_graph)
    return build_confirmatory_data_from_labels(
        pairs, hop, D, S, e5_cache, feature_graph_config, model_path, device
    )


def _one_line(value):
    return str(value).replace("\r", " ").replace("\n", " ")


def render_markdown(result, args, n_pairs, hop_counts, skipped):
    decision = "confirmed" if result["confirmed"] else "not confirmed"
    corpus_note = _one_line(args.corpus_note)
    judge_note = _one_line(args.judge_note)
    return f"""# Confirmatory Σ(hop) run

Preregistration: `PREREG_sigma_hop_confirmatory.md`

```text
fresh corpus dump/root: {corpus_note}
node-overlap check with 100k_cats: passed
n scored pairs: {n_pairs}
hop counts: {hop_counts}
judge/prompt/model: {judge_note}
valid descendant-disjoint splits: {result['valid_splits']}
mean held pairs/split: {result['mean_held_pairs']:.1f}
observed mean gain: {result['mean_gain']:+.6f}
constant-Σ NLL: {result['constant_nll']:.6f}
Σ(hop) NLL: {result['sigma_hop_nll']:.6f}
hop-shuffle null mean: {result['null_mean']:+.6f}
hop-shuffle null 95%ile: {result['null_p95']:+.6f}
K: {result['permutation_k']}
permutation p: {result['permutation_p']:.6f}
decision: {decision}
skipped splits: {len(skipped)}
```
"""


def validate_preregistered_cli(args):
    expected = {
        "splits": 40,
        "held_frac": 0.30,
        "min_train": 30,
        "min_held": 12,
    }
    actual = {
        "splits": args.splits,
        "held_frac": args.held_frac,
        "min_train": args.min_train,
        "min_held": args.min_held,
    }
    mismatches = [f"{k}={actual[k]} (expected {v})" for k, v in expected.items() if actual[k] != v]
    if mismatches:
        raise SystemExit("non-preregistered split protocol: " + "; ".join(mismatches))
    if args.permutations < 1000:
        raise SystemExit("--permutations must be >= 1000 for the preregistered confirmatory test")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-in", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--e5-cache", required=True)
    graph_source = ap.add_mutually_exclusive_group(required=True)
    graph_source.add_argument("--graph", help="child<TAB>parent TSV graph for feature construction")
    graph_source.add_argument("--candidate-lmdb", help="Phase-1 category LMDB graph for feature construction")
    ap.add_argument("--lmdb-root", help="selected retained LMDB slice root, e.g. the sampled root recorded in the manifest")
    ap.add_argument("--title-i2s-db", default="title_i2s", help="LMDB uint32 id -> real category title sub-db")
    ap.add_argument("--title-s2i-db", default="title_s2i", help="LMDB real category title -> uint32 id sub-db")
    ap.add_argument("--lmdb-no-lock", action="store_true", help="open candidate LMDB with lock=False; use only for immutable fixtures")
    ap.add_argument("--model", default="model_prod.pt")
    ap.add_argument("--prefix", default="transitive_h")
    ap.add_argument("--exploratory-graph", required=True, help="PR #3517 100k_cats/category_parent.tsv for required no-overlap check")
    ap.add_argument("--splits", type=int, default=40)
    ap.add_argument("--held-frac", type=float, default=0.30)
    ap.add_argument("--min-train", type=int, default=30)
    ap.add_argument("--min-held", type=int, default=12)
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--perm-seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md-out", default=None)
    ap.add_argument("--corpus-note", required=True, help="fresh corpus dump/root, recorded before scoring")
    ap.add_argument("--judge-note", required=True, help="judge model, prompt template, and frozen predictor model provenance")
    args = ap.parse_args()
    if args.candidate_lmdb and not args.lmdb_root:
        ap.error("--lmdb-root is required when --candidate-lmdb is given")
    if args.lmdb_root and not args.candidate_lmdb:
        ap.error("--lmdb-root requires --candidate-lmdb")

    validate_preregistered_cli(args)
    device = torch.device(args.device)
    try:
        pairs, hop, D, S = load_confirmatory_labels(args.score_in, args.responses, args.prefix, args.exploratory_graph)
        preflight = ConfirmatoryData(pairs=pairs, hop=hop, D=D, S=S, X=np.ones((len(pairs), 1)))
        splits, skipped = valid_splits(
            preflight,
            range(args.splits),
            held_frac=args.held_frac,
            min_train=args.min_train,
            min_held=args.min_held,
        )
        if not splits:
            raise ConfirmatoryInputError("no valid descendant-disjoint splits before model inference")

        cache, idx, pairs, hop, D, S = load_e5_cache_and_filter(pairs, hop, D, S, args.e5_cache)
        validate_hop_range(hop)
        preflight = ConfirmatoryData(pairs=pairs, hop=hop, D=D, S=S, X=np.ones((len(pairs), 1)))
        splits, skipped = valid_splits(
            preflight,
            range(args.splits),
            held_frac=args.held_frac,
            min_train=args.min_train,
            min_held=args.min_held,
        )
        if not splits:
            raise ConfirmatoryInputError("no valid descendant-disjoint splits survived e5-cache filtering")

        feature_graph_config = FeatureGraphConfig(
            graph=args.graph,
            candidate_lmdb=args.candidate_lmdb,
            lmdb_root=args.lmdb_root,
            exploratory_graph=args.exploratory_graph,
            title_i2s_db=args.title_i2s_db,
            title_s2i_db=args.title_s2i_db,
            lmdb_no_lock=args.lmdb_no_lock,
        )
        X = build_confirmatory_features(pairs, cache, idx, feature_graph_config, args.model, device)
        data = ConfirmatoryData(pairs=pairs, hop=hop, D=D, S=S, X=X)
    except ConfirmatoryInputError as exc:
        raise SystemExit(str(exc)) from exc

    result = permutation_test(data, splits, k=args.permutations, seed=args.perm_seed)
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "n_pairs": len(data.pairs),
            "hop_counts": {int(h): int(np.sum(data.hop == h)) for h in sorted(set(data.hop))},
            "valid_splits": len(splits),
            "skipped_splits": skipped,
            "split_weighting": "equal_by_split",
            "decision_rule": "confirmed iff mean_gain > 0 and one-sided hop-shuffle p < 0.01",
        }
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
    if args.md_out:
        md = render_markdown(result, args, len(data.pairs), result["hop_counts"], skipped)
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write(md)


if __name__ == "__main__":
    main()
