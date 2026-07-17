#!/usr/bin/env python3
"""Pearltrees cheap-judge fusion: the first live deployment of the validated labeling recipe.

Pipeline (DESIGN_cheap_judge_pipeline.md, deployed on the Pearltrees campaign):
  1. luna bulk labels + a RANDOM 5.5 overlap (never conflict-selected) — scored upstream.
  2. Debias luna: global per-channel affine fit on the overlap ONLY (calibrate_luna pattern),
     applied before any covariance fit (bias first — the #3648 lesson).
  3. Fusion blocks on the overlap: prior ⊕ graph_D (hit_prob→D affine) ⊕ graph_S (4-feature
     linear) ⊕ luna, 6×6 joint residual covariance, shrinkage 0.05, correlated updates.
  4. Kalman-fused (D, S) targets for every bulk row → the fine-tune's distillation targets.
  5. Conflict routing: top --route-frac of NON-overlap rows by the |graph_D − prior_D| innovation
     → a 5.5 score-input file (extra labels; excluded from the covariance fit by construction).
  6. SHADOW bias-state fit (fit_bias_states — spec §5.1, opt-in after the enwiki encouraging
     null): per-(judge,bin,channel) offsets fit on the overlap-train split, diagnostics printed,
     held-overlap control-vs-treatment NLL reported DESCRIPTIVELY. Never applied to the fused
     targets in this run.

Evaluation frame: every number is fidelity to the gpt-5.5-low operating judge, never "semantic
accuracy". The overlap ladder here is DESCRIPTIVE (n≈300 overlap, one node-disjoint split) — the
confirmatory machinery stays run_sym_channel_fusion.py on enwiki.

  python3 run_pearltrees_fusion.py            # writes /tmp/mu_data/pt_fused_targets.tsv etc.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from emit_transitive_hops import hit_prob
from eval_luna_transfer import load_luna
from fine_tune_channel_heads import load_expanded
from fine_tune_fused_head import agnostic_readouts
from fit_bias_states import fit_bias_states, pair_distance_features
from mu_attention import Tokenizer
from node_disjoint_eval import node_disjoint_pair_split
from product_kalman import fit_residual_covariance
from run_judge_channel import correlated_update_H, nll_mahal
from run_product_kalman_logit import dequant
from run_product_kalman_realdata import affine_calibrate
from run_sym_channel_fusion import H4, RUNGS, calibrate_luna, s_marginal_nll, sym_graph_features
from sample_pearltrees_lateral import build_graph, load_policy
from sample_product_kalman_pearltrees_campaign import load_principal_paths

ROOT = os.path.dirname(os.path.abspath(__file__))
PAIRS_TSV = "/tmp/mu_data/pt_campaign_all_pairs.tsv"
LUNA_TSV = "/tmp/mu_data/pt_campaign_scored_luna.tsv"
TSV_55 = "/tmp/mu_data/pt_campaign_scored_55.tsv"
E5_CACHE = "/tmp/mu_data/pt_campaign_e5.pt"
PATHS_JSONL = os.path.join(ROOT, "..", "..", ".local", "data", "api_tree_paths_v8.jsonl")
TITLES_TSV = os.path.join(ROOT, "..", "..", ".local", "data", "pearltrees_api", "assembled_titles.tsv")


def load_scored_strict(path, expect_judge):
    """Fail-closed scored-TSV load: unique keys, judge column verified on every row."""
    keys, out = set(), {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip().split("\t")
        col = {c: i for i, c in enumerate(header)}
        for want in ("node", "root", "judge"):
            if want not in col:
                raise SystemExit(f"{path}: missing column {want!r}")
        from eval_luna_transfer import DIRR, SYMM
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < len(header):
                raise SystemExit(f"{path}: short row (partial scoring failure?): {ln[:80]!r}")
            if c[col["judge"]] != expect_judge:
                raise SystemExit(f"{path}: judge {c[col['judge']]!r} != expected {expect_judge!r}")
            key = (c[col["node"]], c[col["root"]])
            if key in keys:
                raise SystemExit(f"{path}: duplicate score key {key} — refusing to fail open")
            keys.add(key)
            out[key] = (max(float(c[col[f"mu[{r}]"]]) for r in DIRR),
                        max(float(c[col[f"mu[{r}]"]]) for r in SYMM))
    return out


def load_pearltrees_campaign(pairs_tsv=PAIRS_TSV, luna_tsv=LUNA_TSV, tsv_55=TSV_55,
                             e5_cache=E5_CACHE, paths_jsonl=PATHS_JSONL, titles_tsv=TITLES_TSV,
                             require_55=True):
    """Assemble the Pearltrees campaign dataset: title pairs, id-graph features, judges, tokenizer.

    Returns a ds dict shaped like load_campaign_datasets' (pairs/tags/tok/d/...) plus id-level
    graph handles (parents_id, id_pairs, in_graph) for the bias-state distance features, luna
    arrays over all rows, and the 5.5 overlap (indices + labels).  Join key everywhere is the
    (audited) title pair — the scored-TSV key.
    """
    forest = load_principal_paths(paths_jsonl, titles_tsv)
    parents_id, children_id = build_graph(forest)
    corrections = load_policy()

    def title_of(node_id):
        return corrections.get(node_id, forest["titles"].get(node_id, ""))

    rows = []
    with open(pairs_tsv, encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip().split("\t")
        col = {c: i for i, c in enumerate(header)}
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < len(header):
                continue
            rows.append(dict(pair_id=c[col["pair_id"]], account=c[col["account"]],
                             a_id=c[col["a_id"]], a_title=c[col["a_title"]],
                             b_id=c[col["b_id"]], b_title=c[col["b_title"]],
                             hop=int(c[col["hop"]]), tag=c[col["tag"]]))

    luna_by = load_scored_strict(luna_tsv, "gpt-5.6-luna")
    y55_by = {}
    if tsv_55 and os.path.exists(tsv_55):
        y55_by = load_scored_strict(tsv_55, "gpt-5.5-low")
    elif require_55:
        raise SystemExit(f"missing 5.5 overlap labels: {tsv_55}")

    missing_luna = [r for r in rows if (r["a_title"], r["b_title"]) not in luna_by]
    if missing_luna and require_55:
        raise SystemExit(f"{len(missing_luna)} campaign rows missing luna labels "
                         f"(e.g. {missing_luna[0]['pair_id']}) — refusing to fail open")
    kept = [r for r in rows if (r["a_title"], r["b_title"]) in luna_by]
    pairs = [(r["a_title"], r["b_title"]) for r in kept]
    tags = [r["tag"] for r in kept]
    id_pairs = [((r["account"], r["a_id"]), (r["account"], r["b_id"])) for r in kept]

    cache = torch.load(e5_cache, weights_only=False)
    idx = {name: i for i, name in enumerate(cache["names"])}
    missing_e5 = [pairs[i] for i, (x, y) in enumerate(pairs) if x not in idx or y not in idx]
    if missing_e5:
        raise SystemExit(f"{len(missing_e5)} rows missing from e5 cache (e.g. {missing_e5[0]}) — "
                         "rebuild prep_pearltrees_e5.py output; refusing to fail open")

    # title-keyed parent/degree maps for the Tokenizer's lineage sampling (id graph → titles)
    parents_title, deg_title = {}, {}
    for child, ps in parents_id.items():
        ct = title_of(child[1])
        if not ct:
            continue
        pts = [title_of(p[1]) for p in ps if title_of(p[1])]
        if pts:
            parents_title.setdefault(ct, [])
            parents_title[ct] = sorted(set(parents_title[ct]) | set(pts))
    for parent, kids in children_id.items():
        pt = title_of(parent[1])
        if pt:
            deg_title[pt] = deg_title.get(pt, 0) + len(kids)

    tok = Tokenizer(cache["query"], cache["passage"], idx, parents_title, deg_title)
    d = np.array([hit_prob(parents_id, a, b) for a, b in id_pairs])
    luna = np.array([luna_by[p] for p in pairs])
    overlap_idx = np.array([i for i, p in enumerate(pairs) if p in y55_by], dtype=int)
    y55 = np.array([y55_by[pairs[i]] for i in overlap_idx]) if len(overlap_idx) else np.zeros((0, 2))

    in_graph = set(parents_id) | {p for ps in parents_id.values() for p in ps}
    return dict(rows=kept, pairs=pairs, tags=tags, id_pairs=id_pairs, tok=tok, d=d,
                luna=luna, overlap_idx=overlap_idx, y55=y55, parents_id=parents_id,
                in_graph=in_graph, cache=cache, title_of=title_of)


def group_of(tag):
    return "principal" if tag.startswith("principal") else tag


def campaign_split(pairs, tags, seed=0):
    """The ONE node-disjoint campaign split shared by the target factory, trainer, and eval."""
    from node_disjoint_eval import node_disjoint_pair_split

    return node_disjoint_pair_split(pairs, seed, strata=[group_of(t) for t in tags])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_prod_namecond.pt"))
    ap.add_argument("--shrink", type=float, default=0.05)
    ap.add_argument("--route-frac", type=float, default=0.10,
                    help="fraction of ALL bulk rows routed to 5.5 by |graph_D - prior_D| innovation")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="node-disjoint overlap split seed for the descriptive ladder + shadow fit")
    ap.add_argument(
        "--fit-overlap",
        choices=["full", "train-only"],
        default="full",
        help=("full = production target factory (all overlap rows fit the blocks); train-only = "
              "LEAKAGE-FREE eval targets — calibrations/covariance fit only on overlap rows on the "
              "TRAIN side of the shared campaign split, so held overlap labels never touch the "
              "factory (external review finding 1); output files get an _eval suffix"),
    )
    ap.add_argument("--out-prefix", default="/tmp/mu_data/pt")
    a = ap.parse_args(argv)

    ds = load_pearltrees_campaign()
    pairs, tags, luna, d = ds["pairs"], ds["tags"], ds["luna"], ds["d"]
    ov, y55 = ds["overlap_idx"], ds["y55"]
    print(f"pearltrees campaign: {len(pairs)} luna-scored rows; overlap with 5.5: {len(ov)}")
    if a.fit_overlap == "train-only":
        split_c = campaign_split(pairs, tags, a.split_seed)
        train_side = set(split_c.train.tolist())
        keep_mask = np.array([i in train_side for i in ov])
        ov, y55 = ov[keep_mask], y55[keep_mask]
        a.out_prefix += "_eval" if not a.out_prefix.endswith("_eval") else ""
        print(f"train-only target factory: {len(ov)} overlap rows on the campaign-train side "
              f"(held overlap labels excluded from every fit)")

    ref, _ = load_expanded(a.ckpt, dev="cpu")
    ref.eval()
    ro = agnostic_readouts(ref, {"pairs": pairs, "tok": ds["tok"]}, "cpu")
    prior = np.column_stack([ro["prior_D"], ro["prior_S"]])

    F = sym_graph_features(ds["parents_id"], ds["id_pairs"])
    X = np.column_stack([F, np.ones(len(F))])
    y_ov = dequant(y55.copy())

    # luna tilt table on the overlap (the enwiki-vs-pearltrees comparison for the report)
    print("\nluna-vs-5.5 tilt on the RANDOM overlap (bias = luna - 5.5; fidelity frame):")
    print(f"  {'stratum':14s} {'n':>4s} {'D corr':>8s} {'D bias':>8s} {'S corr':>8s} {'S bias':>8s}")
    tag_groups = sorted(set(tags[i] for i in ov))
    grp = lambda t: "principal" if t.startswith("principal") else t
    for g in sorted(set(map(grp, tag_groups))) + ["ALL"]:
        rows = [k for k, i in enumerate(ov) if g == "ALL" or grp(tags[i]) == g]
        if len(rows) < 3:
            continue
        lv = luna[ov[rows]]
        yv = y55[rows]
        cD = np.corrcoef(lv[:, 0], yv[:, 0])[0, 1] if yv[:, 0].std() > 1e-9 else float("nan")
        cS = np.corrcoef(lv[:, 1], yv[:, 1])[0, 1] if yv[:, 1].std() > 1e-9 else float("nan")
        print(f"  {g:14s} {len(rows):4d} {cD:+8.3f} {(lv[:, 0]-yv[:, 0]).mean():+8.3f} "
              f"{cS:+8.3f} {(lv[:, 1]-yv[:, 1]).mean():+8.3f}")

    # ---- fusion blocks on the full RANDOM overlap (production fit) ----
    graph_D = affine_calibrate(d[ov], y_ov[:, 0], d)
    beta, *_ = np.linalg.lstsq(X[ov], y_ov[:, 1], rcond=None)
    graph_S = X @ beta
    luna_cal = np.column_stack([
        affine_calibrate(luna[ov, 0], y_ov[:, 0], luna[:, 0]),
        affine_calibrate(luna[ov, 1], y_ov[:, 1], luna[:, 1]),
    ])
    meas = np.column_stack([graph_D, graph_S, luna_cal[:, 0], luna_cal[:, 1]])
    E = np.column_stack([y_ov - prior[ov], meas[ov] - y_ov[:, [0, 1, 0, 1]]])
    cov = fit_residual_covariance(E, shrinkage=a.shrink)
    P0, C_pm, R0 = cov[:2, :2], cov[:2, 2:], cov[2:, 2:]
    print(f"\nfusion blocks fit on the {len(ov)}-row RANDOM overlap (shrink {a.shrink}):")
    print(f"  R diag (gD, gS, lunaD, lunaS): {np.round(np.diag(R0), 4).tolist()}")
    print(f"  P0 diag (prior D, S): {np.round(np.diag(P0), 4).tolist()}")

    post = np.zeros((len(pairs), 2))
    post_var = np.zeros((len(pairs), 2))
    for i in range(len(pairs)):
        xp, Pp = correlated_update_H(prior[i], P0, meas[i], R0, C_pm, H4)
        post[i] = xp
        post_var[i] = np.diag(Pp)

    # ---- conflict routing (|graph_D - prior_D| innovation; excluded from any future block fit) ----
    innovation = np.abs(graph_D - prior[:, 0])
    n_route = int(round(a.route_frac * len(pairs)))
    non_overlap = np.array([i for i in range(len(pairs)) if i not in set(ov.tolist())])
    routed = non_overlap[np.argsort(-innovation[non_overlap])][:n_route]
    route_path = f"{a.out_prefix}_conflict_score_in.tsv"
    with open(route_path, "w", encoding="utf-8") as f:
        f.write("# node_title\troot_title\tcur_relation\tconf\tneighborhood\tnode_type\troot_type\traw\n")
        for i in routed:
            rel = "subtopic" if tags[i].startswith("principal") else "assoc"
            f.write(f"{pairs[i][0]}\t{pairs[i][1]}\t{rel}\t1.0\t{tags[i]}\t"
                    f"pearltrees_collection\tpearltrees_collection\t\n")
    print(f"\nconflict routing: top {n_route} non-overlap rows by |graph_D - prior_D| "
          f"(innovation range {innovation[routed].min():.3f}..{innovation[routed].max():.3f}) "
          f"-> {route_path}")

    # ---- fused targets for the fine-tune ----
    targets_path = f"{a.out_prefix}_fused_targets.tsv"
    with open(targets_path, "w", encoding="utf-8") as f:
        f.write("# pair_id\tnode\troot\ttag\thop\tpost_D\tpost_S\tpost_var_D\tpost_var_S\t"
                "prior_D\tprior_S\tgraph_D\tgraph_S\tluna_D_cal\tluna_S_cal\tluna_D_raw\tluna_S_raw\t"
                "d_walk\ty55_D\ty55_S\tis_overlap\trouted\n")
        ov_set, routed_set = set(ov.tolist()), set(routed.tolist())
        for i, r in enumerate(ds["rows"]):
            in_ov = i in ov_set
            k = np.where(ov == i)[0]
            y55d = f"{y55[k[0], 0]:.4f}" if in_ov else ""
            y55s = f"{y55[k[0], 1]:.4f}" if in_ov else ""
            f.write("\t".join(map(str, [
                r["pair_id"], pairs[i][0], pairs[i][1], tags[i], r["hop"],
                f"{post[i, 0]:.4f}", f"{post[i, 1]:.4f}",
                f"{post_var[i, 0]:.5f}", f"{post_var[i, 1]:.5f}",
                f"{prior[i, 0]:.4f}", f"{prior[i, 1]:.4f}",
                f"{graph_D[i]:.4f}", f"{graph_S[i]:.4f}",
                f"{luna_cal[i, 0]:.4f}", f"{luna_cal[i, 1]:.4f}",
                f"{luna[i, 0]:.4f}", f"{luna[i, 1]:.4f}",
                f"{d[i]:.4f}", y55d, y55s, int(in_ov), int(i in routed_set),
            ])) + "\n")
    print(f"fused targets -> {targets_path}")

    # ---- descriptive node-disjoint overlap ladder + SHADOW bias states ----
    ov_pairs = [pairs[i] for i in ov]
    ov_tags = [("principal" if tags[i].startswith("principal") else tags[i]) for i in ov]
    split = node_disjoint_pair_split(ov_pairs, a.split_seed, strata=ov_tags)
    tr_o, he_o = ov[split.train], ov[split.held]
    ytr = dequant(y55[split.train].copy())
    yhe = dequant(y55[split.held].copy())
    gD2 = affine_calibrate(d[tr_o], ytr[:, 0], d)
    beta2, *_ = np.linalg.lstsq(X[tr_o], ytr[:, 1], rcond=None)
    gS2 = X @ beta2
    lc2 = np.column_stack([
        affine_calibrate(luna[tr_o, 0], ytr[:, 0], luna[:, 0]),
        affine_calibrate(luna[tr_o, 1], ytr[:, 1], luna[:, 1]),
    ])
    meas2 = np.column_stack([gD2, gS2, lc2[:, 0], lc2[:, 1]])

    feats = pair_distance_features(ds["parents_id"], ds["id_pairs"], in_graph=ds["in_graph"])
    y_all = np.zeros((len(pairs), 2))
    y_all[tr_o] = ytr
    resid = {
        ("graph", "D"): meas2[:, 0] - y_all[:, 0],
        ("graph", "S"): meas2[:, 1] - y_all[:, 1],
        ("luna", "D"): meas2[:, 2] - y_all[:, 0],
        ("luna", "S"): meas2[:, 3] - y_all[:, 1],
    }
    print(f"\nSHADOW bias states (spec §5.1; NOT applied to the fused targets) — "
          f"fit on {len(tr_o)} overlap-train rows:")
    states = fit_bias_states(feats, tr_o, resid, cv_pairs=pairs, verbose=True)

    E2 = np.column_stack([ytr - prior[tr_o], meas2[tr_o] - ytr[:, [0, 1, 0, 1]]])
    cov2 = fit_residual_covariance(E2, shrinkage=a.shrink)
    variants = {"affine": meas2}
    corr4 = np.column_stack([states.corrections(k) for k in
                             [("graph", "D"), ("graph", "S"), ("luna", "D"), ("luna", "S")]])
    variants["affine+bins(shadow)"] = meas2 - corr4
    print(f"\ndescriptive held-overlap ladder ({len(he_o)} node-disjoint held rows; NLL, fidelity to 5.5):")
    print(f"  {'variant':22s} {'rung':18s} {'joint':>9s} {'S-marg':>9s} {'D-marg':>9s}")
    for vname, mv in variants.items():
        E3 = np.column_stack([ytr - prior[tr_o], mv[tr_o] - ytr[:, [0, 1, 0, 1]]])
        cov3 = fit_residual_covariance(E3, shrinkage=a.shrink)
        P3, C3, R3 = cov3[:2, :2], cov3[:2, 2:], cov3[2:, 2:]
        for rung, selected in RUNGS.items():
            js, ss, dsc = [], [], []
            for k, i in enumerate(he_o):
                if not selected:
                    xp, Pp = prior[i], P3
                else:
                    xp, Pp = correlated_update_H(prior[i], P3, mv[i][selected],
                                                 R3[np.ix_(selected, selected)],
                                                 C3[:, selected], H4[selected])
                js.append(nll_mahal(yhe[k] - xp, Pp)[0])
                ss.append(s_marginal_nll(yhe[k, 1] - xp[1], Pp[1, 1]))
                dsc.append(s_marginal_nll(yhe[k, 0] - xp[0], Pp[0, 0]))
            print(f"  {vname:22s} {rung:18s} {np.mean(js):+9.4f} {np.mean(ss):+9.4f} {np.mean(dsc):+9.4f}")

    manifest = {
        "rows": len(pairs), "overlap": int(len(ov)), "routed": int(n_route),
        "route_frac": a.route_frac, "shrink": a.shrink,
        "R_diag": np.round(np.diag(R0), 5).tolist(),
        "P0_diag": np.round(np.diag(P0), 5).tolist(),
        "ckpt": os.path.basename(a.ckpt),
        "strata": {t: tags.count(t) for t in sorted(set(tags))},
    }
    with open(f"{a.out_prefix}_fusion_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f"\nmanifest -> {a.out_prefix}_fusion_manifest.json")


if __name__ == "__main__":
    main()
