#!/usr/bin/env python3
"""Figures for PAPER_sigma_hop_confirmatory.md, from the completed confirmatory-run artifacts.

All figures are DESCRIPTIVE views of the raw fuzzy labels / recorded split gains — none re-computes the
confirmatory statistic (that stays `sigma_hop_confirmatory.py` + REPORT_sigma_hop_confirmatory.md). The
label-geometry figures use raw (D,S) labels per hop (not mean-model residuals), which is stated in captions.

  python3 make_sigma_hop_paper_figures.py \
      --result-json /tmp/mu_data/sigma_hop_confirmatory_result.json \
      --fresh-scored /tmp/mu_data/sigma_hop_fresh_scored_gpt55low.tsv \
      --expl-score-in /tmp/mu_data/multihop_score_in.tsv --expl-responses /tmp/mu_data/multihop_resp.txt \
      --out-dir figures/sigma_hop
"""
import argparse, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emit_direction_blend import parse_responses

DIR = ["subcategory", "subtopic", "element_of", "super_category"]
SYM = ["see_also", "assoc"]
HOPS = range(1, 6)


def load_fresh(scored_tsv):
    """Fresh Behavior slice: scored TSV already has mu[...] columns."""
    with open(scored_tsv, encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip().split("\t")
        col = {c: i for i, c in enumerate(header)}
        out = {h: ([], []) for h in HOPS}
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < len(header):
                continue
            nb = c[col["neighborhood"]]
            if not nb.startswith("transitive_h"):
                continue
            h = int(nb[len("transitive_h"):])
            D = max(float(c[col[f"mu[{r}]"]]) for r in DIR)
            S = max(float(c[col[f"mu[{r}]"]]) for r in SYM)
            out[h][0].append(D); out[h][1].append(S)
    return {h: (np.array(d), np.array(s)) for h, (d, s) in out.items()}


def load_exploratory(score_in, responses, prefix="transitive_h"):
    """Exploratory multihop pairs: labels come from the raw responses file."""
    rows = [ln.rstrip("\n").split("\t") for ln in open(score_in, encoding="utf-8") if not ln.startswith("#")]
    byid = parse_responses(responses)

    def gmu(o, rel):
        e = o.get(rel, {}) or {}
        return float(e.get("mu_fwd", e.get("mu", 0)) or 0)
    out = {}
    for h in HOPS:
        idx = [i for i, r in enumerate(rows) if len(r) > 4 and r[4] == f"{prefix}{h}" and i in byid]
        D = np.array([max(gmu(byid[i], r) for r in DIR) for i in idx])
        S = np.array([max(gmu(byid[i], r) for r in SYM) for i in idx])
        out[h] = (D, S)
    return out


def cov_ellipse(ax, D, S, **kw):
    """2-sigma empirical covariance ellipse."""
    C = np.cov(D, S)
    w, V = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(V[1, -1], V[0, -1]))
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((D.mean(), S.mean()), 4 * np.sqrt(w[-1]), 4 * np.sqrt(w[0]), angle=ang,
                         fill=False, **kw))


def fig_split_gains(result_json, out):
    r = json.load(open(result_json))
    g = np.array(r["split_gains"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(g, bins=16, color="#4878b0", edgecolor="white")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(r["mean_gain"], color="#d1495b", lw=1.6,
               label=f"mean gain +{r['mean_gain']:.3f}")
    ax.axvline(r["null_p95"], color="#666", ls="--", lw=1.2,
               label=f"hop-shuffle null 95%ile +{r['null_p95']:.4f}")
    ax.set_xlabel("per-split held-out NLL gain: constant-Σ − Σ(hop)")
    ax.set_ylabel("splits")
    ax.set_title(f"Confirmatory split gains ({int((g > 0).sum())}/{len(g)} positive; permutation p={r['permutation_p']:.4g})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_geometry_curves(fresh, expl, out):
    """Per-hop raw-label sigma_D, sigma_S, rho for both corpora — descriptive, raw labels not residuals."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for name, data, colr in [("exploratory (100k_cats)", expl, "#4878b0"),
                             ("confirmatory (Behavior slice)", fresh, "#d1495b")]:
        hs = list(HOPS)
        sD = [data[h][0].std() for h in hs]
        sS = [data[h][1].std() for h in hs]
        rho = [np.corrcoef(data[h][0], data[h][1])[0, 1] for h in hs]
        for ax, y, lab in zip(axes, [sD, sS, rho], ["σ_D (raw labels)", "σ_S (raw labels)", "ρ(D,S) (raw labels)"]):
            ax.plot(hs, y, "o-", color=colr, label=name)
            ax.set_xlabel("hop"); ax.set_title(lab, fontsize=10); ax.set_xticks(hs)
    axes[2].axhline(0, color="k", lw=0.6)
    axes[0].legend(fontsize=7)
    fig.suptitle("Raw-label covariance geometry vs hop (descriptive; the test statistic uses mean-model residuals)",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_fresh_ellipses(fresh, out):
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.0), sharex=True, sharey=True)
    for ax, h in zip(axes, HOPS):
        D, S = fresh[h]
        ax.scatter(D, S, s=8, alpha=0.55, color="#4878b0")
        cov_ellipse(ax, D, S, color="#d1495b", lw=1.4)
        ax.set_title(f"h={h}  ρ={np.corrcoef(D, S)[0, 1]:+.2f}", fontsize=10)
        ax.set_xlabel("D (directional μ)")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("S (symmetric μ)")
    fig.suptitle("Fresh Behavior slice: raw (D,S) labels with 2σ empirical covariance ellipses per hop", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-json", default="/tmp/mu_data/sigma_hop_confirmatory_result.json")
    ap.add_argument("--fresh-scored", default="/tmp/mu_data/sigma_hop_fresh_scored_gpt55low.tsv")
    ap.add_argument("--expl-score-in", default="/tmp/mu_data/multihop_score_in.tsv")
    ap.add_argument("--expl-responses", default="/tmp/mu_data/multihop_resp.txt")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "sigma_hop"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    fresh = load_fresh(a.fresh_scored)
    expl = load_exploratory(a.expl_score_in, a.expl_responses)
    for h in HOPS:
        print(f"h{h}: fresh n={len(fresh[h][0])}  expl n={len(expl[h][0])}")

    fig_split_gains(a.result_json, os.path.join(a.out_dir, "fig1_split_gains.png"))
    fig_geometry_curves(fresh, expl, os.path.join(a.out_dir, "fig2_geometry_vs_hop.png"))
    fig_fresh_ellipses(fresh, os.path.join(a.out_dir, "fig3_fresh_ellipses.png"))
    print(f"wrote 3 figures → {a.out_dir}")


if __name__ == "__main__":
    main()
