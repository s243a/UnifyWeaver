#!/usr/bin/env python3
"""Render the cheap-judge-pipeline architecture diagrams (DESIGN_cheap_judge_pipeline.md) → figures/*.png.

Two figures: (1) the budget/data flow — structural sampling, luna bulk, overlap core, routed calls,
sonnet-5 tiebreaker, block fitting, fused targets, distillation, and the two-timescale maintenance loops;
(2) the fusion architecture — measurement channels + H, the partitioned joint covariance, the correlated
gain, and the amortized-head vs explicit-filter split at inference.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figures")

C = {"cheap": "#dbeafe", "mid": "#fde68a", "gold": "#fca5a5", "free": "#d1fae5",
     "fit": "#e9d5ff", "head": "#fbcfe8", "note": "#f3f4f6"}


def box(ax, x, y, w, h, text, fc, fs=8.5, ec="#374151"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, text="", style="-", color="#374151", fs=7.5, lw=1.2, curve=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 lw=lw, ls=style, color=color, connectionstyle=f"arc3,rad={curve}"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.012, text, ha="center", va="bottom",
                fontsize=fs, color=color)


def fig_dataflow():
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Cheap-judge pipeline — budget & data flow (DESIGN_cheap_judge_pipeline.md §1)",
                 fontsize=12, pad=12)

    box(ax, 0.02, 0.78, 0.20, 0.15,
        "STRUCTURAL SAMPLER\nrandom node → D parent hops\n(h1..h5 quotas + sib/cous/rand)\ncoarse D coverage, pre-scoring", C["free"])
    box(ax, 0.30, 0.80, 0.17, 0.11, "LUNA scores ALL\n(cheap pass, k× volume)", C["cheap"])
    box(ax, 0.30, 0.62, 0.17, 0.10, "D-bin top-up\n(fill ambiguous middle\nusing luna's D)", C["free"])
    arrow(ax, 0.22, 0.855, 0.30, 0.855)
    arrow(ax, 0.385, 0.80, 0.385, 0.72)
    arrow(ax, 0.30, 0.67, 0.11, 0.78, style="--", text="resample")

    box(ax, 0.55, 0.83, 0.19, 0.10, "5.5 RANDOM overlap core\n(~500-700 rows — calibration)", C["mid"])
    box(ax, 0.55, 0.68, 0.19, 0.10, "5.5 ROUTED calls\nconflict = |graph − prior|\n(extra labels, NOT in fit)", C["mid"])
    box(ax, 0.55, 0.53, 0.19, 0.10, "SONNET 5 low tiebreaker\non luna↔5.5 disagreements\n(cross-family; R from random slice)", C["gold"])
    arrow(ax, 0.47, 0.87, 0.55, 0.88)
    arrow(ax, 0.47, 0.83, 0.55, 0.73)
    arrow(ax, 0.74, 0.83, 0.74, 0.63, text="disagree")

    box(ax, 0.55, 0.32, 0.19, 0.13,
        "FIT ON OVERLAP (batch)\naffine bias per channel/stratum\n5×5+ joint covariance\n(shrinkage → SPD)", C["fit"])
    arrow(ax, 0.645, 0.83, 0.645, 0.45)

    box(ax, 0.28, 0.32, 0.19, 0.13,
        "EXPLICIT KALMAN\ncorrelated update over\nprior ⊕ graph_D ⊕ graph_S ⊕ luna\n→ fused posteriors (bulk)", C["fit"])
    arrow(ax, 0.55, 0.385, 0.47, 0.385, text="blocks")
    arrow(ax, 0.385, 0.62, 0.375, 0.45, text="luna bulk")
    box(ax, 0.02, 0.34, 0.17, 0.09, "GRAPH (free)\nwalk d → D\nlateral distance → S", C["free"])
    arrow(ax, 0.19, 0.385, 0.28, 0.385)

    box(ax, 0.28, 0.09, 0.19, 0.12,
        "DISTILL\nkalman-fused name head\n(+ channel heads keep\nraw supervision)", C["head"])
    arrow(ax, 0.375, 0.32, 0.375, 0.21, text="targets")
    box(ax, 0.55, 0.09, 0.19, 0.12,
        "DEPLOY\nmeasurement-free: fused head\nmeasurements in hand:\nexplicit filter (exact)", C["head"])
    arrow(ax, 0.47, 0.15, 0.55, 0.15)

    box(ax, 0.80, 0.32, 0.185, 0.24,
        "TWO TIMESCALES\nfast: judge drift →\nre-fit ~20 block numbers\non a fresh overlap slice\n\nslow: re-distill head\n\noverlap slice doubles as\ndrift monitor (innovation χ²)", C["note"], fs=8)
    arrow(ax, 0.80, 0.42, 0.745, 0.40, style="--")
    ax.text(0.02, 0.02, "colors: green = free/structural   blue = cheap judge   yellow = full-price judge   "
                        "red = escalation   purple = filter   pink = model", fontsize=8, color="#4b5563")
    fig.savefig(os.path.join(OUT, "pipeline_dataflow.png"), dpi=170, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "pipeline_dataflow.svg"), bbox_inches="tight")
    plt.close(fig)


def fig_fusion():
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Fusion architecture — channels, priced correlations, and the two Kalman forms (§2–§3)",
                 fontsize=12, pad=12)

    chans = [("model prior  μ_D μ_S   (state)", C["head"]),
             ("graph walk d  → D        H=[1,0]", C["free"]),
             ("graph lateral dist → S   H=[0,1]", C["free"]),
             ("luna D → D               H=[1,0]", C["cheap"]),
             ("luna S → S               H=[0,1]", C["cheap"]),
             ("5.5 / sonnet (routed rows only)", C["mid"])]
    for i, (t, fc) in enumerate(chans):
        box(ax, 0.02, 0.80 - i * 0.105, 0.24, 0.085, t, fc, fs=8)
    ax.text(0.14, 0.90, "measurement channels (bias-corrected: affine per channel, fit on overlap)",
            fontsize=8.5, ha="center", style="italic")

    box(ax, 0.34, 0.44, 0.24, 0.38,
        "JOINT RESIDUAL COVARIANCE\n(batch fit on overlap, shrinkage)\n\n"
        "⎡ P₀  │  C  ⎤   P₀: prior error\n"
        "⎢─────┼─────⎥   R: channel errors\n"
        "⎣ Cᵀ  │  R  ⎦   C: cross-correlations\n\n"
        "correlated gain\nK = (PHᵀ + C) S⁻¹\nS = HPHᵀ + R + HC + (HC)ᵀ", C["fit"], fs=8.5)
    for i in range(6):
        arrow(ax, 0.26, 0.843 - i * 0.105, 0.34, 0.70 - i * 0.04, lw=0.9)

    box(ax, 0.34, 0.16, 0.24, 0.15,
        "POSTERIOR  (D̂, Ŝ), P_post\nnon-degenerate when the judge\nis noisy (pull 0.08–0.13)", C["fit"])
    arrow(ax, 0.46, 0.44, 0.46, 0.31)

    box(ax, 0.70, 0.62, 0.27, 0.20,
        "AMORTIZED (slow): kalman-fused JUDGE\nname-conditioned head, r = 0 card prior\ntrained by distilling posteriors\n"
        "one forward pass, no measurements\ncorrelations baked in — frozen", C["head"], fs=8.5)
    box(ax, 0.70, 0.34, 0.27, 0.20,
        "EXPLICIT (fast): filter at inference\nruns when measurements exist\n(graph is ALWAYS free)\n"
        "exact; drift → re-fit blocks only\n[open: step-5 head-vs-filter A/B]", C["fit"], fs=8.5)
    arrow(ax, 0.58, 0.28, 0.70, 0.68, text="distill (train time)", curve=0.25)
    arrow(ax, 0.58, 0.23, 0.70, 0.40, text="same update, live inputs", curve=-0.1)

    box(ax, 0.70, 0.08, 0.27, 0.18,
        "⚠ FUSED-JUDGE EXCLUSION RULE\nthe posterior is a linear combination\nof its inputs → near-deterministically\n"
        "correlated with them. It REPLACES\nluna/graph/prior downstream —\nnever fused beside them.", "#fee2e2", fs=8.5)
    ax.text(0.14, 0.13, "batch blocks = time-invariant filter\n(stability; no time axis in exchangeable pairs;\n"
                        "honest splits; external error estimate)\nalternatives ladder: DESIGN §4", fontsize=8,
            ha="center", color="#4b5563")
    fig.savefig(os.path.join(OUT, "fusion_architecture.png"), dpi=170, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fusion_architecture.svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_dataflow()
    fig_fusion()
    print(f"figures → {OUT}/pipeline_dataflow.{{png,svg}}, fusion_architecture.{{png,svg}}")
