#!/usr/bin/env python3
"""Unit tests for the JointPosterior + 1/d additions (PR follow-up to #3488). Pytest-free; run directly.
Proves the estimation logic (AURC, margin gate, the 1/d source) is correct WITHOUT the full dataset —
the end-to-end relation-classification run is data-gated on a graded Wikipedia round overlapping the
struct embedding, but these deterministic checks are what actually need to be right."""
import os, tempfile
import numpy as np
from mu_posterior import aurc, aurc_boot, margin_conf, struct_dist_fn


def test_aurc_monotone():
    # confidence that TRACKS correctness ⇒ low AURC; ANTI-tracks ⇒ high; all-correct ⇒ 0; all-wrong ⇒ 1.
    correct = [1, 1, 0, 0]
    good = [0.9, 0.8, 0.2, 0.1]      # confident-when-correct
    bad = [0.1, 0.2, 0.8, 0.9]       # confident-when-wrong
    assert aurc(good, correct) < aurc(bad, correct), "AURC must reward correctness-tracking confidence"
    assert abs(aurc([0.9, 0.8, 0.7, 0.6], [1, 1, 1, 1]) - 0.0) < 1e-9, "all-correct ⇒ AURC 0"
    assert abs(aurc([0.9, 0.8, 0.7, 0.6], [0, 0, 0, 0]) - 1.0) < 1e-9, "all-wrong ⇒ AURC 1"
    # exact value for the 'good' case: risk_at_k = [0,0,1/3,1/2] ⇒ mean (5/6)/4 = 5/24 ≈ 0.2083
    assert abs(aurc(good, correct) - (0 + 0 + 1/3 + 1/2) / 4) < 1e-9
    print("PASS: AURC monotone + exact")


def test_aurc_boot_ci():
    rng = np.random.default_rng(0)
    correct = (rng.random(300) < 0.7).astype(float)
    conf = correct * 0.5 + rng.random(300) * 0.5      # weakly correctness-tracking
    a, lo, hi = aurc_boot(conf, correct, B=200, seed=1)
    assert lo <= a <= hi, f"point {a} should sit within bootstrap CI [{lo},{hi}]"
    assert hi - lo < 0.3, "CI should be reasonably tight at n=300"
    print(f"PASS: AURC bootstrap CI (a={a:.3f} [{lo:.3f},{hi:.3f}])")


def test_margin_conf():
    proba = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
    m = margin_conf(proba)
    assert np.allclose(m, [0.5, 0.05]), f"margin (top1-top2) wrong: {m}"
    print("PASS: margin_conf (top1−top2)")


def test_struct_dist_fn():
    import torch
    d = tempfile.mkdtemp()
    path = os.path.join(d, "se.pt")
    nodes = ["A", "B"]
    emb = torch.tensor([[0.0, 0.0], [3.0, 4.0]])      # ‖Δ‖ = 5 ⇒ 3/(1+5) = 0.5
    torch.save({"nodes": nodes, "emb": emb}, path)
    dist = struct_dist_fn(path)
    assert abs(dist("A", "B") - 0.5) < 1e-6, f"1/d wrong: {dist('A','B')}"
    assert dist("A", "A") == 3.0, "identical nodes ⇒ 3/(1+0)=3"
    # the /3 normalisation (used in emit_blend_judge / eval): identical ⇒ dist01 = min(1, 3/3) = 1
    assert abs(min(1.0, dist("A", "A") / 3.0) - 1.0) < 1e-9, "dist01(identical) must saturate at 1"
    assert dist("A", "Z") != dist("A", "Z"), "missing node ⇒ NaN"   # NaN != NaN
    print("PASS: struct_dist_fn (1/d = 3/(1+‖Δ‖), /3 normalisation, NaN on miss)")


if __name__ == "__main__":
    test_aurc_monotone()
    test_aurc_boot_ci()
    test_margin_conf()
    test_struct_dist_fn()
    print("ALL PASS")
