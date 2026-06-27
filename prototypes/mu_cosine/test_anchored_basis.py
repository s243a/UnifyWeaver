#!/usr/bin/env python3
"""Tests for the anchored-basis attention module (DESIGN §8/§8b/§1b). Needs torch.
Run: `python3 test_anchored_basis.py`."""
import torch

from anchored_basis import AnchoredBasis


def test_shapes_and_simplex():
    ab = AnchoredBasis(torch.randn(4, 384), n_atoms=5, d_query=6, d_k=64)
    q = torch.randn(8, 6)
    token, w = ab(q)
    assert token.shape == (8, 384)
    assert w.shape == (8, 4 + 5)
    assert torch.allclose(w.sum(-1), torch.ones(8), atol=1e-5)      # simplex: weights sum to 1
    assert (w >= 0).all()                                          # positive


def test_anchor_values_frozen_atoms_learn():
    ab = AnchoredBasis(torch.randn(4, 384), n_atoms=5, d_query=6, d_k=64)
    # anchor_values is a buffer (frozen), not in parameters; atoms + keys + q_proj are
    names = {n for n, _ in ab.named_parameters()}
    assert "anchor_values" not in names                            # frozen value block
    assert "atom_values" in names and "atom_keys" in names and "anchor_keys" in names
    token, w = ab(torch.randn(8, 6))
    token.sum().backward()
    assert ab.atom_values.grad is not None and ab.atom_values.grad.abs().sum() > 0
    assert ab.anchor_keys.grad is not None                         # keys are learnable (KL-calibrated)


def test_anchor_kl_and_utilization():
    ab = AnchoredBasis(torch.randn(4, 384), n_atoms=5, d_query=6, d_k=64)
    _, w = ab(torch.randn(8, 6))
    tgt = torch.full((8, 4), 0.25)                                 # uniform anchor target
    kl = ab.anchor_kl(w, tgt)
    assert kl.shape == () and torch.isfinite(kl)
    u = ab.utilization(w)
    assert len(u["anchor_mass"]) == 4 and len(u["atom_mass"]) == 5
    total = sum(u["anchor_mass"]) + sum(u["atom_mass"])
    assert abs(total - 1.0) < 1e-4                                 # mean mass-shares sum to 1


def test_zero_atoms_is_k0_special_case():
    ab = AnchoredBasis(torch.randn(4, 384), n_atoms=0, d_query=6, d_k=64)   # §8: K=0 = anchors-only
    token, w = ab(torch.randn(8, 6))
    assert w.shape == (8, 4) and token.shape == (8, 384)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"all {len(tests)} anchored_basis tests passed")
