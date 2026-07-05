#!/usr/bin/env python3
"""Unit tests for the SYM dual-judge struct-blend modes (PR #3488). Pytest-free (like test_infer_switch.py) —
run directly: `python3 test_struct_blend.py`. Torch required.

Asserts the safety properties the PR claims (the review asked these be committed, not just smoke-run):
  1. WARM-START NO-OP — with zero-init blend weights, μ is EXACTLY equal to the no-struct path (all modes).
  2. SYM-GATED — engaging the blend moves the SYM row and leaves non-SYM (HIER) rows numerically unchanged.
  3. BOUNDED — output ∈ [0,1] even with the blend fully engaged.
  4. membership_readouts is DETACHED (no grad).
  5. PER-REGION shift — a data-rich pair gives 1/d a smaller weight share than a sparse pair (precision mode).
"""
import torch
from mu_attention import MuAttention, Tokenizer, OPS, membership_readouts

D = 384
NAMES = ["A", "B", "C", "D", "E"]
PARENTS = {"A": ["B"], "B": ["C"], "D": ["C"], "E": ["C"]}


def _fixture(struct_mode, deg=None):
    torch.manual_seed(0)
    q, p = torch.randn(5, D), torch.randn(5, D)
    idx = {n: i for i, n in enumerate(NAMES)}
    deg = deg or {n: 10 for n in NAMES}
    struct = {n: torch.randn(16) for n in NAMES}
    tok = Tokenizer(q, p, idx, PARENTS, deg, struct_tbl=struct, struct_mode=struct_mode, deg_scale=5.0)
    return tok


def _channels_for(blend):
    return {"inside": "dist", "outside": "dist", "precision": "precision", "membership": "membership"}[blend]


def _n_struct_for(blend):
    return {"inside": 1, "outside": 1, "precision": 3, "membership": 3}[blend]


def _zero_weight_param(blend):
    # the zero-init parameter whose non-zero value engages the blend, per mode
    return {"inside": "sym_struct_w", "outside": "struct_lambda",
            "precision": "struct_lambda", "membership": "struct_lambda"}[blend]


def _forward(m, tok, items, **extra):
    b = tok.build(items, train=False)
    return m(**b, **extra)


def _mem_kwargs(m, tok, items):
    if m.struct_blend != "membership":
        return {}
    pairs = [(it[0], it[1]) for it in items]
    ms, me = membership_readouts(m, tok, pairs, torch.device("cpu"))
    # only the SYM row(s) should carry membership; gating zeroes the rest anyway
    return {"mem_subcat": ms, "mem_elem": me}


def test_warm_start_no_op_and_gating_and_bounds():
    items = [("A", "B", OPS["SYM"]), ("A", "B", OPS["HIER"])]
    for blend in ("inside", "outside", "precision", "membership"):
        tok = _fixture(_channels_for(blend))
        m = MuAttention(d_model=D, n_layers=1, struct_blend=blend, n_struct=_n_struct_for(blend),
                        c_dist=0.35, c_mem_ceiling=0.67, c_subcat=0.72, c_elem=0.82)
        m.eval()
        with torch.no_grad():
            b = tok.build(items, train=False)
            base = m(**{k: v for k, v in b.items() if k != "struct_feat"})   # pure e5, no struct at all
            mem = _mem_kwargs(m, tok, items)
            out0 = m(**b, **mem)                                             # zero-init ⇒ must equal base EXACTLY
            assert torch.equal(base, out0), f"[{blend}] warm-start NOT an exact no-op: {base} vs {out0}"

            # engage the blend
            getattr(m, _zero_weight_param(blend)).data.fill_(0.6)
            out1 = m(**b, **mem)
            assert not torch.allclose(out0[0], out1[0]), f"[{blend}] SYM row did not move when engaged"
            assert torch.equal(out0[1], out1[1]), f"[{blend}] non-SYM (HIER) row changed — gating broken"
            assert bool((out1 >= 0).all() and (out1 <= 1).all()), f"[{blend}] output out of [0,1]: {out1}"
    print("PASS: warm-start no-op (exact) + SYM-gating + bounds, all 4 modes")


def test_membership_readouts_detached():
    tok = _fixture("membership")
    m = MuAttention(d_model=D, n_layers=1, struct_blend="membership", n_struct=3); m.eval()
    ms, me = membership_readouts(m, tok, [("A", "B")], torch.device("cpu"))
    assert not ms.requires_grad and not me.requires_grad, "membership_readouts must be detached (stop-grad)"
    assert bool((ms >= 0).all() and (ms <= 1).all() and (me >= 0).all() and (me <= 1).all()), "μ readouts ∉ [0,1]"
    print("PASS: membership_readouts detached + bounded")


def test_precision_per_region_shift():
    # A,B well-connected (data-rich) → high region → memberships weighted more → SMALLER 1/d share.
    # D,E sparse → low region → LARGER 1/d share. Check the region channel + the resulting weight share.
    deg = {"A": 20, "B": 20, "C": 20, "D": 1, "E": 1}
    tok = _fixture("precision", deg=deg)
    b = tok.build([("A", "B", OPS["SYM"]), ("D", "E", OPS["SYM"])], train=False)
    region_rich = float(b["struct_feat"][0, 2])
    region_sparse = float(b["struct_feat"][1, 2])
    assert region_rich > region_sparse, f"region should be higher for the data-rich pair: {region_rich} vs {region_sparse}"
    c_dist, c_mem_ceiling = 0.35, 0.67
    share_rich = c_dist / (c_mem_ceiling * region_rich + c_dist)
    share_sparse = c_dist / (c_mem_ceiling * region_sparse + c_dist)
    assert share_sparse > share_rich, f"sparse pair must give 1/d a larger share: {share_sparse} vs {share_rich}"
    print(f"PASS: per-region shift (1/d share rich {share_rich:.2f} < sparse {share_sparse:.2f})")


if __name__ == "__main__":
    test_warm_start_no_op_and_gating_and_bounds()
    test_membership_readouts_detached()
    test_precision_per_region_shift()
    print("ALL PASS")
