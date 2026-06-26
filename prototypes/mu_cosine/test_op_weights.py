#!/usr/bin/env python3
"""Tests for the BLENDED operator (`op_weights`) path in MuAttention.forward — the enabler for the random
operator embedding. A one-hot op_weights must reproduce the indexed path exactly; a blend must differ;
gradients must reach op_emb AND readout_w through op_weights. Needs torch.
Run: `python3 test_op_weights.py`."""
import torch

from mu_attention import MuAttention, Tokenizer, OPS, load_dag, all_names


def _setup():
    parents, children, deg = load_dag()
    names = all_names(parents, children)
    idx = {n: i for i, n in enumerate(names)}
    d = 384
    torch.manual_seed(0)
    q = torch.randn(len(names), d); q /= q.norm(dim=1, keepdim=True)
    p = torch.randn(len(names), d); p /= p.norm(dim=1, keepdim=True)
    tok = Tokenizer(q, p, idx, parents, deg)
    m = MuAttention(d_model=d, n_layers=2)
    return tok, m, names


def test_one_hot_equals_indexed():
    tok, m, names = _setup(); m.eval()
    a, b = names[100], names[200]
    batch = tok.build([(a, b, OPS["WIKI"]), (a, b, OPS["ELEM"]), (a, b, OPS["SYM"])], train=False)
    n_ops = m.op_emb.weight.shape[0]
    ow = torch.zeros(3, n_ops)
    for k, op in enumerate(("WIKI", "ELEM", "SYM")):
        ow[k, OPS[op]] = 1.0
    with torch.no_grad():
        assert torch.allclose(m(**batch), m(**batch, op_weights=ow), atol=1e-6)


def test_blend_differs_from_endpoints():
    tok, m, names = _setup(); m.eval()
    a, b = names[100], names[200]
    batch = tok.build([(a, b, OPS["WIKI"])], train=False)
    n_ops = m.op_emb.weight.shape[0]
    ow = torch.zeros(1, n_ops); ow[0, OPS["WIKI"]] = 0.5; ow[0, OPS["ELEM"]] = 0.5
    with torch.no_grad():
        wiki = float(m(**batch)[0])
        blend = float(m(**batch, op_weights=ow)[0])
    assert abs(blend - wiki) > 1e-4


def test_gradients_flow_through_op_weights():
    tok, m, names = _setup()
    a, b = names[100], names[200]
    batch = tok.build([(a, b, OPS["WIKI"])], train=True)
    n_ops = m.op_emb.weight.shape[0]
    ow = torch.full((1, n_ops), 1.0 / n_ops)               # uniform blend (detached, like a sampled weight)
    m.zero_grad()
    m(**batch, op_weights=ow).sum().backward()
    # the blend touches every operator's embedding AND readout head ⇒ both get gradient
    assert m.op_emb.weight.grad is not None and m.op_emb.weight.grad.abs().sum() > 0
    assert m.readout_w.grad is not None and m.readout_w.grad.abs().sum() > 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"all {len(tests)} op_weights tests passed")
