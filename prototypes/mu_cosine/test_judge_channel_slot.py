#!/usr/bin/env python3
"""R10 item 3 — the judge channel's slot contract as ASSERTIONS, not prose. The future
custom function-channel encoder must be a drop-in swap for NameFunctionCond's output:
same card-table shape, same output dimensionality, same insertion point (the provenance
token), unit-normed card rows. If any of these breaks, the swap is not a swap."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from fine_tune_channel_heads import ROOT, load_expanded
from judge_cards import JUDGE_CARDS, prompt_text_cards, _card_e5
from mu_attention import JUDGES


def test_slot_contract():
    model, cfg = load_expanded(os.path.join(ROOT, "model_prod_namecond_full.pt"), dev="cpu")
    jn = model.judge_name
    assert jn is not None
    n_judge, d_card = jn.name_e5.shape
    assert n_judge == len(JUDGES), "one card row per judge index"
    # descriptive and prompt-text tables are interchangeable: same shape, unit-normed rows
    for cards, tag in ((JUDGE_CARDS, "descriptive"), (prompt_text_cards(), "prompt")):
        tbl, _ = _card_e5(cards, JUDGES, cache_path=f"/tmp/mu_data/slot_test_{tag}.pt")
        assert tbl.shape == (n_judge, d_card), f"{tag}: card table shape"
        norms = tbl.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), f"{tag}: unit norm"
    # channel output: same dimensionality as the model stream, defined for every judge index
    idx = torch.arange(n_judge)
    out = jn(idx)
    assert out.shape == (n_judge, cfg["d_model"]), "judge channel output is d_model-sized"
    # swap invariance: replacing the card table changes VALUES only, never the interface
    tbl2, _ = _card_e5(prompt_text_cards(), JUDGES,
                       cache_path="/tmp/mu_data/slot_test_prompt.pt")
    jn.name_e5.copy_(tbl2)
    out2 = jn(idx)
    assert out2.shape == out.shape, "swap is a swap: output shape unchanged"


if __name__ == "__main__":
    test_slot_contract()
    print("slot contract holds")
