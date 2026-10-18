#!/usr/bin/env python3
"""PLACEBO / churn control for fine-tuning (suggested by the user). Question: how much of a fine-tune's
apparent change is the NEW DATA vs just more optimization on the same distribution?

Measure, on the discrimination probe nodes (25 nodes × the domain roots), how much the per-root μ vector
(and its softmax over the roots) MOVES from a baseline checkpoint under:
  * PLACEBO  — warm-start + replay but NO new data (continue training on the old set), and
  * REAL     — warm-start + replay + the new-domain data.
If REAL ≈ PLACEBO, the new data did not move the discrimination beyond the churn floor (it may still help
the held-out relatedness RANKING — that is a separate, non-saturated metric). Build the two models first:

  # baseline checkpoint already exists (model_cumulative.pt)
  train_mu_attention.py --pairs OLD  --replay-pairs OLD --replay-frac 0.4 --init-from BASE ... --save model_null_finetune.pt
  train_mu_attention.py --pairs NEW  --replay-pairs OLD --replay-frac 0.4 --init-from BASE ... --save model_eng_finetune.pt
  python3 drift_control.py BASE model_null_finetune model_eng_finetune
"""
import sys
import torch
import torch.nn.functional as Fn

from mu_attention import load_dag, all_names, build_e5_tables, Tokenizer, MuAttention, OPS
import train_mu_attention as T


def main():
    base_n, placebo_n, real_n = (sys.argv[1:4] + ["model_cumulative", "model_null_finetune",
                                                  "model_eng_finetune"])[:3]
    parents, children, deg = load_dag()
    names = all_names(parents, children)
    q, p, idx = build_e5_tables(names, cache_path=os.environ.get("UW_E5_CACHE","e5_tables.pt"))
    tok = Tokenizer(q, p, idx, parents, deg, k=1)
    roots = [r for r in T.DOMAIN_ROOTS if r in idx]
    probe = [(n, dom) for dom in roots for n in T.DOMAIN_PROBE[dom] if n in idx]

    def load(name):
        ck = torch.load(name + ".pt", weights_only=False)
        m = MuAttention(d_model=ck["cfg"]["d_model"], n_heads=ck["cfg"]["heads"], n_layers=ck["cfg"]["layers"])
        m.load_state_dict(ck["state"]); m.eval(); return m

    def muvec(m):
        return torch.tensor([[float(T.mu_batch(m, tok, [(n, r, OPS["SYM"])])[0]) for r in roots]
                             for n, _ in probe])

    base, placebo, real = muvec(load(base_n)), muvec(load(placebo_n)), muvec(load(real_n))

    def report(tag, A, B):
        draw = (B - A).abs().mean().item()
        sa, sb = Fn.softmax(A, dim=1), Fn.softmax(B, dim=1)
        dsoft = 0.5 * (sb - sa).abs().sum(1).mean().item()           # mean total-variation of 5-root softmax
        flips = sum(1 for i in range(len(probe)) if A[i].argmax() != B[i].argmax())
        print(f"  {tag:26} mean|Δμ|={draw:.3f}  ΔsoftmaxTV={dsoft:.3f}  argmax-flips={flips}/{len(probe)}")
        return draw

    print(f"Drift from {base_n} on {len(probe)} probe nodes × {len(roots)} roots:")
    dp = report("PLACEBO (no new data)", base, placebo)
    dr = report("REAL    (new-domain data)", base, real)
    print(f"  → REAL/PLACEBO μ-drift ratio = {dr / max(dp, 1e-9):.2f}×  "
          f"({'beyond' if dr > dp else 'WITHIN'} the churn floor)")


if __name__ == "__main__":
    main()
