#!/usr/bin/env python3
"""Transitive cross-judge superposition (the two-part architecture, DESIGN_cross_judge_direction.md +
REPORT_multihop_direction §e/f). At each hop h, superpose:
  GRAPH part (IS)      — walk hit-prob P(up-walk desc→anc), μ_rev=0 (structural, no second model)
  NON-GRAPH part(OUGHT)— LLM element_of / subcategory membership, scored DIRECTLY on the multi-hop pair
                         (no math composition), carrying the epistemic uncertainty incl. a non-zero μ_rev.

  μ_fwd = w_g·hit_prob + w_e·llm_elem_fwd + w_s·llm_subcat_fwd
  μ_rev = w_g·0        + w_e·llm_elem_rev + w_s·llm_subcat_rev        (w ~ Dirichlet, DRAWN PER PAIR)

w is drawn PER PAIR (not one fixed vector) — deliberate augmentation, the 3-way analog of truncated-λ: spreading
the mixing family teaches the model to SEPARATE the operators rather than fit one recipe. The formulas above show
the STRUCTURE; each pair sees its own w. Reverse: the graph's μ_rev=0 is a HARD constraint (IS + the LLM confirms
rev≈0 every hop), so in practice `mu_r ≈ 0` even through the blend.

So the graph supplies the crisp structural IS and the LLM supplies the OUGHT — the blended μ_rev is NOT 0.
Emitted as HIER rows, judge=dir-blend.

  python3 emit_transitive_superposition.py --score-in /tmp/mu_data/multihop_score_in.tsv \
      --responses /tmp/mu_data/multihop_resp.txt --graph .../100k_cats/category_parent.tsv \
      --mix dirichlet --alpha 4 --out /tmp/mu_data/transitive_superpos.tsv
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import load_dag
from emit_transitive_hops import hit_prob
from emit_direction_blend import parse_responses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-in", required=True, help="the TSV fed to the scorer (id = row order)")
    ap.add_argument("--responses", required=True, help="LLM §14 responses (id → per-relation mu_fwd/mu_rev)")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--mix", choices=["equal", "dirichlet"], default="dirichlet")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--corpus", default="enwiki")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    parents, _, _ = load_dag(a.graph)
    byid = parse_responses(a.responses)

    with open(a.score_in, encoding="utf-8") as fh:
        rows = [ln.rstrip("\n").split("\t") for ln in fh if not ln.startswith("#")]
    # byid is keyed by the scorer's id = the Nth NON-COMMENT row; we strip comments the same way here, so id==index.
    # Guard against a mismatched score_in/responses pairing (review): a stray id past the row count = wrong pairing.
    if byid and max(byid) >= len(rows):
        raise SystemExit(f"score_in/responses mismatch: max response id {max(byid)} >= {len(rows)} rows — "
                         "were they produced from the same --pairs file?")

    def d_of(o, rel, side):
        return float((o.get(rel, {}) or {}).get(side, 0.0))

    out, n, n_missing = [], 0, 0
    for i, r in enumerate(rows):
        if i not in byid:
            n_missing += 1                                                # partial/failed scoring run — count it
            continue
        desc, anc = r[0], r[1]
        g = hit_prob(parents, desc, anc)                                 # graph IS
        ef, er = d_of(byid[i], "element_of", "mu_fwd"), d_of(byid[i], "element_of", "mu_rev")   # LLM OUGHT
        sf, sr = d_of(byid[i], "subcategory", "mu_fwd"), d_of(byid[i], "subcategory", "mu_rev")
        # PER-PAIR Dirichlet by design (the 3-way analog of truncated-λ — spreads the mixing family so the model
        # learns to SEPARATE the operators, not one fixed recipe). Seeded per-row from (seed, i) so it's fully
        # deterministic regardless of skip pattern or row order (review: was global-rng-in-loop, skip-sensitive).
        w = np.array([1/3, 1/3, 1/3]) if a.mix == "equal" else np.random.default_rng([a.seed, i]).dirichlet([a.alpha] * 3)
        mu_f = max(0.0, min(1.0, w[0] * g + w[1] * ef + w[2] * sf))
        # graph contributes 0 to the reverse BY DESIGN — μ_rev=0 is a HARD constraint (structural IS + the LLM
        # confirms rev≈0 at every hop). So w[0] here anchors the reverse toward 0; it is not a wasted weight.
        mu_r = max(0.0, min(1.0, w[0] * 0.0 + w[1] * er + w[2] * sr))
        out.append((desc, anc, mu_f)); out.append((anc, desc, mu_r)); n += 1

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# node\troot\tmu\top\trelation\tnode_type\troot_type\tcorpus\tjudge\tconf\n")
        for na, ro, mu in out:
            # op=HIER, relation=subcategory, conf=1.0 are fixed for THIS corpus: cat-cat pairs where element_of is
            # low/noisy so subcategory dominates (REPORT §1). Add a --relation/--conf arg before reusing on
            # element-heavy corpora, else rows are silently mis-tagged / over-confident.
            f.write(f"{na}\t{ro}\t{mu:.3f}\tHIER\tsubcategory\tcategory\tcategory\t{a.corpus}\tdir-blend\t1.0\n")
    if n_missing:
        print(f"WARNING: dropped {n_missing}/{len(rows)} pairs with no LLM response (partial scoring run?)",
              file=sys.stderr)
    print(f"wrote {len(out)} HIER rows ({n} transitive pairs, graph-walk IS ⊕ LLM OUGHT, per-pair mix={a.mix}) → {a.out}")


if __name__ == "__main__":
    main()
