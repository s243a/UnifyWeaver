#!/usr/bin/env python3
"""Cross-judge DIRECTION superposition (DESIGN_cross_judge_direction.md): mix THREE estimators of the same latent
— direction (which node is broader/container) — and train it as a distinct calibrated judge `dir-blend`:

    d_blend = w_g·d_graph  +  w_e·d_element  +  w_s·d_subcat        (w ~ mix over the 3-simplex)
      d_graph   = (3/(1+up_hops(a→b)) − 3/(1+up_hops(b→a)))/3       structure (graph-judged discrimination)
      d_element = μ_fwd − μ_rev  for element_of                     LLM-judged (raw §14 responses)
      d_subcat  = μ_fwd − μ_rev  for subcategory                    LLM-judged

Emitted as HIER directional rows (the direction carrier): per pair, (node,root)→μ_fwd and (root,node)→μ_rev with
μ_fwd−μ_rev = d_blend, so the operator's asymmetry is trained toward the mixed direction. Mix: fixed-equal, or a
random Dirichlet over the simplex (--mix dirichlet --alpha) — the 3-way analog of truncated-λ. All three agree on
SIGN ~100% on Wikipedia (a clean taxonomy); the mixing content is the MAGNITUDE + the judge-independence test.

  <out>: node root mu op=HIER relation=subcategory node_type root_type corpus judge=dir-blend conf
  python3 emit_direction_blend.py --pairs /tmp/mu_data/wiki_rel_pairs.tsv --responses /tmp/mu_data/wiki_rel_resp.txt \
      --graph ../../data/benchmark/100k_cats/category_parent.tsv --mix dirichlet --alpha 4 --out /tmp/mu_data/dir_blend_pairs.tsv
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import load_dag


def up_hops(parents, x, y, cap=8):
    if x == y:
        return 0
    fr, seen = {x}, {x}
    for h in range(1, cap + 1):
        nx = set()
        for c in fr:
            for p in (parents.get(c) or []):
                if p == y:
                    return h
                if p not in seen:
                    seen.add(p); nx.add(p)
        if not nx:
            break
        fr = nx
    return None


def parse_responses(path):
    """id → {relation: {mu_fwd, mu_rev}} from the concatenated §14 JSON arrays."""
    txt = open(path, encoding="utf-8").read()
    dec = json.JSONDecoder(); i = 0; byid = {}
    while True:
        j = txt.find("[", i)
        if j < 0:
            break
        try:
            arr, end = dec.raw_decode(txt, j)
            if isinstance(arr, list):
                for o in arr:
                    if isinstance(o, dict) and "id" in o:
                        byid[int(o["id"])] = o
            i = end
        except json.JSONDecodeError:
            i = j + 1
    return byid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--mix", choices=["equal", "dirichlet"], default="equal",
                    help="3-simplex mix weights: equal (1/3 each) or a random Dirichlet per pair (the 3-way "
                    "analog of truncated-λ — spreads the family for judge-independence)")
    ap.add_argument("--alpha", type=float, default=4.0, help="Dirichlet concentration (large=near-equal centroid, "
                    "small=peaky/one-operator-ish). 4 ≈ gentle spread, like truncated-normal std 0.15.")
    ap.add_argument("--corpus", default="enwiki")
    ap.add_argument("--no-negatives", action="store_true", help="drop no/contradictory-direction pairs instead of "
                    "emitting them as asymmetry-0 negatives (default: include them — direction needs agreement)")
    ap.add_argument("--held-ids", default=None, help="file of pair indices to EXCLUDE (held-out for the eval)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    parents, _, _ = load_dag(a.graph)
    pairs = [ln.rstrip("\n").split("\t")[:2] for ln in open(a.pairs, encoding="utf-8") if not ln.startswith("#")]
    byid = parse_responses(a.responses)

    def d_of(o, rel):
        e = o.get(rel, {}); return float(e.get("mu_fwd", 0)) - float(e.get("mu_rev", 0))

    held = set(int(x) for x in open(a.held_ids).read().split()) if a.held_ids else set()
    rows, npos, nneg = [], 0, 0
    for idx, (na, ro) in enumerate(pairs):
        if idx not in byid or idx in held:
            continue
        uf, ur = up_hops(parents, na, ro), up_hops(parents, ro, na)
        dg = ((3 / (1 + uf) if uf is not None else 0.0) - (3 / (1 + ur) if ur is not None else 0.0)) / 3.0  # →[-1,1]
        de, ds = d_of(byid[idx], "element_of"), d_of(byid[idx], "subcategory")
        directional = not (abs(dg) < 1e-6 and abs(de) < 1e-6 and abs(ds) < 1e-6)
        if not directional and a.no_negatives:
            continue
        # NEGATIVES (user 2026-07-05): when the operators give NO / contradictory direction, the mix → ~0 ⇒
        # μ_fwd ≈ μ_rev (asymmetry 0). Trains "a direction requires operator AGREEMENT" — the framework handles
        # contradictions gracefully as no-direction cases (matters most on direction-AMBIGUOUS data).
        w = np.array([1/3, 1/3, 1/3]) if a.mix == "equal" else rng.dirichlet([a.alpha, a.alpha, a.alpha])
        d = float(w[0] * dg + w[1] * de + w[2] * ds)        # mixed direction ∈ [-1,1]; ≈0 for no/contradictory dir
        mu_f = max(0.0, min(1.0, 0.5 + d / 2.0))
        mu_r = max(0.0, min(1.0, 0.5 - d / 2.0))            # μ_fwd − μ_rev = d
        rows.append((na, ro, mu_f)); rows.append((ro, na, mu_r))
        npos += directional; nneg += (not directional)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# node\troot\tmu\top\trelation\tnode_type\troot_type\tcorpus\tjudge\tconf\n")
        for na, ro, mu in rows:
            f.write(f"{na}\t{ro}\t{mu:.3f}\tHIER\tsubcategory\tcategory\tcategory\t{a.corpus}\tdir-blend\t1.0\n")
    print(f"wrote {len(rows)} HIER dir-blend rows ({npos} directional + {nneg} no-direction negatives, ×fwd/rev) "
          f"→ {a.out}  (mix={a.mix}{'' if a.mix=='equal' else f' α={a.alpha}'})")


if __name__ == "__main__":
    main()
