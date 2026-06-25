#!/usr/bin/env python3
"""Ensemble bridge judge — score each cross-corpus BRIDGE with several MODELS (embeddings and/or an LLM) and
combine them with a pluggable JUDGE into a keep/quarantine verdict. A bridge asserts "same concept across
corpora" (μ≈0.9); the ensemble flags the ones the models don't support, for review before training
(DESIGN_provenance_and_representation.md §Privacy / REPORT_graded_round.md bridge quality).

ARCHITECTURE
  MODEL  : a named scorer (a_key, b_key) -> float in ~[0,1] ("confidence these are the same/related").
           Provided models, by independence from the bridge label:
             * e5            cosine of frozen e5 title vectors          (independent of ALL labels)
             * model:CKPT    symmetric μ(a|b) from a trained MuAttention, provenance MASKED (agnostic).
                             Use a checkpoint that PREDATES the bridges (e.g. model_nodetype.pt) ⇒ its
                             opinion is unbiased AND richer than e5 (it knows the operators / node-types /
                             relational graph that raw e5 can't see — e.g. it can rate BELBIC↔control-systems
                             plausible where e5 saw only an opaque acronym).
             * llm           (optional, Haiku — budget-gated) same interface; not wired by default.
  JUDGE  : {model_name: score} -> keep(bool). It is GIVEN the model names, so it MAY weight or ignore
           specific models, but need not use them. The DEFAULT is a factory-built closure
           `geomean_judge(threshold=0.7)` (geometric mean ≥ threshold). 0.7 is a by-feel default, not a law —
           pass your own threshold, or any lambda `scores -> bool`, to override.

    python3 bridge_ensemble.py --fused /tmp/cyb_fused_2 --fused /tmp/ds_fused_3 \
        --e5-cache e5_tables_graded.pt --judge-model model_nodetype.pt --threshold 0.7 --out /tmp/bridge_verdict
"""
import argparse
import math
import os


# ----- the JUDGE (pluggable; default is a factory that closes over the threshold) -------------------------
def geomean_judge(threshold=0.7):
    """Factory → judge(scores: dict[name,float]) -> bool. Keep iff the GEOMETRIC MEAN of the model scores
    is ≥ threshold. Ignores the model names (but receives them, so a custom judge could use them)."""
    def judge(scores):
        vals = [max(1e-12, float(v)) for v in scores.values()]
        if not vals:
            return True
        gm = math.exp(sum(math.log(v) for v in vals) / len(vals))
        return gm >= threshold
    judge.label = f"geomean>={threshold:g}"
    return judge


def confirm_judge(model_name, threshold=0.5):
    """Factory → a NAME-AWARE judge: keep iff the specified model confidently CONFIRMS (score ≥ threshold).
    Useful when a model ABSTAINS with a low score (≈0) on inputs it never trained on — there a geomean lets
    abstention veto, but a confirm-judge only trusts a positive vote and defers the rest (e.g. to an LLM).
    Demonstrates that the judge is given the model names and MAY use them (the default geomean does not)."""
    def judge(scores):
        return scores.get(model_name, 0.0) >= threshold
    judge.label = f"confirm[{model_name}]>={threshold:g}"
    return judge


# ----- the MODELS (named scorers) -------------------------------------------------------------------------
def e5_scorer(cache_path):
    import torch
    d = torch.load(cache_path, weights_only=False)
    vidx = {n: i for i, n in enumerate(d["names"])}
    vec = d["passage"]                                     # unit-normed ⇒ dot = cosine

    def score(a, b):
        ia, ib = vidx.get(a), vidx.get(b)
        return float("nan") if ia is None or ib is None else float(vec[ia] @ vec[ib])
    return ("e5", score)


def model_scorer(ckpt_path, e5_cache, name=None, device="cpu"):
    """A trained-MuAttention judge: symmetric μ with provenance MASKED (agnostic). Loads the checkpoint with
    the warm-start GROW logic so an older (smaller-codebook) checkpoint still loads; masked provenance ⇒ the
    grown corpus/judge/account rows are never read."""
    import torch
    from mu_attention import MuAttention, Tokenizer, OPS, load_dag
    d = torch.load(e5_cache, weights_only=False)
    idx = {n: i for i, n in enumerate(d["names"])}
    parents, children, deg = load_dag()
    tok = Tokenizer(d["query"], d["passage"], idx, parents, deg)
    ck = torch.load(ckpt_path, weights_only=False)
    cfg = ck.get("cfg", {})
    model = MuAttention(d_model=d["query"].shape[1], n_heads=cfg.get("heads", 4), n_layers=cfg.get("layers", 3))
    sd, own = ck["state"], model.state_dict()
    for k, v in sd.items():                               # copy matching; grow op-/codebook-indexed rows
        if k in own and own[k].shape == v.shape:
            own[k] = v
        elif k in own and own[k].dim() >= 1 and own[k].shape[0] > v.shape[0] and own[k].shape[1:] == v.shape[1:]:
            t = own[k].clone(); t[:v.shape[0]] = v; t[v.shape[0]:] = v[0]; own[k] = t
    model.load_state_dict(own); model.to(device); model.eval()

    @torch.no_grad()
    def score(a, b):
        if a not in idx or b not in idx:
            return float("nan")
        batch = tok.build([(a, b, OPS["SYM"]), (b, a, OPS["SYM"])], train=False)   # 3-tuple ⇒ masked prov
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        mu = model(**batch).cpu()
        return float((mu[0] + mu[1]) / 2)                 # symmetric: mean of both directions
    return (name or f"model:{os.path.basename(ckpt_path)}", score)


# ----- the ENSEMBLE ---------------------------------------------------------------------------------------
class BridgeEnsemble:
    def __init__(self, models):                           # models: list of (name, scorer_fn)
        self.models = models

    def score(self, a, b):
        return {name: fn(a, b) for name, fn in self.models}

    def review(self, bridges, judge=None):
        judge = judge or geomean_judge()
        keep, quarantine = [], []
        for a, b in bridges:
            s = {k: v for k, v in self.score(a, b).items() if v == v}   # drop NaN (a model didn't know a node)
            (keep if (not s or judge(s)) else quarantine).append((a, b, s))
        return keep, quarantine


def load_bridges(prefixes):
    nodes, bridges = {}, set()
    for pref in prefixes:
        with open(pref + "_nodes.tsv", encoding="utf-8") as f:
            for ln in f:
                if not ln.startswith("#"):
                    c = ln.rstrip("\n").split("\t")
                    if len(c) >= 4:
                        nodes[c[0]] = c[3]                 # key → title
        with open(pref + "_edges.tsv", encoding="utf-8") as f:
            for ln in f:
                if not ln.startswith("#"):
                    c = ln.rstrip("\n").split("\t")
                    if len(c) >= 3 and c[2] == "bridge":
                        bridges.add(tuple(sorted((c[0], c[1]))))
    return nodes, sorted(bridges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", action="append", required=True, help="fused prefix (repeatable)")
    ap.add_argument("--e5-cache", required=True)
    ap.add_argument("--judge-model", default=None, help="MuAttention checkpoint as a 2nd judge model "
                    "(ideally pre-bridge, e.g. model_nodetype.pt)")
    ap.add_argument("--threshold", type=float, default=0.7, help="default judge: geomean ≥ threshold")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nodes, bridges = load_bridges(args.fused)
    models = [e5_scorer(args.e5_cache)]
    if args.judge_model:
        models.append(model_scorer(args.judge_model, args.e5_cache, device=args.device))
    ens = BridgeEnsemble(models)
    judge = geomean_judge(args.threshold)
    keep, quarantine = ens.review(bridges, judge)

    names = [n for n, _ in models]
    print(f"{len(bridges)} bridges · models {names} · judge {judge.label} "
          f"→ keep {len(keep)}, quarantine {len(quarantine)}")
    print("quarantined (judge says NOT same-concept) — review candidates, lowest geomean first:")
    def gm(s): return math.exp(sum(math.log(max(1e-12, v)) for v in s.values()) / len(s)) if s else 1.0
    for a, b, s in sorted(quarantine, key=lambda x: gm(x[2]))[:20]:
        sc = "  ".join(f"{k}={v:.3f}" for k, v in s.items())
        print(f"  gm={gm(s):.3f}  {a}  <->  {b}   [{sc}]")
    if args.out:
        with open(args.out + "_quarantine.tsv", "w", encoding="utf-8") as f:
            f.write("# a\tb\tgeomean\t" + "\t".join(names) + "\ta_title\tb_title\n")
            for a, b, s in sorted(quarantine, key=lambda x: gm(x[2])):
                f.write(f"{a}\t{b}\t{gm(s):.3f}\t" + "\t".join(f"{s.get(n, float('nan')):.3f}" for n in names)
                        + f"\t{nodes.get(a,'')}\t{nodes.get(b,'')}\n")
        print(f"  wrote {args.out}_quarantine.tsv ({len(quarantine)} bridges for review)")


if __name__ == "__main__":
    main()
