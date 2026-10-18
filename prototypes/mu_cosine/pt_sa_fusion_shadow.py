#!/usr/bin/env python3
"""SHADOW: LEARNED S/A FUSION FOR FILING (encoder-lane §5.1 owner direction, PR #4071).
Hypothesis: strong ASYMMETRIC (directional, HIER) signal is direct evidence of where to
file (filing makes the item a child); SYMMETRIC (SYM) relatedness is context. Learn the
blend against filing cross-entropy instead of fixing it.
Arms (query-level 60/40 train/held split, Pearltrees harvested trees, model_pt_filing):
  baseline  e5+mu-max fixed blend (0.1·e5z + 0.9·max(elem,hier,sym)z — the tuned default)
  linear    score = we·e5z + ws·Sz + wa·Az, 3 learned weights, slate CE over folders
  gate      score = we·e5z + g·Az + (1-g)·Sz, g = sigmoid(MLP_8([Sz,Az])) per pair —
            can express "trust A when A is strong, fall back to S otherwise"
Reporting: recall@1/5/10, MRR on held; strata = A-margin (strong vs weak top1-top2 of Az)
and true-folder size; plus the learned gate's SHAPE g(S,A) on a grid — does it reproduce
strong-A-dominates? Standing R10 config (prompt-text cards + judge dropout .3) noted:
filing scoring is provenance-masked (3-tuple items), so the judge channel — and therefore
the card choice — is structurally inert in this scorer; recorded, not hidden."""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from eval_filing import load_filing, score_mu, metrics
from fine_tune_channel_heads import load_expanded
from mu_attention import OPS, Tokenizer, build_e5_tables, E5_REVISION

TREES = "/home/s243a/Projects/UnifyWeaver/.local/data/pearltrees_api/trees"
CKPT, SEED, MAXQ = "model_pt_filing.pt", 7, 500

def nzrow(M):
    lo = M.min(dim=1, keepdim=True).values
    hi = M.max(dim=1, keepdim=True).values
    return (M - lo) / (hi - lo + 1e-9)

def ranks_from(M, truepos):
    return [1 + int(((M[r] > M[r][truepos[r]]) |
                     ((M[r] == M[r][truepos[r]]) & (torch.arange(M.shape[1]) < truepos[r]))).sum().item())
            for r in range(M.shape[0])]

def run():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if os.environ.get("SOURCE") == "simplewiki":
        from eval_filing import load_membership, GRAPH
        queries, cand = load_membership(GRAPH, 3)
    else:
        queries, cand = load_filing(TREES, 3)
    rng = random.Random(SEED)
    if len(queries) > MAXQ:
        queries = rng.sample(queries, MAXQ)
    f_keys = [f"F:{t}" for t in cand]
    q_keys = [f"B:{i}" for i in range(len(queries))]
    texts = {f"F:{t}": cand[t] for t in cand} | {f"B:{i}": q for i, (q, _) in enumerate(queries)}
    qtbl, ptbl, idx = build_e5_tables(sorted(texts), cache_path=None, texts=texts,
                                      model_revision=E5_REVISION)
    tok = Tokenizer(qtbl, ptbl, idx, {}, {})
    model, cfg = load_expanded(os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT),
                               dev=dev)
    model.eval()
    n_ops = model.op_emb.weight.shape[0]
    ow = lambda op: torch.zeros(1, n_ops).index_fill_(1, torch.tensor([OPS[op]]), 1.0)
    sm = lambda o: torch.tensor(score_mu(model, tok, idx, q_keys, f_keys, o, dev))
    S_elem, S_hier, S_sym = sm(ow("ELEM")), sm(ow("HIER")), sm(ow("SYM"))
    tid_list = list(cand)
    truepos = [tid_list.index(t) for _, t in queries]
    C = (qtbl[[idx[k] for k in q_keys]] @ ptbl[[idx[k] for k in f_keys]].T).cpu()
    Cz, Sz, Az = nzrow(C), nzrow(S_sym), nzrow(S_hier)
    Mz = nzrow(torch.maximum(torch.maximum(S_elem, S_hier), S_sym))
    tp = torch.tensor(truepos)
    order = np.random.default_rng(SEED).permutation(len(queries))
    tr, he = order[:int(0.6 * len(order))], order[int(0.6 * len(order)):]
    tr_t, he_t = torch.tensor(tr), torch.tensor(he)

    def train_blend(param_fn, params, steps=400, lr=0.05):
        opt = torch.optim.Adam(params, lr=lr)
        for s in range(steps):
            sc = param_fn(Cz[tr_t], Sz[tr_t], Az[tr_t])
            loss = torch.nn.functional.cross_entropy(sc * 8.0, tp[tr_t])
            opt.zero_grad(); loss.backward(); opt.step()
        return float(loss)

    arms0_placeholder = None
    res = {"stamp": "shadow-exploratory-tier1-not-decision-bearing",
           "n_queries": len(queries), "n_folders": len(cand),
           "split": {"train": len(tr), "held": len(he)}}
    arms = {"baseline_e5_mumax": 0.1 * Cz + 0.9 * Mz, "e5_only": Cz}
    # linear
    w = torch.nn.Parameter(torch.tensor([0.1, 0.45, 0.45]))
    train_blend(lambda c, s, a: w[0] * c + w[1] * s + w[2] * a, [w])
    res["linear_weights"] = {"e5": float(w[0]), "S": float(w[1]), "A": float(w[2])}
    arms["linear"] = (w[0] * Cz + w[1] * Sz + w[2] * Az).detach()
    # gate
    mlp = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    we = torch.nn.Parameter(torch.tensor(0.1))
    def gated(c, s, a):
        g = torch.sigmoid(mlp(torch.stack([s, a], -1)).squeeze(-1))
        return we * c + g * a + (1 - g) * s
    train_blend(gated, list(mlp.parameters()) + [we])
    with torch.no_grad():
        arms["gate"] = gated(Cz, Sz, Az)
        res["gate_we"] = float(we)
        grid = {}
        for sv in (0.1, 0.5, 0.9):
            for av in (0.1, 0.5, 0.9):
                g = torch.sigmoid(mlp(torch.tensor([[sv, av]])).squeeze()).item()
                grid[f"S={sv},A={av}"] = round(g, 3)
        res["gate_shape_g(S,A)"] = grid
    # strata on held: A-margin strength + true-folder size
    a_top2 = Az.topk(2, dim=1).values
    a_margin = a_top2[:, 0] - a_top2[:, 1]
    med = a_margin[he_t].median()
    folder_sz = torch.tensor([sum(1 for _, t in queries if t == tid_list[truepos[r]])
                              for r in range(len(queries))])
    strata = {"all": he_t,
              "strongA": he_t[a_margin[he_t] > med], "weakA": he_t[a_margin[he_t] <= med],
              "bigfolder": he_t[folder_sz[he_t] >= folder_sz[he_t].float().median()],
              "smallfolder": he_t[folder_sz[he_t] < folder_sz[he_t].float().median()]}
    for nm, M in arms.items():
        res[nm] = {}
        for st, ix_t in strata.items():
            r = ranks_from(M[ix_t], [truepos[i] for i in ix_t.tolist()])
            res[nm][st] = {k: round(v, 4) for k, v in metrics(r).items()}
    print(json.dumps(res, indent=1), flush=True)

run()
