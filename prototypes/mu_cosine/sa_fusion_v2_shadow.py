#!/usr/bin/env python3
"""SHADOW: S/A fusion second iteration (encoder-lane §5.1, PR #4074). Order per ruling:
D0 hop-distance mechanism check (in-domain): A/S vs TRUE hop distance from the query's
   true folder to candidate folders, upward over parents; full_dag (all parents) AND
   principal_tree (first parent). NOTE, not resolved: typos masquerade as drift (§7 decay
   fork) — the curve is an upper bound on true semantic drift.
D1 per-channel STANDALONE OOD (Pearltrees): S / A / ELEM alone vs e5 and chance — signal-
   free inputs cannot be repaired by any blend form.
A2 (S, A, ELEM) learned blend + gate (estimand family split: symmetric / directional /
   membership), same protocol+strata as v1; gate weights = softmax(MLP([S,A,E])); readouts:
   channel-dominance ordering + surface vs S/A ratio; explicit-ratio input arm if D0 curve
   is clean.
A3 domain-conditioned gate: + per-corpus substrate stats (density, branching, folder-size
   dist) as inputs, ONE function trained jointly on both corpora; success = transfer vs
   per-domain retrained gates.
Standing R10 config noted: filing scoring is provenance-masked — judge cards inert here."""
import os, sys, json, random
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from eval_filing import load_filing, load_membership, score_mu, metrics
from fine_tune_channel_heads import load_expanded
from mu_attention import OPS, Tokenizer, build_e5_tables, E5_REVISION, load_dag, GRAPH

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

def corpus(source, model, dev):
    if source == "simplewiki":
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
    ow = lambda op: torch.zeros(1, model.op_emb.weight.shape[0]).index_fill_(
        1, torch.tensor([OPS[op]]), 1.0)
    sm = lambda o: torch.tensor(score_mu(model, tok, idx, q_keys, f_keys, o, dev))
    E, A, S = sm(ow("ELEM")), sm(ow("HIER")), sm(ow("SYM"))
    tid_list = list(cand)
    truepos = [tid_list.index(t) for _, t in queries]
    C = (qtbl[[idx[k] for k in q_keys]] @ ptbl[[idx[k] for k in f_keys]].T).cpu()
    order = np.random.default_rng(SEED).permutation(len(queries))
    tr, he = torch.tensor(order[:300]), torch.tensor(order[300:])
    return dict(queries=queries, cand=cand, tid_list=tid_list, truepos=torch.tensor(truepos),
                Cz=nzrow(C), Sz=nzrow(S), Az=nzrow(A), Ez=nzrow(E), tr=tr, he=he)

def hop_diag(cw):
    parents, children, _ = load_dag(GRAPH)
    tid_ix = {t: j for j, t in enumerate(cw["tid_list"])}
    out = {}
    for mode in ("full_dag", "principal_tree"):
        rows = {}
        for r, (_, true_t) in enumerate(cw["queries"]):
            seen, frontier, hop = {true_t}, [true_t], 0
            while frontier and hop < 6:
                hop += 1
                nxt = []
                for x in frontier:
                    ps = sorted(parents.get(x, []))    # sets in the DAG; sort = deterministic principal
                    if mode == "principal_tree":
                        ps = ps[:1]
                    for p in ps:
                        if p not in seen:
                            seen.add(p); nxt.append(p)
                            if p in tid_ix:
                                rows.setdefault(hop, []).append(
                                    (float(cw["Az"][r, tid_ix[p]]),
                                     float(cw["Sz"][r, tid_ix[p]])))
                frontier = nxt
        out[mode] = {h: {"n": len(v),
                         "A": round(float(np.mean([a for a, _ in v])), 4),
                         "S": round(float(np.mean([s for _, s in v])), 4),
                         "A_over_S": round(float(np.mean([a for a, _ in v]) /
                                           max(np.mean([s for _, s in v]), 1e-6)), 4)}
                     for h, v in sorted(rows.items())}
    return out

def gate3(extra_dim=0):
    return torch.nn.Sequential(torch.nn.Linear(3 + extra_dim, 12), torch.nn.Tanh(),
                               torch.nn.Linear(12, 3))

def gate_score(mlp, we, cw, ix, extra=None):
    X = torch.stack([cw["Sz"][ix], cw["Az"][ix], cw["Ez"][ix]], -1)
    if extra is not None:
        X = torch.cat([X, extra], -1)
    w = torch.softmax(mlp(X), -1)
    chan = torch.stack([cw["Sz"][ix], cw["Az"][ix], cw["Ez"][ix]], -1)
    return we * cw["Cz"][ix] + (w * chan).sum(-1), w

def train(fn, params, steps=400, lr=0.05):
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        loss = fn()
        opt.zero_grad(); loss.backward(); opt.step()

def evaluate(M, cw, ix):
    return {k: round(v, 4) for k, v in
            metrics(ranks_from(M[ix] if M.shape[0] != len(ix) else M,
                               [int(cw["truepos"][i]) for i in ix.tolist()])).items()}

def run():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    here = os.path.dirname(os.path.abspath(__file__))
    model, _ = load_expanded(os.path.join(here, CKPT), dev=dev)
    model.eval()
    res = {"stamp": "shadow-exploratory-tier1-not-decision-bearing"}
    cp = corpus("pearltrees", model, dev)
    cw = corpus("simplewiki", model, dev)
    # D0 hop-distance mechanism
    res["D0_hop_curve"] = hop_diag(cw)
    print("[D0] " + json.dumps(res["D0_hop_curve"]), flush=True)
    # D1 standalone OOD channels
    res["D1_standalone_pearltrees"] = {
        nm: evaluate(cp[k], cp, cp["he"]) for nm, k in
        (("S_alone", "Sz"), ("A_alone", "Az"), ("ELEM_alone", "Ez"), ("e5_only", "Cz"))}
    nf = len(cp["tid_list"])
    res["D1_chance_MRR"] = round(sum(1.0 / r for r in range(1, nf + 1)) / nf, 4)
    print("[D1] " + json.dumps(res["D1_standalone_pearltrees"]), flush=True)
    # A2 per-corpus 3-channel arms
    for tag, cx in (("pearltrees", cp), ("simplewiki", cw)):
        w = torch.nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25]))
        train(lambda: torch.nn.functional.cross_entropy(
            (w[0] * cx["Cz"][cx["tr"]] + w[1] * cx["Sz"][cx["tr"]] +
             w[2] * cx["Az"][cx["tr"]] + w[3] * cx["Ez"][cx["tr"]]) * 8.0,
            cx["truepos"][cx["tr"]]), [w])
    # keep the last-trained per-corpus results via re-loop (explicit, simple)
    a2 = {}
    per_domain_gate = {}
    for tag, cx in (("pearltrees", cp), ("simplewiki", cw)):
        w = torch.nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25]))
        train(lambda: torch.nn.functional.cross_entropy(
            (w[0] * cx["Cz"][cx["tr"]] + w[1] * cx["Sz"][cx["tr"]] +
             w[2] * cx["Az"][cx["tr"]] + w[3] * cx["Ez"][cx["tr"]]) * 8.0,
            cx["truepos"][cx["tr"]]), [w])
        lin = (w[0] * cx["Cz"] + w[1] * cx["Sz"] + w[2] * cx["Az"] + w[3] * cx["Ez"]).detach()
        mlp, we = gate3(), torch.nn.Parameter(torch.tensor(0.2))
        train(lambda: torch.nn.functional.cross_entropy(
            gate_score(mlp, we, cx, cx["tr"])[0] * 8.0, cx["truepos"][cx["tr"]]),
            list(mlp.parameters()) + [we])
        with torch.no_grad():
            gs, gw = gate_score(mlp, we, cx, torch.arange(len(cx["queries"])))
        per_domain_gate[tag] = evaluate(gs, cx, cx["he"])
        mumax = nzrow(torch.maximum(torch.maximum(cx["Sz"], cx["Az"]), cx["Ez"]))
        a2[tag] = {
            "linear_weights": {"e5": round(float(w[0]), 3), "S": round(float(w[1]), 3),
                               "A": round(float(w[2]), 3), "E": round(float(w[3]), 3)},
            "gate_we": round(float(we), 3),
            "gate_mean_channel_weights_held": {
                k: round(float(gw[cx["he"]][..., i].mean()), 3)
                for i, k in enumerate(("S", "A", "E"))},
            "mumax_fixed": evaluate(0.1 * cx["Cz"] + 0.9 * mumax, cx, cx["he"]),
            "linear": evaluate(lin, cx, cx["he"]),
            "gate": per_domain_gate[tag],
        }
        # gate surface vs S/A ratio (E fixed mid): does the gate track ratio contours?
        with torch.no_grad():
            surf = {}
            for sv in (0.2, 0.5, 0.8):
                for av in (0.2, 0.5, 0.8):
                    wsurf = torch.softmax(mlp(torch.tensor([[sv, av, 0.5]])), -1)[0]
                    surf[f"S={sv},A={av},E=0.5"] = [round(float(x), 3) for x in wsurf]
            a2[tag]["gate_surface_SAE_weights"] = surf
        print(f"[A2 {tag}] " + json.dumps(a2[tag]), flush=True)
    res["A2"] = a2
    # A3 domain-conditioned joint gate: substrate stats as inputs (constant per corpus)
    def stats_of(cx, tag):
        sizes = torch.tensor([sum(1 for _, t in cx["queries"] if t == tid)
                              for tid in cx["tid_list"]], dtype=torch.float32)
        if tag == "simplewiki":
            parents, children, _ = load_dag(GRAPH)
            dens = sum(len(v) for v in parents.values()) / max(len(parents), 1)
            branch = float(np.mean([len(v) for v in children.values()]))
        else:
            dens, branch = 1.0, float(sizes.mean())
        return torch.tensor([dens, branch / 50.0, float(sizes.float().mean()) / 50.0])
    st = {"pearltrees": stats_of(cp, "pearltrees"), "simplewiki": stats_of(cw, "simplewiki")}
    mlp, we = gate3(extra_dim=3), torch.nn.Parameter(torch.tensor(0.2))
    def joint_loss():
        tot = 0
        for tag, cx in (("pearltrees", cp), ("simplewiki", cw)):
            ex = st[tag].expand(len(cx["tr"]), len(cx["tid_list"]), 3)
            s, _ = gate_score(mlp, we, cx, cx["tr"], extra=ex)
            tot = tot + torch.nn.functional.cross_entropy(s * 8.0, cx["truepos"][cx["tr"]])
        return tot
    train(joint_loss, list(mlp.parameters()) + [we])
    res["A3_joint_gate"] = {}
    for tag, cx in (("pearltrees", cp), ("simplewiki", cw)):
        with torch.no_grad():
            ex = st[tag].expand(len(cx["queries"]), len(cx["tid_list"]), 3)
            s, _ = gate_score(mlp, we, cx, torch.arange(len(cx["queries"])), extra=ex)
        res["A3_joint_gate"][tag] = {"joint": evaluate(s, cx, cx["he"]),
                                     "per_domain_retrained": per_domain_gate[tag]}
    print(json.dumps(res, indent=1), flush=True)

run()
