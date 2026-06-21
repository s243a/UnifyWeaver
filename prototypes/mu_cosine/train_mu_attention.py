#!/usr/bin/env python3
"""Train + validate `MuAttention` (DESIGN_directional_attention.md) — multi-task over operators.

Operators (one shared attention backbone, per-operator readout):
  * **WIKI** (free, dense): for a Wikipedia edge `child → parent`, a directional **margin** loss
    `μ(child|parent) − μ(parent|child) ≥ m` plus an in-batch uniform negative
    `μ(child|parent) − μ(child|parent_neg) ≥ m`. Zero LLM budget — the explicit wrong-order penalty.
  * **SYM** (cheap): order-invariant MSE on the scored `mu_pairs_scored.tsv` (feed both orders, same μ).
  * **LLM** (optional, `--llm`): directional-μ MSE on the ALREADY-BOUGHT cutoff-band fixture
    `tests/fixtures/wikipedia_physics_boundary_haiku.tsv` (root = Physics) — **no new Haiku spend**.

Balanced per-operator batches (WIKI down-weighted to parity so the millions of free edges don't drown
the few hundred SYM/LLM labels). Frozen e5 + AdamW weight decay on the learned head.

Validation (per operator) + head-to-head vs the #3287 MiniLM-symmetric CONTROL (REPORT_control_baseline):
held-out μ corr / SYM symmetry, WIKI held-out edge order-accuracy, gate-leak (5-probe + OOD), each dense
map through check_feeds_rust.py. SECONDARY: lin-agreement on the NODE-gated IC (#3296), not path-gated.

    python3 train_mu_attention.py --steps 1200 --save model.pt
    python3 train_mu_attention.py --steps 1200 --llm        # + the already-bought LLM operator
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
from collections import deque

import torch
import torch.nn.functional as F

from mu_attention import (OPS, GRAPH, load_dag, all_names, build_e5_tables, Tokenizer, MuAttention)

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
PAIRS = os.path.join(ROOT, "mu_pairs_scored.tsv")
BOUNDARY = os.path.join(REPO, "tests", "fixtures", "wikipedia_physics_boundary_haiku.tsv")
NONPHYS = ["Music", "Cooking", "Religious_buildings", "Politics", "Fashion"]


def pearson(xs, ys):
    xs, ys = torch.as_tensor(xs, dtype=torch.float64), torch.as_tensor(ys, dtype=torch.float64)
    xs, ys = xs - xs.mean(), ys - ys.mean()
    return float((xs @ ys) / (xs.norm() * ys.norm()).clamp_min(1e-12))


def load_edges(path=GRAPH):
    edges = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("child"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[0] != p[1]:
                edges.append((p[0], p[1]))               # (child, parent)
    return edges


def load_pairs(path=PAIRS):
    pos, neg = [], []
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 5 or p[4].strip() == "":
                continue
            # any non-neg stratum (pos / pos_phys / pos_chem / cross) is a graded positive; neg is μ=0
            (neg if p[2] == "neg" else pos).append((p[0], p[1], float(p[4])))
    return pos, neg


def load_mu(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                try:
                    out[p[0]] = float(p[1])
                except ValueError:
                    pass
    return out


def bfs_dist(adj, src):
    dist = {src: 0}
    q = deque([src])
    while q:
        x = q.popleft()
        for y in adj.get(x, ()):
            if y not in dist:
                dist[y] = dist[x] + 1
                q.append(y)
    return dist


@torch.no_grad()
def mu_batch(model, tok, items, bs=256):
    out = []
    for i in range(0, len(items), bs):
        out.append(model(**tok.build(items[i:i + bs], train=False)))
    return torch.cat(out) if out else torch.zeros(0)


@torch.no_grad()
def emit_dense(model, tok, names, op, root="Physics", bs=512):
    items = [(n, root, OPS[op]) for n in names]
    mu = mu_batch(model, tok, items, bs)
    return {n: float(m) for n, m in zip(names, mu.tolist())}


def write_map(mu_map, path, header):
    with open(path, "w") as f:
        f.write(f"# {header}\n")
        for n, m in mu_map.items():
            f.write(f"{n}\t{m:.4f}\n")


def check_feeds(path):
    r = subprocess.run(["python3", os.path.join(ROOT, "check_feeds_rust.py"), "--mu-file", path],
                       cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip().splitlines()


# --------------------------------------------------------------------------------------------------
def train(args):
    parents, children, deg = load_dag()
    names = all_names(parents, children)
    adj = {}
    for c, ps in parents.items():
        for p in ps:
            adj.setdefault(c, set()).add(p)
            adj.setdefault(p, set()).add(c)
    q, p, idx = build_e5_tables(names, cache_path=os.path.join(ROOT, "e5_tables.pt"))
    tok = Tokenizer(q, p, idx, parents, deg, k=args.k, beta=1.0, max_anc=args.max_anc)

    rng = random.Random(args.seed)
    edges = [e for e in load_edges() if e[0] in idx and e[1] in idx]
    rng.shuffle(edges)
    n_eh = int(0.1 * len(edges))
    edges_hold, edges_tr = edges[:n_eh], edges[n_eh:]
    pos, neg = load_pairs(args.pairs)
    pos = [r for r in pos if r[0] in idx and r[1] in idx]
    neg = [r for r in neg if r[0] in idx and r[1] in idx]
    rng.shuffle(pos)
    n_ph = int(0.2 * len(pos))
    pos_hold, pos_tr = pos[:n_ph], pos[n_ph:]
    sym_tr = pos_tr + neg
    print(f"WIKI edges: {len(edges_tr)} train / {len(edges_hold)} held-out")
    print(f"SYM pairs: {len(pos_tr)} pos + {len(neg)} neg train / {len(pos_hold)} held-out positives")

    llm = []
    if args.llm:
        bmu = load_mu(BOUNDARY)
        llm = [(n, "Physics", m) for n, m in bmu.items() if n in idx]
        print(f"LLM (already-bought boundary fixture, no new spend): {len(llm)} directional labels")

    torch.manual_seed(args.seed)
    model = MuAttention(d_model=q.shape[1], n_heads=args.heads, n_layers=args.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    m = args.margin

    eps = 1e-6
    bce = lambda x, t: F.binary_cross_entropy(x.clamp(eps, 1 - eps), torch.full_like(x, float(t)))
    for step in range(args.steps):
        model.train()
        opt.zero_grad()
        L_wiki = torch.zeros(())
        if not args.sym_only:
            # WIKI — one build+forward for [child|parent ; parent|child ; child|random], then split.
            # PIN the negatives low with BCE and PUSH μ(child|parent) above them with weighted margins
            # (no bce(·,1) ceiling on μ(child|parent) — that creates a "predict the mean (1/3)" collapse).
            eb = [edges_tr[rng.randrange(len(edges_tr))] for _ in range(args.bs)]
            negpar = [eb[(i + 1) % len(eb)][1] for i in range(len(eb))]
            wiki_items = ([(c, par, OPS["WIKI"]) for c, par in eb]
                          + [(par, c, OPS["WIKI"]) for c, par in eb]
                          + [(c, negpar[i], OPS["WIKI"]) for i, (c, par) in enumerate(eb)])
            mu_w = model(**tok.build(wiki_items, train=True, rng=rng))
            mu_cp, mu_pc, mu_cpn = mu_w[:args.bs], mu_w[args.bs:2 * args.bs], mu_w[2 * args.bs:]
            L_wiki = (bce(mu_pc, 0) + bce(mu_cpn, 0)
                      + args.wiki_abs * bce(mu_cp, 0.9)
                      + args.margin_weight * (F.relu(m - (mu_cp - mu_pc)).mean()
                                              + F.relu(m - (mu_cp - mu_cpn)).mean()))
        # SYM (order-invariant) — BALANCED pos/neg so the μ=0 negatives can't collapse μ→0
        half = args.bs // 2
        sb = ([pos_tr[rng.randrange(len(pos_tr))] for _ in range(half)] +
              [neg[rng.randrange(len(neg))] for _ in range(args.bs - half)])
        sym_items = ([(a, b, OPS["SYM"]) for a, b, _ in sb] + [(b, a, OPS["SYM"]) for a, b, _ in sb])
        tgt = torch.tensor([mu for _, _, mu in sb])
        mu_s = model(**tok.build(sym_items, train=True, rng=rng))
        mu_ab, mu_ba = mu_s[:len(sb)], mu_s[len(sb):]
        L_sym = F.mse_loss(mu_ab, tgt) + F.mse_loss(mu_ba, tgt)
        loss = args.sym_weight * L_sym + (0.0 if args.sym_only else args.wiki_weight * L_wiki)
        # LLM (optional, already-bought) — skipped in single-task SYM mode
        L_llm = torch.zeros(())
        if llm and not args.sym_only:
            lb = [llm[rng.randrange(len(llm))] for _ in range(args.bs)]
            li = [(n, r, OPS["LLM"]) for n, r, _ in lb]
            lt = torch.tensor([mu for _, _, mu in lb])
            L_llm = F.mse_loss(model(**tok.build(li, train=True, rng=rng)), lt)
            loss = loss + args.llm_weight * L_llm
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, args.steps // 6) == 0:
            print(f"  step {step+1:5d}  L_sym {float(L_sym):.4f}  L_wiki {float(L_wiki):.4f}"
                  + (f"  L_llm {float(L_llm):.4f}" if (llm and not args.sym_only) else ""))

    model.eval()
    validate(model, tok, names, idx, adj, edges_hold, pos_hold, llm, args)
    if args.save:
        torch.save({"state": model.state_dict(), "cfg": {"d_model": q.shape[1], "heads": args.heads,
                    "layers": args.layers}}, args.save)
        print(f"\nsaved model → {args.save}")


def validate(model, tok, names, idx, adj, edges_hold, pos_hold, llm, args):
    print("\n=== PER-OPERATOR VALIDATION ===")
    ops = ["SYM"] if args.sym_only else (["SYM", "WIKI"] + (["LLM"] if llm else []))

    # --- WIKI: held-out edge order-accuracy ---
    if not args.sym_only:
        cp = mu_batch(model, tok, [(c, p, OPS["WIKI"]) for c, p in edges_hold])
        pc = mu_batch(model, tok, [(p, c, OPS["WIKI"]) for c, p in edges_hold])
        acc = float((cp > pc).float().mean())
        print(f"[WIKI] held-out edge ORDER-accuracy μ(child|parent)>μ(parent|child): {acc*100:.1f}% "
              f"({len(edges_hold)} edges)  mean μ(c|p)={float(cp.mean()):.3f} μ(p|c)={float(pc.mean()):.3f}")

    # --- SYM: held-out corr + symmetry (the headline metric for this work) ---
    ab = mu_batch(model, tok, [(a, b, OPS["SYM"]) for a, b, _ in pos_hold])
    ba = mu_batch(model, tok, [(b, a, OPS["SYM"]) for a, b, _ in pos_hold])
    tgt = [mu for _, _, mu in pos_hold]
    corr = pearson(((ab + ba) / 2).tolist(), tgt)
    sym_gap = float((ab - ba).abs().mean())
    print(f"[SYM]  held-out μ corr (control +0.726, #3302 +0.335): {corr:+.3f}  ({len(pos_hold)} "
          f"positives, MSE {F.mse_loss((ab+ba)/2, torch.tensor(tgt)):.3f})  symmetry gap {sym_gap:.3f}")

    # --- gate-leak 5-probe (control 0/5), per operator ---
    for op in ops:
        pr = mu_batch(model, tok, [(n, "Physics", OPS[op]) for n in NONPHYS])
        leak = int((pr >= 0.3).sum())
        print(f"[{op}] gate-leak 5-probe (non-physics μ(·|Physics)≥0.3): {leak}/5  " +
              "  ".join(f"{n.split('_')[0]}={v:.2f}" for n, v in zip(NONPHYS, pr.tolist())))

    # --- Physics-vs-Chemistry DISCRIMINATION: μ(node|Physics) vs μ(node|Chemistry), check ordering ---
    discrimination_probe(model, tok, idx)

    if args.quick_val:
        print("(quick-val: skipping dense-map emission / check_feeds / lin-agreement)")
        return

    # --- dense maps per operator → check_feeds_rust + OOD gate-leak (control 1.1%) ---
    dist = bfs_dist(adj, "Physics")
    far = [n for n in names if dist.get(n, 99) >= 5]
    for op in ops:
        dm = emit_dense(model, tok, names, op)
        path = os.path.join(ROOT, f"dense_mu_attn_{op.lower()}.tsv")
        write_map(dm, path, f"MuAttention {op} μ(X|Physics) — frozen e5 + learned tags (regenerable)")
        ood = [dm[n] for n in far]
        leak = sum(1 for v in ood if v >= 0.3)
        print(f"\n[{op}] dense map → {os.path.basename(path)}  | OOD gate-leak (dist≥5, {len(far)} nodes, "
              f"control 1.1%): {100*leak/max(len(ood),1):.1f}% (μ̄ {sum(ood)/max(len(ood),1):.3f})")
        for ln in check_feeds(path):
            if "guards OK" in ln or "PASS" in ln or "general→specific" in ln:
                print("   " + ln.strip())

    # --- SECONDARY: lin-agreement on NODE-gated IC (#3296), not path-gated ---
    node_gated_lin_agreement(model, tok, idx)


DOMAIN_ROOTS = ["Physics", "Chemistry", "Mathematics", "Computer_science", "Engineering"]
DOMAIN_PROBE = {
    "Physics": ["Thermodynamics", "Optics", "Mechanics", "Electromagnetism", "Motion_(physics)"],
    "Chemistry": ["Periodic_table", "Acids", "Chemical_compounds", "Oxygen", "Chemical_reactions"],
    "Mathematics": ["Calculus", "Differential_equations", "Mathematical_analysis", "Logic",
                    "Fields_of_mathematics"],
    "Computer_science": ["Software", "Computer_hardware", "Operating_systems", "Computer_networking",
                         "Computer_architecture"],
    "Engineering": ["Mechanical_engineering", "Civil_engineering", "Engineering_disciplines",
                    "Machines", "Infrastructure"],
}
BORDER_PROBE = ["Atoms", "Electronics", "Measurement", "Materials", "Energy"]


def discrimination_probe(model, tok, idx):
    """MULTI-domain discrimination: for clear nodes of each domain, μ(node|own-root) should be the ARGMAX
    over all DOMAIN_ROOTS {Physics, Chemistry, Mathematics, Computer_science, Engineering} (SYM operator).
    Reports the per-domain accuracy and the full confusion (true domain × argmax root)."""
    roots = [r for r in DOMAIN_ROOTS if r in idx]
    if len(roots) < 2:
        print("[DISCRIM] <2 domain roots in graph — skipped")
        return
    ab = {r: r[:4] for r in roots}
    print(f"\n[DISCRIM] {len(roots)}-domain (argmax μ over {', '.join(roots)}):")
    confusion = {d: {r: 0 for r in roots} for d in roots}
    correct = total = 0
    for dom in roots:
        nodes = [n for n in DOMAIN_PROBE[dom] if n in idx]
        mus = {r: mu_batch(model, tok, [(n, r, OPS["SYM"]) for n in nodes]) for r in roots}
        for i, n in enumerate(nodes):
            vals = {r: float(mus[r][i]) for r in roots}
            pred = max(roots, key=lambda r: vals[r])
            confusion[dom][pred] += 1
            ok = pred == dom
            correct += ok
            total += 1
            scores = "  ".join(f"{ab[r]}={vals[r]:.2f}" for r in roots)
            print(f"    [{ab[dom]}] {n:22} {scores}  →{ab[pred]} {'✓' if ok else '✗'}")
    # borderline (no ground truth — just show the argmax)
    bnodes = [n for n in BORDER_PROBE if n in idx]
    if bnodes:
        for n in bnodes:
            vals = {r: float(mu_batch(model, tok, [(n, r, OPS["SYM"])])[0]) for r in roots}
            pred = max(roots, key=lambda r: vals[r])
            print(f"    [border] {n:22} " + "  ".join(f"{ab[r]}={vals[r]:.2f}" for r in roots)
                  + f"  →{ab[pred]}")
    print("  confusion (true → argmax):")
    print("            " + "".join(f"{ab[r]:>7}" for r in roots))
    for d in roots:
        print(f"    {ab[d]:>7} " + "".join(f"{confusion[d][r]:>7}" for r in roots))
    print(f"  multi-domain discrimination accuracy: {correct}/{total} "
          f"({100*correct/max(total,1):.0f}%)")


def node_gated_lin_agreement(model, tok, idx):
    from itertools import combinations
    from node_gated_ic import (load_graph as ng_load, load_mu as ng_mu, node_gated_mass, ic_map,
                               lin as ng_lin, FIXTURE, THRESH, SAT)
    parents, children = ng_load(GRAPH)
    mu = ng_mu(FIXTURE)
    total = sum(mu.values())
    allnodes = set(parents) | set(children)
    ic_node = ic_map(allnodes, children, mu, THRESH, total, node_gated_mass)
    scored = [n for n in mu if n in parents and mu[n] >= THRESH and n in idx]
    pairs, lin_v = [], []
    for a, b in combinations(scored, 2):
        lv, _ = ng_lin(a, b, parents, ic_node)
        if lv is not None:
            pairs.append((a, b))
            lin_v.append(lv)
    sym = mu_batch(model, tok, [(a, b, OPS["SYM"]) for a, b in pairs])
    sym2 = mu_batch(model, tok, [(b, a, OPS["SYM"]) for a, b in pairs])
    pred = ((sym + sym2) / 2).tolist()
    nonsat = [(l, pr) for l, pr in zip(lin_v, pred) if l < SAT]
    r_all = pearson(lin_v, pred) if pairs else float("nan")
    r_ns = pearson([l for l, _ in nonsat], [pr for _, pr in nonsat]) if len(nonsat) > 2 else float("nan")
    sat = sum(1 for l in lin_v if l >= SAT)
    print(f"\n[SECONDARY] node-gated lin-agreement (#3296 IC, NOT path-gated): {len(pairs)} scored-node "
          f"pairs, {sat} saturated ({100*sat/max(len(pairs),1):.1f}%).  Pearson(SYM μ, node-gated Lin) "
          f"all={r_all:+.3f}  non-saturated={r_ns:+.3f} ({len(nonsat)} pairs; control non-sat +0.124)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--margin-weight", type=float, default=2.0)
    ap.add_argument("--wiki-abs", type=float, default=0.5,
                    help="weight on bce(μ(child|parent),0.9) — absolute anchor for the WIKI map")
    ap.add_argument("--wiki-weight", type=float, default=1.0, help="WIKI loss weight (parity)")
    ap.add_argument("--llm-weight", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=1, help="ancestor depth (1 = node+parents)")
    ap.add_argument("--max-anc", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--llm", action="store_true", help="add the LLM operator (already-bought fixture)")
    ap.add_argument("--pairs", default=PAIRS, help="scored SYM pairs file (use mu_pairs_scored_large.tsv)")
    ap.add_argument("--sym-weight", type=float, default=1.0, help="SYM loss weight (ablation lever b)")
    ap.add_argument("--sym-only", action="store_true", help="single-task SYM head (ablation lever c)")
    ap.add_argument("--quick-val", action="store_true", help="skip dense-map emission/lin-agreement")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
