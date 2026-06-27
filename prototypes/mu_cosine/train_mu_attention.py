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

from mu_attention import (OPS, CORPORA, JUDGES, NODETYPE, GRAPH, load_dag, all_names, build_e5_tables,
                          Tokenizer, MuAttention)

# map the pair-file a_type/b_type strings to NODETYPE ids (category=0 default; pearltrees "collection" → 3)
_NT = {"category": 0, "page": 1, "mindmap_node": 2, "collection": 3, "pearltrees_collection": 3}
def nt(s):
    return _NT.get(s, 0)


# relation → (operator, forward target, reverse target) for the INFER-BLEND path (§5b: blend the operator
# embedding at OPERATOR level, but keep the μ target at RELATION level — SYM's relations have different μ).
REL_SPEC = {
    "element_of":     ("ELEM", 0.90, 0.10), "subcategory": ("WIKI", 0.90, 0.10),
    "subtopic":       ("WIKI", 0.85, 0.12), "super_category": ("WIKI", 0.85, 0.12),
    "see_also":       ("SYM",  0.40, 0.40), "assoc": ("SYM", 0.30, 0.30),
    "bridge":         ("SYM",  0.90, 0.90), "bridge_neg": ("SYM", 0.10, 0.10),
}


def switched_op(op, rel, conf, breadth, base, scale, rng):
    """For an INFERRED `element_of` (conf<1.0), stochastically return WIKI (subcategory) instead of ELEM, with
    p = base · min(1, breadth/scale) · (1−conf). Pure + module-level so it is unit-testable and so its RNG is
    ISOLATED from the training rng (passed in) — switch-on/off then preserves the same sampling trajectory.
    Tagged rows (conf≥1.0) and non-element_of rows are returned unchanged (p=0)."""
    if op == "ELEM" and rel == "element_of" and conf < 1.0:
        p = base * min(1.0, breadth / max(1.0, scale)) * (1.0 - conf)
        if rng.random() < p:
            return "WIKI"                                # train this untagged membership as a subcategory
    return op

# PART B provenance tags for the current single-corpus data. corpus = simplewiki (the 10k graph);
# judge = haiku for bought LLM labels (SYM positives, LLM fixture), graph for the free graph-derived
# labels (WIKI edges, and the μ=0 SYM negatives = non-edges). Threaded as the 4th/5th item fields so the
# provenance token can be masked (default) or revealed during training.
SW, HAIKU, GRAPHJ = CORPORA["simplewiki"], JUDGES["haiku"], JUDGES["graph"]

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
            # any non-neg stratum (pos / pos_phys / pos_chem / cross) is a graded positive; neg is μ=0.
            # 4th field = relation (extended page/pearltrees rows carry it; default subcat_of/SYM-style).
            rel = p[5].strip() if len(p) > 5 and p[5].strip() else "subcat_of"
            at = p[6].strip() if len(p) > 6 and p[6].strip() else "category"   # a_type (root endpoint)
            bt = p[7].strip() if len(p) > 7 and p[7].strip() else "category"   # b_type (node endpoint)
            (neg if p[2] == "neg" else pos).append((p[0], p[1], float(p[4]), rel, at, bt))
    return pos, neg


def load_graded(pairs_path, nodes_path=None):
    """Read the fused multi-corpus graded round (build_graded_round.py). Returns (rows, node_text):
      rows      = (node, root, mu, op, relation, node_type, root_type, corpus, judge) directional targets
      node_text = key → embed_text (the e5 string per fused node; key is corpus-prefixed mm:/pt:/wiki:)."""
    if nodes_path is None:
        nodes_path = pairs_path.replace("_pairs.tsv", "_nodes.tsv")
    node_text = {}
    if os.path.exists(nodes_path):
        with open(nodes_path) as f:
            for ln in f:
                if ln.startswith("#"):
                    continue
                c = ln.rstrip("\n").split("\t")          # key, corpus, node_type, title, embed_text
                if len(c) >= 5:
                    node_text[c[0]] = c[4] or c[3]
    rows = []
    with open(pairs_path) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split("\t")              # node root mu op relation n_type r_type corpus judge
            if len(c) >= 9:
                conf = float(c[9]) if len(c) > 9 and c[9] else 1.0   # 1.0 tagged; <1.0 INFERRED
                rows.append((c[0], c[1], float(c[2]), c[3], c[4], c[5], c[6], c[7], c[8], conf))
    return rows, node_text


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
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(items), bs):
        b = tok.build(items[i:i + bs], train=False)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        out.append(model(**b).cpu())
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
    dev_arg = getattr(args, "device", "auto")
    device = torch.device("cuda" if (dev_arg == "cuda" or (dev_arg == "auto" and torch.cuda.is_available()))
                          else "cpu")
    print(f"device: {device}")
    parents, children, deg = load_dag()
    names = all_names(parents, children)
    adj = {}
    for c, ps in parents.items():
        for p in ps:
            adj.setdefault(c, set()).add(p)
            adj.setdefault(p, set()).add(c)
    # union in every node referenced by --pairs/--replay-pairs/LLM so cold-start nodes (pairs whose
    # endpoints aren't in the GRAPH — e.g. cross-slice, page, or pearltrees nodes) still get a frozen e5
    # embedding instead of being silently dropped by the `in idx` filters below.
    extra, extra_texts = set(), {}
    for _pth in (args.pairs, args.replay_pairs):
        if _pth and os.path.exists(_pth):
            _pp, _nn = load_pairs(_pth)
            for _r in _pp + _nn:
                extra.add(_r[0]); extra.add(_r[1])
    if args.llm and os.path.exists(BOUNDARY):
        extra.update(load_mu(BOUNDARY).keys())
    # fused graded round: union its corpus-prefixed keys, each embedding its TITLE (embed_text), not the key.
    graded_rows, graded_text = ([], {})
    if args.graded and os.path.exists(args.graded):
        graded_rows, graded_text = load_graded(args.graded, args.graded_nodes)
        for _k, _t in graded_text.items():
            if _k not in set(names):
                extra.add(_k); extra_texts[_k] = _t
    names = list(dict.fromkeys(list(names) + sorted(extra - set(names))))
    cache = os.environ.get("UW_E5_CACHE", os.path.join(ROOT, "e5_tables.pt"))
    if args.graded:                                       # keep the base cache pristine; graded set differs
        cache = cache.replace(".pt", "_graded.pt")
    q, p, idx = build_e5_tables(names, cache_path=cache, texts=extra_texts, device=str(device),
                                batch_size=128 if device.type == "cuda" else 512)
    tok = Tokenizer(q, p, idx, parents, deg, k=args.k, beta=1.0, max_anc=args.max_anc)

    rng = random.Random(args.seed)
    edges = [e for e in load_edges() if e[0] in idx and e[1] in idx]
    rng.shuffle(edges)
    n_eh = int(0.1 * len(edges))
    edges_hold, edges_tr = edges[:n_eh], edges[n_eh:]
    pos, neg = load_pairs(args.pairs)
    pos = [r for r in pos if r[0] in idx and r[1] in idx]
    neg = [r for r in neg if r[0] in idx and r[1] in idx]
    # split by RELATION: element-of (page / collection membership) trains on the ELEM operator; everything
    # else (subcat_of / association / default) on SYM. a=category/topic (root), b=page/member (node).
    elem_pos = [r for r in pos if r[3] == "element_of"]
    elem_neg = [r for r in neg if r[3] == "element_of"]
    pos = [r for r in pos if r[3] != "element_of"]
    neg = [r for r in neg if r[3] != "element_of"]
    rng.shuffle(pos)
    n_ph = int(0.2 * len(pos))
    pos_hold, pos_tr = pos[:n_ph], pos[n_ph:]
    rng.shuffle(elem_pos)
    n_eh2 = int(0.2 * len(elem_pos))
    elem_hold, elem_tr_pos = elem_pos[:n_eh2], elem_pos[n_eh2:]
    elem_tr = elem_tr_pos + elem_neg
    print(f"WIKI edges: {len(edges_tr)} train / {len(edges_hold)} held-out")
    print(f"SYM pairs: {len(pos_tr)} pos + {len(neg)} neg train / {len(pos_hold)} held-out positives")
    print(f"ELEM pairs: {len(elem_tr_pos)} pos + {len(elem_neg)} neg train / {len(elem_hold)} held-out positives")

    # FINE-TUNE-WITH-REPLAY (continual learning): when --replay-pairs is given, --pairs is the NEW data
    # (e.g. the engineering build-out) and the replay file is the cumulative scored set. Each SYM batch
    # mixes a `--replay-frac` fraction of OLD (replay) examples with the new ones, so the warm-started
    # head does not catastrophically forget the existing domains while it learns the new one.
    replay_pos, replay_neg = [], []
    if args.replay_pairs:
        rp, rn = load_pairs(args.replay_pairs)
        replay_pos = [r for r in rp if r[0] in idx and r[1] in idx]
        replay_neg = [r for r in rn if r[0] in idx and r[1] in idx]
        print(f"REPLAY: {len(replay_pos)} pos + {len(replay_neg)} neg from {os.path.basename(args.replay_pairs)} "
              f"(replay-frac {args.replay_frac:.2f} of each SYM batch)")

    llm = []
    if args.llm:
        bmu = load_mu(BOUNDARY)
        llm = [(n, "Physics", m) for n, m in bmu.items() if n in idx]
        print(f"LLM (already-bought boundary fixture, no new spend): {len(llm)} directional labels")

    # fused graded round (mixed-operator, hand-curated): hold out 15% for eval; bridges down-weighted in loss
    graded_rows = [r for r in graded_rows if r[0] in idx and r[1] in idx]
    rng.shuffle(graded_rows)
    n_gh = int(0.15 * len(graded_rows))
    graded_hold, graded_tr = graded_rows[:n_gh], graded_rows[n_gh:]
    if args.graded:
        from collections import Counter as _C
        n_inf = sum(1 for r in graded_tr if len(r) > 9 and r[9] < 1.0)
        print(f"GRADED round: {len(graded_tr)} train / {len(graded_hold)} held-out "
              f"(ops {dict(_C(r[3] for r in graded_tr))}; bridge weight {args.bridge_weight:g}; "
              f"inferred {n_inf}" + (f", infer-switch p={args.infer_subcat:g}×breadth×(1−conf), isolated rng"
                                     if args.infer_switch else "")
              + ")")

    # INFERRED relations (confidence<1, e.g. a typo'd / missing section) are untagged ⇒ the operator is
    # uncertain. Stochastically switch an inferred `element_of` → `subcategory` (same μ, directional, but the
    # WIKI operator) with probability ∝ the container's BREADTH (a broad domain is likelier to hold
    # subcategories than leaf elements). This injects operator noise + models the element/subcategory
    # ambiguity for untagged relations; TAGGED rows (conf 1.0) always keep their operator.
    elem_breadth = {}                                    # container key → # element_of MEMBERS (breadth proxy)
    for r in graded_tr:                                  # count the FORWARD row only (μ≥0.5) so an edge counts
        if r[4] == "element_of" and r[2] >= 0.5:         # once — not 2× for its fwd+rev rows
            elem_breadth[r[1]] = elem_breadth.get(r[1], 0) + 1
    # ISOLATED rng for the switch: switch-on vs switch-off then consumes NO draws from the training rng, so
    # the batch-sampling/masking trajectory is identical between the A/B arms (only the operator differs).
    switch_rng = random.Random(args.seed + 8191) if args.infer_switch else None

    def graded_op(r):
        if not (args.infer_switch and len(r) > 9):
            return r[3]
        container = r[1] if r[2] >= 0.5 else r[0]        # breadth keyed on the container (root in fwd rows)
        return switched_op(r[3], r[4], r[9], elem_breadth.get(container, 0),
                           args.infer_subcat, args.breadth_scale, switch_rng)

    torch.manual_seed(args.seed)
    model = MuAttention(d_model=q.shape[1], n_heads=args.heads, n_layers=args.layers)
    if args.init_from:                                    # warm start — DON'T reinit the head (fine-tune)
        ck = torch.load(args.init_from, weights_only=False)
        sd, own = ck["state"], model.state_dict()
        grown = []
        for k, v in sd.items():
            if k not in own:
                continue
            if own[k].shape == v.shape:
                own[k] = v
            elif own[k].dim() >= 1 and own[k].shape[0] > v.shape[0] and own[k].shape[1:] == v.shape[1:]:
                # op-indexed tensor grew (new ELEM operator). Copy the old rows; seed each NEW row from
                # the SYM row (op 0) so ELEM starts SYM-like (membership) and specialises during fine-tune.
                t = own[k].clone()
                t[:v.shape[0]] = v
                t[v.shape[0]:] = v[OPS["SYM"]]
                own[k] = t
                grown.append(k)
        model.load_state_dict(own)
        print(f"WARM START from {os.path.basename(args.init_from)} (fine-tune, head NOT reinitialised) "
              f"@ lr {args.lr:g}" + (f"; grew {grown} for new op(s), ELEM seeded from SYM" if grown else ""))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    m = args.margin

    def bld(items, **kw):                                  # tokenize on CPU, move the batch to the device
        b = tok.build(items, **kw)
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

    # ---- INFER-BLEND (§5b): random operator embedding from the fitted joint P(relation|μ_vec) ----
    import numpy as _np
    from mu_posterior import JointPosterior, sample_operator_weights
    blend_rng = random.Random(args.seed + 9973)            # ISOLATED rng for inferred-row sampling
    blend_nprng = _np.random.default_rng(args.seed + 9973)  # ISOLATED rng for the Dirichlet draws
    graded_inf = [r for r in graded_tr if len(r) > 9 and r[9] < 1.0]   # inferred rows (conf<1) for the blend
    graded_tag = [r for r in graded_tr if not (len(r) > 9 and r[9] < 1.0)]
    relset = sorted(set(r[4] for r in graded_tag))
    blend_state = {"pop": None, "tgt": None}               # per-inferred-row P(op) [N,n_ops] + blended target
    # tagged-blend (DESIGN §7): when --blend-tagged-conf c < 1.0, ALSO blend TAGGED rows as a REGULARIZER with
    # a STATIC label-smoothed op distribution p = c·onehot(label) + (1-c)·uniform (the label is a sharp prior,
    # not a delta). Feeds the regularizer the whole set, not just the (small) inferred remnant.
    blend_tag_on = bool(args.infer_blend and args.blend_tagged_conf < 1.0)
    tag_pop = None
    if blend_tag_on:
        c = args.blend_tagged_conf
        tag_pop = _np.full((len(graded_tag), len(OPS)), (1.0 - c) / len(OPS))
        for i, r in enumerate(graded_tag):
            tag_pop[i, OPS[graded_op(r)]] += c
    blend_all = (graded_inf + graded_tag) if blend_tag_on else graded_inf   # inferred first, then tagged

    @torch.no_grad()
    def measure_readouts(pairs):                           # [N,6]: e5, sym, wiki_fwd/rev, elem_fwd/rev (current model)
        model.eval()
        def mb(items):
            return mu_batch(model, tok, items).numpy()
        e5 = _np.array([float(tok.p[idx[a]] @ tok.q[idx[b]]) if a in idx and b in idx else _np.nan for a, b in pairs])
        cols = [e5, mb([(a, b, OPS["SYM"]) for a, b in pairs]),
                mb([(a, b, OPS["WIKI"]) for a, b in pairs]), mb([(b, a, OPS["WIKI"]) for a, b in pairs]),
                mb([(a, b, OPS["ELEM"]) for a, b in pairs]), mb([(b, a, OPS["ELEM"]) for a, b in pairs])]
        model.train()
        return _np.stack(cols, 1)

    def refresh_blend():                                   # fit the joint head on tagged, predict P(op)+target for inferred
        tag = graded_tag if len(graded_tag) <= 1500 else [graded_tag[i] for i in
              random.Random(0).sample(range(len(graded_tag)), 1500)]
        Xt = measure_readouts([(r[0], r[1]) for r in tag])
        jp = JointPosterior(relset, n_features=6, hidden=args.blend_hidden).fit(Xt, [r[4] for r in tag])
        Pr = jp.proba(measure_readouts([(r[0], r[1]) for r in graded_inf]))      # [N, n_rel]
        pop = _np.zeros((len(graded_inf), len(OPS)))
        tgt = _np.zeros(len(graded_inf))
        for j, rel in enumerate(relset):
            op_i = OPS[REL_SPEC[rel][0]]
            for i, r in enumerate(graded_inf):
                pop[i, op_i] += Pr[i, j]
                tgt[i] += Pr[i, j] * REL_SPEC[rel][1 if r[2] >= 0.5 else 2]      # relation-level dir-specific target
        blend_state["pop"], blend_state["tgt"] = pop, tgt

    new_corpus = CORPORA.get(args.pairs_corpus, SW)        # corpus tag for the --pairs (new) data
    def draw_sym(pool, replay, k):
        """Draw k SYM examples as (item, corpus_id): replay-frac OLD (replay, corpus=simplewiki) mixed with
        new (pool, corpus=--pairs-corpus) when replay is active."""
        if replay and args.replay_frac > 0:
            n_old = int(round(k * args.replay_frac))
            old = [(replay[rng.randrange(len(replay))], SW) for _ in range(n_old)]
            new = [(pool[rng.randrange(len(pool))], new_corpus) for _ in range(k - n_old)] if pool else []
            return old + new
        return [(pool[rng.randrange(len(pool))], new_corpus) for _ in range(k)]

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
            wiki_items = ([(c, par, OPS["WIKI"], SW, GRAPHJ) for c, par in eb]
                          + [(par, c, OPS["WIKI"], SW, GRAPHJ) for c, par in eb]
                          + [(c, negpar[i], OPS["WIKI"], SW, GRAPHJ) for i, (c, par) in enumerate(eb)])
            mu_w = model(**bld(wiki_items, train=True, rng=rng, p_mask_prov=args.prov_mask))
            mu_cp, mu_pc, mu_cpn = mu_w[:args.bs], mu_w[args.bs:2 * args.bs], mu_w[2 * args.bs:]
            L_wiki = (bce(mu_pc, 0) + bce(mu_cpn, 0)
                      + args.wiki_abs * bce(mu_cp, 0.9)
                      + args.margin_weight * (F.relu(m - (mu_cp - mu_pc)).mean()
                                              + F.relu(m - (mu_cp - mu_cpn)).mean()))
        # SYM (order-invariant) — BALANCED pos/neg so the μ=0 negatives can't collapse μ→0
        half = args.bs // 2
        sb = (draw_sym(pos_tr, replay_pos, half) +
              draw_sym(neg, replay_neg, args.bs - half))     # list of ((a,b,mu), corpus_id)
        sb_j = [HAIKU] * half + [GRAPHJ] * (args.bs - half)             # pos=bought, neg=free non-edge
        sym_items = ([(it[0], it[1], OPS["SYM"], cid, j) for (it, cid), j in zip(sb, sb_j)] +
                     [(it[1], it[0], OPS["SYM"], cid, j) for (it, cid), j in zip(sb, sb_j)])
        tgt = torch.tensor([it[2] for it, _ in sb]).to(device)
        mu_s = model(**bld(sym_items, train=True, rng=rng, p_mask_prov=args.prov_mask))
        mu_ab, mu_ba = mu_s[:len(sb)], mu_s[len(sb):]
        L_sym = F.mse_loss(mu_ab, tgt) + F.mse_loss(mu_ba, tgt)
        # ELEM (element-of: directional + graded) — μ(page|category) toward the graded centrality target,
        # plus a directional margin pushing it above the reverse μ(category|page) on positives. a=category
        # (root), b=page (node). Its own operator token + readout row → page-membership as a distinct function.
        L_elem = torch.zeros(())
        if elem_tr and not args.sym_only:
            eb2 = [elem_tr[rng.randrange(len(elem_tr))] for _ in range(args.bs)]
            # node-type tags (opt-in): node=b (member, b_type=r[5]), root=a (topic, a_type=r[4])
            if args.use_nodetype:
                fwd = [(r[1], r[0], OPS["ELEM"], new_corpus, HAIKU, nt(r[5]), nt(r[4])) for r in eb2]
                rev = [(r[0], r[1], OPS["ELEM"], new_corpus, HAIKU, nt(r[4]), nt(r[5])) for r in eb2]
            else:
                fwd = [(r[1], r[0], OPS["ELEM"], new_corpus, HAIKU) for r in eb2]   # μ(page|cat)
                rev = [(r[0], r[1], OPS["ELEM"], new_corpus, HAIKU) for r in eb2]   # μ(cat|page)
            mu_e = model(**bld(fwd + rev, train=True, rng=rng, p_mask_prov=args.prov_mask))
            mu_ef, mu_er = mu_e[:args.bs], mu_e[args.bs:]
            tgt_e = torch.tensor([r[2] for r in eb2]).to(device)
            posmask = (tgt_e > 0).float()
            L_elem = (F.mse_loss(mu_ef, tgt_e)
                      + args.margin_weight * (F.relu(m - (mu_ef - mu_er)) * posmask).sum()
                      / posmask.sum().clamp_min(1.0))
        # GRADED (fused multi-corpus, hand-curated) — per-row operator, weighted MSE to the calibrated μ.
        # bridges (μ≈0.9 "same concept", the bulk) are DOWN-WEIGHTED so they don't swamp the directional
        # WIKI/ELEM signal. Endpoints carry corpus/judge (maskable) + node-type tags (when --use-nodetype).
        L_graded = torch.zeros(())
        # with infer-blend, the blend handles inferred separately. With tagged-blend, tagged rows train HARD
        # until the warmup completes — the blend's joint posterior + μ readouts must be established BEFORE the
        # blend consumes them (DESIGN §4/§7) — then move into the blend.
        tag_blended_now = blend_tag_on and step >= args.blend_warmup
        graded_pool = (([] if tag_blended_now else graded_tag) if args.infer_blend else graded_tr)
        if graded_pool and not args.sym_only:
            gb = [graded_pool[rng.randrange(len(graded_pool))] for _ in range(args.bs)]
            if args.use_nodetype:
                gi = [(r[0], r[1], OPS[graded_op(r)], CORPORA[r[7]], JUDGES[r[8]], nt(r[5]), nt(r[6])) for r in gb]
            else:
                gi = [(r[0], r[1], OPS[graded_op(r)], CORPORA[r[7]], JUDGES[r[8]]) for r in gb]
            gt = torch.tensor([r[2] for r in gb]).to(device)
            gw = torch.tensor([args.bridge_weight if r[4] == "bridge" else 1.0 for r in gb]).to(device)
            mu_g = model(**bld(gi, train=True, rng=rng, p_mask_prov=args.prov_mask))
            L_graded = (gw * (mu_g - gt) ** 2).sum() / gw.sum().clamp_min(1.0)
        # INFER-BLEND: inferred rows trained with a RANDOM operator embedding from the fitted joint posterior
        L_blend = torch.zeros(())
        blend_on = (args.infer_blend and (graded_inf or blend_tag_on)
                    and step >= args.blend_warmup and not args.sym_only)
        if blend_on:
            if (step - args.blend_warmup) % args.blend_refresh == 0 and graded_inf:   # refresh inferred posterior
                refresh_blend()
            n_inf_b = len(graded_inf)
            bi = [blend_rng.randrange(len(blend_all)) for _ in range(args.bs)]
            pops, tgts, items = [], [], []                                 # op=SYM is just the position marker;
            for i in bi:                                                   # op_weights overrides token + readout
                r = blend_all[i]
                if i < n_inf_b:                                            # inferred: μ-posterior P(op) + blended target
                    pops.append(blend_state["pop"][i]); tgts.append(blend_state["tgt"][i]); co, ju = new_corpus, HAIKU
                else:                                                      # tagged: label-smoothed P(op) + real μ + own provenance
                    pops.append(tag_pop[i - n_inf_b]); tgts.append(r[2]); co, ju = CORPORA[r[7]], JUDGES[r[8]]
                items.append((r[0], r[1], OPS["SYM"], co, ju, nt(r[5]), nt(r[6])) if args.use_nodetype
                             else (r[0], r[1], OPS["SYM"], co, ju))
            ow = _np.stack([sample_operator_weights(p, args.blend_alpha, blend_nprng) for p in pops])
            op_w = torch.tensor(ow, dtype=torch.float32).to(device)
            bt = torch.tensor(tgts, dtype=torch.float32).to(device)
            mu_b = model(**bld(items, train=True, rng=rng, p_mask_prov=args.prov_mask), op_weights=op_w)
            L_blend = F.mse_loss(mu_b, bt)
        loss = (args.sym_weight * L_sym + (0.0 if args.sym_only else args.wiki_weight * L_wiki)
                + (args.elem_weight * L_elem if (elem_tr and not args.sym_only) else 0.0)
                + (args.graded_weight * L_graded if (graded_tr and not args.sym_only) else 0.0)
                + (args.blend_weight * L_blend if blend_on else 0.0))
        # LLM (optional, already-bought) — skipped in single-task SYM mode
        L_llm = torch.zeros(())
        if llm and not args.sym_only:
            lb = [llm[rng.randrange(len(llm))] for _ in range(args.bs)]
            li = [(n, r, OPS["LLM"], SW, HAIKU) for n, r, _ in lb]
            lt = torch.tensor([mu for _, _, mu in lb]).to(device)
            L_llm = F.mse_loss(model(**bld(li, train=True, rng=rng, p_mask_prov=args.prov_mask)), lt)
            loss = loss + args.llm_weight * L_llm
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, args.steps // 6) == 0:
            print(f"  step {step+1:5d}  L_sym {float(L_sym):.4f}  L_wiki {float(L_wiki):.4f}"
                  + (f"  L_elem {float(L_elem):.4f}" if (elem_tr and not args.sym_only) else "")
                  + (f"  L_graded {float(L_graded):.4f}" if (graded_tr and not args.sym_only) else "")
                  + (f"  L_blend {float(L_blend):.4f}" if blend_on else "")
                  + (f"  L_llm {float(L_llm):.4f}" if (llm and not args.sym_only) else ""))

    model.eval()
    validate(model, tok, names, idx, adj, edges_hold, pos_hold, llm, args, elem_hold=elem_hold,
             graded_hold=graded_hold)
    if args.save:
        torch.save({"state": model.state_dict(), "cfg": {"d_model": q.shape[1], "heads": args.heads,
                    "layers": args.layers}}, args.save)
        print(f"\nsaved model → {args.save}")


def validate(model, tok, names, idx, adj, edges_hold, pos_hold, llm, args, elem_hold=None, graded_hold=None):
    print("\n=== PER-OPERATOR VALIDATION ===")
    ops = ["SYM"] if args.sym_only else (["SYM", "WIKI"] + (["LLM"] if llm else []))

    # --- GRADED: held-out fused-round fit, per operator (revealed provenance + node-type as trained) ---
    if graded_hold and not args.sym_only:
        from collections import defaultdict
        byop = defaultdict(list)
        for r in graded_hold:
            byop[r[3]].append(r)
        print(f"[GRADED] held-out fused round ({len(graded_hold)} targets) — μ vs calibrated target:")
        for op in sorted(byop):
            rs = byop[op]
            if args.use_nodetype:
                items = [(r[0], r[1], OPS[op], CORPORA[r[7]], JUDGES[r[8]], nt(r[5]), nt(r[6])) for r in rs]
            else:
                items = [(r[0], r[1], OPS[op], CORPORA[r[7]], JUDGES[r[8]]) for r in rs]
            pred = mu_batch(model, tok, items).tolist()
            tgt = [r[2] for r in rs]
            mse = sum((pred[i] - tgt[i]) ** 2 for i in range(len(rs))) / len(rs)
            corr = pearson(pred, tgt) if len(set(tgt)) > 1 else float("nan")
            print(f"  {op:5s} n={len(rs):4d}  MSE {mse:.4f}  r {corr:+.3f}")

    # --- WIKI: held-out edge order-accuracy ---
    if not args.sym_only:
        cp = mu_batch(model, tok, [(c, p, OPS["WIKI"]) for c, p in edges_hold])
        pc = mu_batch(model, tok, [(p, c, OPS["WIKI"]) for c, p in edges_hold])
        acc = float((cp > pc).float().mean())
        print(f"[WIKI] held-out edge ORDER-accuracy μ(child|parent)>μ(parent|child): {acc*100:.1f}% "
              f"({len(edges_hold)} edges)  mean μ(c|p)={float(cp.mean()):.3f} μ(p|c)={float(pc.mean()):.3f}")

    # --- SYM: held-out corr + symmetry (the headline metric for this work) ---
    ab = mu_batch(model, tok, [(a, b, OPS["SYM"]) for a, b, *_ in pos_hold])
    ba = mu_batch(model, tok, [(b, a, OPS["SYM"]) for a, b, *_ in pos_hold])
    tgt = [r[2] for r in pos_hold]
    corr = pearson(((ab + ba) / 2).tolist(), tgt)
    sym_gap = float((ab - ba).abs().mean())
    print(f"[SYM]  held-out μ corr (control +0.726, #3302 +0.335): {corr:+.3f}  ({len(pos_hold)} "
          f"positives, MSE {F.mse_loss((ab+ba)/2, torch.tensor(tgt)):.3f})  symmetry gap {sym_gap:.3f}")

    # --- ELEM: held-out page/collection-membership centrality corr + direction (the new operator) ---
    if elem_hold and not args.sym_only:
        # match training: node-type tags only when --use-nodetype (else plain 3-tuples)
        if args.use_nodetype:
            ef = mu_batch(model, tok, [(r[1], r[0], OPS["ELEM"], None, None, nt(r[5]), nt(r[4]))
                                       for r in elem_hold])                      # μ(page|category)
            er = mu_batch(model, tok, [(r[0], r[1], OPS["ELEM"], None, None, nt(r[4]), nt(r[5]))
                                       for r in elem_hold])                      # μ(category|page)
        else:
            ef = mu_batch(model, tok, [(r[1], r[0], OPS["ELEM"]) for r in elem_hold])   # μ(page|category)
            er = mu_batch(model, tok, [(r[0], r[1], OPS["ELEM"]) for r in elem_hold])   # μ(category|page)
        etgt = [r[2] for r in elem_hold]
        ecorr = pearson(ef.tolist(), etgt)
        edir = float((ef > er).float().mean())
        print(f"[ELEM] held-out centrality μ(page|cat) corr: {ecorr:+.3f}  ({len(elem_hold)} positives, "
              f"MSE {F.mse_loss(ef, torch.tensor(etgt)):.3f})  direction μ(page|cat)>μ(cat|page) {edir*100:.0f}%")

    # --- gate-leak 5-probe (control 0/5), per operator --- (filter to nodes present in this graph;
    # science-only slices like wide_enwiki won't contain Music/Cooking/etc.)
    nonphys = [n for n in NONPHYS if n in idx]
    for op in (ops if nonphys else []):
        pr = mu_batch(model, tok, [(n, "Physics", OPS[op]) for n in nonphys])
        leak = int((pr >= 0.3).sum())
        print(f"[{op}] gate-leak probe (non-physics μ(·|Physics)≥0.3): {leak}/{len(nonphys)}  " +
              "  ".join(f"{n.split('_')[0]}={v:.2f}" for n, v in zip(nonphys, pr.tolist())))
    if not nonphys:
        print("[gate-leak] none of the NONPHYS probe nodes are in this graph — skipped")

    # --- Physics-vs-Chemistry DISCRIMINATION: μ(node|Physics) vs μ(node|Chemistry), check ordering ---
    discrimination_probe(model, tok, idx)

    # --- PART B: provenance-token structural probe (masked-default vs revealed source) ---
    provenance_probe(model, tok, idx)

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


# Artificial_intelligence is a candidate 6th root (the enwiki widening round). The probe filters to roots
# present in the loaded graph, so on the 10k/simplewiki graphs (no AI) it stays 5-way automatically.
DOMAIN_ROOTS = ["Physics", "Chemistry", "Mathematics", "Computer_science", "Engineering",
                "Artificial_intelligence"]
DOMAIN_PROBE = {
    # Physics probe carries CLASSICAL nodes (comparable to prior rounds) + MODERN nodes (the enwiki
    # widening — tests whether the now-present modern-physics subfields read as Physics).
    "Physics": ["Thermodynamics", "Optics", "Mechanics", "Electromagnetism", "Motion_(physics)",
                "Quantum_field_theory", "Particle_physics", "General_relativity", "Condensed_matter_physics"],
    "Chemistry": ["Periodic_table", "Acids", "Chemical_compounds", "Oxygen", "Chemical_reactions"],
    "Mathematics": ["Calculus", "Differential_equations", "Mathematical_analysis", "Logic",
                    "Fields_of_mathematics", "Number_theory", "Group_theory", "Topology",
                    "Real_analysis", "Complex_analysis"],
    "Computer_science": ["Software", "Computer_hardware", "Operating_systems", "Computer_networking",
                         "Computer_architecture"],
    "Engineering": ["Mechanical_engineering", "Civil_engineering", "Engineering_disciplines",
                    "Machines", "Infrastructure"],
    "Artificial_intelligence": ["Machine_learning", "Neural_networks", "Deep_learning",
                                "Computer_vision", "Natural_language_processing"],
}
BORDER_PROBE = ["Atoms", "Electronics", "Measurement", "Materials", "Energy"]


def discrimination_probe(model, tok, idx):
    """MULTI-domain discrimination: for clear nodes of each domain, μ(node|own-root) over all DOMAIN_ROOTS
    {Physics, Chemistry, Mathematics, Computer_science, Engineering} (SYM operator). Reports BOTH metrics:
      * hard ARGMAX accuracy + confusion (is the true root strictly the top-1?) — brittle for a node that
        is genuinely high-μ to several roots (multi-membership), and
      * a RANKING/MARGIN view (#3314 finding): the true root's RANK among the 5, the signed margin
        μ(true) − max-other (>0 ⇔ argmax-correct), and top-1 / top-2 rates. If ranking is strong even
        where argmax flips, the "brittleness" is largely a metric artifact (correct multi-membership)."""
    roots = [r for r in DOMAIN_ROOTS if r in idx]
    if len(roots) < 2:
        print("[DISCRIM] <2 domain roots in graph — skipped")
        return
    ab = {r: r[:4] for r in roots}
    print(f"\n[DISCRIM] {len(roots)}-domain (argmax μ over {', '.join(roots)}):")
    confusion = {d: {r: 0 for r in roots} for d in roots}
    rankrows = {d: [] for d in roots}            # per-domain list of (rank, signed_margin)
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
            rank = 1 + sum(1 for r in roots if r != dom and vals[r] > vals[dom])
            best_other = max(vals[r] for r in roots if r != dom)
            rankrows[dom].append((rank, vals[dom] - best_other))
            scores = "  ".join(f"{ab[r]}={vals[r]:.2f}" for r in roots)
            print(f"    [{ab[dom]}] {n:22} {scores}  →{ab[pred]} {'✓' if ok else '✗'}  "
                  f"rank{rank} m{vals[dom]-best_other:+.2f}")
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
    print(f"  multi-domain discrimination accuracy (hard argmax): {correct}/{total} "
          f"({100*correct/max(total,1):.0f}%)")
    # --- ranking / margin view (the #3314 honest re-measure) ---
    print("  ranking/margin (true-root rank among the 5; signed margin μ(true)−max-other; top-1/top-2):")
    print(f"    {'domain':>7}  {'meanRank':>8}  {'meanMargin':>10}  {'top1':>5}  {'top2':>5}")
    allr = []
    for d in roots:
        rs = rankrows[d]
        allr += rs
        mr = sum(x[0] for x in rs) / max(len(rs), 1)
        mm = sum(x[1] for x in rs) / max(len(rs), 1)
        t1 = sum(1 for x in rs if x[0] == 1) / max(len(rs), 1)
        t2 = sum(1 for x in rs if x[0] <= 2) / max(len(rs), 1)
        print(f"    {ab[d]:>7}  {mr:>8.2f}  {mm:>+10.2f}  {100*t1:>4.0f}%  {100*t2:>4.0f}%")
    mr = sum(x[0] for x in allr) / max(len(allr), 1)
    t2 = sum(1 for x in allr if x[0] <= 2) / max(len(allr), 1)
    print(f"    {'ALL':>7}  {mr:>8.2f}  {'':>10}  {100*correct/max(total,1):>4.0f}%  {100*t2:>4.0f}%")


def provenance_probe(model, tok, idx):
    """PART B structural validation. For a fixed set of (node, root) SYM queries, compare μ with the
    provenance token MASKED (3-tuple → the default agnostic inference path) against μ with the source
    REVEALED (5-tuple → corpus_emb+judge_emb added). Confirms (1) the slot exists and is wired, (2)
    masking flips the input and is read by the model, (3) HONESTLY how much the (near-constant, single-
    corpus) provenance token shifts μ — small Δ is expected and is the point: masking marginalizes it out."""
    probe = [("Optics", "Physics"), ("Mechanics", "Physics"), ("Calculus", "Mathematics"),
             ("Acids", "Chemistry"), ("Software", "Computer_science"), ("Machines", "Engineering")]
    probe = [(n, r) for n, r in probe if n in idx and r in idx]
    if not probe:
        print("[PROV] no probe nodes in graph — skipped")
        return
    SW, HK, GR = CORPORA["simplewiki"], JUDGES["haiku"], JUDGES["graph"]
    masked = mu_batch(model, tok, [(n, r, OPS["SYM"]) for n, r in probe])
    rev_hk = mu_batch(model, tok, [(n, r, OPS["SYM"], SW, HK) for n, r in probe])
    rev_gr = mu_batch(model, tok, [(n, r, OPS["SYM"], SW, GR) for n, r in probe])
    d_hk = float((rev_hk - masked).abs().mean())
    d_gr = float((rev_gr - masked).abs().mean())
    print(f"\n[PROV] provenance token (corpus=simplewiki; judge=haiku|graph) — masked-default vs revealed:")
    for i, (n, r) in enumerate(probe):
        print(f"    {n:22} ({r[:4]})  masked={float(masked[i]):.3f}  "
              f"haiku={float(rev_hk[i]):.3f}  graph={float(rev_gr[i]):.3f}")
    print(f"  mean |Δμ| revealing source: haiku {d_hk:.3f}  graph {d_gr:.3f}  "
          f"(small ⇒ near-constant token, single-corpus; masking marginalizes it — the default path)")


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
    ap.add_argument("--elem-weight", type=float, default=1.0, help="ELEM (element-of/page-membership) loss weight")
    ap.add_argument("--use-nodetype", action="store_true", help="tag ELEM endpoints with the node-type token "
                    "(DESIGN §7). DEFAULT OFF: at current data operator and node-type are COLLINEAR "
                    "(ELEM⟺page), so it's redundant and hurts (REPORT_nodetype.md); enable once data has "
                    "within-operator type diversity (pearltrees collections / mindmap nodes).")
    ap.add_argument("--llm-weight", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=1, help="ancestor depth (1 = node+parents)")
    ap.add_argument("--max-anc", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3, help="transformer depth — bumped 2→3 once the ELEM "
                    "operator landed: 2 layers couldn't co-serve discrimination + page-centrality at full "
                    "strength (REPORT_element_operator.md); 3 resolves the interference. ~3.6M params.")
    ap.add_argument("--llm", action="store_true", help="add the LLM operator (already-bought fixture)")
    ap.add_argument("--pairs", default=PAIRS, help="scored SYM pairs file (use mu_pairs_scored_large_260620-223001.tsv)")
    ap.add_argument("--graded", default=None, help="fused multi-corpus graded round (build_graded_round.py "
                    "<out>_pairs.tsv); its <out>_nodes.tsv supplies the e5 embed_text per fused node")
    ap.add_argument("--graded-nodes", default=None, help="override the graded nodes file (default: derive "
                    "from --graded by _pairs→_nodes)")
    ap.add_argument("--graded-weight", type=float, default=1.0, help="graded-round loss weight")
    ap.add_argument("--bridge-weight", type=float, default=0.2, help="down-weight bridge targets in the "
                    "graded loss (they dominate at μ≈0.9; keep them from swamping directional signal)")
    ap.add_argument("--infer-switch", action="store_true", help="for INFERRED graded relations (confidence "
                    "<1, e.g. a typo'd/missing section), stochastically switch element_of→subcategory "
                    "(operator noise + element/subcategory ambiguity for untagged relations)")
    ap.add_argument("--infer-subcat", type=float, default=0.4, help="base switch prob (× container breadth)")
    ap.add_argument("--breadth-scale", type=float, default=20.0, help="member count at which breadth→1.0")
    ap.add_argument("--infer-blend", action="store_true", help="§5b: train INFERRED graded rows with a RANDOM "
                    "operator embedding drawn from the fitted joint P(relation|μ_vec) (supersedes --infer-switch)")
    ap.add_argument("--blend-warmup", type=int, default=200, help="steps tagged-only before the blend starts")
    ap.add_argument("--blend-refresh", type=int, default=100, help="re-measure readouts + re-fit joint head every N steps")
    ap.add_argument("--blend-alpha", type=float, default=20.0, help="Dirichlet concentration (low=more op noise)")
    ap.add_argument("--blend-hidden", type=int, default=0, help="joint head hidden units (0=logistic regression)")
    ap.add_argument("--blend-weight", type=float, default=1.0, help="infer-blend loss weight")
    ap.add_argument("--blend-tagged-conf", type=float, default=1.0, help="DESIGN §7: if <1.0, ALSO push TAGGED "
                    "rows through the operator superposition as a REGULARIZER — op ~ Dirichlet(alpha · "
                    "[c·onehot(label) + (1-c)·uniform]). c=1.0 (default) = tagged rows trained hard "
                    "(inferred-only blend). Feeds the regularizer the whole set; capacity-bounded.")
    ap.add_argument("--sym-weight", type=float, default=1.0, help="SYM loss weight (ablation lever b)")
    ap.add_argument("--sym-only", action="store_true", help="single-task SYM head (ablation lever c)")
    ap.add_argument("--quick-val", action="store_true", help="skip dense-map emission/lin-agreement")
    ap.add_argument("--prov-mask", type=float, default=0.5, help="PART B: prob of masking the provenance "
                    "token during training (1.0 = always-masked = the provenance-OFF ablation control)")
    ap.add_argument("--init-from", default=None, help="warm-start checkpoint for FINE-TUNING (head NOT "
                    "reinitialised); pair with a LOWER --lr (~1/3–1/5 of from-scratch)")
    ap.add_argument("--replay-pairs", default=None, help="cumulative scored set to REPLAY while fine-tuning "
                    "on --pairs (the new data); prevents catastrophic forgetting")
    ap.add_argument("--replay-frac", type=float, default=0.4, help="fraction of each SYM batch drawn from "
                    "the replay set (0.3–0.5 typical)")
    ap.add_argument("--pairs-corpus", default="simplewiki", choices=list(CORPORA),
                    help="provenance corpus tag for the --pairs (new) data; replay stays simplewiki")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto = cuda if available else cpu; keep --bs modest on a small GPU")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
