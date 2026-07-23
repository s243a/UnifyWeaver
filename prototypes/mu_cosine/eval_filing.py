#!/usr/bin/env python3
"""eval_filing.py — bookmark-filing retrieval eval on REAL Pearltrees filing decisions.

The formal, NON-CIRCULAR retrieval eval the applications roadmap asked for (DESIGN_model_applications.md
build-plan §1-2). Ground truth = where each bookmark is *actually filed* (its `treeId`) — a real human
decision, not graph distance. Task: rank candidate folders by μ(bookmark|folder) and measure recall@k / MRR.

Six rankers, head-to-head — isolating what the model adds over plain similarity, and whether the review-hardened
recipe transfers OOD (μ trained on Wikipedia categories → REAL personal bookmark filing):
  * e5-cos      — raw e5 cosine(query: bookmark, passage: folder) — the symmetric, no-model baseline.
  * mu-super    — μ under the equal-weight operator superposition (unconditional relatedness).
  * mu-elem     — μ under the ELEM (element_of) operator alone.
  * mu-max      — max(μ-elem, μ-wiki, μ-sym): the non-linear operator-OR that won the Wikipedia retrieval eval.
  * e5+mu-max   — e5 + 0.9·μ-max, the coverage-insurance blend (the tuned default; small e5 = untrained-region catch-all).
  * margin-gate — per-query blend weight from the μ-max MARGIN (top1−top2 over folders): the self-annealing/#3391
                  finding operationalised — lean on μ where its margin is sharp, fall back to e5 where it's flat.
This is the OOD test of the central deferred claim: the margin gate / coverage-insurance were predicted to pay off
exactly HERE (real data μ hasn't seen), where low confidence coincides with μ being wrong, not just unhelpful.

The connection to the bookmarking agent (scripts/infer_pearltrees_federated.py): that agent solves the SAME
rank-folders-by-relatedness task with a Procrustes-projection + cosine engine. This eval is the apparatus to
later drop that engine in as a 4th ranker. Filing data lives in .local (gitignored harvest); this script is
committable, the data is not.

Usage:
  python3 eval_filing.py --ckpt model_nodetype.pt --trees ../../.local/data/pearltrees_api/trees \
      --min-bm 3 --max-queries 500 [--seed 7]
"""
import argparse, json, os, random, collections
import torch
from mu_attention import build_e5_tables, Tokenizer, MuAttention, OPS, load_dag, GRAPH
from filing_privacy import build_pearltrees_privacy_index
from privacy import is_private_title, vis_private


import re
# TIER 1 — truly meaningless: maintenance / template / navigation categories whose TITLE is procedural noise
# (zero topical content). Neither e5-cos nor μ can rank them; no training data fixes a meaningless label. Always
# safe to drop (pure noise removal).
_JUNK = re.compile(r"catautotoc|navseasoncats|navbox|wikipedia|^articles?[ _]|^pages?[ _]|^categor|"
                   r"[ _]stubs?\b|templates?|redirects?|hidden|tracking|maintenance|disambiguation", re.I)
# TIER 2 — loosely semantic: structural / temporal / by-X groupings (years, places, nationalities). These DO
# carry meaning (just lower semantic density), so dropping them makes the eval EASIER on valid-but-hard targets.
# Keep by default; down-SAMPLE (not drop) in training. Drop only with --drop-structural (to decompose the gain).
_STRUCTURAL = re.compile(r"by[ _]nationality|by[ _]country|by[ _]year|by[ _]decade|by[ _]century|by[ _]date|"
                         r"by[ _]state|establishments|disestablishments|introductions|^years[ _]of|^decades|"
                         r"^centuries", re.I)
def is_admin(name, level="junk"):
    """level='junk' → Tier-1 only (meaningless); level='all' → Tier-1 + Tier-2 (also loosely-semantic structural)."""
    n = name.replace("_", " ")
    return bool(_JUNK.search(n) or (level == "all" and _STRUCTURAL.search(n)))


def load_membership(graph_path, min_bm, holdout=None, drop_admin=None):
    """IN-DOMAIN home-turf analog of filing: the simplewiki category DAG. A 'folder' = a category; its
    'bookmarks' = its child nodes. Same (member → container) shape as filing, but in the model's TRAINED
    region. Returns (queries=[(child_title, parent_id)], cand={parent_id: parent_title}).

    `holdout` (optional set of RAW node names): restrict QUERIES to members in this set — for the node-holdout
    eval (members the checkpoint NEVER trained on; candidates stay the full folder set). Kills the memorisation
    caveat: ranking never-seen nodes vs e5-cos is pure generalisation."""
    parents, children, deg = load_dag(graph_path)
    disp = lambda s: s.replace("_", " ")
    ok = (lambda n: not is_admin(n, drop_admin)) if drop_admin else (lambda n: True)
    cand = {par: disp(par) for par, kids in children.items() if len(kids) >= min_bm and ok(par)}
    queries = [(disp(c), par) for par in cand for c in children[par]
               if ok(c) and (holdout is None or c in holdout)]
    return queries, cand


def load_filing(trees_dir, min_bm, *, return_privacy=False, paths_jsonl=None):
    """Parse a certified-public harvested snapshot into filing queries/folders.

    Privacy filtering is unconditional: there is deliberately no
    ``include_private`` escape hatch.  Candidate eligibility and ``min_bm`` are
    computed *after* private-title bookmarks and private/quarantined folders
    have been removed, so excluded content cannot influence ranking margins or
    routing.  With ``return_privacy=True`` the exact privacy index used for the
    population is returned as a third value.
    """
    privacy = build_pearltrees_privacy_index(trees_dir, paths_jsonl=paths_jsonl)
    folders = {}                                  # tid -> title
    by_folder = collections.defaultdict(list)     # tid -> [bookmark_title, ...]
    for tid_key, t in privacy.tree_payloads.items():
        if tid_key not in privacy.public_ids:
            continue
        tid, ttitle = privacy.tree_value_ids[tid_key], t.get("title")
        if tid is None or not ttitle:
            raise ValueError(f"certified-public tree {tid_key} lacks a title")
        folders[tid] = ttitle
        for p in t.get("pearls", []) if isinstance(t.get("pearls"), list) else []:
            if (
                isinstance(p, dict)
                and str(p.get("contentType")) == "1"
                and p.get("title")
                and not is_private_title(p["title"])
                and not vis_private(p.get("visibility"))
            ):
                by_folder[tid].append(p["title"])
    cand = {tid: folders[tid] for tid, bms in by_folder.items() if len(bms) >= min_bm and tid in folders}
    queries = [(bt, tid) for tid in cand for bt in by_folder[tid]]   # (bookmark_title, true_folder_tid)
    if return_privacy:
        return queries, cand, privacy
    return queries, cand


@torch.no_grad()
def score_mu(model, tok, idx, q_keys, f_keys, op_weights_row, dev, batch=512):
    """μ(bookmark|folder) for every (query, folder) pair → [Q, F] score matrix, for a fixed op_weights row."""
    n_ops = model.op_emb.weight.shape[0]
    items = [(qk, fk, 0) for qk in q_keys for fk in f_keys]        # (node=bookmark, root=folder, op placeholder)
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        b = tok.build(chunk, train=False)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        ow = op_weights_row.expand(len(chunk), n_ops).to(dev)
        out += model(**b, op_weights=ow).cpu().tolist()
    F = len(f_keys)
    return [out[r * F:(r + 1) * F] for r in range(len(q_keys))]    # row-major → [Q][F]


def metrics(ranks):
    """ranks: list of the true folder's 1-based rank per query. → recall@k, MRR, median rank."""
    n = len(ranks)
    r = {f"recall@{k}": sum(x <= k for x in ranks) / n for k in (1, 5, 10)}
    r["MRR"] = sum(1.0 / x for x in ranks) / n
    r["median_rank"] = sorted(ranks)[n // 2]
    return r


def rank_all(model, tok, qtbl, ptbl, idx, q_keys, f_keys, truepos, dev):
    """Build all six filing rankers → (rank_of, ordered-names). Caller owns the q_keys/truepos split (so this is
    reusable for a FAIR held-out in-domain eval from train_filing, not just the zero-shot eval here)."""
    n_ops = model.op_emb.weight.shape[0]
    ow_of = lambda op: torch.zeros(1, n_ops).index_fill_(1, torch.tensor([OPS[op]]), 1.0)
    sm = lambda ow: torch.tensor(score_mu(model, tok, idx, q_keys, f_keys, ow, dev))   # [Q,F] μ score matrix
    S_elem, S_hier, S_sym = sm(ow_of("ELEM")), sm(ow_of("HIER")), sm(ow_of("SYM"))
    S_super = sm(torch.full((1, n_ops), 1.0 / n_ops))
    S_max = torch.maximum(torch.maximum(S_elem, S_hier), S_sym)   # operator-OR — the hierarchy-eval winner
    C = (qtbl[[idx[k] for k in q_keys]] @ ptbl[[idx[k] for k in f_keys]].T).cpu()       # e5 cosine [Q,F], unit-normed
    def nzrow(M):                                                 # per-query min-max → [0,1] across folders
        lo = M.min(dim=1, keepdim=True).values; hi = M.max(dim=1, keepdim=True).values
        return (M - lo) / (hi - lo + 1e-9)
    Cz, Sz, Se = nzrow(C), nzrow(S_max), nzrow(S_elem)
    S_blend = 0.1 * Cz + 0.9 * Sz                                 # e5 + 0.9·μ-max (coverage insurance, tuned α)
    S_blend_e = 0.1 * Cz + 0.9 * Se                               # e5 + 0.9·μ-elem (blend on the single trained op)
    top2 = S_max.topk(min(2, S_max.shape[1]), dim=1).values       # per-query μ-max margin = top1 − top2 over folders
    margin = (top2[:, 0] - top2[:, 1]).clamp(min=0) if top2.shape[1] > 1 else torch.zeros(S_max.shape[0])
    mq = margin.argsort().argsort().float() / max(1, len(margin) - 1)   # cross-query quantile position (scale-free)
    alpha = (0.3 + 0.6 * mq).unsqueeze(1)                         # low margin ⇒ lean e5; high ⇒ lean μ (the gate)
    S_gate = (1 - alpha) * Cz + alpha * Sz
    def ranks_from(M):                                            # per-query 1-based rank of the true folder
        return [1 + int(((M[r] > M[r][truepos[r]]) |
                         ((M[r] == M[r][truepos[r]]) & (torch.arange(M.shape[1]) < truepos[r]))).sum().item())
                for r in range(M.shape[0])]
    rank_of = {nm: ranks_from(M) for nm, M in (
        ("e5-cos", C), ("mu-super", S_super), ("mu-elem", S_elem), ("mu-max", S_max),
        ("e5+mu-max", S_blend), ("e5+mu-elem", S_blend_e), ("margin-gate", S_gate))}
    return rank_of, ("e5-cos", "mu-super", "mu-elem", "mu-max", "e5+mu-max", "e5+mu-elem", "margin-gate")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", choices=("pearltrees", "simplewiki"), default="pearltrees",
                    help="pearltrees = OOD bookmark filing (.local); simplewiki = IN-DOMAIN home-turf "
                         "category membership (isolates OOD by changing only the domain, lineage held off)")
    ap.add_argument("--trees", default=None, help="dir of harvested per-tree JSONs (.local; pearltrees source)")
    ap.add_argument("--graph", default=GRAPH, help="category_parent.tsv (simplewiki source)")
    ap.add_argument("--holdout-nodes", default=None, help="JSON list of RAW node names never seen in training; "
                    "restricts simplewiki queries to these (node-holdout: pure generalisation, no retrain)")
    ap.add_argument("--drop-admin", nargs="?", const="junk", default=None, choices=("junk", "all"),
                    help="drop categories: 'junk' = Tier-1 meaningless only (maintenance/template); "
                         "'all' = also Tier-2 loosely-semantic structural (by-year/country/…). Default off.")
    ap.add_argument("--min-bm", type=int, default=3, help="min bookmarks for a folder to be a candidate")
    ap.add_argument("--max-queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache", default="/tmp/filing_e5.pt", help="e5 table cache (regenerable)")
    ap.add_argument("--core-anchors", default="Physics,Mathematics,Chemistry,Computer science,Engineering",
                    help="comma-sep terms naming the model's TRAINED region; queries are stratified by their "
                         "true-folder's max e5-similarity to these (tests 'μ only helps inside its region')")
    a = ap.parse_args()

    if a.source == "simplewiki":
        held = set(json.load(open(a.holdout_nodes))) if a.holdout_nodes else None
        if held is not None:
            print(f"[HOLDOUT] restricting queries to {len(held)} never-trained nodes")
        queries, cand = load_membership(a.graph, a.min_bm, holdout=held, drop_admin=a.drop_admin)
    else:
        assert a.trees, "--trees required for --source pearltrees"
        queries, cand = load_filing(a.trees, a.min_bm)
    print(f"[DATA] source={a.source}: {len(cand)} candidate folders (>= {a.min_bm} members), "
          f"{len(queries)} eligible members")
    rng = random.Random(a.seed)
    if len(queries) > a.max_queries:
        queries = rng.sample(queries, a.max_queries)
    print(f"[DATA] evaluating {len(queries)} sampled queries against {len(cand)} folders "
          f"(random@{len(cand)} ≈ recall@10 {10/len(cand):.3f}, MRR {sum(1/r for r in range(1,len(cand)+1))/len(cand):.3f})")

    # vocab: folders F:<tid>, query bookmarks B:<i> — unique keys, real titles via `texts`
    f_keys = [f"F:{tid}" for tid in cand]
    ftext = {f"F:{tid}": cand[tid] for tid in cand}
    q_keys = [f"B:{i}" for i in range(len(queries))]
    qtext = {f"B:{i}": queries[i][0] for i in range(len(queries))}
    anchors = [s.strip() for s in a.core_anchors.split(",") if s.strip()]
    a_keys = [f"A:{i}" for i in range(len(anchors))]
    atext = {f"A:{i}": anchors[i] for i in range(len(anchors))}
    names = f_keys + q_keys + a_keys
    texts = {**ftext, **qtext, **atext}
    qtbl, ptbl, idx = build_e5_tables(names, cache_path=a.cache, texts=texts, device=a.device)

    dev = torch.device(a.device)
    tok = Tokenizer(qtbl, ptbl, idx, parents={}, deg={})          # no DAG: bookmarks/folders have no lineage
    ck = torch.load(a.ckpt, weights_only=False)
    sd = ck["state"]
    cfg = ck.get("cfg", {"d_model": qtbl.shape[1], "heads": 4, "layers": 3})
    # provenance-axis sizes vary by checkpoint vintage — infer from the state dict; account_emb (if absent) is
    # a zero-init no-op for our 3-tuple (provenance-masked) items, so load non-strict.
    sz = lambda k, d: sd[k].shape[0] if k in sd else d
    model = MuAttention(d_model=cfg["d_model"], n_heads=cfg["heads"], n_layers=cfg["layers"],
                        n_ops=sz("op_emb.weight", len(OPS)), n_corpus=sz("corpus_emb.weight", 2),
                        n_judge=sz("judge_emb.weight", 2), n_nodetype=sz("nodetype_emb.weight", 4)).to(dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert all(("account" in k or "prefix" in k) for k in missing), f"unexpected missing keys: {missing}"
    model.eval()
    print(f"[MODEL] {os.path.basename(a.ckpt)}  {cfg}  (n_ops={model.op_emb.weight.shape[0]})")

    true_tids = [t for _, t in queries]
    f_order = list(cand)                                          # tid order aligned with f_keys
    truepos = [f_order.index(t) for t in true_tids]

    rank_of, order = rank_all(model, tok, qtbl, ptbl, idx, q_keys, f_keys, truepos, dev)
    fn = ptbl[[idx[k] for k in f_keys]]                           # folder passage-embeddings (for core-sim stratification)

    def table(title, qsel):
        print(f"\n{title}  (n={len(qsel)})")
        print(f"  {'ranker':10} {'recall@1':>9} {'recall@5':>9} {'recall@10':>10} {'MRR':>7} {'med.rank':>9}")
        for name in order:
            m = metrics([rank_of[name][i] for i in qsel])
            print(f"  {name:10} {m['recall@1']:9.3f} {m['recall@5']:9.3f} {m['recall@10']:10.3f} "
                  f"{m['MRR']:7.3f} {m['median_rank']:9d}")

    table("[OVERALL]", list(range(len(queries))))

    # ---- stratify by distance to the model's TRAINED region (the user's "outside the physics core" test) ----
    an = ptbl[[idx[k] for k in a_keys]]                           # anchor passage-embeddings [A, d]
    core_sim = (fn @ an.T).max(dim=1).values                      # per-folder max cosine to any core anchor
    qcore = [core_sim[f_order.index(t)].item() for t in true_tids]    # per-query: its true folder's core-sim
    order_by_core = sorted(range(len(queries)), key=lambda i: qcore[i])
    third = len(queries) // 3
    bins = [("FAR from core", order_by_core[:third]),
            ("MID", order_by_core[third:2 * third]),
            ("NEAR core (STEM)", order_by_core[2 * third:])]
    print(f"\n[STRATIFIED by true-folder e5-similarity to core anchors: {anchors}]")
    for label, qsel in bins:
        lo, hi = qcore[qsel[0]], qcore[qsel[-1]]
        table(f"  {label}  [core-sim {lo:.2f}..{hi:.2f}]", qsel)
    print("\n  → if μ closes the gap to e5-cos NEAR the core but loses FAR, the model helps only in its trained "
          "region\n    (a single global transform); the fix is per-region/mixture μ — cf. the bookmarking agent's "
          "routed per-cluster Procrustes.")


if __name__ == "__main__":
    main()
