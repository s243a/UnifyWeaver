#!/usr/bin/env python3
"""`MuAttention` — directional, multi-relational μ over FROZEN e5 + learned tags.

Implements `DESIGN_directional_attention.md`: a tiny permutation-invariant transformer over a short
**set** of tagged tokens

    { operator(op), anchor(root), node(X)@gen0, {ancestors(X)}@gen_d (min-hop on the DAG), ⌀@gen=noise }

with a sigmoid μ-readout ∈ [0,1]. e5 embeddings are **frozen**; the only learned parameters are the tags
(`op_emb` codebook, `anchor_tag`, per-generation `gen_emb`), the 1–2-layer attention block, and a
per-operator linear readout. Every node = frozen e5 + shared tags ⇒ all 8,247 categories covered
(cold-start safe; no per-node table).

Key design points enforced here (see the doc):
  * **asymmetry is structural** — the `anchor`(root) and `node`(X) tokens carry different tags AND
    different e5 *roles* (root = e5 `query:`, candidate/ancestors = e5 `passage:`), so μ(X|root)≠μ(root|X).
  * **absent / dropped lineage = OFF-MANIFOLD NOISE** (random unit vector matched to the unit-normed e5
    magnitude) + its `gen_emb` tag — never a learned token or a zero. Same noise is the dropout
    regulariser (replace present ancestors p≈0.2 each, whole lineage p≈0.1). No dropout at inference;
    a per-node seed fixes the (rare) empty-lineage noise so the dense map is deterministic.
  * **keep the explicit anchor(root) token** — lineage-only can't score a root that isn't an ancestor
    (Music vs Physics, the gate-leak), which is exactly the case we care about.
"""
from __future__ import annotations

import hashlib
import math
import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
GRAPH = os.environ.get("UW_MU_GRAPH", os.path.join(REPO, "data", "benchmark", "10k", "category_parent.tsv"))
E5_MODEL = "intfloat/e5-small-v2"
E5_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"

# operator codebook (the relation axis). OPERATORS = pure RELATIONS (what the edge is). SOURCE lives on other axes:
# corpus_emb (enwiki/simplewiki/pearltrees/mindmap) = which knowledge base; judge_emb (graph/haiku/…) = how labeled.
#   SYM = symmetric relatedness · HIER = category-hierarchy (subcategory/subtopic/super_category; was "WIKI") ·
#   ELEM = element_of (membership). Index 2 = DEPRECATED (was "LLM" — but the LLM is a *judge* on judge_emb, not a
#   relation; the prompt gives the relation. The --llm fixture now routes to ELEM+haiku.) Kept only so n_ops stays 4
#   (op_emb loads by shape); no relation maps to it.
OPS = {"SYM": 0, "HIER": 1, "_DEPRECATED_LLM": 2, "ELEM": 3, "LINEAGE": 4, "LINEAGE_RANK": 5}   # 2 kept for parity; LINEAGE = mindmap graded-MSE (magnitude), LINEAGE_RANK = candidate-softmax CE (order) — see DESIGN_mindmap_lineage.md §3c
# ELEM = element-of (page-in-category / collection-membership): directional like WIKI (μ(page|category)
# high, reverse low) but graded like SYM. Its own operator token + readout row on the shared trunk, so
# page-membership trains as a DISTINCT relation instead of being conflated into SYM (see
# DESIGN_calibrated_judges.md §7 and REPORT_train_consolidation.md — the empirical motivation).

# PROVENANCE codebooks (PART B) — the judge axis, generalized into "where did this label come from".
# A single provenance TOKEN carries a FACTORED embedding corpus_emb[corpus] + judge_emb[judge], so the
# corpus⊗judge product is representable while it presents to the set as one input. The token is MASKABLE
# (off-manifold noise, exactly like an absent ancestor slot); masking ⇒ provenance-AGNOSTIC μ, which is
# the DEFAULT inference path (marginalize over sources). Reserved entries (enwiki) are for later corpora.
CORPORA = {"simplewiki": 0, "enwiki": 1, "pearltrees": 2, "mindmap": 3}
JUDGES = {"haiku": 0, "graph": 1, "human": 2, "sonnet": 3, "opus": 4, "gemini": 5, "gpt-5.5-low": 6, "blend": 7, "dir-blend": 8, "gpt-5.6-luna": 9, "kalman-fused": 10}    # learned judge_emb; each = own calibration row. "blend" = SYM DUAL judge (2 inputs, emit_blend_judge.py); "dir-blend" = 3-ESTIMATOR cross-judge DIRECTION superposition (graph-discrim ⊕ LLM-element ⊕ LLM-subcat, emit_direction_blend.py). "gpt-5.6-luna" (B2 step 3) and "kalman-fused" (the amortized-filter head, DESIGN_amortized_fusion_heads) onboard under NameFunctionCond at r=0 = pure name prior. Checkpoints saved after these rows expect judge_emb num_embeddings >= 11; older checkpoints load fine (new rows are zero-init/name-prior, unreferenced). Rationale in DESIGN_calibrated_judges.md. Post-migration (NameFunctionCond) every judge also needs a card in judge_cards.py.
                                         # (SYM/LLM); graph = a Wikipedia edge / non-edge (WIKI, free μ=0 SYM
                                         # negatives); human = a hand-curated edge (mindmap/pearltrees);
                                         # sonnet/opus = stronger-model judgments (escalated tie-breaks, §14)
# NODE-TYPE (DESIGN_calibrated_judges.md §7): a factored per-ENDPOINT token saying WHAT each node is — a
# `category`/`mindmap_node` is an internal node that can have children; a `page` is a LEAF; a
# `pearltrees_collection` is a curated container. Orthogonal to the OPERATOR (which says what the RELATION
# is). e5 still encodes the title text; the node-type token only adds the structural role the title can't
# convey. category=0 is the implicit default; the embedding is zero-initialised so it is a no-op at warm
# start and the type signal is learned during fine-tuning.
NODETYPE = {"category": 0, "page": 1, "mindmap_node": 2, "pearltrees_collection": 3}
# ACCOUNT (DESIGN_provenance_and_representation.md): a SECOND maskable provenance axis — which Pearltrees
# account a label came from — factored into the SAME provenance token as corpus⊗judge (so it is masked /
# marginalised out together with them). Small + closed (two accounts), so a learned token is right. It is a
# no-op until items actually carry an account id (position 7); zero-initialised so warm-starts are safe and
# the signal is learned only once BOTH accounts are harvested (single-account data is collinear with corpus).
ACCOUNTS = {"s243a": 0, "s243a_groups": 1}


# --------------------------------------------------------------------------------------------------
# graph: directed parent map (child -> parents) for min-hop ancestry; undirected degree for hubs
# --------------------------------------------------------------------------------------------------
def load_dag(path=GRAPH):
    parents, children, deg = {}, {}, {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("child"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            c, par = p[0], p[1]
            parents.setdefault(c, set()).add(par)
            children.setdefault(par, set()).add(c)
            parents.setdefault(par, parents.get(par, set()))
            children.setdefault(c, children.get(c, set()))
            deg[c] = deg.get(c, 0) + 1
            deg[par] = deg.get(par, 0) + 1
    return parents, children, deg


def all_names(parents, children):
    return sorted(set(parents) | set(children))


def _seed_of(name, salt=0):
    h = hashlib.blake2b(f"{salt}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


# --------------------------------------------------------------------------------------------------
# ancestor sampling — direct parents always; gen-2+ via hub-down-weighted walk (1/deg^β), min-hop tag
# --------------------------------------------------------------------------------------------------
def sample_ancestors(node, parents, deg, k=1, beta=1.0, stop=0.33, depth_cap=5, max_anc=8,
                     rng=None):
    """Return [(ancestor_name, min_hop_d)] for `node`, tagged by min-hop distance d∈[1,k].

    gen-1 = ALL direct parents (always included). gen-2+ (only if k≥2): a hub-down-weighted random walk
    UP (step to a parent with prob ∝ 1/deg^β, per-step stop, depth ≤ depth_cap), so the walk avoids the
    generic apex hubs (Main_topic_classifications, …) that carry no membership signal. Deduped, min-hop."""
    import random as _r
    rng = rng or _r.Random()
    min_hop = {}
    for p in parents.get(node, ()):                       # gen-1: always all parents
        min_hop[p] = 1
    if k >= 2:
        frontier = list(parents.get(node, ()))
        d = 1
        while frontier and d < k and d < depth_cap:
            nxt = []
            for x in frontier:
                ps = list(parents.get(x, ()))
                if not ps or rng.random() < stop:
                    continue
                w = [1.0 / (deg.get(p, 1) ** beta) for p in ps]
                tot = sum(w) or 1.0
                r, acc, chosen = rng.random() * tot, 0.0, ps[-1]
                for p, wi in zip(ps, w):
                    acc += wi
                    if r <= acc:
                        chosen = p
                        break
                if chosen not in min_hop:
                    min_hop[chosen] = d + 1
                nxt.append(chosen)
            frontier = nxt
            d += 1
    anc = sorted(min_hop.items(), key=lambda kv: (kv[1], kv[0]))
    if len(anc) > max_anc:                                # keep the closest (smallest min-hop) ancestors
        anc = anc[:max_anc]
    return anc


# --------------------------------------------------------------------------------------------------
# e5 embedding cache (frozen) — query: for the root/anchor, passage: for candidate + ancestors
# --------------------------------------------------------------------------------------------------
def build_e5_tables(
    names,
    cache_path=None,
    model_name=E5_MODEL,
    batch_size=512,
    device=None,
    texts=None,
    model_revision=None,
):
    """Return (query_tbl, passage_tbl, idx) — two [N,384] unit-normed frozen e5 tables. Cached to disk
    (regenerable, git-ignored). `query:`/`passage:` are e5's asymmetric prefixes — the directional
    motivation for choosing e5 (the root is the query, the candidate/ancestors are passages).

    `texts` (optional {name: text}) overrides the embedded string for a name — used for fused nodes whose
    KEY (e.g. `mm:cybernetics`) is not its text; they embed their title/embed_text instead of the key.

    ``model_revision`` is an immutable Hub commit when reproducibility is decision-bearing. Cache reuse
    requires the same model ID and revision; legacy or mismatched caches are regenerated rather than
    silently crossing model versions.
    """
    idx = {n: i for i, n in enumerate(names)}
    texts = texts or {}
    human = [texts.get(n, n.replace("_", " ")) for n in names]
    if cache_path and os.path.exists(cache_path):
        d = torch.load(cache_path, weights_only=False)
        if (
            d.get("names") == list(names)
            and d.get("human") == human
            and d.get("model_name") == model_name
            and d.get("model_revision") == model_revision
        ):
            return d["query"], d["passage"], idx
    from sentence_transformers import SentenceTransformer
    model_kwargs = {"device": device}
    if model_revision is not None:
        model_kwargs["revision"] = model_revision
    model = SentenceTransformer(model_name, **model_kwargs)
    q = model.encode(["query: " + h for h in human], batch_size=batch_size,
                     convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    p = model.encode(["passage: " + h for h in human], batch_size=batch_size,
                     convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    qt, pt = torch.tensor(q, dtype=torch.float32), torch.tensor(p, dtype=torch.float32)
    if cache_path:
        torch.save(
            {
                "names": list(names),
                "human": human,
                "model_name": model_name,
                "model_revision": model_revision,
                "query": qt,
                "passage": pt,
            },
            cache_path,
        )
    return qt, pt, idx


def unit_noise(n, d, generator=None):
    v = torch.randn(n, d, generator=generator)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


# --------------------------------------------------------------------------------------------------
# tokenizer: turn (node, root, op) examples into padded token tensors for the model
# --------------------------------------------------------------------------------------------------
class Tokenizer:
    """Builds the token set per example and pads a batch. Holds the frozen e5 tables + the DAG."""

    def __init__(self, query_tbl, passage_tbl, idx, parents, deg, k=1, beta=1.0, max_anc=8,
                 struct_tbl=None, struct_mode="dist", struct_cap=8, deg_scale=5.0,
                 root_lineage=False, root_lineage_depth=5):
        self.q, self.p, self.idx = query_tbl, passage_tbl, idx
        self.parents, self.deg = parents, deg
        # root_lineage: ALSO emit the anchor/root's ancestors as `anc` tokens (candidate-lineage-
        # conditioned μ — DESIGN filing §7). At filing the node is a lineage-less bookmark and the
        # useful known structure is the candidate FOLDER's principal path, which the default (node-
        # only) ancestor sampling never supplies. Reuses the `anc` role → no new model params, no
        # forward change; the model must be trained with this on to use it. Default off (backward
        # compatible with every existing caller).
        self.root_lineage = root_lineage
        self.root_lineage_depth = root_lineage_depth
        self.k, self.beta, self.max_anc = k, beta, max_anc
        self.d = query_tbl.shape[1]
        # DUAL-JUDGE (step 3): {name: struct-emb vector} for the O(1) graph-judge proxy `3/(1+‖Δ‖)`. When set,
        # build() emits a per-example `struct_feat` VECTOR [K] of graph signals. None ⇒ omitted.
        self.struct = struct_tbl
        # struct_mode selects what build() emits (see DESIGN_sym_dual_judge.md "Confidence architecture"):
        #   "dist"      ⇒ K=1: [ 3/(1+‖Δ‖) ]                                   distance predictor only
        #   "dir"       ⇒ K=3: [ dist, 3/(1+up_hops(a→b)), 3/(1+up_hops(b→a)) ] free-weight predictors
        #   "precision" ⇒ K=3: [ dist, mem, region ] — the LOCKED design:
        #       mem    = (3/(1+up_hops(a→b)) + 3/(1+up_hops(b→a)))/2   membership signal (data-rich)
        #       region = min(deg_a,deg_b)/(min(deg_a,deg_b)+deg_scale) ∈[0,1) — the PER-REGION c_mem factor
        #                (#3356 §3 source (c) measurement uncertainty: sparse-data ⇒ noisier μ ⇒ lower confidence)
        # up_hops = directed DAG ancestry (graph proxy for subcategory membership, not the model's μ ⇒ no
        # feedback loop; cheap local parent-climb). n_struct on the model must match K (1 for "dist", else 3).
        self.struct_mode, self.struct_cap, self.deg_scale = struct_mode, struct_cap, deg_scale

    def _up_hops(self, x, y, cap):
        """min hops climbing PARENT edges from x up to y (y an ancestor of x); None if not within cap."""
        if x == y:
            return 0
        frontier, seen = {x}, {x}
        for h in range(1, cap + 1):
            nxt = set()
            for cur in frontier:
                ps = self.parents.get(cur)
                if not ps:
                    continue
                for pp in (ps if isinstance(ps, (set, list, tuple)) else [ps]):
                    if pp == y:
                        return h
                    if pp not in seen:
                        seen.add(pp); nxt.add(pp)
            if not nxt:
                break
            frontier = nxt
        return None

    def _principal_path(self, node, cap):
        """Deterministic parent-climb: [(ancestor, depth)] up the FIRST-parent chain, depth 1..cap.
        The candidate folder's materialized path (§7) — the full lineage, not just gen-1 parents."""
        out, cur, seen = [], node, {node}
        for d in range(1, cap + 1):
            ps = self.parents.get(cur)
            if not ps:
                break
            nxt = next(iter(ps)) if isinstance(ps, (set, frozenset)) else ps[0]
            if nxt in seen:
                break
            seen.add(nxt)
            out.append((nxt, d))
            cur = nxt
        return out

    def _anc_for(self, node, train, rng):
        import random as _r
        if rng is None:
            rng = _r.Random(_seed_of(node, salt=7))        # inference: deterministic per node
        return sample_ancestors(node, self.parents, self.deg, k=self.k, beta=self.beta,
                                max_anc=self.max_anc, rng=rng)

    def build(self, items, train=False, rng=None, p_drop_anc=0.2, p_drop_lineage=0.1,
              p_mask_prov=0.5):
        """items: list of (node, root, op_id) OR (node, root, op_id, corpus_id, judge_id). Returns a dict
        of padded tensors for MuAttention.forward.

        Every example also gets ONE provenance token (PART B). With a 3-tuple the provenance is MASKED
        (off-manifold noise ⇒ provenance-agnostic μ — the default inference path). With a 5-tuple the
        token carries (corpus_id, judge_id); during training it is still masked with prob `p_mask_prov`
        so the model learns BOTH provenance-conditioned and provenance-agnostic μ."""
        B = len(items)
        rows = []          # per-example list of token dicts
        ntypes, rtypes, has_nt = [], [], []   # node-type of node/root per example; off unless item carries it
        for it in items:
            node, root, op = it[0], it[1], it[2]
            corpus_id, judge_id = (it[3], it[4]) if len(it) >= 5 else (None, None)
            account_id = it[7] if len(it) >= 8 else None             # optional 2nd provenance axis (account)
            if len(it) >= 7:
                ntypes.append(it[5]); rtypes.append(it[6]); has_nt.append(True)
            else:                               # no node-type tags ⇒ leave nodetype_of=-1 (emb not applied)
                ntypes.append(0); rtypes.append(0); has_nt.append(False)
            toks = []
            toks.append(("op", None, op, 0))                              # operator token
            toks.append(("anchor", root, None, 0))                       # anchor(root) — e5 query:
            toks.append(("node", node, None, 0))                         # node(X)@gen0 — e5 passage:
            anc = self._anc_for(node, train, rng)
            drop_lineage = train and rng is not None and rng.random() < p_drop_lineage
            if anc and not drop_lineage:
                for (a, d) in anc:
                    if train and rng is not None and rng.random() < p_drop_anc:
                        toks.append(("noise", node, None, d))            # present ancestor → noise
                    else:
                        toks.append(("anc", a, None, d))
            else:
                toks.append(("noise", node, None, 1))                    # absent lineage → one noise@gen1
            if self.root_lineage:                                        # candidate-folder lineage (§7)
                rpath = self._principal_path(root, self.root_lineage_depth)
                if rpath and not (train and rng is not None and rng.random() < p_drop_lineage):
                    for (a, d) in rpath:
                        if a in self.idx and not (train and rng is not None and rng.random() < p_drop_anc):
                            toks.append(("anc", a, None, d))
            # provenance token — masked (agnostic) for 3-tuples, or for tagged items with prob p_mask_prov
            masked = corpus_id is None or (train and rng is not None and rng.random() < p_mask_prov)
            if masked:
                toks.append(("prov_mask", node, None, 0))                # off-manifold noise ⇒ agnostic
            else:
                toks.append(("prov", None, (corpus_id, judge_id, account_id), 0))  # corpus+judge(+account)
            rows.append(toks)

        T = max(len(r) for r in rows)
        content = torch.zeros(B, T, self.d)
        gen_id = torch.full((B, T), -1, dtype=torch.long)
        is_anchor = torch.zeros(B, T, dtype=torch.bool)
        op_pos = torch.full((B, T), -1, dtype=torch.long)
        is_prov = torch.zeros(B, T, dtype=torch.bool)                     # the provenance slot (any state)
        corpus_of = torch.full((B, T), -1, dtype=torch.long)
        judge_of = torch.full((B, T), -1, dtype=torch.long)
        account_of = torch.full((B, T), -1, dtype=torch.long)             # 2nd provenance axis (account)
        nodetype_of = torch.full((B, T), -1, dtype=torch.long)            # per-endpoint structural role
        prefix_of = torch.full((B, T), -1, dtype=torch.long)              # e5 prefix regime (0 query:/1 passage:/2 none)
        pad = torch.ones(B, T, dtype=torch.bool)                          # True = pad (ignored)
        op_of = torch.tensor([it[2] for it in items], dtype=torch.long)

        for bi, toks in enumerate(rows):
            for ti, (kind, name, op, d) in enumerate(toks):
                pad[bi, ti] = False
                if kind == "op":
                    op_pos[bi, ti] = op
                elif kind == "anchor":
                    content[bi, ti] = self.q[self.idx[name]]
                    is_anchor[bi, ti] = True
                    prefix_of[bi, ti] = 0                                 # e5 query: regime
                    if has_nt[bi]:
                        nodetype_of[bi, ti] = rtypes[bi]                  # the root's type
                elif kind == "node":
                    content[bi, ti] = self.p[self.idx[name]]
                    gen_id[bi, ti] = 0
                    prefix_of[bi, ti] = 1                                 # e5 passage: regime
                    if has_nt[bi]:
                        nodetype_of[bi, ti] = ntypes[bi]                  # the candidate node's type
                elif kind == "anc":
                    content[bi, ti] = self.p[self.idx[name]]
                    gen_id[bi, ti] = d
                    prefix_of[bi, ti] = 1                                 # e5 passage: regime
                    if has_nt[bi]:
                        nodetype_of[bi, ti] = NODETYPE["category"]        # ancestors are categories
                elif kind == "prov":
                    is_prov[bi, ti] = True                                # content stays 0; forward adds
                    corpus_of[bi, ti], judge_of[bi, ti], _acc = op        # corpus_emb + judge_emb + prov_tag
                    if _acc is not None:
                        account_of[bi, ti] = _acc                         # + account_emb (if item carries it)
                elif kind in ("noise", "prov_mask"):
                    if train:                                 # fast path: fresh noise, no per-seed RNG
                        v = torch.randn(self.d)
                    else:                                     # inference: per-node seed ⇒ deterministic map
                        salt = (200 if kind == "prov_mask" else 100) + d
                        g = torch.Generator().manual_seed(_seed_of(name, salt=salt))
                        v = torch.randn(self.d, generator=g)
                    content[bi, ti] = v / v.norm().clamp_min(1e-9)
                    if kind == "prov_mask":
                        is_prov[bi, ti] = True                            # masked provenance slot (agnostic)
                    else:
                        gen_id[bi, ti] = d
        out = {"content": content, "gen_id": gen_id, "is_anchor": is_anchor, "op_pos": op_pos,
               "is_prov": is_prov, "corpus_of": corpus_of, "judge_of": judge_of,
               "account_of": account_of, "nodetype_of": nodetype_of, "prefix_of": prefix_of,
               "op_of": op_of, "pad": pad}
        if self.struct is not None:                          # DUAL-JUDGE: per-pair graph signal vector [B, K]
            mode = self.struct_mode
            K = {"dist": 1, "membership": 2}.get(mode, 3)    # membership: [dist, region] (mem from model kwargs)
            sf = torch.zeros(B, K)
            for bi, it in enumerate(items):
                a, b = it[0], it[1]                          # (node, root)
                va, vb = self.struct.get(a), self.struct.get(b)
                sf[bi, 0] = float(3.0 / (1.0 + (va - vb).norm())) if (va is not None and vb is not None) else 0.0
                if mode == "membership":                     # [dist, region]; memberships arrive as detached kwargs
                    mind = min(self.deg.get(a, 0), self.deg.get(b, 0))
                    sf[bi, 1] = mind / (mind + self.deg_scale)   # per-region data-density factor ∈[0,1)
                elif mode != "dist":
                    fh = self._up_hops(a, b, self.struct_cap)   # forward: b is an ancestor of a (a subcat_of b)
                    bh = self._up_hops(b, a, self.struct_cap)   # backward: a is an ancestor of b
                    fwd = 3.0 / (1.0 + fh) if fh is not None else 0.0
                    bwd = 3.0 / (1.0 + bh) if bh is not None else 0.0
                    if mode == "dir":                        # free-weight predictors [dist, fwd, bwd]
                        sf[bi, 1] = fwd; sf[bi, 2] = bwd
                    else:                                    # "precision": [dist, mem, region]
                        sf[bi, 1] = (fwd + bwd) / 2.0        # membership signal (up_hops proxy)
                        mind = min(self.deg.get(a, 0), self.deg.get(b, 0))
                        sf[bi, 2] = mind / (mind + self.deg_scale)   # per-region c_mem factor ∈[0,1)
            out["struct_feat"] = sf
        return out


# --------------------------------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------------------------------
class NameFunctionCond(nn.Module):
    """Identity conditioning as a FUNCTION of a frozen e5 name-card embedding — cond_j = W·e_j + r_j
    (REPORT_channel_campaign.md §5-6; DESIGN_amortized_fusion_heads.md). The anchored-basis idiom
    (anchored_basis.py) applied to provenance: e_j (the judge-card embedding, judge_cards.py) is FROZEN —
    the pinned interpretable part; W is a learned translation amplifying the calibration-relevant axes of
    name space; r_j holds only what the card doesn't say (regularize ‖r‖ toward 0 so the name prior stays
    the default — resid_penalty(), the anchor-KL's anti-drift role here). A NEW judge onboards at r=0 =
    pure name prior, so family-graded transfer falls out of the name geometry (luna starts at 0.97 of
    gpt-5.5's conditioning instead of a zero row that barely learns — the probe's zero-init finding).
    Migration from an indexed judge_emb is behavior-preserving: migrate_judge_names.py fits W by ridge
    least squares to the old rows and sets r_j = judge_emb[j] − W·e_j (exact reproduction at init).
    Same mechanism applies verbatim to OPS/CORPORA when their turn comes."""

    def __init__(self, name_e5, d_model):
        super().__init__()
        self.register_buffer("name_e5", name_e5.clone())     # FROZEN cards (like AnchoredBasis values)
        self.W = nn.Linear(name_e5.shape[1], d_model, bias=False)
        self.resid = nn.Embedding(name_e5.shape[0], d_model)
        nn.init.zeros_(self.resid.weight)                    # default = pure name prior

    def forward(self, idx):
        return self.W(self.name_e5[idx]) + self.resid(idx)

    def table(self):
        """The full conditioning matrix [n, d_model] — for consumers that need every row at once
        (the blended-operator path `op_weights @ table`, DESIGN_inferred_operator_superposition)."""
        return self.W(self.name_e5) + self.resid.weight

    def resid_penalty(self):
        return self.resid.weight.pow(2).sum(-1).mean()


class MuAttention(nn.Module):
    def __init__(self, d_model=384, n_ops=len(OPS), n_heads=4, n_layers=1, max_gen=5,
                 dim_ff=None, dropout=0.0, n_corpus=len(CORPORA), n_judge=len(JUDGES),
                 n_nodetype=len(NODETYPE), n_account=len(ACCOUNTS), struct_blend="inside", n_struct=1,
                 c_dist=1.0, c_mem_ceiling=1.0, c_subcat=1.0, c_elem=1.0, judge_name_e5=None,
                 op_name_e5=None, corpus_name_e5=None):
        # NB c_subcat/c_elem default to a NEUTRAL 1.0 here — the measured +0.72/+0.82 are LEAKAGE-INFLATED
        # (training-pair measurement, PR #3488 review) and are supplied ONLY via the CLI on the superseded
        # `membership` ablation path (see DESIGN_sym_estimation_integration.md), not baked into the module.
        super().__init__()
        self.d = d_model
        self.struct_blend = struct_blend            # DUAL-JUDGE combine: "inside"|"outside"|"precision"|"membership"
        self.op_emb = nn.Embedding(n_ops, d_model)
        # NODE-TYPE: per-endpoint structural-role token (category/page/mindmap/pearltrees). Zero-init so it
        # is a no-op at warm start (category=0 default) and the type signal is learned during fine-tuning.
        self.nodetype_emb = nn.Embedding(n_nodetype, d_model)
        # e5 PREFIX REGIME token (0 query: / 1 passage: / 2 none): lets the model know whether an input embedding
        # carries e5's role prefix, so it can handle mixed/ablated regimes. Zero-init ⇒ no-op until used (direction
        # survives with OR without the prefix — eval_prefix_ablation.py — so this is safe to add).
        self.prefix_emb = nn.Embedding(3, d_model)
        self.gen_emb = nn.Embedding(max_gen + 1, d_model)                 # gen 0..max_gen
        self.anchor_tag = nn.Parameter(torch.randn(d_model) * 0.02)
        # PROVENANCE (PART B): factored corpus_emb + judge_emb on a single maskable token. `prov_tag`
        # marks the slot itself (present in BOTH the tagged and the masked/agnostic state), so the model
        # can locate "the provenance input" regardless of whether its source is revealed.
        self.corpus_emb = nn.Embedding(n_corpus, d_model)
        self.judge_emb = nn.Embedding(n_judge, d_model)
        # NAME-FUNCTION judge conditioning (post-migration): when a card table is supplied, the judge
        # condition is NameFunctionCond (W·e5(card) + residual) and judge_emb is BYPASSED in forward —
        # kept in the module so pre-migration checkpoints load and the migration can read its rows.
        if judge_name_e5 is not None:
            assert judge_name_e5.shape[0] == n_judge, \
                f"judge card table has {judge_name_e5.shape[0]} rows, n_judge={n_judge}"
            self.judge_name = NameFunctionCond(judge_name_e5, d_model)
        else:
            self.judge_name = None
        # same mechanism for OPS/CORPORA (§6.7): when a card table is supplied the indexed embedding is
        # bypassed in forward (kept for pre-migration checkpoints). The per-operator READOUT stays indexed.
        if op_name_e5 is not None:
            assert op_name_e5.shape[0] == n_ops
            self.op_name = NameFunctionCond(op_name_e5, d_model)
        else:
            self.op_name = None
        if corpus_name_e5 is not None:
            assert corpus_name_e5.shape[0] == n_corpus
            self.corpus_name = NameFunctionCond(corpus_name_e5, d_model)
        else:
            self.corpus_name = None
        # ACCOUNT: a 2nd factored provenance axis on the SAME maskable token. Zero-init ⇒ a no-op at warm
        # start (and whenever items carry no account), so the signal is learned only once both accounts exist.
        self.account_emb = nn.Embedding(n_account, d_model)
        self.prov_tag = nn.Parameter(torch.randn(d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=dim_ff or 2 * d_model,
                                           dropout=dropout, batch_first=True, activation="gelu",
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.readout_w = nn.Parameter(torch.randn(n_ops, d_model) * (1.0 / math.sqrt(d_model)))
        self.readout_b = nn.Parameter(torch.zeros(n_ops))
        # DUAL-JUDGE (SYM = e5 ⊕ graph, DESIGN_sym_dual_judge.md step 3): a learned scale on the O(1)
        # structural feature `3/(1+‖Δ struct-emb‖)` (the cheap graph-judge proxy). Added to the SYM logit
        # ONLY (gated by op). Zero-init ⇒ exact warm-start no-op; learns the graph half during SYM training.
        self.sym_struct_w = nn.Parameter(torch.zeros(n_struct))  # INSIDE: logit += Σ wₖ·struct_featₖ (learned per-channel)
        # OUTSIDE mode (user's point): blend two BOUNDED judges — μ = μ_e5 + λ·(μ_graph − μ_e5), where the e5 μ
        # passes through untouched and only the unbounded graph term gets its own squash μ_graph = σ(Σ gₖ·featₖ + h).
        # λ zero-init ⇒ pure-e5 ⇒ exact warm-start no-op; g,h unused at init (λ=0 gates them out).
        self.struct_lambda = nn.Parameter(torch.zeros(1))
        self.struct_g = nn.Parameter(torch.zeros(n_struct))
        self.struct_h = nn.Parameter(torch.zeros(1))
        # PRECISION mode (LOCKED design, DESIGN_sym_dual_judge.md "Confidence architecture"):
        #   μ_graph = σ(prec_g·pw + prec_h),  pw = (c_mem·mem + c_dist·dist)/(c_mem + c_dist)
        #   c_mem   = c_mem_ceiling · region   (per-pair; ceiling = 1/error_converged, region from data density)
        # c_dist / c_mem_ceiling are MEASURED constants (LLM-agreement / converged membership error), NOT learned
        # ⇒ registered buffers set from config. prec_g/prec_h just calibrate pw→[0,1]. The e5↔graph blend reuses
        # struct_lambda (zero-init ⇒ warm-start no-op; learned ~0.5 as a COMPLEMENTARY superposition, not gated).
        self.register_buffer("c_dist", torch.tensor(float(c_dist)))
        self.register_buffer("c_mem_ceiling", torch.tensor(float(c_mem_ceiling)))
        # "membership" mode (ELEM follow-up): the graph judge fuses THREE structural signals — distance (siblings/
        # lateral) + subcategory membership + element membership — each weighted by its MEASURED reliability
        # (register_buffers, not learned). μ_HIER/μ_ELEM come in as detached kwargs (mem_subcat/mem_elem). Per
        # DESIGN §Confidence: c per OPERATOR (subcat vs elem separate), region (data density) modulates memberships.
        self.register_buffer("c_subcat", torch.tensor(float(c_subcat)))
        self.register_buffer("c_elem", torch.tensor(float(c_elem)))
        self.prec_g = nn.Parameter(torch.tensor(1.0))
        self.prec_h = nn.Parameter(torch.tensor(0.0))
        for emb in (self.op_emb, self.gen_emb, self.corpus_emb, self.judge_emb):
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.zeros_(self.nodetype_emb.weight)            # no-op at warm start; learned during fine-tune
        nn.init.zeros_(self.prefix_emb.weight)              # no-op until mixed prefix regimes are used
        nn.init.zeros_(self.account_emb.weight)             # no-op until items carry an account id

    def forward(self, content, gen_id, is_anchor, op_pos, op_of, pad,
                is_prov=None, corpus_of=None, judge_of=None, account_of=None, nodetype_of=None,
                prefix_of=None, op_weights=None, struct_feat=None, mem_subcat=None, mem_elem=None):
        # op_weights [B, n_ops] (optional): a BLENDED operator — a weight vector over operators that replaces
        # the one-hot op token AND the per-operator readout head (a random superposition for inferred rows;
        # a one-hot for tagged rows reproduces the indexed path exactly). See
        # DESIGN_inferred_operator_superposition.md (random operator embedding).
        emb = content.clone()
        gmask = (gen_id >= 0).unsqueeze(-1)
        emb = emb + self.gen_emb(gen_id.clamp(min=0)) * gmask
        emb = emb + self.anchor_tag * is_anchor.unsqueeze(-1)
        if nodetype_of is not None:                          # per-endpoint structural role
            emb = emb + self.nodetype_emb(nodetype_of.clamp(min=0)) * (nodetype_of >= 0).unsqueeze(-1)
        if prefix_of is not None:                            # e5 prefix regime (query:/passage:/none); zero-init no-op
            emb = emb + self.prefix_emb(prefix_of.clamp(min=0)) * (prefix_of >= 0).unsqueeze(-1)
        omask = (op_pos >= 0).unsqueeze(-1)
        opT = self.op_name.table() if self.op_name is not None else self.op_emb.weight
        if op_weights is None:
            emb = emb + opT[op_pos.clamp(min=0)] * omask                             # indexed (one-hot) op token
        else:
            emb = emb + (op_weights @ opT).unsqueeze(1) * omask                      # blended op token
        if is_prov is not None:
            emb = emb + self.prov_tag * is_prov.unsqueeze(-1)            # mark the provenance slot
            if corpus_of is not None:                                    # add factored source (if revealed)
                cmask = (corpus_of >= 0).unsqueeze(-1)
                ccond = (self.corpus_name(corpus_of.clamp(min=0)) if self.corpus_name is not None
                         else self.corpus_emb(corpus_of.clamp(min=0)))
                emb = emb + ccond * cmask
                jcond = (self.judge_name(judge_of.clamp(min=0)) if self.judge_name is not None
                         else self.judge_emb(judge_of.clamp(min=0)))
                emb = emb + jcond * cmask
            if account_of is not None:                                   # 2nd provenance axis (account)
                amask = (account_of >= 0).unsqueeze(-1)
                emb = emb + self.account_emb(account_of.clamp(min=0)) * amask
        h = self.encoder(emb, src_key_padding_mask=pad)
        # CLS-style readout: the operator token (always position 0, never padded) attends over the whole
        # set, giving a relation-conditioned summary. Cleaner than mean-pool, which dilutes the input
        # signal with the large learned op-embedding and collapses each operator's readout to a constant.
        pooled = h[:, 0, :]
        if op_weights is None:
            w, b = self.readout_w[op_of], self.readout_b[op_of]                      # per-operator head
        else:
            w, b = op_weights @ self.readout_w, op_weights @ self.readout_b          # blended head
        logit = (pooled * w).sum(-1) + b
        if struct_feat is not None:                          # DUAL-JUDGE: graph predictors [B,K] → SYM only
            sym_gate = op_weights[:, OPS["SYM"]] if op_weights is not None else (op_of == OPS["SYM"]).to(logit.dtype)
            if self.struct_blend == "precision":             # LOCKED: precision-weighted(mem,1/d) ⊕ e5, per-region c_mem
                dist, mem, region = struct_feat[:, 0], struct_feat[:, 1], struct_feat[:, 2]
                c_mem = self.c_mem_ceiling * region                             # per-pair membership confidence
                pw = (c_mem * mem + self.c_dist * dist) / (c_mem + self.c_dist + 1e-6)   # inverse-variance fusion
                mu_graph = torch.sigmoid(self.prec_g * pw + self.prec_h)        # graph judge ∈[0,1]
                mu_e5 = torch.sigmoid(logit)                                    # complementary e5 judge
                mu = mu_e5 + sym_gate * self.struct_lambda * (mu_graph - mu_e5)  # λ=0 ⇒ pure e5 (no-op); ~0.5 learned
                return mu.clamp(0.0, 1.0)
            if self.struct_blend == "membership":            # ELEM: fuse dist(siblings) + subcat + elem memberships
                # ⚠ SUPERSEDED ABLATION (PR #3488 review): this hand-set fusion still gates the memberships by the
                # graph-degree `region` proxy, which DESIGN_sym_estimation_integration.md argues is the wrong
                # data-limit proxy (confidence should be learned/calibrated via JointPosterior, not a per-item
                # weight). Kept only as an A/B control — NOT the recommended path; don't cite it as evidence for it.
                dist, region = struct_feat[:, 0], struct_feat[:, 1]            # struct_feat = [dist, region]
                ms = mem_subcat if mem_subcat is not None else torch.zeros_like(dist)   # detached μ_HIER (max fwd/bwd)
                me = mem_elem if mem_elem is not None else torch.zeros_like(dist)        # detached μ_ELEM (max fwd/bwd)
                cs, ce = region * self.c_subcat, region * self.c_elem          # per-region membership confidences
                num = self.c_dist * dist + cs * ms + ce * me                   # precision (inverse-variance) fusion
                pw = num / (self.c_dist + cs + ce + 1e-6)
                mu_graph = torch.sigmoid(self.prec_g * pw + self.prec_h)
                mu_e5 = torch.sigmoid(logit)
                mu = mu_e5 + sym_gate * self.struct_lambda * (mu_graph - mu_e5)  # λ=0 ⇒ pure e5 (no-op)
                return mu.clamp(0.0, 1.0)
            if self.struct_blend == "outside":               # blend two BOUNDED judges in μ-space (no outer sigmoid)
                mu_e5 = torch.sigmoid(logit)                                     # e5 judge — already ∈[0,1]
                mu_graph = torch.sigmoid((struct_feat * self.struct_g).sum(-1) + self.struct_h)  # squash the graph terms
                mu = mu_e5 + sym_gate * self.struct_lambda * (mu_graph - mu_e5)  # λ=0 ⇒ pure e5 (no-op)
                return mu.clamp(0.0, 1.0)
            logit = logit + sym_gate * (struct_feat * self.sym_struct_w).sum(-1)  # INSIDE: Σ wₖ·featₖ, one sigmoid
        return torch.sigmoid(logit)


@torch.no_grad()
def membership_readouts(model, tok, pairs, dev, batch=512):
    """Detached μ_HIER / μ_ELEM memberships for the 'membership' graph judge (DESIGN §ELEM follow-up).
    pairs = [(a, b), ...] (node, root). Returns (mem_subcat, mem_elem) tensors [N] on `dev`:
        mem_subcat = max(μ_HIER(a|b), μ_HIER(b|a)),   mem_elem = max(μ_ELEM(a|b), μ_ELEM(b|a))
    STOP-GRAD (features, not targets): SYM training must not back-prop into / corrupt the HIER/ELEM operators.
    Correct even in 'membership' blend mode: a non-SYM readout has sym_gate=0 ⇒ the struct branch returns μ_e5,
    i.e. the plain operator readout."""
    n_ops = model.op_emb.num_embeddings

    def score(op, ordered):
        ow = torch.zeros(1, n_ops, device=dev); ow[0, OPS[op]] = 1.0
        out = []
        for i in range(0, len(ordered), batch):
            ch = ordered[i:i + batch]
            bd = tok.build([(x, y, 0) for x, y in ch], train=False)
            bd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bd.items()}
            out.append(model(**bd, op_weights=ow.expand(len(ch), n_ops)))
        return torch.cat(out) if out else torch.zeros(0, device=dev)

    fwd = list(pairs); rev = [(b, a) for a, b in pairs]
    ms = torch.maximum(score("HIER", fwd), score("HIER", rev))
    me = torch.maximum(score("ELEM", fwd), score("ELEM", rev))
    return ms.detach(), me.detach()


if __name__ == "__main__":
    parents, children, deg = load_dag()
    names = all_names(parents, children)
    print(f"DAG: {len(names)} nodes")
    print("sample ancestors of Optics (k=1):", sample_ancestors("Optics", parents, deg, k=1))
    print("sample ancestors of Optics (k=3):", sample_ancestors("Optics", parents, deg, k=3,
          rng=__import__("random").Random(1)))
