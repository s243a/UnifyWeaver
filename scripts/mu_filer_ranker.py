#!/usr/bin/env python3
"""mu_filer_ranker.py — the mu_cosine μ-attention filing ranker, packaged as the "mu" routing alternative for
infer_pearltrees_federated.py (vs the rotational/weighted W-projection over the federated nomic model).

Scores a query (bookmark) against candidate folders with the tuned combiner from the mu_cosine work:
    score = 0.1 · norm(e5-cos)  +  0.9 · norm( max(μ-elem, μ-hier, μ-sym) )
i.e. e5 coarse relevance + a 0.9-weighted operator-OR over directional membership (the "coverage insurance" blend;
`eval_filing.py` / DESIGN_model_applications.md). Frozen e5 + a small permutation-invariant attention head, so it
generalises to arbitrary folder titles — no per-folder training.

e5 ROLES (must match the training/eval convention):
  * folder = ANCHOR/root  → e5 `query:`   (qtbl)
  * bookmark = NODE/X     → e5 `passage:`  (ptbl)
  * e5-cosine coarse term = bookmark-`query:` · folder-`passage:`
So a live query is encoded with BOTH prefixes.

mu_cosine lives in prototypes/mu_cosine (not a package) — added to sys.path here; imports are heavy
(torch + sentence_transformers) so only pay them when the "mu" routing method is actually selected.
"""
import os, sys
import numpy as np
import torch

_MU_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prototypes", "mu_cosine")
if _MU_DIR not in sys.path:
    sys.path.insert(0, _MU_DIR)
from mu_attention import build_e5_tables, Tokenizer, MuAttention, OPS, E5_MODEL   # noqa: E402


class MuRanker:
    """Loads a mu_cosine checkpoint + e5, precomputes candidate-folder embeddings, scores live queries."""

    def __init__(self, ckpt_path, folder_titles, folder_ids, device="cpu", cache_path=None,
                 mu_weight=0.9, e5_weight=0.1):
        self.dev = torch.device(device)
        self.mu_weight, self.e5_weight = mu_weight, e5_weight
        self.folder_ids = list(folder_ids)
        self.folder_keys = ["F:%d" % i for i in range(len(folder_titles))]
        ftext = {k: (folder_titles[i] or str(folder_ids[i])) for i, k in enumerate(self.folder_keys)}

        # candidate-folder e5 tables (both roles), reordered to folder_keys order
        fq, fp, fidx = build_e5_tables(self.folder_keys, cache_path=cache_path, texts=ftext, device=device)
        order = [fidx[k] for k in self.folder_keys]
        self.folder_q = fq[order]        # folder as anchor  → query:
        self.folder_p = fp[order]        # folder as passage → for the e5-cosine coarse term
        self.F = len(self.folder_keys)

        # persistent e5 encoder for live queries (matches build_e5_tables params exactly)
        from sentence_transformers import SentenceTransformer
        self._e5 = SentenceTransformer(E5_MODEL, device=device)

        # load the mu_cosine checkpoint (infer axis sizes from the state dict, non-strict — as eval_filing.py)
        ck = torch.load(ckpt_path, weights_only=False)
        sd = ck["state"]
        cfg = ck.get("cfg", {"d_model": self.folder_q.shape[1], "heads": 4, "layers": 3})
        sz = lambda k, d: sd[k].shape[0] if k in sd else d
        self.model = MuAttention(d_model=cfg["d_model"], n_heads=cfg["heads"], n_layers=cfg["layers"],
                                 n_ops=sz("op_emb.weight", len(OPS)), n_corpus=sz("corpus_emb.weight", 2),
                                 n_judge=sz("judge_emb.weight", 2), n_nodetype=sz("nodetype_emb.weight", 4)).to(self.dev)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not unexpected, "unexpected keys: %s" % unexpected
        self.model.eval()
        self.n_ops = self.model.op_emb.weight.shape[0]
        self.ow = {op: torch.zeros(1, self.n_ops).index_fill_(1, torch.tensor([OPS[op]]), 1.0)
                   for op in ("ELEM", "HIER", "SYM")}

    def _encode(self, text, prefix):
        v = self._e5.encode([prefix + ": " + text], convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)[0]
        return torch.tensor(v, dtype=torch.float32)

    @torch.no_grad()
    def _mu(self, tok, op, batch=512):
        items = [("Q", fk, 0) for fk in self.folder_keys]        # node=bookmark(Q), root=folder, op placeholder
        ow = self.ow[op]
        out = []
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            b = tok.build(chunk, train=False)
            b = {k: (v.to(self.dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            out += self.model(**b, op_weights=ow.expand(len(chunk), self.n_ops).to(self.dev)).cpu().tolist()
        return torch.tensor(out)

    @torch.no_grad()
    def score(self, query):
        """→ np.ndarray[F]: blended per-folder score, aligned with the folder_titles/ids order given at init."""
        qv, pv = self._encode(query, "query"), self._encode(query, "passage")
        q = torch.cat([self.folder_q, qv[None]]); p = torch.cat([self.folder_p, pv[None]])
        idx = {k: i for i, k in enumerate(self.folder_keys)}; idx["Q"] = self.F
        tok = Tokenizer(q, p, idx, parents={}, deg={})
        S_elem, S_hier, S_sym = self._mu(tok, "ELEM"), self._mu(tok, "HIER"), self._mu(tok, "SYM")
        S_max = torch.maximum(torch.maximum(S_elem, S_hier), S_sym)      # operator-OR
        C = qv @ self.folder_p.T                                         # e5-cos: bookmark query: · folder passage:
        nz = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-9)
        return (self.e5_weight * nz(C) + self.mu_weight * nz(S_max)).numpy()


if __name__ == "__main__":                                              # smoke test on toy folders
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--query", default="quantum entanglement in photons")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    titles = ["Physics", "Quantum mechanics", "Cooking recipes", "Machine learning", "Astronomy"]
    r = MuRanker(a.ckpt, titles, [str(i) for i in range(len(titles))], device=a.device, cache_path=None)
    s = r.score(a.query)
    order = np.argsort(-s)
    print("query:", a.query)
    for rank, i in enumerate(order, 1):
        print("  %d. %-20s %.3f" % (rank, titles[i], s[i]))
