#!/usr/bin/env python3
"""Label-data μ→relation distribution estimator — the foundation for inferred-operator assignment
(DESIGN_inferred_operator_superposition.md) and multi-source combination
(DESIGN_mu_sources_and_estimation.md).

From the TAGGED pairs (whose relation we know) we estimate, per μ SOURCE (frozen e5, the training model,
later an LLM), the generative `P(μ | relation)` — one smoothed histogram per relation — and Bayes-invert to
the posterior `P(relation | μ)`. Two uses:
  1. **Inferred-operator assignment**: for an untagged pair, measure μ and read the posterior over relations
     (→ operator), instead of v1's fixed-breadth heuristic.
  2. **Label-anomaly review** (the side-note rule): a tagged label keeps confidence 1.0 UNLESS its measured μ
     is OUTSIDE the expected band for that relation — then it is flagged for LLM/human review.

Sources are NOT independent (the model consumes e5) — combine with weighted product-of-experts, not a naive
product; estimate the e5↔model dependence before trusting a tight combined posterior (see the design doc).

This module is pure-Python + numpy (no torch); the static **e5** source runs now from an e5 cache.

    python3 mu_posterior.py --pairs /tmp/graded_pairs.tsv --e5-cache e5_tables_graded.pt
"""
import argparse
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[m] - xs[m].mean(), ys[m] - ys[m].mean()
    d = (np.linalg.norm(xs) * np.linalg.norm(ys))
    return float((xs @ ys) / d) if d else float("nan")


class MuPosterior:
    """Per-(source, relation) μ histograms; Bayes posterior P(relation | μ_1..μ_K); per-relation expected band."""

    def __init__(self, nbins=20, lo=0.0, hi=1.0, smoothing=1.0):
        self.nbins, self.lo, self.hi, self.smoothing = nbins, lo, hi, smoothing
        self.edges = np.linspace(lo, hi, nbins + 1)
        self.dens = {}                                    # (source, relation) -> density array (sums to 1)
        self.raw = defaultdict(list)                      # (source, relation) -> list of μ (for quantile bands)
        self.prior = {}                                   # relation -> P(relation)
        self.weights = {}                                 # source -> weight in the product-of-experts
        self.sources = []

    def _bin(self, mu):
        return int(np.clip(np.searchsorted(self.edges, mu, side="right") - 1, 0, self.nbins - 1))

    def fit_source(self, source, rel_mu, weight=1.0):
        """rel_mu: iterable of (relation, μ). Builds a smoothed density per relation for this source."""
        if source not in self.sources:
            self.sources.append(source)
        self.weights[source] = weight
        by_rel = defaultdict(list)
        for rel, mu in rel_mu:
            if mu == mu:                                  # drop NaN
                by_rel[rel].append(float(mu))
        for rel, mus in by_rel.items():
            h = np.histogram(mus, bins=self.edges)[0].astype(float) + self.smoothing
            self.dens[(source, rel)] = h / h.sum()
            self.raw[(source, rel)] = sorted(mus)
        # prior from the (first-source) relation frequencies
        if not self.prior:
            tot = sum(len(v) for v in by_rel.values())
            self.prior = {rel: len(v) / tot for rel, v in by_rel.items()}

    def relations(self):
        return sorted(self.prior)

    def posterior(self, mu_by_source, candidates=None):
        """mu_by_source: {source: μ}. Returns {relation: P(relation | μ's)} over `candidates` (default all)."""
        rels = candidates or self.relations()
        logp = {}
        for rel in rels:
            lp = math.log(self.prior.get(rel, 1e-9))
            for src, mu in mu_by_source.items():
                if mu != mu or (src, rel) not in self.dens:
                    continue
                lp += self.weights.get(src, 1.0) * math.log(self.dens[(src, rel)][self._bin(mu)] + 1e-12)
            logp[rel] = lp
        m = max(logp.values())
        exp = {r: math.exp(lp - m) for r, lp in logp.items()}
        z = sum(exp.values()) or 1.0
        return {r: v / z for r, v in exp.items()}

    def band(self, source, relation, q=0.05):
        """Expected μ band for a relation under a source = its [q, 1−q] quantiles on the tagged data."""
        xs = self.raw.get((source, relation))
        if not xs:
            return (self.lo, self.hi)
        return (float(np.quantile(xs, q)), float(np.quantile(xs, 1 - q)))

    def is_anomalous(self, source, relation, mu, q=0.05):
        """True if a TAGGED pair's measured μ is outside its relation's expected band ⇒ needs review."""
        if mu != mu:
            return False
        lo, hi = self.band(source, relation, q)
        return not (lo <= mu <= hi)

    def separability(self, source):
        """How well this source's μ separates the relations: pairwise band overlap summary + the mutual-info-ish
        spread of per-relation means. Low ⇒ a weak source ⇒ small weight."""
        means = {rel: float(np.mean(self.raw[(source, rel)])) for rel in self.relations() if self.raw.get((source, rel))}
        if len(means) < 2:
            return 0.0, means
        return float(np.std(list(means.values()))), means


class JointPosterior:
    """JOINT conditional P(relation | μ_vector) — a small discriminative head over the FULL readout vector,
    NOT a product of per-source marginals. Captures (a) source correlations (the joint fit down-weights
    redundant features automatically) and (b) the fwd×rev ASYMMETRY interaction a product of 1-D marginals
    cannot. `hidden=0` ⇒ multinomial logistic regression (the asymmetry `fwd−rev` is a linear feature combo,
    so LR already captures it); `hidden>0` ⇒ a 1-hidden-layer MLP for higher-order interactions.

    Inputs are standardised (z-scored); NaN readouts are imputed to the feature mean (0 after standardising)."""

    def __init__(self, relations, n_features, hidden=0, seed=0):
        import torch
        torch.manual_seed(seed)
        self.relations = list(relations)
        self.ri = {r: i for i, r in enumerate(self.relations)}
        C = len(self.relations)
        self.net = (torch.nn.Sequential(torch.nn.Linear(n_features, hidden), torch.nn.ReLU(),
                                        torch.nn.Linear(hidden, C)) if hidden > 0
                    else torch.nn.Linear(n_features, C))
        self.mean = self.std = None

    def _z(self, X):
        X = np.asarray(X, float)
        return np.nan_to_num((X - self.mean) / self.std, nan=0.0)

    def fit(self, X, rels, epochs=500, lr=0.05, weight_decay=2e-3):
        import torch
        X = np.asarray(X, float)
        self.mean, self.std = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
        Xt = torch.tensor(self._z(X), dtype=torch.float32)
        y = torch.tensor([self.ri[r] for r in rels])
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        lf = torch.nn.CrossEntropyLoss()
        for _ in range(epochs):
            opt.zero_grad(); lf(self.net(Xt), y).backward(); opt.step()
        return self

    def proba(self, X):
        import torch
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(self._z(X), dtype=torch.float32)), -1).numpy()


def _eval(proba, rels, ri, bins=10):
    """accuracy + cross-entropy (log-loss) + ECE (expected calibration error) of a [N,C] proba."""
    import numpy as _np
    y = _np.array([ri[r] for r in rels])
    pred, conf = proba.argmax(1), proba.max(1)
    acc = float((pred == y).mean())
    ll = float(-_np.log(_np.clip(proba[_np.arange(len(y)), y], 1e-9, 1.0)).mean())
    ece, edges = 0.0, _np.linspace(0, 1, bins + 1)        # |confidence − accuracy|, confidence-binned
    for b in range(bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.sum():
            ece += m.mean() * abs(conf[m].mean() - (pred[m] == y[m]).mean())
    return acc, ll, float(ece)


# ---------- consume the fitted posterior: a RANDOM operator embedding (superposition) -------------------
def sample_operator_weights(probs, alpha=20.0, rng=None):
    """Random CONVEX weights over the candidate operators ~ Dirichlet(alpha · probs). Mean ≈ probs; `alpha`
    is the concentration — LOW alpha ⇒ more spread ⇒ more operator noise, alpha→∞ ⇒ the deterministic mean.
    (The posterior's own entropy already spreads the mean; alpha adds sampling variance on top.)"""
    p = np.asarray(probs, float)
    p = p / max(p.sum(), 1e-12)
    rng = rng or np.random.default_rng()
    return rng.dirichlet(alpha * p + 1e-9)


def random_operator_embedding(probs, op_emb, alpha=20.0, out_of_set_noise=0.0, rng=None):
    """Generate a RANDOM operator embedding for an INFERRED row: a random superposition of the candidate
    operator embeddings, `w @ op_emb` with `w ~ Dirichlet(alpha · probs)`, plus an optional OUT-OF-SET noise
    term (the mass that the true operator is *none* of the candidates). Realises "operator = random
    superposition of the possibilities + noise" with the noise decomposition's two knobs:
      alpha           → (a) μ-residual / posterior spread + (c,d) measurement/churn variance
      out_of_set_noise→ (b) the true operator is outside the candidate set.
    Inputs: probs [K] over candidates; op_emb [K, d] the candidate operator embeddings. Returns [d].

    In training this is the OP-TOKEN OVERRIDE for inferred rows (tagged rows keep their fixed op_emb[op]);
    the matmul `w @ op_emb` is done in torch so gradients still flow to the operator embeddings (w is a
    detached random constant, like a dropout mask). This is the numpy reference + the testable core."""
    rng = rng or np.random.default_rng()
    w = sample_operator_weights(probs, alpha, rng)
    emb = np.asarray(op_emb, float)
    out = w @ emb
    if out_of_set_noise > 0:
        out = out + out_of_set_noise * rng.standard_normal(emb.shape[-1])
    return out


# ---------- static e5 source: μ_e5(node, root) = cos(query[root], passage[node]) -------------------------
def e5_mu_fn(cache_path):
    import torch
    d = torch.load(cache_path, weights_only=False)
    idx = {n: i for i, n in enumerate(d["names"])}
    q, p = d["query"].numpy(), d["passage"].numpy()       # unit-normed

    def mu(node, root):
        ni, ri = idx.get(node), idx.get(root)
        return float("nan") if ni is None or ri is None else float(p[ni] @ q[ri])
    return mu


def model_readout_fn(ckpt_path, e5_cache, device="cpu", bs=2048):
    """Returns readouts(pairs) → {source: np.array} for the FULL μ-readout vector: the symmetric SYM μ and,
    for the ASYMMETRIC operators (WIKI, ELEM), BOTH directions as separate features (wiki_fwd/wiki_rev,
    elem_fwd/elem_rev). The posterior conditions on each of these; conditioning on fwd+rev separately captures
    the directional asymmetry without hand-designing it (NB the operator readouts are circular — the model was
    trained on these relations — so the estimator should down-weight them by their correlation)."""
    import torch
    from mu_attention import MuAttention, Tokenizer, OPS, load_dag
    d = torch.load(e5_cache, weights_only=False)
    idx = {n: i for i, n in enumerate(d["names"])}
    parents, children, deg = load_dag()
    tok = Tokenizer(d["query"], d["passage"], idx, parents, deg)
    ck = torch.load(ckpt_path, weights_only=False); cfg = ck.get("cfg", {})
    model = MuAttention(d_model=d["query"].shape[1], n_heads=cfg.get("heads", 4), n_layers=cfg.get("layers", 3))
    sd, own = ck["state"], model.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
        elif k in own and own[k].dim() >= 1 and own[k].shape[0] > v.shape[0] and own[k].shape[1:] == v.shape[1:]:
            t = own[k].clone(); t[:v.shape[0]] = v; t[v.shape[0]:] = v[0]; own[k] = t
    model.load_state_dict(own); model.to(device); model.eval()

    @torch.no_grad()
    def _mu(pairs, op):
        out = []
        for i in range(0, len(pairs), bs):
            b = tok.build([(a, bb, OPS[op]) for a, bb in pairs[i:i + bs]], train=False)
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out.append(model(**b).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    def readouts(pairs):                                   # pairs: list of (node, root) both in idx
        rev = [(b, a) for a, b in pairs]
        return {"sym": _mu(pairs, "SYM"),
                "wiki_fwd": _mu(pairs, "HIER"), "wiki_rev": _mu(rev, "HIER"),
                "elem_fwd": _mu(pairs, "ELEM"), "elem_rev": _mu(rev, "ELEM")}
    readouts.idx = idx
    return readouts


def struct_dist_fn(struct_emb_path):
    """The DECORRELATED lateral/sibling source (DESIGN_sym_estimation_integration.md): `1/d` via the learned
    structural embedding, `3/(1+‖Δ‖)`. Trained separately on graph distance ⇒ not a re-reading of e5, and it
    covers the sibling axis the vertical membership operators miss. NaN if either endpoint is absent."""
    import torch
    se = torch.load(struct_emb_path, weights_only=False)
    sv = {n: v for n, v in zip(se["nodes"], se["emb"])}

    def dist(node, root):
        va, vb = sv.get(node), sv.get(root)
        return float("nan") if va is None or vb is None else float(3.0 / (1.0 + (va - vb).norm()))
    return dist


def aurc(conf, correct):
    """Area under the risk-coverage curve (Geifman & El-Yaniv). conf = per-item confidence, correct ∈ {0,1}.
    Sort by confidence desc; selective risk at coverage k = error rate among the k most-confident. Lower = the
    confidence signal routes errors to the low-confidence tail better."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf)
    err = 1.0 - correct[order]
    risk_at_k = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risk_at_k.mean())


def aurc_boot(conf, correct, B=500, seed=0):
    """AURC point estimate + percentile bootstrap 95% CI (AURC is noisy on small held-out sets)."""
    rng = np.random.default_rng(seed)
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    n = len(conf)
    vals = [aurc(conf[ix], correct[ix]) for ix in (rng.integers(0, n, n) for _ in range(B))]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return aurc(conf, correct), float(lo), float(hi)


def margin_conf(proba):
    """#3391 gate: top1 − top2 of the posterior (margin, not level)."""
    s = np.sort(np.asarray(proba, float), axis=1)
    return s[:, -1] - s[:, -2]


def load_pairs(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            c = ln.rstrip("\n").split("\t")               # node root mu op relation ... corpus judge conf
            if len(c) >= 5:
                conf = float(c[9]) if len(c) > 9 and c[9] else 1.0
                rows.append({"node": c[0], "root": c[1], "mu": float(c[2]), "op": c[3], "rel": c[4], "conf": conf})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="graded _pairs.tsv (uses the TAGGED rows, conf≥1.0)")
    ap.add_argument("--e5-cache", required=True)
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--q", type=float, default=0.05, help="anomaly band quantile")
    ap.add_argument("--model", default=None, help="a trained MuAttention checkpoint → adds the full μ-READOUT "
                    "VECTOR (sym + wiki_fwd/rev + elem_fwd/rev) and reports per-source separability + the "
                    "correlation matrix")
    ap.add_argument("--reject-outliers", action="store_true", help="drop tagged labels whose e5 μ is out of "
                    "their relation's band BEFORE fitting (the design's outlier rejection, all relation types)")
    ap.add_argument("--hidden", type=int, default=0, help="JOINT head hidden units (0 = logistic regression)")
    ap.add_argument("--held-frac", type=float, default=0.25)
    ap.add_argument("--struct-emb", default=None, help="structural_embedding.py .pt → adds the DECORRELATED "
                    "lateral/sibling source `1/d`; triggers the with-vs-without-1/d ablation (does it earn its keep?)")
    ap.add_argument("--split", choices=["node-disjoint", "random"], default="node-disjoint",
                    help="held-out split: node-disjoint (no node in both train+held — guards leakage, the #3488 "
                    "lesson) or random pairs.")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples for the AURC 95%% CI")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_pairs(args.pairs)
    tagged = [r for r in rows if r["conf"] >= 1.0]
    e5mu = e5_mu_fn(args.e5_cache)
    mu_e5 = {(r["node"], r["root"]): e5mu(r["node"], r["root"]) for r in tagged}

    # provisional e5 fit → bands → OUTLIER REJECTION (the side-note rule, applied to ALL relation types)
    prov = MuPosterior(nbins=args.nbins)
    prov.fit_source("e5", ((r["rel"], mu_e5[(r["node"], r["root"])]) for r in tagged))
    flag_idx = {i for i, r in enumerate(tagged)
                if prov.is_anomalous("e5", r["rel"], mu_e5[(r["node"], r["root"])], args.q)}
    flagged = [tagged[i] for i in flag_idx]
    fit_set = [r for i, r in enumerate(tagged) if i not in flag_idx] if args.reject_outliers else tagged
    print(f"tagged pairs: {len(tagged)} / {len(rows)} total; out-of-band {len(flagged)}"
          + (f" → REJECTED, fitting on {len(fit_set)}" if args.reject_outliers else " (kept; --reject-outliers to drop)"))
    # per-relation RATE, not count: the per-relation quantile band flags ~2q of EACH class by construction, so
    # the absolute count tracks class size — this is a review QUEUE, not a relative-noise diagnostic.
    rel_n = Counter(r["rel"] for r in tagged)
    rel_f = Counter(r["rel"] for r in flagged)
    print(f"  out-of-band by relation (rate; ~{2*args.q:.0%} expected per class by construction):")
    for rel in sorted(rel_n, key=lambda r: -rel_f[r] / max(rel_n[r], 1)):
        print(f"    {rel:14s} {rel_f[rel]:4d}/{rel_n[rel]:<4d} = {rel_f[rel]/max(rel_n[rel],1):.1%}")

    # build the SOURCE table: e5 (static) + the model μ-readout vector (sym, wiki_fwd/rev, elem_fwd/rev)
    post = MuPosterior(nbins=args.nbins)
    src_vals = {"e5": [mu_e5[(r["node"], r["root"])] for r in fit_set]}
    post.fit_source("e5", ((r["rel"], v) for r, v in zip(fit_set, src_vals["e5"])))
    if args.model:
        ro = model_readout_fn(args.model, args.e5_cache, device=args.device)
        pairs = [(r["node"], r["root"]) for r in fit_set]
        rdv = ro(pairs)                                    # {source: array aligned with fit_set}
        for src in ("sym", "wiki_fwd", "wiki_rev", "elem_fwd", "elem_rev"):
            src_vals[src] = list(rdv[src])
            post.fit_source(src, ((r["rel"], v) for r, v in zip(fit_set, rdv[src])))
    if args.struct_emb:                                    # the decorrelated lateral/sibling source `1/d`
        sdist = struct_dist_fn(args.struct_emb)
        src_vals["dist"] = [sdist(r["node"], r["root"]) for r in fit_set]
        # restrict to rows where 1/d is available so the with-vs-without ablation is on the SAME pairs
        keep = [i for i, v in enumerate(src_vals["dist"]) if v == v]   # not NaN
        if len(keep) < len(fit_set):
            print(f"struct-emb: {len(keep)}/{len(fit_set)} fit pairs have both nodes in the embedding "
                  f"→ ablation restricted to those")
            fit_set = [fit_set[i] for i in keep]
            src_vals = {s: [vals[i] for i in keep] for s, vals in src_vals.items()}
        post.fit_source("dist", ((r["rel"], v) for r, v in zip(fit_set, src_vals["dist"])))

    sources = list(src_vals)
    print(f"\nper-source separability (μ-mean spread across relations — higher = more discriminative):")
    for src in sources:
        sp, _ = post.separability(src)
        print(f"  {src:9s} {sp:.3f}")
    if len(sources) > 1:
        print(f"\ncorrelation matrix (|r| ⇒ redundancy; the operator readouts are CIRCULAR — trained on these):")
        print("            " + " ".join(f"{s[:8]:>8s}" for s in sources))
        for a in sources:
            print(f"  {a:9s} " + " ".join(f"{pearson(src_vals[a], src_vals[b]):+7.2f} " for b in sources))

    # JOINT conditional P(relation | μ_vector) vs the FACTORED product-of-marginals, on a held-out split
    if args.model and len(sources) > 1:
        rels = [r["rel"] for r in fit_set]
        relset = sorted(set(rels))
        X = np.array([[src_vals[s][i] for s in sources] for i in range(len(fit_set))], float)
        ri = {r: i for i, r in enumerate(relset)}
        # HELD-OUT SPLIT — node-disjoint by default (no node in both train & held) to guard the leakage the
        # #3488 review flagged; pairs spanning the two node-sets are dropped so the split is strictly disjoint.
        if args.split == "node-disjoint":
            nodes = sorted({r["node"] for r in fit_set} | {r["root"] for r in fit_set})
            random.Random(0).shuffle(nodes)
            hn = set(nodes[:int(args.held_frac * len(nodes))])
            held = [i for i, r in enumerate(fit_set) if r["node"] in hn and r["root"] in hn]
            train = [i for i, r in enumerate(fit_set) if r["node"] not in hn and r["root"] not in hn]
            print(f"\nsplit=node-disjoint: {len(train)} train / {len(held)} held "
                  f"({len(fit_set)-len(train)-len(held)} cross-split pairs dropped)")
        else:
            order = list(range(len(fit_set))); random.Random(0).shuffle(order)
            nh = int(args.held_frac * len(order)); held, train = order[:nh], order[nh:]
            print(f"\nsplit=random: {len(train)} train / {len(held)} held")
        Xtr = X[train]; rtr = [rels[i] for i in train]
        Xhe = X[held]; rhe = [rels[i] for i in held]

        joint = JointPosterior(relset, n_features=len(sources), hidden=args.hidden).fit(Xtr, rtr)
        jacc, jll, jece = _eval(joint.proba(Xhe), rhe, ri)

        def factored(weights):                             # product-of-experts on train, with per-source weights
            fac = MuPosterior(nbins=args.nbins)
            for s in sources:
                fac.fit_source(s, ((rtr[k], src_vals[s][i]) for k, i in enumerate(train)), weight=weights[s])
            fp = np.array([[fac.posterior({s: src_vals[s][i] for s in sources}).get(r, 1e-9) for r in relset]
                           for i in held])
            return _eval(fp, rhe, ri)

        eacc, ell, eece = factored({s: 1.0 for s in sources})           # naive equal-weight product
        sep = {s: post.separability(s)[0] for s in sources}             # corrected PoE: weight ∝ separability
        wacc, wll, wece = factored(sep)                                 # (down-weights weak/redundant sources)

        print(f"\nJOINT vs FACTORED on held-out ({len(held)} pairs, {len(relset)} relations; "
              f"majority baseline {max(Counter(rhe).values())/len(rhe):.1%}):")
        print(f"  factored PoE  (equal weights)   : acc {eacc*100:5.1f}%  log-loss {ell:.3f}  ECE {eece:.3f}")
        print(f"  factored PoE  (sep-weighted, #3357): acc {wacc*100:5.1f}%  log-loss {wll:.3f}  ECE {wece:.3f}")
        print(f"  joint head    ({'LR' if args.hidden==0 else f'MLP-{args.hidden}':6s})        : "
              f"acc {jacc*100:5.1f}%  log-loss {jll:.3f}  ECE {jece:.3f}")
        if args.hidden == 0:                              # show LR captured the asymmetry: fwd vs rev opposite signs
            W = joint.net.weight.detach().numpy()         # [C, K]
            for rel in ("subcategory", "element_of", "see_also", "assoc"):
                if rel in ri:
                    w = W[ri[rel]]
                    print(f"  LR weights[{rel:12s}] " + " ".join(f"{s}:{w[j]:+.2f}" for j, s in enumerate(sources)))

        # ── THE HONEST TEST: does the decorrelated 1/d source earn its keep in the calibrated joint head? ──
        # Same held-out split; refit the joint head WITH vs WITHOUT 1/d; compare acc / log-loss / ECE(bins=10)
        # and AURC gated by MARGIN (#3391), with a bootstrap 95% CI. If 1/d is redundant, the intervals overlap.
        if "dist" in sources:
            true_he = np.array([ri[r] for r in rhe])
            print(f"\n── ABLATION: 1/d contribution (JOINT {'LR' if args.hidden==0 else f'MLP-{args.hidden}'}, "
                  f"held-out {len(held)}, split={args.split}; AURC=margin gate, bins=10 ECE) ──")

            def fit_joint(sub):
                cols = [sources.index(s) for s in sub]
                pr = JointPosterior(relset, n_features=len(cols), hidden=args.hidden).fit(Xtr[:, cols], rtr).proba(Xhe[:, cols])
                acc, ll, ece = _eval(pr, rhe, ri)
                correct = (pr.argmax(1) == true_he).astype(float)
                a, lo, hi = aurc_boot(margin_conf(pr), correct, B=args.boot)
                return acc, ll, ece, a, lo, hi

            no_dist = [s for s in sources if s != "dist"]
            for label, sub in [("without 1/d", no_dist), ("with 1/d   ", list(sources))]:
                acc, ll, ece, a, lo, hi = fit_joint(sub)
                print(f"  {label}: acc {acc*100:5.1f}%  log-loss {ll:.3f}  ECE {ece:.3f}  "
                      f"AURC(margin) {a:.3f} [{lo:.3f}, {hi:.3f}]")
            print(f"  (dist separability {post.separability('dist')[0]:.3f}; "
                  f"1/d 'earns its keep' iff the with-1/d AURC CI sits below the without-1/d point estimate)")

    # the side-note rule: out-of-band tagged labels ⇒ review (these were rejected from the fit above)
    print(f"\nLABEL-ANOMALY review (out-of-band tagged labels → LLM/human): {len(flagged)}/{len(tagged)}")
    flagged = [(r, mu_e5[(r["node"], r["root"])]) for r in flagged]
    for r, mu in sorted(flagged, key=lambda x: x[1])[:15]:
        print(f"  μ_e5={mu:.3f}  [{r['rel']}]  {r['node']} ∈/~ {r['root']}")
    if args.out and flagged:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# node\troot\trelation\tmu_e5\treason=out_of_band\n")
            for r, mu in sorted(flagged, key=lambda x: x[1]):
                f.write(f"{r['node']}\t{r['root']}\t{r['rel']}\t{mu:.3f}\n")
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
