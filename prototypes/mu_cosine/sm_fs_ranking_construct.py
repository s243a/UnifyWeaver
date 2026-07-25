#!/usr/bin/env python3
"""SM-FS lineage-ranking negative constructor — PROTOCOL_sm_fs_lineage_ranking.md §2–4.

Pure constructor (prereg 0235521270c565a4…: constructor_implementation_authorized=true,
model_fitting_authorized=false). Consumes ONLY the certified lineage_fs_targets.tsv (byte-bound to
the frozen hashes) and this specification; no ledger, no reserve rows, no embeddings, no model
scores, no network. Every frozen gate blocks on mismatch rather than adapting:

  inventory 361/359/129,599/2,792/126,807 with hardness 1,819/7,814/117,174;
  fold structure cap 28 / 14 deepened prefixes / 82 blocks / maps 73-72-72-72-72 /
  blocks 16-16-16-17-17 / assignment sha b5439b1acdfb…

Targets: positives = the certified six-decimal ASCII values parsed once to binary64 (never
recomputed); non-ancestors = max(1/50, (17/20)^h · ℓ/|d_q|) computed as an exact reduced rational
then rounded once to binary64, with the reduced rational + float.hex() recorded per row.
Cross-root pairs record the unreachable marker and the 1/50 floor. Weights: per query, positives
share total mass 1/2 uniformly; non-ancestors share 1/2 across nonempty hard:medium:easy = 3:2:1
(renormalized; empty buckets stay empty), uniform within a bucket; recorded as exact fractions.

  python3 sm_fs_ranking_construct.py            # -> ~/mu_data/sm_fs_ranking_v1 (0700/0600)
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BUNDLE = os.path.expanduser("~/mu_data/sm_fs_v3")
FROZEN = {
    "manifest.json": "466adfcf2e7c5914dad27b548c3f009804fec12e633498e187b0b4df71a98d61",
    "ledger.json": "b45ff8ded88f2e5d5ded78664b2d381a017a5722464f956f6b44d660f5060b40",
    "lineage_fs_targets.tsv":
        "c3b298d5335ee901111f4985bbf5f7c5feb017c503a8c81db60dba1b947ac051",
}
COUNTS = {"queries": 361, "candidates": 359, "pairs": 129599, "positives": 2792,
          "nonancestors": 126807, "hard": 1819, "medium": 7814, "easy": 117174}
FOLD_SALT = b"sm-fs-lineage-ranking-fold-v1"
ASSIGN_SHA = "b5439b1acdfb5020ecb7fb4fad437fd183242408d09056e1ed7ec90d33335f37"
CAP = max(1, int(0.08 * 361))                     # 28
SCHEMA = "unifyweaver.sm-fs-lineage-ranking-bundle.v1"


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def canon(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def gate(cond, msg):
    if not cond:
        raise ValueError(f"FROZEN GATE: {msg}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/mu_data/sm_fs_ranking_v1"))
    a = ap.parse_args(argv)

    for name, want in FROZEN.items():
        got = sha_file(os.path.join(a.bundle, name))
        gate(got == want, f"{name} sha {got[:16]} != frozen {want[:16]}")

    # parse the certified targets — the ONLY data input
    rows = []
    for ln in open(os.path.join(a.bundle, "lineage_fs_targets.tsv"), encoding="utf-8"):
        if ln.startswith("#"):
            if ln[1:].strip().startswith("map_path"):
                cols = ln[1:].strip().split("\t")
            continue
        rows.append(dict(zip(cols, ln.rstrip("\n").split("\t"))))
    gate(len(rows) == COUNTS["positives"], f"positive rows {len(rows)}")

    anc = defaultdict(dict)                       # map_path -> {ancestor_path: (hop, ascii_target)}
    for r in rows:
        anc[r["map_path"]][r["ancestor_path"]] = (int(r["hop"]), r["target"])
    queries = sorted(anc)
    gate(len(queries) == COUNTS["queries"], f"queries {len(queries)}")
    dest = {q: next(p for p, (h, _) in anc[q].items() if h == 1) for q in queries}
    catalog = sorted({p for q in queries for p in anc[q]})
    gate(len(catalog) == COUNTS["candidates"], f"catalog {len(catalog)}")

    def parts(p):
        return p.split("/")

    pairs = []
    tallies = defaultdict(int)
    for q in queries:
        d = dest[q]
        dp = parts(d)
        for c in catalog:
            if c in anc[q]:
                hop, ascii_t = anc[q][c]
                cls = "positive_parent" if c == d else "positive_ancestor"
                t = float(ascii_t)                 # certified six-decimal ASCII, parsed once
                rec = {"query": q, "candidate": c, "class": cls, "hop": hop,
                       "target": t, "target_hex": t.hex(), "target_ascii": ascii_t}
            else:
                cp = parts(c)
                l = 0
                for x, y in zip(dp, cp):
                    if x != y:
                        break
                    l += 1
                if l == 0:
                    relation, h_rec = "cross_root", "unreachable"
                    frac = Fraction(1, 50)
                    hard = "easy"
                else:
                    h = len(dp) + len(cp) - 2 * l
                    if len(dp) == l:               # d is strict prefix of c (c deeper)
                        relation = "descendant"
                    elif len(dp) == len(cp) and l == len(dp) - 1:
                        relation = "sibling"
                    elif h <= 4:
                        relation = "near_branch"
                    else:
                        relation = "same_root_far"
                    frac = max(Fraction(1, 50),
                               Fraction(17, 20) ** h * Fraction(l, len(dp)))
                    h_rec = h
                    hard = "hard" if h <= 2 else ("medium" if h <= 4 else "easy")
                tallies[hard] += 1
                t = float(frac)
                rec = {"query": q, "candidate": c, "class": "structural_nonancestor",
                       "relation": relation, "distance": h_rec,
                       "lca_fraction": [l, len(dp)] if l else [0, len(dp)],
                       "target_rational": [frac.numerator, frac.denominator],
                       "target": t, "target_hex": t.hex(), "hardness": hard}
            pairs.append(rec)
    gate(len(pairs) == COUNTS["pairs"], f"pairs {len(pairs)}")
    n_pos = sum(1 for p in pairs if p["class"].startswith("positive"))
    gate(n_pos == COUNTS["positives"], f"positives {n_pos}")
    gate(len(pairs) - n_pos == COUNTS["nonancestors"], "nonancestors")
    for k in ("hard", "medium", "easy"):
        gate(tallies[k] == COUNTS[k], f"{k} {tallies[k]} != {COUNTS[k]}")
    print(f"inventory gate PASSED: {COUNTS}")

    # weights: exact fractions, per query
    by_q = defaultdict(list)
    for p in pairs:
        by_q[p["query"]].append(p)
    for q, ps in by_q.items():
        pos = [p for p in ps if p["class"].startswith("positive")]
        for p in pos:
            w = Fraction(1, 2) / len(pos)
            p["weight_rational"] = [w.numerator, w.denominator]
            p["weight"] = float(w)
        buckets = defaultdict(list)
        for p in ps:
            if p["class"] == "structural_nonancestor":
                buckets[p["hardness"]].append(p)
        ratio = {"hard": 3, "medium": 2, "easy": 1}
        denom = sum(ratio[b] for b in buckets)
        for b, members in buckets.items():
            mass = Fraction(1, 2) * Fraction(ratio[b], denom)
            for p in members:
                w = mass / len(members)
                p["weight_rational"] = [w.numerator, w.denominator]
                p["weight"] = float(w)

    # frozen five-fold split over adaptive destination-lineage blocks
    def block_of(dst, deepened):
        pp = parts(dst)
        k = 3
        while True:
            b = "/".join(pp[:k])
            if b not in deepened or k >= len(pp):
                return b
            k += 1
    deepened = set()
    while True:
        bmap = {q: block_of(dest[q], deepened) for q in queries}
        counts = defaultdict(int)
        for b in bmap.values():
            counts[b] += 1
        big = {b for b, c in counts.items() if c > CAP
               and any(len(parts(dest[q])) > len(parts(b)) for q in queries if bmap[q] == b)}
        new = big - deepened
        if not new:
            break
        deepened |= new
    blocks = sorted(set(bmap.values()))
    gate(len(deepened) == 14, f"deepened prefixes {len(deepened)}")
    gate(len(blocks) == 82, f"blocks {len(blocks)}")
    gate(max(counts.values()) <= 28 and min(counts.values()) >= 1, "block size range")

    def digest(b):
        return hashlib.sha256(FOLD_SALT + b"\x00" + b.encode()).hexdigest()
    order = sorted(blocks, key=lambda b: (-counts[b], digest(b), b.encode()))
    fold_maps, fold_blocks, assign = [0] * 5, [0] * 5, {}
    for b in order:
        f = min(range(5), key=lambda i: (fold_maps[i], i))
        assign[b] = f
        fold_maps[f] += counts[b]
        fold_blocks[f] += 1
    gate(fold_maps == [73, 72, 72, 72, 72], f"fold map counts {fold_maps}")
    gate(sorted(fold_blocks) == [16, 16, 16, 17, 17], f"fold block counts {fold_blocks}")
    lines = sorted(f"{b}\t{assign[b]}\t{counts[b]}" for b in blocks)
    got = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    gate(got == ASSIGN_SHA, f"assignment sha {got[:16]} != frozen {ASSIGN_SHA[:16]}")
    print(f"fold gate PASSED: maps {fold_maps}, blocks {fold_blocks}, sha {got[:16]}")
    for p in pairs:
        p["fold"] = assign[bmap[p["query"]]]

    os.makedirs(a.out_dir, mode=0o700, exist_ok=True)
    os.chmod(a.out_dir, 0o700)
    pairs_bytes = b"".join(canon(p) for p in pairs)
    fold_bytes = ("\n".join(lines) + "\n").encode()
    manifest = {
        "schema": SCHEMA, "source_bundle": a.bundle,
        "source_hashes": FROZEN, "counts": dict(COUNTS),
        "fold": {"cap": CAP, "deepened_prefixes": len(deepened), "blocks": len(blocks),
                 "fold_map_counts": fold_maps, "fold_block_counts": fold_blocks,
                 "assignment_sha256": got},
        "constructor_sha256": sha_file(os.path.abspath(__file__)),
        "outputs": {"pairs.jsonl": hashlib.sha256(pairs_bytes).hexdigest(),
                    "fold_assignment.tsv": hashlib.sha256(fold_bytes).hexdigest()},
        "prereg_schema": "unifyweaver.sm-fs-lineage-ranking-prereg.v1",
        "model_fitting_authorized": False,
    }
    for name, data in (("pairs.jsonl", pairs_bytes), ("fold_assignment.tsv", fold_bytes),
                       ("manifest.json", canon(manifest))):
        path = os.path.join(a.out_dir, name)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    print(f"bundle -> {a.out_dir} (pairs {len(pairs)}, manifest sha "
          f"{hashlib.sha256(canon(manifest)).hexdigest()[:16]})")


if __name__ == "__main__":
    main()
