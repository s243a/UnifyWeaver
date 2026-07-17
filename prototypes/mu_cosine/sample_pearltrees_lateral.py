#!/usr/bin/env python3
"""Lateral strata (sib / cous / rand) for the Pearltrees labeling campaign, plus campaign merge.

The Codex principal-path sampler (sample_product_kalman_pearltrees_campaign.py) covers the
directional strata (principal_h1..h5) only.  The cheap-judge fusion needs S-channel variance, which
lives on LATERAL pairs (the enwiki campaign lesson: sib/cous rows are where S varies).  This script
samples them from the SAME principal-path forest (load_principal_paths — same privacy filtering,
same title table):

  pt_sib  — two children of a shared parent, neither an ancestor of the other
  pt_cous — shared grandparent, disjoint parents, not siblings, neither an ancestor
  pt_rand — random titled node pairs with none of the above relations

Determinism: pools are ranked by blake2b(seed:a:b) (the Codex candidate_rank idiom) with
round-robin over the shared parent/grandparent group (sib/cous) so one large folder cannot
dominate a stratum; no RNG state.

Title policy: the audited Pearltrees corrections (title_policies/product_kalman_pearltrees_titles
.json — 36 unambiguous spelling fixes, frozen 2026-07-09) are applied BY ENDPOINT ID to every
emitted title, here and (via --merge-*) to the principal rows.  Raw titles remain in the unified
pairs TSV alias columns; corrections never come from judge labels (the title-policy review rule).

Outputs the same judge-ready score-input schema as the principal sampler, PLUS a unified campaign
pairs TSV (principal + lateral, one row per pair with ids, audited titles, hop, stratum tag) that
downstream fusion/training loaders consume.

  python3 sample_pearltrees_lateral.py \
      --paths-jsonl ../../.local/data/api_tree_paths_v8.jsonl \
      --titles-tsv ../../.local/data/pearltrees_api/assembled_titles.tsv \
      --merge-pairs /tmp/mu_data/pt_campaign_pairs.tsv \
      --out-score-in /tmp/mu_data/pt_campaign_all_score_in.tsv \
      --out-pairs /tmp/mu_data/pt_campaign_all_pairs.tsv \
      --manifest /tmp/mu_data/pt_campaign_lateral_manifest.json
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_product_kalman_pearltrees_campaign import load_principal_paths, sha256_path

ROOT = os.path.dirname(os.path.abspath(__file__))
POLICY = os.path.join(ROOT, "title_policies", "product_kalman_pearltrees_titles.json")
SCORE_HEADER = "# node_title\troot_title\tcur_relation\tconf\tneighborhood\tnode_type\troot_type\traw\n"
NODE_TYPE = "pearltrees_collection"


def load_policy(path=POLICY):
    with open(path, encoding="utf-8") as f:
        policy = json.load(f)
    return {c["endpoint_id"]: c["audited_title"] for c in policy.get("corrections", [])}


def build_graph(forest):
    parents, children = defaultdict(set), defaultdict(set)
    for record in forest["records"]:
        keyed = tuple((record["account"], nid) for nid in record["path_ids"])
        for parent, child in zip(keyed, keyed[1:]):
            if parent != child:
                parents[child].add(parent)
                children[parent].add(child)
    return parents, children


def ancestors(parents, node, hmax=6):
    out, frontier = {}, {node}
    for hop in range(1, hmax + 1):
        nxt = set()
        for n in frontier:
            for p in parents.get(n, ()):
                if p not in out and p != node:
                    out[p] = hop
                    nxt.add(p)
        frontier = nxt
        if not frontier:
            break
    return out


def rank(seed, *parts):
    return hashlib.blake2b(":".join([str(seed), *map(str, parts)]).encode(), digest_size=8).hexdigest()


def titled(forest, node):
    return bool(forest["titles"].get(node[1]))


def sample_lateral(forest, parents, children, n_sib, n_cous, n_rand, seed, hmax):
    anc_cache = {}

    def anc(node):
        if node not in anc_cache:
            anc_cache[node] = ancestors(parents, node, hmax)
        return anc_cache[node]

    def related(a, b):
        return b in anc(a) or a in anc(b)

    def grouped_take(groups, want, kind):
        """Round-robin over hash-ordered groups of hash-ordered candidate pairs."""
        ordered = sorted(groups.items(), key=lambda kv: rank(seed, kind, kv[0]))
        pools = [sorted(v, key=lambda ab: rank(seed, kind, *ab)) for _, v in ordered]
        out, seen = [], set()
        while len(out) < want and any(pools):
            for pool in pools:
                while pool:
                    a, b = pool.pop(0)
                    key = tuple(sorted((a, b)))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((a, b))
                    break
                if len(out) >= want:
                    break
        return out, seen

    sib_groups = defaultdict(list)
    for parent, kids in children.items():
        kids = [k for k in kids if titled(forest, k)]
        for i, a in enumerate(sorted(kids)):
            for b in sorted(kids)[i + 1:]:
                if not related(a, b):
                    sib_groups[parent].append((a, b))
    sibs, sib_seen = grouped_take(sib_groups, n_sib, "sib")

    cous_groups = defaultdict(list)
    for gp, mids in children.items():
        mids = sorted(mids)
        for i, pa in enumerate(mids):
            for pb in mids[i + 1:]:
                for a in sorted(children.get(pa, ())):
                    for b in sorted(children.get(pb, ())):
                        if a == b or not titled(forest, a) or not titled(forest, b):
                            continue
                        if parents.get(a, set()) & parents.get(b, set()):
                            continue  # actually siblings through another shared parent
                        if related(a, b):
                            continue
                        cous_groups[gp].append((a, b))
    cous_pool_seen = sib_seen
    cous, cous_seen = grouped_take(
        {k: [p for p in v if tuple(sorted(p)) not in cous_pool_seen] for k, v in cous_groups.items()},
        n_cous,
        "cous",
    )

    nodes = sorted(n for n in forest["nodes"] if titled(forest, n))
    taken = sib_seen | cous_seen
    rand_pairs = []
    ordered = sorted(
        ((a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:] if a[0] == b[0]),
        key=lambda ab: rank(seed, "rand", *ab),
    ) if len(nodes) <= 400 else None
    if ordered is None:
        # large node set: hash-walk candidate pairs instead of materializing all O(n²)
        ordered = []
        idx = sorted(range(len(nodes)), key=lambda i: rank(seed, "rw", nodes[i]))
        for step, i in enumerate(idx):
            j = idx[(step * 2654435761 + 1) % len(idx)]
            if i != j and nodes[i][0] == nodes[j][0]:
                ordered.append(tuple(sorted((nodes[i], nodes[j]))))
        ordered = sorted(set(ordered), key=lambda ab: rank(seed, "rand", *ab))
    for a, b in ordered:
        if len(rand_pairs) >= n_rand:
            break
        key = tuple(sorted((a, b)))
        if key in taken or related(a, b):
            continue
        if parents.get(a, set()) & parents.get(b, set()):
            continue
        ga = {g for p in parents.get(a, ()) for g in parents.get(p, ())}
        gb = {g for p in parents.get(b, ()) for g in parents.get(p, ())}
        if ga & gb:
            continue
        taken.add(key)
        rand_pairs.append((a, b))
    return {"pt_sib": sibs, "pt_cous": cous, "pt_rand": rand_pairs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths-jsonl", required=True)
    ap.add_argument("--titles-tsv", required=True)
    ap.add_argument("--sib", type=int, default=150)
    ap.add_argument("--cous", type=int, default=150)
    ap.add_argument("--rand", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hmax", type=int, default=6)
    ap.add_argument("--merge-pairs", help="principal pairs TSV from the Codex sampler to merge in")
    ap.add_argument("--out-score-in", required=True)
    ap.add_argument("--out-pairs", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--overlap", type=int, default=300,
                    help="size of the RANDOM 5.5 calibration overlap drawn from the unified rows; "
                         "0 disables. Emitted as <out-score-in base>_overlap.tsv + manifest entry")
    ap.add_argument("--overlap-seed", type=int, default=0)
    a = ap.parse_args(argv)

    forest = load_principal_paths(a.paths_jsonl, a.titles_tsv)
    parents, children = build_graph(forest)
    corrections = load_policy()

    def title_of(node_id):
        return corrections.get(node_id, forest["titles"].get(node_id, ""))

    laterals = sample_lateral(forest, parents, children, a.sib, a.cous, a.rand, a.seed, a.hmax)

    unified = []  # (pair_id, account, a_id, a_title, b_id, b_title, hop, tag)
    corrected_count = 0
    if a.merge_pairs:
        with open(a.merge_pairs, encoding="utf-8") as f:
            header = f.readline().lstrip("#").strip().split("\t")
            col = {c: i for i, c in enumerate(header)}
            for ln in f:
                c = ln.rstrip("\n").split("\t")
                if len(c) < len(header):
                    continue
                d_id, an_id = c[col["descendant_id"]], c[col["ancestor_id"]]
                d_t, an_t = title_of(d_id), title_of(an_id)
                corrected_count += int(d_t != c[col["descendant_title"]]) + int(an_t != c[col["ancestor_title"]])
                unified.append((c[col["pair_id"]], c[col["account"]], d_id, d_t, an_id, an_t,
                                int(c[col["hop"]]), f"principal_h{c[col['hop']]}"))
    for tag, pairs_list in laterals.items():
        for i, (na, nb) in enumerate(pairs_list):
            unified.append((f"pearltrees-{tag}-{i:04d}", na[0], na[1], title_of(na[1]),
                            nb[1], title_of(nb[1]), -1, tag))

    # duplicate (title, title) keys would collide in the scored-TSV join — drop later duplicates
    seen_keys, rows = set(), []
    dropped_dup = 0
    for row in unified:
        key = (row[3], row[5])
        if key in seen_keys or not row[3] or not row[5] or row[3] == row[5]:
            dropped_dup += 1
            continue
        seen_keys.add(key)
        rows.append(row)

    os.makedirs(os.path.dirname(a.out_pairs), exist_ok=True)
    with open(a.out_pairs, "w", encoding="utf-8") as f:
        f.write("# pair_id\taccount\ta_id\ta_title\tb_id\tb_title\thop\ttag\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")
    with open(a.out_score_in, "w", encoding="utf-8") as f:
        f.write(SCORE_HEADER)
        for _, _, _, a_t, _, b_t, hop, tag in rows:
            rel = "subtopic" if tag.startswith("principal") else "assoc"
            f.write(f"{a_t}\t{b_t}\t{rel}\t1.0\t{tag}\t{NODE_TYPE}\t{NODE_TYPE}\t\n")

    overlap_manifest = None
    if a.overlap:
        # deterministic RANDOM overlap (never conflict-selected — the covariance-fit rule); indices
        # over the unified score-in data rows, reproducing the original ad-hoc draw exactly
        import numpy as np

        idx_sel = sorted(np.random.default_rng(a.overlap_seed)
                         .choice(len(rows), size=min(a.overlap, len(rows)), replace=False).tolist())
        overlap_path = a.out_score_in.rsplit(".tsv", 1)[0] + "_overlap.tsv"
        with open(overlap_path, "w", encoding="utf-8") as f:
            f.write(SCORE_HEADER)
            for i in idx_sel:
                _, _, _, a_t, _, b_t, hop, tag = rows[i]
                rel = "subtopic" if tag.startswith("principal") else "assoc"
                f.write(f"{a_t}\t{b_t}\t{rel}\t1.0\t{tag}\t{NODE_TYPE}\t{NODE_TYPE}\t\n")
        overlap_manifest = {
            "path": os.path.basename(overlap_path),
            "size": len(idx_sel),
            "seed": a.overlap_seed,
            "row_indices_sha256": hashlib.sha256(
                ",".join(map(str, idx_sel)).encode()).hexdigest(),
            "file_sha256": sha256_path(overlap_path),
        }
        print(f"overlap ({len(idx_sel)} rows, seed {a.overlap_seed}) -> {overlap_path}")

    counts = {}
    for row in rows:
        counts[row[7]] = counts.get(row[7], 0) + 1
    manifest = {
        "corpus": "pearltrees",
        "seed": a.seed,
        "hmax": a.hmax,
        "strata": counts,
        "dropped_duplicate_or_empty_title_keys": dropped_dup,
        "title_policy": os.path.basename(POLICY),
        "title_corrections_applied": corrected_count + sum(
            1 for row in rows if row[7].startswith("pt_") and (
                row[3] != forest["titles"].get(row[2], row[3]) or row[5] != forest["titles"].get(row[4], row[5]))),
        "inputs_sha256": {
            "paths_jsonl": sha256_path(a.paths_jsonl),
            "titles_tsv": sha256_path(a.titles_tsv),
            "title_policy": sha256_path(POLICY),
        },
        "note": "lateral pools drawn from the SAME privacy-filtered principal-path forest as the "
                "Codex principal sampler; pt_rand additionally excludes anc/sib/cous relations",
        "overlap": overlap_manifest,
    }
    with open(a.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f"strata: {counts}; dropped {dropped_dup} dup/empty-title rows")
    print(f"pairs -> {a.out_pairs}\nscore input -> {a.out_score_in}\nmanifest -> {a.manifest}")


if __name__ == "__main__":
    main()
