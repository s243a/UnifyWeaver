#!/usr/bin/env python3
"""SHADOW: ROOT-NODE-TEXT RE-BASELINE (owner title-source directive, resolved form) — the
directive said map names should come from the root node, not the filename. Corpus scan
found root-node NOTES essentially absent (13/4385, all content excerpts), but root-node
TEXT differs substantively from the filename stem in 1514/4385 maps (e.g. 'USA' ->
'USA (United States of America)', but also stale 'Afghanistan' -> 'Central Asia').
This runner swaps QUERY titles (the .smmx side) to root-node text and re-scores the
frozen-e5 leaf-cosine baseline on all 5 folds. Candidates (dir leaves) unchanged —
directories carry no richer source. Reproduces the stem baseline in the same process
as a control."""
import json, os, re, sys, zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import sm_fs_ranking_pipeline as pl
import sm_fs_ranking_shadow as sh
from mu_attention import E5_REVISION, build_e5_tables

DROPBOX = "/mnt/c/Users/johnc/Dropbox/root"

def root_text(rel):
    p = os.path.join(DROPBOX, rel)
    try:
        with zipfile.ZipFile(p) as z:
            data = z.read(next(n for n in z.namelist() if n.endswith("mindmap.xml")))
        rt = next((el for el in ET.fromstring(data).iter("topic")
                   if el.get("parent") == "-1"), None)
        t = (rt.get("text") or "") if rt is not None else ""
        t = re.sub(r"\s+", " ", t.replace("\\N", " ")).strip()
        return t or None
    except Exception:
        return None

def mrr(titles, pairs, dest_of, held):
    qt, pt, ix = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                 model_revision=E5_REVISION)
    catalog = sorted({p["candidate"] for p in pairs})
    cv = np.stack([pt.numpy()[ix[c]] for c in catalog])
    ci = {c: j for j, c in enumerate(catalog)}
    out = {}
    for f in range(5):
        rrs = []
        for q in sorted(held[f]):
            cos = qt.numpy()[ix[q]] @ cv.T
            rrs.append(1.0 / pl.recompute_rank([float(v) for v in cos], catalog,
                                               ci[dest_of[q]]))
        out[f] = float(np.mean(rrs))
    return out

def run():
    pl.enforce_environment()
    titles = sh._titles()
    man, pairs, _ = pl.load_pairs_verified()
    dest_of = {p["query"]: p["candidate"] for p in pairs if p["class"] == "positive_parent"}
    held = defaultdict(lambda: defaultdict(dict))
    for p in pairs:
        held[p["fold"]][p["query"]][p["candidate"]] = p
    titles2, changed, missing = dict(titles), 0, 0
    for k in titles:
        if not k.endswith(".smmx"):
            continue
        t = root_text(k)
        if t is None:
            missing += 1
        elif t != titles[k]:
            titles2[k] = t; changed += 1
    base = mrr(titles, pairs, dest_of, held)
    root = mrr(titles2, pairs, dest_of, held)
    print(json.dumps({
        "stamp": "shadow-exploratory-tier1-not-decision-bearing",
        "queries_changed": changed, "root_unreadable": missing,
        "stem_baseline_by_fold": base, "stem_baseline": float(np.mean(list(base.values()))),
        "roottext_by_fold": root, "roottext": float(np.mean(list(root.values()))),
        "delta": float(np.mean(list(root.values())) - np.mean(list(base.values()))),
    }, indent=1), flush=True)

run()
