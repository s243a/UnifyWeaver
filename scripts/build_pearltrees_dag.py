#!/usr/bin/env python3
"""build_pearltrees_dag.py — assemble the multi-parent Pearltrees DAG from ALL local sources, and emit the
harvest-queue augmentation for the trees still missing (the RDF-export truncation gap).

Background: the Pearltrees RDF export silently truncates at ~24MB / ~5004 trees, so the API-only harvest looked
93% incomplete. Unioning the RDF exports (both accounts) + API trees + api_tree_paths + reports recovers most of it.
See project_pearltrees_rdf_export_bug (memory) and prototypes/mu_cosine/DESIGN_path_operator.md.

Outputs (all under .local/data/pearltrees_api/, gitignored):
  - assembled_dag.tsv        child<TAB>parent   (the full multi-parent DAG)
  - assembled_titles.tsv     id<TAB>title       (RDF + API, incl. titles for dangling children from RefPearls)
  - harvest_queue_augmented.json   {count, maps:[...]}  — existing queue UNION the still-missing referenced trees,
      in the same schema infer_pearltrees_federated.py / the browser-automation harvester consume. REVIEW then
      swap over harvest_queue.json to have the next harvest run fetch the gap.

Re-run after a fresh (chunked) RDF re-export or an API backfill to fold the new data in. Read-only on all inputs.
"""
import json, re, glob, os, sys
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(BASE, ".local/data/pearltrees_api")
OUT_DAG = os.path.join(API_DIR, "assembled_dag.tsv")
OUT_TITLES = os.path.join(API_DIR, "assembled_titles.tsv")
OUT_QUEUE = os.path.join(API_DIR, "harvest_queue_augmented.json")
EXISTING_QUEUE = os.path.join(API_DIR, "harvest_queue.json")

NS = {'rdf': "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
      'rdfs': "http://www.w3.org/2000/01/rdf-schema#",
      'dcterms': "http://purl.org/dc/elements/1.1/",
      'pt': "http://www.pearltrees.com/rdf/0.1/#"}
def q(p, t): return "{%s}%s" % (NS[p], t)
ID_RE = re.compile(r'/id(\d+)')

def id_from_uri(uri):
    m = ID_RE.search(uri) if uri else None
    return m.group(1) if m else None

def acct_from_uri(uri):
    m = re.search(r'pearltrees\.com/([^/]+)/', uri) if uri else None
    if not m:
        return None
    return 'groups' if m.group(1) == 't' else m.group(1)

edges = defaultdict(set)             # child -> {parents}
edge_source = defaultdict(set)       # (child,parent) -> {sources}
node_account = {}                    # id -> account
title_of = {}                        # id -> title
def add_edge(child, parent, src):
    if child and parent and child != parent:
        edges[child].add(parent); edge_source[(child, parent)].add(src)

RDF_FILES = [(os.path.join(BASE, "context/PT/pearltrees_export_s243a_2026-01-02.rdf"), "s243a"),
             (os.path.join(BASE, "context/PT/pearltrees_export_s243a_grous_2026-01-02.rdf"), "groups")]
rdf_defined = defaultdict(set); rdf_refd = defaultdict(set)

def title_text(elem):
    t = elem.find(q('dcterms', 'title'))
    return t.text.strip() if t is not None and t.text else None

for path, acct in RDF_FILES:
    if not os.path.exists(path):
        print("WARN: missing RDF", path); continue
    root = ET.parse(path).getroot()
    for tr in root.iter(q('pt', 'Tree')):
        tid = id_from_uri(tr.get(q('rdf', 'about')))
        if tid:
            rdf_defined[path].add(tid)
            node_account.setdefault(tid, acct_from_uri(tr.get(q('rdf', 'about'))) or acct)
            tt = title_text(tr)
            if tt:
                title_of[tid] = tt
    for tag, src in (('RefPearl', 'rdf'), ('AliasPearl', 'rdf_alias')):
        for rp in root.iter(q('pt', tag)):
            par_el = rp.find(q('pt', 'parentTree')); see = rp.find(q('rdfs', 'seeAlso'))
            if par_el is None or see is None:
                continue
            parent = id_from_uri(par_el.get(q('rdf', 'resource')))
            child = id_from_uri(see.get(q('rdf', 'resource')))
            if child:
                if src == 'rdf':
                    rdf_refd[path].add(child)
                tt = title_text(rp)                       # RefPearl title = the referenced tree's display title
                if tt:
                    title_of.setdefault(child, tt)
                node_account.setdefault(child, acct_from_uri(see.get(q('rdf', 'resource'))) or acct)
            add_edge(child, parent, src)

# API trees dir (contentTree.id child <- pearl.treeId parent) + titles
tree_files = glob.glob(os.path.join(API_DIR, "trees/*.json")) + glob.glob(os.path.join(API_DIR, "*_tree.json"))
for tf in tree_files:
    try:
        d = json.load(open(tf, encoding="utf-8"))
    except Exception:
        continue
    def walk(o):
        if isinstance(o, dict):
            t = o.get("api_response", {}).get("tree") if "api_response" in o else None
            if isinstance(t, dict) and t.get("id") and t.get("title"):
                title_of.setdefault(str(t["id"]), t["title"])
            if isinstance(o.get("pearls"), list):
                for p in o["pearls"]:
                    if not isinstance(p, dict):
                        continue
                    parent = str(p["treeId"]) if p.get("treeId") else None
                    ct = p.get("contentTree")
                    if isinstance(ct, dict) and ct.get("id"):
                        child = str(ct["id"]); add_edge(child, parent, 'api')
                        node_account.setdefault(child, 's243a')
                        if ct.get("title"):
                            title_of.setdefault(child, ct["title"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)

# api_tree_paths jsonl (parent_tree_id + path_ids chains) + reports
for jf in glob.glob(os.path.join(BASE, ".local/data/api_tree_paths_v*.jsonl")):
    for line in open(jf, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        tid = str(r["tree_id"]) if r.get("tree_id") else None
        if tid and r.get("account"):
            node_account.setdefault(tid, r["account"])
        if tid and r.get("parent_tree_id"):
            add_edge(tid, str(r["parent_tree_id"]), 'paths')
        clean = [str(x) for x in (r.get("path_ids") or []) if not str(x).startswith('account:')]
        for i in range(1, len(clean)):
            add_edge(clean[i], clean[i-1], 'paths')

# write DAG + titles
os.makedirs(API_DIR, exist_ok=True)
pairs = sorted((c, p) for c, ps in edges.items() for p in ps)
with open(OUT_DAG, "w", encoding="utf-8") as f:
    for c, p in pairs:
        f.write("%s\t%s\n" % (c, p))
with open(OUT_TITLES, "w", encoding="utf-8") as f:
    for k, v in sorted(title_of.items()):
        f.write("%s\t%s\n" % (k, v))

# structural-only view (no alias) → the missing referenced TREES (the harvest worklist)
struct = defaultdict(set)
for (c, p), srcs in edge_source.items():
    if any(s != 'rdf_alias' for s in srcs):
        struct[c].add(p)
snodes = set(struct) | {p for ps in struct.values() for p in ps}
defined_all = set().union(*rdf_defined.values()) if rdf_defined else set()
api_ids = set(os.path.basename(x)[:-5] for x in glob.glob(os.path.join(API_DIR, "trees/*.json")))
have = defined_all | api_ids
missing = sorted(snodes - have, key=lambda x: (len(x), x))

# augment the existing harvest queue with the missing trees
existing = {"count": 0, "maps": []}
if os.path.exists(EXISTING_QUEUE):
    existing = json.load(open(EXISTING_QUEUE))
seen = {str(m.get("tree_id")) for m in existing.get("maps", [])}
maps = list(existing.get("maps", []))
added = 0
for tid in missing:
    if tid in seen:
        continue
    acct = node_account.get(tid, "s243a")
    seg = "t/s243a" if acct == "groups" else acct       # groups (team spaces) live under /t/s243a/
    maps.append({"tree_id": tid, "title": title_of.get(tid, ""), "account": acct,
                 "uri": "https://www.pearltrees.com/%s/id%s" % (seg, tid),
                 "queued_by": "build_pearltrees_dag", "reason": "missing_tree_content"})
    added += 1
json.dump({"count": len(maps), "maps": maps}, open(OUT_QUEUE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

alln = set(edges) | {p for ps in edges.values() for p in ps}
multi = sum(1 for c, ps in edges.items() if len(ps) > 1)
print("=== ASSEMBLED DAG ===")
print("  nodes %d | edges %d | multi-parent %d | titles %d (%.0f%%)"
      % (len(alln), len(pairs), multi, len(title_of), 100*sum(1 for n in alln if n in title_of)/max(1, len(alln))))
print("  DAG    -> %s" % OUT_DAG)
print("  titles -> %s" % OUT_TITLES)
print("=== HARVEST WORKLIST ===")
print("  structural nodes %d | defined/harvested %d | MISSING trees %d" % (len(snodes), len(have), len(missing)))
print("  queue was %d, added %d missing -> %d total" % (len(existing.get("maps", [])), added, len(maps)))
print("  augmented queue -> %s   (REVIEW, then swap over harvest_queue.json)" % OUT_QUEUE)
