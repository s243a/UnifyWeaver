#!/usr/bin/env python3
"""Parse a SimpleMind mindmap (.smmx = zip → document/mindmap.xml) into a TYPED node/edge listing, applying
the map's relation semantics so the graph can feed the mu-attention pipeline (corpus=mindmap, judge=human)
AND so Haiku can read it. See SKILL_understand_smmx.md for the human/Haiku-readable version of these rules.

Relation semantics (the structural container retypes its descendants' edge to the node it hangs off):
  * plain hierarchy parent→child          → `subtopic`        (membership / narrower)
  * under  See Also / Via Link / Related  → `see_also`        (associative, weakly related)
  * under  Super Categories / Super Cat.  → `super_category`  (broader / parent category)
  * `cloudmapref` (relative path to another .smmx, when no container tags it) → DIRECTIONAL by the path:
       `../` UP to a parent folder → `super_category` (target map is a broader PARENT tree);
       DOWN into a subfolder       → `subcategory`    (target map is narrower / a child)
  * explicit <relation source target>     → `assoc`           (cross-link)
  * a child labelled "wiki"/"Wikipedia", or any node with a direct en.wikipedia.org urllink
       → NOT a node: it sets the enwiki ANCHOR of the node it is attached to (the join key into enwiki)

Node identity = Pearltrees slug (from a pearltrees urllink) if present, else the normalised title. Each node
also carries: title, pearltrees id, enwiki_alias.

    python3 parse_smmx.py path/to/Map.smmx --out-prefix cyber     # → cyber_nodes.tsv, cyber_edges.tsv
"""
import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

SEE_ALSO = {"See Also", "Via Link", "Related"}
SUPER = {"Super Categories", "Super Category", "Navigate Up"}
STRUCTURAL = SEE_ALSO | SUPER | {""}            # "" = blank grouping/section node
WIKI_LABEL = re.compile(r"^\s*(wiki|wikipedia)\s*$", re.I)
PEARL = re.compile(r"pearltrees\.com/[^/]+/([^/\"]+)/id(\d+)")
WIKI_URL = re.compile(r"en\.wikipedia\.org/wiki/(.+)$")


def load_xml(path):
    if path.endswith(".smmx") or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("mindmap.xml"))
            return ET.fromstring(z.read(name))
    return ET.parse(path).getroot()


def label(t):
    return (t.get("text") or "").replace("\\N", " ").strip()


def slug_of(t):
    for l in t.findall("link"):
        u = l.get("urllink") or ""
        m = PEARL.search(u)
        if m:
            return m.group(1), m.group(2)          # slug, pearltrees id
    return None, None


def wiki_of(t):
    for l in t.findall("link"):
        m = WIKI_URL.search(l.get("urllink") or "")
        if m:
            return m.group(1)
    return None


def cloud_of(t):
    for l in t.findall("link"):
        if l.get("cloudmapref"):
            return l.get("cloudmapref"), l.get("element")
    return None, None


def _key_from(slug, title):
    return slug or re.sub(r"[^a-z0-9]+", "_", (title or "").lower()).strip("_")


_ROOT_CACHE = {}
def root_identity(smmx_path):
    """Open a linked .smmx and read its ROOT node's identity — because super-category/parent (and many
    cloudmapref) holder nodes are UNNAMED in the source map; the real name + Pearltrees slug lives on the
    linked map's central theme. Returns (key, title, slug, pid) or None if the file can't be read."""
    if smmx_path in _ROOT_CACHE:
        return _ROOT_CACHE[smmx_path]
    out = None
    try:
        r = load_xml(smmx_path)
        ts = r.findall(".//topic")
        rt = next((t for t in ts if (t.get("parent") in (None, "-1"))), ts[0] if ts else None)
        if rt is not None:
            sl, pid = slug_of(rt)
            out = (_key_from(sl, label(rt)), label(rt), sl or "", pid or "")
    except Exception:
        out = None
    _ROOT_CACHE[smmx_path] = out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("smmx")
    ap.add_argument("--out-prefix", default=None, help="write <prefix>_nodes.tsv + <prefix>_edges.tsv")
    ap.add_argument("--no-resolve", action="store_true", help="don't open linked maps to name cloudmapref "
                    "targets (offline / single-file mode); fall back to the filename")
    args = ap.parse_args()
    base_dir = os.path.dirname(os.path.abspath(args.smmx))

    root = load_xml(args.smmx)
    topics = {t.get("id"): t for t in root.findall(".//topic")}
    parent = {i: t.get("parent") for i, t in topics.items()}

    def key(t):
        s, _ = slug_of(t)
        return s or re.sub(r"[^a-z0-9]+", "_", label(t).lower()).strip("_")

    # pass 1: classify; collect enwiki anchors (wiki-labelled children + direct wiki urllinks)
    enwiki = {}                                     # node-id → enwiki title
    is_anchor = set()                              # ids that are wiki-anchor holders (not real nodes)
    for i, t in topics.items():
        w = wiki_of(t)
        if WIKI_LABEL.match(label(t)) and w and parent[i] in topics:
            enwiki[parent[i]] = w                  # "wiki" child → anchor of its parent
            is_anchor.add(i)
        elif w:
            enwiki[i] = w                          # node with its own wiki urllink

    def is_struct(i):
        return i in topics and label(topics[i]) in STRUCTURAL

    def eff_parent_and_rel(i):
        """Walk up through structural containers → (effective-parent-id, relation)."""
        rel = "subtopic"
        p = parent.get(i)
        first = True
        while p in topics and is_struct(p):
            lab = label(topics[p])
            if first and lab in SEE_ALSO:
                rel = "see_also"
            elif first and lab in SUPER:
                rel = "super_category"
            if lab:                                # non-blank container fixes the relation (nearest wins)
                first = False
            p = parent.get(p)
        return p, rel

    nodes, edges = {}, []
    for i, t in topics.items():
        if i in is_anchor or label(topics[i]) in STRUCTURAL:
            continue                               # skip wiki-anchor holders and structural containers
        k = key(t)
        sl, pid = slug_of(t)
        nodes[k] = {"title": label(t), "slug": sl or "", "pid": pid or "",
                    "enwiki": enwiki.get(i, "")}
        # hierarchy edge to effective (non-structural) parent
        pp, rel = eff_parent_and_rel(i)
        if pp in topics and pp not in is_anchor:
            edges.append((key(topics[pp]), k, rel))

    # cross-map links (cloudmapref) — over ALL topics, since a blank "link-holder" child often carries the
    # ref; attribute it to the holder if it's a real node, else to its effective (non-structural) parent.
    for i, t in topics.items():
        ref, el = cloud_of(t)
        if not ref:
            continue
        if label(t) in STRUCTURAL or i in is_anchor:
            src_id, _ = eff_parent_and_rel(i)
        else:
            src_id = i
        if src_id in topics and src_id not in is_anchor:
            # DIRECTION from the relative path (when no container tags the relation): a ref UP to a parent
            # folder (`../`) ⇒ the target map is a broader PARENT (`super_category`); a ref DOWN into a
            # subfolder ⇒ the target is narrower (`subcategory`). Mirrors the directory-as-taxonomy layout.
            segs = [s for s in re.split(r"[\\/]+", ref) if s and s != "."]
            rel = "super_category" if ".." in segs else "subcategory"
            # NAME the target from the linked map's ROOT node (holder/parent nodes are usually unnamed);
            # fall back to the filename if the linked file can't be opened.
            ident = None if args.no_resolve else root_identity(
                os.path.normpath(os.path.join(base_dir, ref.replace("\\", "/"))))
            if ident:
                tkey, ttitle, tsl, tpid = ident
                nodes.setdefault(tkey, {"title": ttitle, "slug": tsl, "pid": tpid, "enwiki": ""})
            else:
                fn = re.sub(r"\.smmx$", "", os.path.basename(ref.rstrip("/")))
                tkey = re.sub(r"[^a-z0-9]+", "_", fn.lower()).strip("_")
                nodes.setdefault(tkey, {"title": fn, "slug": "", "pid": "", "enwiki": ""})
            edges.append((key(topics[src_id]), tkey, rel))

    # explicit <relation source target> → assoc
    for r in root.findall(".//relation"):
        s, d = topics.get(r.get("source")), topics.get(r.get("target"))
        if s is not None and d is not None:
            edges.append((key(s), key(d), "assoc"))

    # de-dup edges, drop self/empty
    seen, uniq = set(), []
    for a, b, rel in edges:
        if a and b and a != b and (a, b, rel) not in seen:
            seen.add((a, b, rel)); uniq.append((a, b, rel))

    from collections import Counter
    rc = Counter(r for _, _, r in uniq)
    print(f"{os.path.basename(args.smmx)}: {len(nodes)} nodes, {len(uniq)} edges  {dict(rc)}")
    print(f"  enwiki-anchored nodes: {sum(1 for n in nodes.values() if n['enwiki'])}")

    if args.out_prefix:
        with open(args.out_prefix + "_nodes.tsv", "w", encoding="utf-8") as f:
            f.write("# node_key\ttitle\tpearltrees_slug\tpearltrees_id\tenwiki_alias\n")
            for k, n in sorted(nodes.items()):
                f.write(f"{k}\t{n['title']}\t{n['slug']}\t{n['pid']}\t{n['enwiki']}\n")
        with open(args.out_prefix + "_edges.tsv", "w", encoding="utf-8") as f:
            f.write("# src_key\tdst_key\trelation  (subtopic|subcategory|see_also|super_category|assoc)\n")
            for a, b, rel in uniq:
                f.write(f"{a}\t{b}\t{rel}\n")
        print(f"  wrote {args.out_prefix}_nodes.tsv + {args.out_prefix}_edges.tsv")
    else:
        for a, b, rel in uniq[:40]:
            print(f"    {a}  --{rel}-->  {b}")


if __name__ == "__main__":
    main()
