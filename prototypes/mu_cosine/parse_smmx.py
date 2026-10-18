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
  * explicit <relation source target>, or an INTRA-MAP `cloudmapref="."` (an empty list/chapter connector
       referring to an adjacent node by `element` GUID) → `assoc` (cross-link)
  * `element` GUID on a cloudmapref → resolve to the SPECIFIC target node (not the map root)
  * a child labelled "wiki"/"Wikipedia"/"enwiki", or any node with a direct en.wikipedia.org urllink
       → a `bridge` edge: the node ↔ the SAME concept in enwiki, a node of type `category` (Category:…) or
       `page`. Same concept, DIFFERENT node-type (mindmap_node ⟷ category/page), possibly different name —
       the cross-corpus join key. (Scarce in the maps; most live in the Pearltrees data.)

PRIVACY (scrub-everywhere): a topic labelled "private" drops itself AND its subtree; a private ROOT drops the
whole map. Dropped at parse time, never emitted — private data never reaches the public dataset. See
DESIGN_provenance_and_representation.md §Privacy.

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
from urllib.parse import unquote

SEE_ALSO = {"See Also", "Via Link", "Related"}
SUPER = {"Super Categories", "Super Category", "Navigate Up"}
STRUCTURAL = SEE_ALSO | SUPER | {""}            # "" = blank grouping/section node
WIKI_LABEL = re.compile(r"^\s*(wiki|wikipedia|enwiki)\s*$", re.I)
# NAVIGATION nodes in book/index maps: page markers (pg12, Pg6) and section numbers (1.1, 2.3, A.4) —
# reading-order scaffolding, NOT concepts. Tagged node_type="navigation"; <relation> chains between them
# are `sequence` (reading order), not membership. The real concepts hang off `See Also` nodes.
NAV_LABEL = re.compile(r"^([Pp][gG]\.?\s*\d+|\d+(\.\d+)+|[A-Za-z]\.\d+)$")
PEARL = re.compile(r"pearltrees\.com/[^/]+/([^/\"]+)/id(\d+)")
WIKI_URL = re.compile(r"en\.wikipedia\.org/wiki/(.+)$")
# PRIVACY (scrub-everywhere — see DESIGN_provenance_and_representation.md §Privacy + privacy.py): a node whose
# label contains the word "private" marks itself AND its whole subtree (children inherit); a private ROOT ⇒
# the whole map. Dropped at parse time so private data never reaches the public dataset; we LOG every scrub.
from privacy import is_private_title          # noqa: E402  (single source of truth for the privacy rule)


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


_GUID_CACHE = {}
def node_by_guid(smmx_path, guid):
    """Resolve a cloudmapref `element` GUID to the SPECIFIC node it targets in the linked map (not the
    root). Returns (key, title, slug, pid) or None."""
    if smmx_path not in _GUID_CACHE:
        d = {}
        try:
            for t in load_xml(smmx_path).findall(".//topic"):
                g = t.get("guid")
                if g:
                    sl, pid = slug_of(t)
                    d[g] = (_key_from(sl, label(t)), label(t), sl or "", pid or "")
        except Exception:
            pass
        _GUID_CACHE[smmx_path] = d
    return _GUID_CACHE[smmx_path].get(guid)


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

    # PRIVACY: seed = topics labelled "private"; inherit DOWN the tree (a private node takes its subtree).
    children_of = {}
    for i, p in parent.items():
        children_of.setdefault(p, []).append(i)
    private_ids, fr = set(), [i for i, t in topics.items() if is_private_title(label(t))]
    while fr:
        x = fr.pop()
        if x in private_ids:
            continue
        private_ids.add(x)
        fr.extend(children_of.get(x, []))
    root_id = next((i for i in topics if parent.get(i) in (None, "-1")), None)
    map_private = bool(root_id) and root_id in private_ids        # private root ⇒ the whole map is private

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

    nodes, edges, id2key = {}, [], {}
    for i, t in topics.items():
        if i in is_anchor or label(topics[i]) in STRUCTURAL or i in private_ids:
            continue                               # skip wiki-anchor holders, containers, and PRIVATE nodes
        k = key(t)
        sl, pid = slug_of(t)
        ntype = "navigation" if NAV_LABEL.match(label(t)) else "mindmap_node"
        nodes[k] = {"title": label(t), "slug": sl or "", "pid": pid or "",
                    "enwiki": enwiki.get(i, ""), "ntype": ntype}
        id2key[i] = k
        # hierarchy edge to effective (non-structural) parent
        pp, rel = eff_parent_and_rel(i)
        if pp in topics and pp not in is_anchor and pp not in private_ids:
            edges.append((key(topics[pp]), k, rel))

    # cross-map links (cloudmapref) — over ALL topics, since a blank "link-holder"/connector child often
    # carries the ref; attribute it to the holder if real, else to its effective (non-structural) parent.
    guid2t = {t.get("guid"): t for t in topics.values() if t.get("guid")}
    for i, t in topics.items():
        ref, el = cloud_of(t)
        if not ref:
            continue
        src_id = eff_parent_and_rel(i)[0] if (label(t) in STRUCTURAL or i in is_anchor) else i
        if src_id not in topics or src_id in is_anchor or src_id in private_ids:
            continue
        segs = [s for s in re.split(r"[\\/]+", ref) if s and s != "."]
        if not segs:
            # cloudmapref="." ⇒ INTRA-MAP reference to a specific node by `element` GUID. An empty
            # "connector" node doing this is often joining a node to an ADJACENT one (list/chapter
            # structure) ⇒ treat as an associative cross-link.
            tt = guid2t.get(el)
            if tt is None or tt.get("id") in private_ids:    # don't link to a PRIVATE intra-map target
                continue
            tsl, tpid = slug_of(tt)
            tkey, rel = key(tt), "assoc"
            nodes.setdefault(tkey, {"title": label(tt), "slug": tsl or "", "pid": tpid or "",
                                    "enwiki": "", "ntype": "mindmap_node"})
        else:
            # DIRECTION from the relative path: `../` UP ⇒ broader (`super_category`); subfolder DOWN ⇒
            # narrower (`subcategory`). NAME the target from the linked map — the SPECIFIC node if an
            # `element` GUID is given, else the map's ROOT (holder/parent nodes are usually unnamed).
            rel = "super_category" if ".." in segs else "subcategory"
            tgt_path = os.path.normpath(os.path.join(base_dir, ref.replace("\\", "/")))
            ident = None
            if not args.no_resolve:
                ident = (node_by_guid(tgt_path, el) if el else None) or root_identity(tgt_path)
            if ident:
                tkey, ttitle, tsl, tpid = ident
                nodes.setdefault(tkey, {"title": ttitle, "slug": tsl, "pid": tpid,
                                        "enwiki": "", "ntype": "mindmap_node"})
            else:
                fn = re.sub(r"\.smmx$", "", os.path.basename(ref.rstrip("/")))
                tkey = re.sub(r"[^a-z0-9]+", "_", fn.lower()).strip("_")
                nodes.setdefault(tkey, {"title": fn, "slug": "", "pid": "",
                                        "enwiki": "", "ntype": "mindmap_node"})
        edges.append((key(topics[src_id]), tkey, rel))

    # BRIDGE edges: a node tagged wiki/Wikipedia/enwiki ⇒ the SAME concept in enwiki, but a DIFFERENT
    # node-type (mindmap_node ⟷ category/page) and possibly a different name. These cross-corpus identity
    # edges are scarce in the maps (most enwiki links live in the Pearltrees data) and therefore valuable —
    # and the differing endpoint TYPES under one `bridge` relation are exactly the within-operator type
    # diversity that makes the node-type token informative (REPORT_nodetype.md).
    for i, w in enwiki.items():
        if i not in id2key or not w or i in private_ids:
            continue
        ww = unquote(w).split("#")[0]
        if ww.startswith("Category:"):
            ek, etype = ww[len("Category:"):], "category"
        else:
            ek, etype = ww, "page"
        ek = ek.replace(" ", "_").strip("_")
        if not ek:
            continue
        nodes.setdefault(ek, {"title": ek.replace("_", " "), "slug": "", "pid": "",
                              "enwiki": ek, "ntype": etype})
        edges.append((id2key[i], ek, "bridge"))

    # explicit <relation source target>: a chain between NAVIGATION nodes (page/section) is reading-order
    # `sequence` (book/course list structure); otherwise a genuine `assoc` cross-link.
    for r in root.findall(".//relation"):
        si, di = r.get("source"), r.get("target")
        s, d = topics.get(si), topics.get(di)
        if s is not None and d is not None and si not in private_ids and di not in private_ids:
            rel = "sequence" if (NAV_LABEL.match(label(s)) and NAV_LABEL.match(label(d))) else "assoc"
            edges.append((key(s), key(d), rel))

    # de-dup edges, drop self/empty
    seen, uniq = set(), []
    for a, b, rel in edges:
        if a and b and a != b and (a, b, rel) not in seen:
            seen.add((a, b, rel)); uniq.append((a, b, rel))

    from collections import Counter
    rc = Counter(r for _, _, r in uniq)
    ntc = Counter(n.get("ntype", "mindmap_node") for n in nodes.values())
    print(f"{os.path.basename(args.smmx)}: {len(nodes)} nodes {dict(ntc)}, {len(uniq)} edges {dict(rc)}")
    print(f"  bridges (mindmap↔enwiki): {rc.get('bridge', 0)}")
    if map_private:
        print(f"  PRIVACY: ROOT is marked private ⇒ WHOLE MAP scrubbed ({len(private_ids)} topics, nothing emitted)")
    elif private_ids:
        print(f"  PRIVACY scrubbed (dropped, never emitted): {len(private_ids)} topics (a node + its subtree)")

    if args.out_prefix:
        with open(args.out_prefix + "_nodes.tsv", "w", encoding="utf-8") as f:
            f.write("# node_key\tnode_type\ttitle\tpearltrees_slug\tpearltrees_id\tenwiki_alias\n")
            for k, n in sorted(nodes.items()):
                f.write(f"{k}\t{n.get('ntype','mindmap_node')}\t{n['title']}\t{n['slug']}\t{n['pid']}\t{n['enwiki']}\n")
        with open(args.out_prefix + "_edges.tsv", "w", encoding="utf-8") as f:
            f.write("# src_key\tdst_key\trelation  "
                    "(subtopic|subcategory|see_also|super_category|bridge|sequence|assoc)\n")
            for a, b, rel in uniq:
                f.write(f"{a}\t{b}\t{rel}\n")
        print(f"  wrote {args.out_prefix}_nodes.tsv + {args.out_prefix}_edges.tsv")
    else:
        for a, b, rel in uniq[:40]:
            print(f"    {a}  --{rel}-->  {b}")


if __name__ == "__main__":
    main()
