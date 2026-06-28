#!/usr/bin/env python3
"""Augment the INFERRED TAIL with Haiku-estimated cell distributions (DESIGN §9/§14).

The inferred tail = the conf<1.0 rows of the fused neighbourhoods (`context/*_edges.tsv`) — the untagged
relations the anchored basis exists to serve. This script (a) EXTRACTS those pairs with readable
titles+context, (b) builds the §14 prompt for a Haiku subagent, (c) INGESTS the returned JSON into a cached
partition + E[μ] per pair (feeding `cell_sampler`). Haiku is the distribution SOURCE; the draw/closure is ours.

Spends NO budget itself — extraction + ingest are pure I/O; the Haiku call is a separate subagent step.

  python3 score_inferred_tail.py extract --out inferred_tail_pilot.tsv [--neighborhoods lti circuit]
  python3 score_inferred_tail.py prompt  --pairs inferred_tail_pilot.tsv [--batch 10]   # prints §14 prompt(s)
  python3 score_inferred_tail.py ingest  --pairs inferred_tail_pilot.tsv --responses haiku_resp.json \
                                         --out haiku_scored_tail.tsv
"""
import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CTX = os.path.join(ROOT, "context")

# the untagged relations §14 estimates (bridge = title-match IDENTITY heuristic, excluded — not a "which
# relation" question). conf<1.0 AND relation in this set = the genuinely-untagged tail.
TAIL_RELS = {"assoc", "subtopic", "element_of", "see_also", "super_category", "subcategory"}

NAMED_DIR = ["element_of", "subcategory", "subtopic", "super_category"]   # directional (mu_fwd + mu_rev)
NAMED_SYM = ["see_also", "assoc"]                                         # symmetric (single mu)
NAMED = NAMED_DIR + NAMED_SYM


def _load_nodes(prefix):
    m = {}
    p = os.path.join(CTX, prefix + "_nodes.tsv")
    if not os.path.exists(p):
        return m
    for ln in open(p, encoding="utf-8"):
        if ln.startswith("#"):
            continue
        c = ln.rstrip("\n").split("\t")
        if len(c) >= 4:
            m[c[0]] = (c[3], c[2])            # key -> (title, node_type)
    return m


def extract(neighborhoods, out):
    prefixes = []
    for p in sorted(glob.glob(os.path.join(CTX, "*_edges.tsv"))):
        base = os.path.basename(p)[:-len("_edges.tsv")]
        if neighborhoods and not any(base.startswith(n) for n in neighborhoods):
            continue
        prefixes.append(base)
    seen, rows = set(), []
    for base in prefixes:
        nodes = _load_nodes(base)
        for ln in open(os.path.join(CTX, base + "_edges.tsv"), encoding="utf-8"):
            if ln.startswith("#"):
                continue
            c = (ln.rstrip("\n").split("\t") + ["", "", "", "", ""])[:5]
            a, b, rel, conf, raw = c
            if not a or not b or rel not in TAIL_RELS:
                continue
            try:
                if float(conf) >= 1.0:
                    continue
            except ValueError:
                continue
            at, atype = nodes.get(a, (a.split(":", 1)[-1].replace("-", " "), "?"))
            bt, btype = nodes.get(b, (b.split(":", 1)[-1].replace("-", " "), "?"))
            key = (at.lower(), bt.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append((at, bt, rel, conf, base, atype, btype, raw))
    with open(out, "w", encoding="utf-8") as f:
        f.write("# node_title\troot_title\tcur_relation\tconf\tneighborhood\tnode_type\troot_type\traw\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    from collections import Counter
    print(f"extracted {len(rows)} inferred-tail pairs → {out}")
    print(f"  by relation: {dict(Counter(r[2] for r in rows))}")
    print(f"  by neighborhood: {dict(Counter(r[4] for r in rows))}")
    return rows


PROMPT_HEADER = """You estimate fuzzy DIRECTIONAL relationships between two concept nodes in a knowledge graph.
For each pair (NODE, ROOT) below, rate how NODE relates to ROOT.

RELATIONS — directional (give mu_fwd AND mu_rev):
  element_of      NODE is a member/instance/element of ROOT
  subcategory     NODE is a narrower category beneath ROOT (taxonomic child)
  subtopic        NODE is a subtopic within ROOT's subject
  super_category  NODE is a broader category that CONTAINS ROOT (taxonomic parent)
RELATIONS — symmetric (give a SINGLE mu):
  see_also        a lateral relation — NODE and ROOT directly reference each other (a deliberate cross-link)
  assoc           a lateral relation that is looser/weaker than see_also (incidental co-occurrence)
  Note: see_also and assoc are close and overlap. If laterally related but you can't cleanly tell which,
  put mass on BOTH (lean see_also for a deliberate link, assoc for a vague one). Don't agonize over it.

Two OPEN-WORLD slots:
  none     applies = probability NO listed relation holds (pair unrelated); its mu is 0
  unknown  applies = probability a REAL relation holds but is NOT listed; give its mu_fwd / mu_rev too

TWO RULES — read carefully:
  (1) mu_* are FUZZY-SET memberships in [0,1], NOT a distribution. Independent; need NOT sum to 1; their
      sum MAY EXCEED 1 when relations overlap (a pair can be element_of AND subtopic). DO NOT normalize.
  (2) the `applies` values over the LISTED relations need NOT sum to 1 and SHOULD sum to LESS than 1 when
      unsure — leftover mass belongs to none + unknown. DO NOT inflate the listed ones to reach 1.

Output STRICT JSON: a list, one object per pair IN ORDER, each:
  {"id": <int>, "element_of":{"mu_fwd":_,"mu_rev":_,"applies":_}, "subcategory":{...}, "subtopic":{...},
   "super_category":{...}, "see_also":{"mu":_,"applies":_}, "assoc":{"mu":_,"applies":_},
   "none":{"applies":_}, "unknown":{"mu_fwd":_,"mu_rev":_,"applies":_}}
BE TERSE: output COMPACT single-line JSON objects (no extra whitespace/newlines inside an object), no prose,
no explanation, do NOT restate these rules. Just the raw JSON array.

PAIRS:
"""


def build_prompts(pairs, batch):
    out = []
    for s in range(0, len(pairs), batch):
        chunk = pairs[s:s + batch]
        lines = [PROMPT_HEADER]
        for i, r in enumerate(chunk):
            ctx = f'  (NODE type={r[5]}, ROOT type={r[6]}' + (f', section="{r[7]}"' if r[7] else "") + ")"
            lines.append(f'{s + i}. NODE: "{r[0]}"   ROOT: "{r[1]}"\n{ctx}')
        out.append("\n".join(lines))
    return out


def _cell_partition(obj):
    """§14 → §12(5) closure: applies[named]++none++unknown, renormalised to a categorical that sums to 1.
    Returns (cells, probs, mus) where mus[i] is the per-cell μ (forward) for E[μ]; symmetric cells use mu."""
    cells, weights, mus = [], [], []
    for r in NAMED_DIR:
        e = obj.get(r, {})
        cells.append(r); weights.append(float(e.get("applies", 0))); mus.append(float(e.get("mu_fwd", 0)))
    for r in NAMED_SYM:
        e = obj.get(r, {})
        cells.append(r); weights.append(float(e.get("applies", 0))); mus.append(float(e.get("mu", 0)))
    cells.append("unknown"); u = obj.get("unknown", {})
    weights.append(float(u.get("applies", 0))); mus.append(float(u.get("mu_fwd", 0)))
    cells.append("none"); weights.append(float(obj.get("none", {}).get("applies", 0))); mus.append(0.0)
    tot = sum(weights) or 1.0
    probs = [w / tot for w in weights]
    return cells, probs, mus


def ingest(pairs_path, responses_path, out):
    pairs = [ln.rstrip("\n").split("\t") for ln in open(pairs_path, encoding="utf-8") if not ln.startswith("#")]
    raw = open(responses_path, encoding="utf-8").read()
    raw = raw.replace("```json", " ").replace("```", " ")          # strip any markdown fences
    # robustly consume one-or-more concatenated JSON arrays/objects, separated by any whitespace
    dec, objs, i = json.JSONDecoder(), [], 0
    while i < len(raw):
        while i < len(raw) and raw[i].isspace():
            i += 1
        if i >= len(raw):
            break
        val, i = dec.raw_decode(raw, i)
        objs.extend(val if isinstance(val, list) else [val])
    by_id = {int(o["id"]): o for o in objs if "id" in o}
    cells = NAMED_DIR + NAMED_SYM + ["unknown", "none"]
    with open(out, "w", encoding="utf-8") as f:
        f.write("# node\troot\tcur_rel\tneighborhood\t" + "\t".join(f"P[{c}]" for c in cells)
                + "\t" + "\t".join(f"mu[{c}]" for c in cells) + "\tE_mu_fwd\n")
        n = 0
        for i, p in enumerate(pairs):
            o = by_id.get(i)
            if not o:
                continue
            cs, probs, mus = _cell_partition(o)
            emu = sum(pr * mu for pr, mu in zip(probs, mus))
            f.write("\t".join([p[0], p[1], p[2], p[4]]) + "\t"
                    + "\t".join(f"{x:.3f}" for x in probs) + "\t"
                    + "\t".join(f"{x:.3f}" for x in mus) + f"\t{emu:.3f}\n")
            n += 1
    print(f"ingested {n}/{len(pairs)} pairs → {out}")


def flag(scored_path, pairs_path, none_min, out):
    """Escalation selector: every tail row IS a graph edge (conf<1.0 asserts a relation), so a high Haiku
    `P[none]` is a direct graph↔Haiku contradiction. Emit those rows (none >= none_min) as a pairs file to
    re-judge with a STRONGER model (Sonnet ¼-budget, Opus less) — the selective multi-source tie-break."""
    scored = {}
    sh = open(scored_path, encoding="utf-8").readline().strip().split("\t")
    ni = sh.index("P[none]")
    for r in (l.rstrip("\n").split("\t") for l in open(scored_path, encoding="utf-8") if not l.startswith("#")):
        scored[(r[0].lower(), r[1].lower())] = float(r[ni])
    kept = []
    for ln in open(pairs_path, encoding="utf-8"):
        if ln.startswith("#"):
            continue
        p = ln.rstrip("\n").split("\t")
        if scored.get((p[0].lower(), p[1].lower()), 0.0) >= none_min:
            kept.append(ln.rstrip("\n"))
    with open(out, "w", encoding="utf-8") as f:
        f.write("# node_title\troot_title\tcur_relation\tconf\tneighborhood\tnode_type\troot_type\traw\n")
        f.write("\n".join(kept) + ("\n" if kept else ""))
    print(f"flagged {len(kept)} sharp rows (Haiku P[none] >= {none_min}) → {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--out", required=True)
    e.add_argument("--neighborhoods", nargs="*", default=None)
    pr = sub.add_parser("prompt"); pr.add_argument("--pairs", required=True); pr.add_argument("--batch", type=int, default=10)
    ig = sub.add_parser("ingest"); ig.add_argument("--pairs", required=True)
    ig.add_argument("--responses", required=True); ig.add_argument("--out", required=True)
    fl = sub.add_parser("flag"); fl.add_argument("--scored", required=True); fl.add_argument("--pairs", required=True)
    fl.add_argument("--none-min", type=float, default=0.3); fl.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.neighborhoods, a.out)
    elif a.cmd == "flag":
        flag(a.scored, a.pairs, a.none_min, a.out)
    elif a.cmd == "prompt":
        pairs = [ln.rstrip("\n").split("\t") for ln in open(a.pairs, encoding="utf-8") if not ln.startswith("#")]
        for i, pr_text in enumerate(build_prompts(pairs, a.batch)):
            print(f"\n===== BATCH {i} =====\n{pr_text}")
    elif a.cmd == "ingest":
        ingest(a.pairs, a.responses, a.out)


if __name__ == "__main__":
    main()
