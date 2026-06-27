#!/usr/bin/env python3
"""Section categorizer — map a Pearltrees SECTION label to a graph RELATION, as a SEPARATE, re-runnable step
over the harvester's RAW capture. The harvester records section text by position id and makes NO relation
judgement; categorization happens here (and in categorize_sections.py / inline in the consumers), so we can
re-categorise cached harvests WITHOUT re-harvesting, and upgrade the matching method over time.

Recording the METHOD makes the categorisation itself PROVENANCE — it slots into the same judge axis as the
rest of the pipeline: `exact_phrase` is deterministic (graph-judge-like, confidence 1.0); `fuzzy` /
`llm_template` are probabilistic (typically <1.0 confidence) and a loose match could later carry a looser μ.

Shared by parse_pearltrees.py, fuse_corpus.py and categorize_sections.py so the rule lives in ONE place.
"""

# category → the graph relation it implies. `reference` (a wiki/encyclopedia reference LIST) is associative.
CATEGORY_RELATION = {
    "subcategory": "subcategory",
    "element_of": "element_of",
    "super_category": "super_category",
    "see_also": "see_also",
    "reference": "see_also",
}


def section_mode(text):
    """EXACT-PHRASE (case-insensitive substring) categoriser: a section label → category, or None when no
    rule fires (the caller then falls back to the structural default — see_also is NOT assumed)."""
    t = (text or "").lower()
    if "subcategor" in t:
        return "subcategory"                          # narrower category
    if "subtopic" in t:
        return "element_of"                           # element / membership
    if ((("super" in t or "broad" in t or "parent" in t) and ("categor" in t or "topic" in t))
            or "navigate up" in t):
        return "super_category"                       # broader / parent ("Broad Categories", "Super Topics")
    if "see also" in t or "via link" in t or "related" in t:
        return "see_also"                             # associative
    if "wiki" in t or "encyclopedia" in t or "reference" in t:
        return "reference"                            # a reference list (→ see_also)
    return None


# FUZZY (typo-robust, lexical): canonical section phrase → category, matched by edit-distance so misspelled
# headers (`Subtoipcs`, `More Subtoipcs`, `Subcatagories`) are still categorised instead of falling through to
# the structural default. This GROWS the labelled set + shrinks the inferred noise (REPORT_infer_blend.md).
# Catches TYPOS; an EMBEDDING method (synonyms/paraphrases) and an LLM method (the hard residual) are the next
# escalation layers — cheap → expensive, each catching what the previous misses.
import difflib
import re as _re

FUZZY_KEYS = [
    ("subcategories", "subcategory"), ("subtopics", "element_of"), ("more subtopics", "element_of"),
    ("super categories", "super_category"), ("navigate up", "super_category"),
    ("see also", "see_also"), ("related", "see_also"),
    ("references", "reference"), ("wiki encyclopedia references", "reference"),
]


def _norm(s):
    return " ".join(_re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


# parent-signal guard: a "Broad Categories" / "Super Topics" header is a BROADER (parent) grouping, but a bare
# content token ("categories", "topics") is lexically closer to the SUB* keys ("subcategories", "subtopics")
# than to "super categories" — so unguarded fuzzy flips the DIRECTION (super→sub). When a parent signal pairs
# with a structural noun, force super_category (the user's Pearltrees convention). Requires BOTH so a plain
# "Super Cool Resources" / "My Groups" is NOT swept up.
_PARENT_SIGNAL = _re.compile(r"\b(super|broad(?:er)?|parent|ancestor)\b")
_STRUCT_NOUN = _re.compile(r"(categor|topic|group|theme)")


# tag/qualifier headers: a section header often pairs a category TAG with a free-text qualifier, either across
# a DASH — in EITHER order, "A-E -- Subtopics", "IT - Subtopics" (qualifier first), "Subtopics - old" (tag
# first) — or in PARENTHESES, "Subtopics (old)", "Further reading (from wikipedia)". Split on a
# WHITESPACE-SURROUNDED dash (so alphabetical-range qualifiers "A-E"/"0-9"/"N-Z" are NOT split) OR a paren,
# then categorise WHICHEVER segment matches: the qualifier can't dilute the fuzzy ratio or override the tag,
# and a keyword INSIDE the parens ("from wikipedia") is still found. Real-harvest examples informed this.
_QUALIFIER = _re.compile(r"\s+(?:--+|[-–—])\s+|\s*[()]\s*")   # whitespace-surrounded dash, or a parenthesis


def _segments(text):
    parts = [s.strip() for s in _QUALIFIER.split(text or "") if s.strip()]
    return list(dict.fromkeys(parts + [text]))             # each segment (in order), then the whole label


def fuzzy_mode(text, threshold=0.78):
    """Best edit-distance match of a section label to a canonical keyword → (category, similarity), or
    (None, best) below threshold. Also tries the label's individual TOKENS so a typo'd keyword embedded in a
    longer header still matches."""
    t = _norm(text)
    if not t:
        return (None, 0.0)
    cands = [t] + t.split()
    if (_PARENT_SIGNAL.search(t) and _STRUCT_NOUN.search(t)) or "navigate up" in t:
        r = max(difflib.SequenceMatcher(None, c, "super categories").ratio() for c in cands)
        return ("super_category", max(r, threshold))   # parent-signal guard — never a sub* match
    best_cat, best = None, 0.0
    for key, cat in FUZZY_KEYS:
        r = max(difflib.SequenceMatcher(None, c, key).ratio() for c in cands)
        if r > best:
            best, best_cat = r, cat
    return (best_cat, best) if best >= threshold else (None, best)


# EMBEDDING (semantic) layer — the escalation after exact + fuzzy. Catches SYNONYMS / PARAPHRASES that share
# no edit-distance with a keyword (`Members`, `Narrower areas`, `Parent concepts`). Each category has a few
# canonical EXEMPLARS; a label is e5-encoded and cosine-matched (best per category). Calibration on the real
# harvest (REPORT_section_embedding.md) showed the post-fuzzy residual is mostly TOPICAL/junk sitting in a
# narrow cosine band where `see also` is a generic attractor — so this layer uses a CONSERVATIVE
# threshold + a 1st-vs-2nd MARGIN gate to reject that pack, and an embedding hit carries confidence = the
# cosine (a GRADED, <1.0 tier — a soft relation prior feeding the operator-noise model, not a hard label).
EMBED_EXEMPLARS = {
    "element_of":    ["members", "elements", "instances", "subtopics", "things filed here"],
    "subcategory":   ["subcategories", "narrower categories", "child categories", "more specific topics"],
    "super_category": ["super categories", "broader categories", "parent categories", "more general topics"],
    "see_also":      ["see also", "related topics", "cross references", "associated pages"],
    "reference":     ["references", "sources", "further reading", "bibliography", "external links"],
}


def embed_mode(text, encoder, threshold=0.88, margin=0.02):
    """Semantic match of a section label to the nearest category EXEMPLAR set → (category, cosine), or
    (None, best) when the top cosine is below `threshold` OR within `margin` of the 2nd category (ambiguous,
    likely a topical name). `encoder(texts)->[N,384]` unit-normed (see section_embed.e5_encoder); the label
    is the `query:`, exemplars are `passage:` (e5's asymmetric prefixes)."""
    import numpy as np
    t = _norm(text)
    if not t:
        return (None, 0.0)
    cats = list(EMBED_EXEMPLARS)
    ex = [e for c in cats for e in EMBED_EXEMPLARS[c]]
    vecs = np.asarray(encoder(["query: " + t] + ["passage: " + e for e in ex]))
    sims = vecs[1:] @ vecs[0]
    per, i = [], 0
    for c in cats:
        n = len(EMBED_EXEMPLARS[c])
        per.append((float(sims[i:i + n].max()), c)); i += n
    per.sort(reverse=True)
    (best, best_cat), second = per[0], (per[1][0] if len(per) > 1 else -1.0)
    return (best_cat, best) if (best >= threshold and best - second >= margin) else (None, best)


def categorize(text, method="exact_phrase", fuzzy_threshold=0.78, encoder=None,
               embed_threshold=0.88, embed_margin=0.02):
    """Section label → (category, method, confidence). Escalation ladder (cheap→expensive, each catching what
    the previous misses): `exact_phrase` literal match (conf 1.0) → `fuzzy` edit-distance (conf 1.0, audited)
    → `embedding` semantic match (conf = cosine, a GRADED <1.0 tier; needs `encoder`). A given `method`
    runs that rung and all cheaper ones."""
    if method not in ("exact_phrase", "fuzzy", "embedding"):
        raise ValueError(f"unknown categorization method: {method!r} (have: exact_phrase, fuzzy, embedding)")
    segs = _segments(text)                                 # each dash/paren segment (tag & qualifier), then full
    for s in segs:
        cat = section_mode(s)
        if cat:
            return (cat, "exact_phrase", 1.0)
    if method in ("fuzzy", "embedding"):
        for s in segs:
            fcat, r = fuzzy_mode(s, fuzzy_threshold)
            if fcat:
                return (fcat, "fuzzy", 1.0)
    if method == "embedding":
        if encoder is None:
            raise ValueError("method='embedding' needs an encoder (section_embed.e5_encoder())")
        for s in segs:
            ecat, sim = embed_mode(s, encoder, embed_threshold, embed_margin)
            if ecat:
                return (ecat, "embedding", round(float(sim), 3))
    return (None, method, 0.0)


def relation_for(text, method="exact_phrase", fuzzy_threshold=0.78, encoder=None,
                 embed_threshold=0.88, embed_margin=0.02):
    """Convenience: section label → (relation, method, confidence). `relation` is None when no rule fires."""
    cat, m, conf = categorize(text, method, fuzzy_threshold, encoder, embed_threshold, embed_margin)
    return (CATEGORY_RELATION.get(cat), m, conf)
